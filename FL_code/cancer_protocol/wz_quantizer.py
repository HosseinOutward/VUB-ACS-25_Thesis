from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Literal, NamedTuple, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from FL_code.FL_core.cache import cache_call, model_training_hash
from FL_code.FL_core.utils import create_training_progress_bar

from .brent_wz_models import EncoderDecoderLayeredRNN
from .prior_code import DedupedPriorCalculator, PriorCalculator


OUTLIER_CHUNK = 2**8
# Below this point, exact grouping costs more than the avoided RNN work.
TRAINING_DEDUPLICATION_MIN_FRACTION = 0.1


class OutlierMetadata(NamedTuple):
    positions: tuple[np.ndarray, ...]
    scale: np.ndarray | None
    signs: np.ndarray


class PreprocessMetadata(NamedTuple):
    norm_factors: torch.Tensor
    outliers: OutlierMetadata


def normalization_params(values: torch.Tensor) -> tuple[float, float]:
    """Estimate a robust center and scale from a seeded quantile-trimmed sample.

    Seeded so encoder and decoder derive bitwise-identical factors from the same values.
    Factors are rounded to float16 because that is the precision the decoder receives
    them in; the encoder must normalize with the exact same values.
    """
    sample_size = min(200_000, values.numel())
    generator = torch.Generator(device=values.device).manual_seed(0)

    def random_sample() -> torch.Tensor:
        return values[torch.randint(
            values.numel(), (sample_size,), device=values.device, generator=generator)].float()

    centers: list[float] = []
    for _ in range(5):
        sample = random_sample()
        q02, q98 = torch.quantile(sample, torch.tensor([0.02, 0.98], device=sample.device))
        centers.append(sample[(sample >= q02) & (sample <= q98)].mean().item())
    center = float(np.float16(np.mean(centers)))

    scales: list[float] = []
    for _ in range(5):
        centered = random_sample() - center
        q01, q99 = torch.quantile(centered, torch.tensor([0.01, 0.99], device=values.device))
        scales.append(centered.clamp(q01, q99).square().mean().sqrt().item())
    scale = float(np.float16(np.mean(scales)))
    assert scale != 0
    return scale, center


def outlier_metadata(values: torch.Tensor, threshold: float) -> tuple[torch.Tensor, OutlierMetadata]:
    """Move values beyond threshold into a separately coded tail representation."""
    positions = torch.nonzero(values.abs() > threshold, as_tuple=False).flatten()
    if positions.numel() == 0:
        return values, OutlierMetadata((), None, np.array([], dtype=np.bool_))

    outliers = values[positions]
    tail = outliers.abs() - threshold
    # Rounded to float16 first because the decoder rescales with the float16 metadata value.
    scale = float(np.float16(torch.quantile(tail.float(), 0.99).item() / threshold))
    assert scale != 0
    values = values.clone()
    values[positions] = tail * outliers.sign() / scale
    positions_np = positions.cpu().numpy()
    return values, OutlierMetadata(
        tuple(
            (positions_np[(positions_np >= start) & (positions_np < start + OUTLIER_CHUNK)] - start).astype(np.uint8)
            for start in range(0, values.numel(), OUTLIER_CHUNK)
        ),
        np.array(scale, dtype=np.float16),
        (outliers > 0).cpu().numpy().astype(np.bool_),
    )


def new_rnn_model(
    num_planes: int,
    bins_per_plane: int,
    side_info_size: int,
    marginal: bool,
    hidden_dim: int,
    layers: int,
    shared_heads: bool = True,
) -> EncoderDecoderLayeredRNN:
    """Create the RNN network shared by the WZ quantizer and the conditional prior model."""
    return EncoderDecoderLayeredRNN(
        num_planes=num_planes,
        bins_per_plane=bins_per_plane,
        side_info_size=max(1, side_info_size),
        input_dim=1,
        layers=layers,
        hidden_dim=hidden_dim,
        marginal=marginal,
        shared_encoder=shared_heads,
        shared_decoder=shared_heads,
    )


def batch_loop(
    func: Callable[[int, int], torch.Tensor],
    model: EncoderDecoderLayeredRNN,
    input_size: int,
    batch_size: int,
    cat_dim: int | None = None,
) -> torch.Tensor:
    """Run an inference callback over contiguous batches on the best device and concatenate the outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.inference_mode():
        batches = [func(start, min(start + batch_size, input_size)) for start in range(0, input_size, batch_size)]
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(batches, dim=cat_dim if cat_dim is not None else (1 if batches[0].ndim == 3 else 0))


def _sample_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Average per-sample values while preserving duplicate multiplicities when supplied."""
    return values.mean() if sample_weights is None else (values * sample_weights).sum() / sample_weights.sum()


def float_prefix(values: torch.Tensor, match_bits: int) -> torch.Tensor:
    """Return float32 values truncated to a shared leading-bit prefix."""
    assert values.dtype == torch.float32
    assert 1 <= match_bits <= 32
    return (
        values if match_bits == 32
        else (values.view(torch.int32) & -(1 << (32 - match_bits))).view(torch.float32)
    )


def wz_model_training_loop(
    compute_loss: Callable[..., torch.Tensor],
    model: EncoderDecoderLayeredRNN,
    parameters: Iterator[Parameter],
    training_inputs: tuple[torch.Tensor, ...],
    side_info: torch.Tensor,
    c_cfg: WZcfgQuant,
    batch_size: int = 50_000,
    label: str = "Training",
    metrics_fn: Callable[[EncoderDecoderLayeredRNN, tuple[torch.Tensor, ...], torch.Tensor], str] | None = None,
    **loss_fn_options,
) -> float:
    """Train one model, adaptively collapsing duplicate rows and preserving their loss mass."""
    model.to('cuda').train()
    training_inputs = tuple(value.to('cuda') for value in training_inputs)
    assert training_inputs
    assert all(value.shape[0] == training_inputs[0].shape[0] for value in training_inputs)
    side_info = side_info.to('cuda')

    input_size = training_inputs[0].shape[0]
    total_samples = min(c_cfg.train_sample_size, input_size)
    batch_draws = min(batch_size, total_samples)
    expected_index_duplicate_fraction = 1 - (
        input_size * (1 - (1 - 1 / input_size) ** batch_draws) / batch_draws
        if input_size > 1 else 1
    )
    row_groups = None
    if c_cfg.deduplicate_training:
        probe_size = min(batch_size, input_size)
        probe_indices = (
            torch.arange(probe_size, device=side_info.device) * input_size // probe_size
        )
        probe_rows = torch.cat([
            value[probe_indices].flatten(start_dim=1)
            for value in (*training_inputs, side_info)
        ], dim=1)
        probe_rows = float_prefix(probe_rows, c_cfg.training_match_bits)
        content_duplicate_fraction = (
            1 - torch.unique(probe_rows, dim=0).shape[0] / probe_rows.shape[0]
        )
        if content_duplicate_fraction >= TRAINING_DEDUPLICATION_MIN_FRACTION:
            rows = torch.cat([
                value.flatten(start_dim=1) for value in (*training_inputs, side_info)
            ], dim=1)
            rows = float_prefix(rows, c_cfg.training_match_bits)
            row_groups = torch.unique(rows, dim=0, return_inverse=True)[1]
            del rows
        elif expected_index_duplicate_fraction >= TRAINING_DEDUPLICATION_MIN_FRACTION:
            row_groups = torch.arange(input_size, device=side_info.device)
        del probe_rows

    # Both training losses are scaled by 1/K relative to their previous forms.
    optimizer_epsilon = 1e-8 / model.num_planes
    optimizer = torch.optim.Adam(
        parameters,
        fused=c_cfg.fused_optimizer,
        lr=c_cfg.lr,
        weight_decay=0,
        eps=optimizer_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c_cfg.lr_step, gamma=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=c_cfg.mixed_precision)

    total_batches = (total_samples + batch_size - 1) // batch_size
    pbar = create_training_progress_bar(
        c_cfg.train_epochs * total_batches,
        desc=label,
        disable=not c_cfg.training_progress_bar,)

    for epoch in range(c_cfg.train_epochs):
        epoch_started = perf_counter()
        indices = torch.randint(input_size, (total_samples,), device=training_inputs[0].device)
        epoch_loss = 0.0
        epoch_model_samples = 0
        for start in range(0, total_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            sample_weights = None
            if row_groups is not None:
                _, inverse, counts = torch.unique(
                    row_groups[batch_indices], return_inverse=True, return_counts=True
                )
                if counts.numel() < batch_indices.numel():
                    representatives = torch.full_like(counts, batch_indices.numel())
                    representatives.scatter_reduce_(
                        0,
                        inverse,
                        torch.arange(batch_indices.numel(), device=inverse.device),
                        reduce="amin",
                    )
                    batch_indices = batch_indices[representatives]
                    sample_weights = counts.to(training_inputs[0].dtype)
            input_batch = tuple(value[batch_indices] for value in training_inputs)
            si_batch = side_info[batch_indices]
            epoch_model_samples += batch_indices.numel()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=c_cfg.mixed_precision):
                loss = compute_loss(
                    model, input_batch, si_batch, epoch, c_cfg,
                    sample_weights=sample_weights, **loss_fn_options
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if c_cfg.training_progress_bar:
                pbar.set_postfix({"loss": f"{loss.item():.2f}"})
                pbar.update(1)
        scheduler.step()
        epoch_loss /= total_batches
        if c_cfg.training_log_every_epochs and (
            (epoch + 1) % c_cfg.training_log_every_epochs == 0 or epoch + 1 == c_cfg.train_epochs
        ):
            epoch_seconds = perf_counter() - epoch_started
            metrics = ""
            if metrics_fn is not None:
                model.eval()
                with torch.inference_mode():
                    metrics = f", {metrics_fn(model, training_inputs, side_info)}"
                model.train()
            print(
                f"{label} epoch {epoch + 1}/{c_cfg.train_epochs}: "
                f"loss={epoch_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.3e}, "
                f"seconds={epoch_seconds:.2f}, samples_per_second={total_samples / epoch_seconds:.0f}, "
                f"model_samples={epoch_model_samples}/{total_samples}{metrics}",
                flush=True,
            )

    pbar.close()
    model.cpu()
    torch.cuda.empty_cache()
    return epoch_loss
    

class WZcfgQuant(BaseModel):
    """Quantizer and training configuration consumed by WZQuantizerCancer."""
    # best config for FL delta (laplace like with b=0.356 but sharp peak around 0)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    bins_per_plane: int
    num_planes: int
    norm_slices: Sequence[slice] | None = None
    outlier_threshold: float | None = None
    marginal_loss: bool = False
    max_side_info_count: int = 5
    pretrain_pth_dir: Path = Path("FL_code/data/pre_trained_pth")

    train_epochs: int = 120
    reconst_ld: float = 36.73683767756248
    train_sample_size: int = 750_000
    train_batch_size: int = 12_500
    lr: float = 0.004530430161538338
    lr_step: int = 37
    tau: float = 2.3406302935091077
    quantizer_train_repeats: int = 2
    prior_train_repeats: int = 2

    training_progress_bar: bool = False
    training_log_every_epochs: int | None = None
    tf32: bool = False
    fused_optimizer: bool = True
    mixed_precision: bool = False
    deduplicate_training: bool = True
    training_match_bits: int = Field(default=26, ge=1, le=32)
    source_perturbation: Literal["none", "draw_discard", "add"] = "none"
    shared_heads: bool = True
    prior_context: Literal["hard", "soft"] = "hard"
    prior_probabilities: Literal["categorical", "gumbel"] = "categorical"
    hidden_dim: int = 74
    layers: int = 6
    cache_dir: Path | None = None
    cache_overwrite: bool = False


class WZQuantizerCancer:
    """Learned Wyner-Ziv quantizer for one flattened Cancer protocol update vector."""

    prior_calculator: ClassVar[type[PriorCalculator]] = PriorCalculator

    @property
    def device(self) -> torch.device:
        """Return the accelerator used by quantizer inference when available."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, name: str) -> Any:
        """Read missing quantizer attributes from c_cfg when they are configuration fields."""
        c_cfg = self.__dict__.get("c_cfg")
        if c_cfg is not None and name in type(c_cfg).model_fields:
            return getattr(c_cfg, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent instance attributes from silently shadowing c_cfg fields."""
        c_cfg = self.__dict__.get("c_cfg")
        assert name == "c_cfg" or c_cfg is None or name not in type(c_cfg).model_fields, (
            f"{name!r} is a WZcfgQuant field; update c_cfg instead of shadowing it on the quantizer.")
        super().__setattr__(name, value)

    def __init__(
        self,
        c_cfg: WZcfgQuant,
        si_size: int,
    ) -> None:
        self.c_cfg: WZcfgQuant = c_cfg
        self.no_side_info: bool = si_size == 0
        assert self.marginal_loss or not self.no_side_info, (
            "si_size=0 requires marginal_loss=True; no-SI quantizers must be explicit.")
        
        self.side_info_size: int = max(si_size, 1)
        
        self.coding_model: EncoderDecoderLayeredRNN = new_rnn_model(
            num_planes=self.num_planes,
            bins_per_plane=self.bins_per_plane,
            side_info_size=self.side_info_size,
            marginal=self.marginal_loss,
            shared_heads=self.shared_heads,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
        )

        self.side_info_list_used: list[torch.Tensor] | None = None
        self.vector_size: int | None = None
        self.training_prior_stage_rates: tuple[float, ...] | None = None
        self.training_prior_rate: float | None = None

    @staticmethod
    def training_metrics(
        model: EncoderDecoderLayeredRNN,
        training_inputs: tuple[torch.Tensor, ...],
        side_info: torch.Tensor,
    ) -> str:
        """Summarize hard-code stage distortion and cumulative rate on a fixed training sample."""
        x = training_inputs[0][:10_000]
        y = side_info[:x.shape[0]]
        bins, codes = model.encode(x)
        reconstructions = model.decode(codes, y)
        priors = model.get_priors(codes, y)
        sample_indices = torch.arange(x.shape[0], device=x.device)
        plane_rates = [
            -torch.log2(prior[sample_indices, symbols] + 1e-12).mean().item()
            for prior, symbols in zip(priors, bins, strict=True)
        ]
        return (
            f"hard_mse={[round(F.mse_loss(reconstruction, x).item(), 6) for reconstruction in reconstructions]}, "
            f"cumulative_rate={[round(float(rate), 4) for rate in np.cumsum(plane_rates)]}"
        )

    @staticmethod
    def compute_loss(
        model: EncoderDecoderLayeredRNN,
        training_input: tuple[torch.Tensor],
        si_batch: torch.Tensor,
        epoch: int,
        c_cfg: WZcfgQuant,
        sample_weights: torch.Tensor | None,
        wmspe_denom: float,
        perturb_generator: torch.Generator,
    ) -> torch.Tensor:
        """Compute the stage-averaged reconstruction-plus-rate loss for one training batch."""
        x_batch, = training_input
        if c_cfg.source_perturbation != "none":
            perturbation = torch.randn(
                x_batch.shape, device=x_batch.device, dtype=x_batch.dtype, generator=perturb_generator
            ) * (1e-5 * x_batch.abs().mean())
            if c_cfg.source_perturbation == "add":
                x_batch = x_batch + perturbation
        progress = epoch / c_cfg.train_epochs
        tau = c_cfg.tau * np.exp(progress * np.log(0.1 / c_cfg.tau))
        reconstructions, bins, soft_codes, priors = model(x_batch, si_batch, tau=tau)

        loss = torch.zeros((), device=x_batch.device)
        sample_indices = torch.arange(x_batch.shape[0], device=x_batch.device)
        for plane in range(model.num_planes):
            distortion = _sample_mean(
                F.mse_loss(
                    reconstructions[plane], x_batch, reduction="none"
                ).flatten(start_dim=1).mean(dim=1),
                sample_weights,
            )

            posterior_prob = soft_codes[plane][sample_indices, bins[plane]]
            prior_prob = priors[plane][sample_indices, bins[plane]]
            rate = _sample_mean(
                torch.log((posterior_prob + 1e-12) / (prior_prob + 1e-12)),
                sample_weights,
            )

            new_loss = (c_cfg.reconst_ld / wmspe_denom) * distortion + rate
            loss = loss + new_loss / model.num_planes
        return loss

    def train_model(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[torch.Tensor] | None
    ) -> None:
        """Train repeated quantizer attempts and keep the finite attempt with the lowest loss."""
        if self.cache_dir is None:
            self._train_model_uncached(x_raw, si_raw_list)
            return

        state_dict, training_prior_stage_rates = cache_call(
            _train_quantizer_parameters,
            self.cache_dir / "quantizers",
            model_training_hash,
            self.cache_overwrite,
            # Inference-only subclasses share this training implementation and
            # restore the resulting state into their own instance below.
            quantizer_class=WZQuantizerCancer,
            c_cfg=self.c_cfg,
            x_raw=x_raw,
            side_information=tuple(si_raw_list or ()),
        )
        # Cache only portable training results; bind the current decoder-available
        # side information to this fresh quantizer instance.
        assert self.side_info_list_used is None
        if self.no_side_info:
            assert not si_raw_list
            self.side_info_list_used = []
        else:
            assert si_raw_list
            self.set_side_information(si_raw_list)
        self.vector_size = x_raw.numel()
        self.coding_model.load_state_dict(state_dict)
        self.training_prior_stage_rates = training_prior_stage_rates
        self.training_prior_rate = sum(training_prior_stage_rates)

    def retrain_model(
        self,
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
        wmspe_denom: float,
    ) -> float:
        """Retrain from the current model and restore the cached final model when available."""
        state_dict, loss = cache_call(
            _retrain_quantizer_parameters,
            None if self.cache_dir is None else self.cache_dir / "quantizers",
            model_training_hash,
            self.cache_overwrite,
            quantizer_class=WZQuantizerCancer,
            c_cfg=c_cfg,
            initial_state=self.coding_model.state_dict(),
            x_prep=x_prep,
            side_info=side_info,
            wmspe_denom=wmspe_denom,
        )
        # The cache boundary is the completed retraining process, not its
        # optimizer steps or failed attempts.
        self.coding_model.load_state_dict(state_dict)
        return loss

    def _train_model_uncached(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[torch.Tensor] | None,
    ) -> None:
        assert self.side_info_list_used is None
        if self.no_side_info:
            assert not si_raw_list, "Marginal quantizer training expects no side information."
            self.side_info_list_used = []
        else:
            assert si_raw_list
            self.set_side_information(si_raw_list)
            
        x_prep, metadata = self.preprocess_x(x_raw)
        wmspe_denom = (x_prep.float().square().mean().item() / 2) + 1e-8
        side_info = self.side_info_tensor(metadata)

        self.coding_model = self._train_finite_candidate_and_retrieve_best(x_prep, side_info, wmspe_denom)

        bins, _ = self._encode_preprocessed(x_prep, 500_000)
        temp = self.prior_calculator.compute_prior_from_network(
                self.coding_model, bins, side_info).to(torch.float16)
        self.training_prior_stage_rates = self.prior_calculator.compute_plane_rates(
            temp, bins, self.c_cfg.num_planes)
        self.training_prior_rate = sum(self.training_prior_stage_rates)

    def _train_finite_candidate_and_retrieve_best(
        self, x_prep: torch.Tensor,
        side_info: torch.Tensor, wmspe_denom: float, 
        failure_tolerance: int = 2
    ) -> EncoderDecoderLayeredRNN:
        made_quants_and_stat: list[tuple[EncoderDecoderLayeredRNN, float]] = []
        for attempt_index in range(self.quantizer_train_repeats + failure_tolerance):
            if len(made_quants_and_stat) == self.quantizer_train_repeats:
                break

            model = self.coding_model if attempt_index == 0 else new_rnn_model(
                num_planes=self.num_planes,
                bins_per_plane=self.bins_per_plane,
                side_info_size=side_info.shape[1],
                marginal=self.marginal_loss,
                hidden_dim=self.hidden_dim,
                layers=self.layers,
                shared_heads=self.shared_heads,
            )
            perturb_generator = torch.Generator(device="cuda").manual_seed(torch.initial_seed())
            loss = wz_model_training_loop(
                self.compute_loss, model, model.parameters(),
                (x_prep,), side_info, self.c_cfg, batch_size=self.train_batch_size,
                label=f"Quantizer attempt {attempt_index + 1}",
                metrics_fn=self.training_metrics,
                wmspe_denom=wmspe_denom,
                perturb_generator=perturb_generator)

            if np.isfinite(loss):
                def reconstruct_batch(start: int, end: int) -> torch.Tensor:
                    _, codes = model.encode(x_prep[start:end])
                    return model.decode(codes, side_info[start:end])[-1].cpu()

                reconstruction = batch_loop(
                    reconstruct_batch, model, x_prep.shape[0], 500_000).reshape(-1)
                if torch.isfinite(reconstruction).all():
                    attempt = (model, loss)
                    made_quants_and_stat.append(attempt)
        assert len(made_quants_and_stat) == self.quantizer_train_repeats, "Too many failed training attempts."

        return min(made_quants_and_stat, key=lambda attempt: attempt[1])[0]

    def encoding_process(
        self, x_raw: torch.Tensor, batch_size: int = 500_000,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PreprocessMetadata]:
        """Preprocess and quantize one raw vector into per-plane bin indices and soft code posteriors.

        Estimator subclasses that bypass the neural encoder return ``None`` soft codes.
        """
        x_prep, metadata = self.preprocess_x(x_raw)
        bins, soft_codes = self._encode_preprocessed(x_prep, batch_size)
        return bins, soft_codes, metadata

    def _encode_preprocessed(self, x_prep: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode normalized model inputs into per-plane bin indices and soft code posteriors."""
        soft_code_batches: list[torch.Tensor] = []

        def encode_batch(start: int, end: int) -> torch.Tensor:
            bins, soft_codes = self.coding_model.encode(x_prep[start:end], force_softmax=True)
            soft_code_batches.append(torch.stack(soft_codes).cpu())
            return torch.stack(bins).cpu()
        bins = batch_loop(encode_batch, self.coding_model, x_prep.shape[0], batch_size, cat_dim=1)
        dtype = torch.uint8 if self.bins_per_plane <= 2**8 else torch.uint16
        return bins.to(dtype), torch.cat(soft_code_batches, dim=1)

    def decoding_process(
        self,
        payload_content: tuple[torch.Tensor, PreprocessMetadata],
        batch_size: int = 500_000,
    ) -> torch.Tensor:
        """Decode bin indices with side information and invert preprocessing metadata."""
        bins, metadata = payload_content
        assert self.vector_size is not None
        assert bins.shape == (self.num_planes, self.vector_size)
        assert bins.max().item() < self.bins_per_plane
        side_info = self.side_info_tensor(metadata)
        recons = self._decode_preprocessed(bins, side_info, batch_size)
        return self.postprocess(recons, metadata)

    def decoding_stages_process(
        self,
        payload_content: tuple[torch.Tensor, PreprocessMetadata],
        batch_size: int = 500_000,
    ) -> torch.Tensor:
        """Decode and postprocess the reconstruction produced by every refinement stage."""
        bins, metadata = payload_content
        assert self.vector_size is not None
        assert bins.shape == (self.num_planes, self.vector_size)
        side_info = self.side_info_tensor(metadata)
        normalized = self._decode_stages_preprocessed(bins, side_info, batch_size)
        return torch.stack([self.postprocess(stage, metadata) for stage in normalized])

    def _decode_stages_preprocessed(
        self,
        bins: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode every normalized reconstruction stage from bins and side information."""
        def decode_batch(start: int, end: int) -> torch.Tensor:
            codes = [
                F.one_hot(plane.long().to(self.device), num_classes=self.bins_per_plane).float()
                for plane in bins[:, start:end]
            ]
            return torch.stack(self.coding_model.decode(codes, side_info[start:end])).cpu()

        return batch_loop(
            decode_batch, self.coding_model, bins.shape[1], batch_size, cat_dim=1
        ).squeeze(-1)

    def _decode_preprocessed(
        self,
        bins: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode the final normalized reconstruction stage."""
        return self._decode_stages_preprocessed(bins, side_info, batch_size)[-1]

    def preprocess_x(
        self,
        x_raw: torch.Tensor,
        skip_outliers: bool = False,
        normalization_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Normalize a raw vector per slice, optionally extract outliers, and shape it for the model."""
        if self.vector_size is None:
            self.vector_size = x_raw.numel()
        assert self.vector_size == x_raw.numel(), f"Expected vector size {self.vector_size}, got {x_raw.numel()}."

        x_prep = x_raw.clone()
        assert normalization_factors is None or normalization_factors.shape == (len(self.norm_slices), 2)
        norm_factors: list[tuple[float, float]] = []
        for index, vector_slice in enumerate(self.norm_slices):
            scale, center = (
                normalization_params(x_prep[vector_slice])
                if normalization_factors is None
                else map(float, normalization_factors[index])
            )
            x_prep[vector_slice] = (x_prep[vector_slice] - center) / scale
            norm_factors.append((scale, center))

        outliers = OutlierMetadata((), None, np.array([], dtype=np.bool_))
        if self.outlier_threshold is not None and not skip_outliers:
            x_prep, outliers = outlier_metadata(x_prep, self.outlier_threshold)

        metadata = PreprocessMetadata(torch.tensor(norm_factors, dtype=torch.float16), outliers)
        return x_prep.to(self.device).unsqueeze(1).to(torch.float32).contiguous(), metadata

    def postprocess(self, recons_raw: torch.Tensor, metadata: PreprocessMetadata) -> torch.Tensor:
        """Invert outlier handling and per-slice normalization."""
        recons = recons_raw.clone()
        if metadata.outliers.positions:
            positions = torch.from_numpy(np.concatenate([
                chunk.astype(np.int64) + start
                for chunk, start in zip(
                    metadata.outliers.positions,
                    range(0, recons.numel(), OUTLIER_CHUNK),
                    strict=True,
                )
            ])).to(recons.device)
            signs = torch.from_numpy(metadata.outliers.signs).to(recons.device, torch.float32) * 2 - 1
            assert signs.numel() == positions.numel()
            assert metadata.outliers.scale is not None and self.outlier_threshold is not None
            outlier_scale = float(metadata.outliers.scale)
            recons[positions] = (recons[positions].abs() * outlier_scale + self.outlier_threshold) * signs

        for vector_slice, (scale, center) in zip(self.norm_slices, metadata.norm_factors, strict=True):
            recons[vector_slice] = recons[vector_slice] * scale + center
        return recons

    def side_info_tensor(self, source_metadata: PreprocessMetadata) -> torch.Tensor:
        """Normalize and stack the quantizer's stored side information."""
        assert self.side_info_list_used is not None, "Side information must be set before formatting."
        return self.format_side_info_tensor(source_metadata, self.side_info_list_used)

    def format_side_info_tensor(
        self,
        source_metadata: PreprocessMetadata,
        side_information: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize and stack a supplied SI list without reading or changing stored SI."""
        assert self.vector_size is not None, "Vector size must be known before side information is formatted."
        assert all(value.numel() == self.vector_size for value in side_information), (
            f"Each side-information vector must contain {self.vector_size} values.")

        tensors = [
            self.preprocess_x(
                si,
                skip_outliers=True,
                normalization_factors=source_metadata.norm_factors,
            )[0].squeeze(1)
            for si in side_information
        ]
        if len(tensors) == 0:
            tensors = [torch.zeros(self.vector_size, device=self.device)]

        return torch.stack(tensors, dim=1).to(
            self.device, dtype=torch.float32).contiguous()

    def set_side_information(self, side_info: Sequence[torch.Tensor]) -> None:
        """Set the decoder-available side-information vectors used by decoding and conditional priors."""
        assert not self.no_side_info, "A no-side-information quantizer cannot accept side information."
        assert len(side_info) == self.side_info_size, (
            f"Quantizer requires {self.side_info_size} side-information vectors, got {len(side_info)}.")
        assert self.vector_size is None or all(value.numel() == self.vector_size for value in side_info), (
            f"Each side-information vector must contain {self.vector_size} values.")
        assert self.side_info_list_used is None, "Side information can only be set once per quantizer instance."
        self.side_info_list_used = list(side_info)


def _train_quantizer_parameters(
    quantizer_class: type[WZQuantizerCancer],
    c_cfg: WZcfgQuant,
    x_raw: torch.Tensor,
    side_information: tuple[torch.Tensor, ...],
) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
    quantizer = quantizer_class(c_cfg=c_cfg, si_size=len(side_information))
    quantizer._train_model_uncached(x_raw, side_information)
    assert quantizer.training_prior_stage_rates is not None
    return quantizer.coding_model.state_dict(), quantizer.training_prior_stage_rates


def _retrain_quantizer_parameters(
    quantizer_class: type[WZQuantizerCancer],
    c_cfg: WZcfgQuant,
    initial_state: dict[str, torch.Tensor],
    x_prep: torch.Tensor,
    side_info: torch.Tensor,
    wmspe_denom: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run sampled retraining and return only its final portable result."""
    quantizer = quantizer_class(c_cfg=c_cfg, si_size=side_info.shape[1])
    quantizer.coding_model.load_state_dict(initial_state)
    perturb_generator = torch.Generator(device="cuda").manual_seed(torch.initial_seed())
    loss = wz_model_training_loop(
        quantizer.compute_loss,
        quantizer.coding_model,
        quantizer.coding_model.parameters(),
        (x_prep,),
        side_info,
        c_cfg,
        batch_size=c_cfg.train_batch_size,
        label="Sample retraining",
        metrics_fn=quantizer.training_metrics,
        wmspe_denom=wmspe_denom,
        perturb_generator=perturb_generator,
    )
    return quantizer.coding_model.state_dict(), loss


def unique_row_groups(
    bins: torch.Tensor,
    side_info: torch.Tensor,
    match_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group rows by exact bins and float-prefix side information."""
    device = side_info.device
    bins_device = bins.to(device, torch.long)
    features = torch.cat([
        float_prefix(side_info, match_bits),
        bins_device.T.to(side_info.dtype),
    ], dim=1)
    generator = torch.Generator(device=device).manual_seed(0)
    projection = torch.randn(features.shape[1], generator=generator, device=device, dtype=torch.float64)

    unique_keys, inverse = torch.unique(features.double() @ projection, return_inverse=True)
    representatives = torch.empty(unique_keys.numel(), dtype=torch.long, device=device)
    representatives[inverse] = torch.arange(features.shape[0], device=device)
    collisions = (features != features[representatives[inverse]]).any(dim=1).nonzero(as_tuple=False).flatten()
    return bins_device, representatives, inverse, collisions


class DeduplicationStats(NamedTuple):
    """Diagnostics from one encoder or decoder deduplication pass."""

    vector_size: int
    unique_count: int
    reused_count: int
    collision_count: int


class DedupedWZQuantizerCancer(WZQuantizerCancer):
    """WZ quantizer that reuses encoder, decoder, and prior results across float-prefix rows."""

    prior_calculator: ClassVar[type[PriorCalculator]] = DedupedPriorCalculator
    match_bits: ClassVar[int] = 24

    last_encoding_stats: DeduplicationStats | None = None
    last_decoding_stats: DeduplicationStats | None = None

    def _encode_preprocessed(
        self,
        x_prep: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one representative per normalized float-prefix group."""
        empty_bins = torch.empty((0, x_prep.shape[0]), dtype=torch.long, device=self.device)
        _, representatives, inverse, collisions = unique_row_groups(
            empty_bins, x_prep, self.match_bits
        )
        bins, soft_codes = super()._encode_preprocessed(x_prep[representatives], batch_size)
        inverse_cpu = inverse.cpu()
        bins = bins[:, inverse_cpu]
        soft_codes = soft_codes[:, inverse_cpu]
        if collisions.numel():
            collision_bins, collision_codes = super()._encode_preprocessed(
                x_prep[collisions], batch_size
            )
            collisions_cpu = collisions.cpu()
            bins[:, collisions_cpu] = collision_bins
            soft_codes[:, collisions_cpu] = collision_codes
        self.last_encoding_stats = DeduplicationStats(
            x_prep.shape[0],
            representatives.numel(),
            x_prep.shape[0] - representatives.numel() - collisions.numel(),
            collisions.numel(),
        )
        return bins, soft_codes

    def _decode_stages_preprocessed(
        self,
        bins: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode every stage once per symbol-and-SI float-prefix group."""
        bins_device, representatives, inverse, collisions = unique_row_groups(
            bins, side_info, self.match_bits
        )
        recons = super()._decode_stages_preprocessed(
            bins_device[:, representatives], side_info[representatives], batch_size
        ).to(self.device)[:, inverse]
        if collisions.numel():
            recons[:, collisions] = super()._decode_stages_preprocessed(
                bins_device[:, collisions], side_info[collisions], batch_size
            ).to(self.device)

        self.last_decoding_stats = DeduplicationStats(
            bins.shape[1],
            representatives.numel(),
            bins.shape[1] - representatives.numel() - collisions.numel(),
            collisions.numel(),
        )
        return recons.cpu()
