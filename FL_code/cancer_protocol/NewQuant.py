from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch
import torch.nn.functional as F

from FL_code.FL_core.utils import create_training_progress_bar

from .brent_wz_models import EncoderDecoderLayeredRNN
from .NewPrior import PriorCalculator, batch_loop, new_rnn_model

if TYPE_CHECKING:
    from .NewCancer import NewCancerConfig

PRIOR_CACHE_NO_RETRAIN = "flag_no_retrain"
PriorCache: TypeAlias = dict[str, torch.Tensor | str]
OUTLIER_CHUNK = 2**8


class OutlierMetadata(NamedTuple):
    """Outlier correction data produced by preprocessing and consumed by reconstruction postprocessing.

    ``positions`` holds per-256-element-chunk uint8 offsets; ``scale`` is a 0-d float16 array
    (an array, not a scalar, so the shared payload size accounting can measure it).
    """

    positions: tuple[np.ndarray, ...]
    scale: np.ndarray | None
    signs: np.ndarray


class PreprocessMetadata(NamedTuple):
    """Per-vector preprocessing data produced during encoding and consumed during decoding.

    A NamedTuple so payload size accounting and serialization treat it as a plain tuple.
    """

    norm_factors: torch.Tensor
    outliers: OutlierMetadata


def normalization_params(values: torch.Tensor) -> tuple[float, float]:
    """Estimate a robust center and scale from a random quantile-trimmed sample."""
    sample_size = min(200_000, values.numel())

    def random_sample() -> torch.Tensor:
        return values[torch.randint(values.numel(), (sample_size,), device=values.device)].float()

    centers: list[float] = []
    for _ in range(5):
        sample = random_sample()
        q02, q98 = torch.quantile(sample, torch.tensor([0.02, 0.98], device=sample.device))
        centers.append(sample[(sample >= q02) & (sample <= q98)].mean().item())
    center = float(np.mean(centers))

    scales: list[float] = []
    for _ in range(5):
        q01, q99 = torch.quantile(
            random_sample() - center,
            torch.tensor([0.01, 0.99], device=values.device),
        ).abs()
        scales.extend((q01.item(), q99.item()))
    scale = float(np.mean(scales))
    assert scale != 0
    return scale, center


def outlier_metadata(values: torch.Tensor, threshold: float) -> tuple[torch.Tensor, OutlierMetadata]:
    """Move values beyond threshold into a separately coded tail representation."""
    positions = torch.nonzero(values.abs() > threshold, as_tuple=False).flatten()
    if positions.numel() == 0:
        return values, OutlierMetadata((), None, np.array([], dtype=np.bool_))

    outliers = values[positions]
    tail = outliers.abs() - threshold
    scale = torch.quantile(tail.float(), 0.99).item() / threshold
    assert scale != 0
    values = values.clone()
    values[positions] = tail * outliers.sign() / scale
    positions_np = positions.cpu().numpy()
    return values, OutlierMetadata(
        tuple(
            (positions_np[(positions_np >= start) & (positions_np < start + OUTLIER_CHUNK)] - start).astype(np.uint8)
            for start in range(0, values.numel() + OUTLIER_CHUNK - 1, OUTLIER_CHUNK)
        ),
        np.array(scale, dtype=np.float16),
        (outliers > 0).cpu().numpy().astype(np.bool_),
    )


class WZQuantizerCancer:
    """Learned Wyner-Ziv quantizer for one flattened Cancer protocol update vector."""

    def __init__(
        self,
        c_cfg: NewCancerConfig,
        num_planes: int,
        bins_per_plane: int,
        si_size: int,
        marginal_loss: bool = False,
        norm_slices: Sequence[slice] | None = None,
        outlier_threshold: float | None = None,
        extra_si_for_prior: Sequence[torch.Tensor] = (),
    ) -> None:
        self.c_cfg: NewCancerConfig = c_cfg
        self.norm_slices: list[slice] = list(norm_slices or [slice(0, None)])
        self.outlier_threshold: float | None = outlier_threshold
        self.extra_si_for_prior: list[torch.Tensor] = list(extra_si_for_prior)
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.no_side_info: bool = si_size == 0
        assert marginal_loss or not self.no_side_info, (
            "si_size=0 requires marginal_loss=True; no-SI quantizers must be explicit."
        )
        self.side_info_size: int = max(si_size, 1)
        self.coding_model: EncoderDecoderLayeredRNN = new_rnn_model(
            num_planes, bins_per_plane, self.side_info_size, marginal_loss
        )

        # "P": untrained zero-SI marginal (pretrained weights may be loaded); None: awaiting train_model().
        self.side_info_list_used: list[torch.Tensor] | str | None = "P" if si_size == 0 else None
        self.vector_size: int | None = None
        self.prior_cache: PriorCache = {}

    @property
    def num_planes(self) -> int:
        return self.coding_model.num_planes

    @property
    def bins_per_plane(self) -> int:
        return self.coding_model.bins_per_plane

    @property
    def bin_count(self) -> int:
        return self.coding_model.bin_count

    @staticmethod
    def compute_loss(
        model: EncoderDecoderLayeredRNN,
        x_batch: torch.Tensor,
        si_batch: torch.Tensor,
        epoch: int,
        c_cfg: NewCancerConfig,
        wmspe_denom: float,
    ) -> torch.Tensor:
        """Compute reconstruction-plus-rate loss for one quantizer training batch."""
        progress = epoch / (c_cfg.train_epochs + 1)
        tau = c_cfg.tau * np.exp(progress * np.log(0.1 / c_cfg.tau))
        reconstructions, bins, soft_codes, priors = model(x_batch, si_batch, tau=tau)

        loss = torch.zeros((), device=x_batch.device)
        sample_indices = torch.arange(x_batch.shape[0], device=x_batch.device)
        for plane in range(model.num_planes):
            distortion = F.mse_loss(reconstructions[plane], x_batch) / wmspe_denom
            posterior_prob = soft_codes[plane][sample_indices, bins[plane]]
            prior_prob = priors[plane][sample_indices, bins[plane]]
            rate = torch.log((posterior_prob + 1e-12) / (prior_prob + 1e-12)).mean()
            loss = loss + c_cfg.reconst_ld * distortion + rate
        return loss / model.num_planes

    def train_model(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[torch.Tensor] | None,
        batch_size: int = 50_000,
    ) -> None:
        """Train repeated quantizer attempts and keep the finite attempt with the lowest loss."""
        assert self.side_info_list_used is None or self.side_info_list_used == "P", (
            "This quantizer instance has already been trained."
        )
        if self.no_side_info:
            assert not si_raw_list, "Marginal quantizer training expects no side information."
            self.side_info_list_used = []
        else:
            assert si_raw_list and len(si_raw_list) == self.side_info_size, (
                f"Quantizer training requires {self.side_info_size} side-information vectors."
            )
            self.side_info_list_used = list(si_raw_list)

        wmspe_denom = (x_raw.float().square().mean().item() / 2) + 1e-8
        x_prep, _ = self.preprocess_x(x_raw)
        side_info = self.side_info_tensor()

        attempts: list[tuple[EncoderDecoderLayeredRNN, float]] = []
        tries = 0
        while len(attempts) < self.c_cfg.quantizer_train_repeats:
            assert tries <= self.c_cfg.quantizer_train_repeats * 5, "Too many failed training attempts."
            tries += 1
            self.coding_model = new_rnn_model(
                self.num_planes, self.bins_per_plane, side_info.shape[1], self.coding_model.marginal
            )
            loss = self.train_attempt(self.coding_model, x_prep, side_info, self.c_cfg, wmspe_denom, batch_size)
            if np.isfinite(loss) and torch.isfinite(self.decoding_process(self.encoding_process(x_raw))).all():
                attempts.append((self.coding_model, loss))

        self.coding_model = min(attempts, key=lambda attempt: attempt[1])[0]
        self.prior_cache[PriorCalculator.get_hash(x_raw)] = PRIOR_CACHE_NO_RETRAIN

    @classmethod
    def train_attempt(
        cls,
        model: EncoderDecoderLayeredRNN,
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: NewCancerConfig,
        wmspe_denom: float,
        batch_size: int = 50_000,
    ) -> float:
        """Train one quantizer initialization and return its final epoch loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if c_cfg.tf32 and device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            fused=c_cfg.fused_optimizer and device.type == "cuda",
            lr=c_cfg.lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(c_cfg.train_epochs * np.ceil(c_cfg.lr_step / 180)),
            gamma=0.3,
        )

        model.to(device).train()
        x_prep = x_prep.to(device)
        side_info = side_info.to(device)
        use_amp = c_cfg.mixed_precision and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        total_samples = min(c_cfg.train_sample_size, x_prep.shape[0])
        total_batches = (total_samples + batch_size - 1) // batch_size
        pbar = create_training_progress_bar(
            c_cfg.train_epochs * total_batches,
            desc="Training Quantizer",
            disable=not c_cfg.training_progress_bar,
        )

        epoch_loss = float("inf")
        for epoch in range(c_cfg.train_epochs):
            indices = torch.randint(x_prep.shape[0], (total_samples,), device=x_prep.device)
            epoch_loss = 0.0
            for start in range(0, total_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                x_batch = x_prep[batch_indices].clone()
                si_batch = side_info[batch_indices]
                x_batch = x_batch + torch.randn_like(x_batch) * (1e-5 * x_batch.abs().mean())

                optimizer.zero_grad()
                if use_amp:
                    assert scaler is not None
                    with torch.amp.autocast("cuda"):
                        loss = cls.compute_loss(model, x_batch, si_batch, epoch, c_cfg, wmspe_denom)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = cls.compute_loss(model, x_batch, si_batch, epoch, c_cfg, wmspe_denom)
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                if c_cfg.training_progress_bar:
                    pbar.set_postfix({"loss": f"{loss.item():.2f}"})
                    pbar.update(1)
            scheduler.step()
            epoch_loss /= total_batches

        pbar.close()
        model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return epoch_loss

    def encoding_process(
        self,
        x_raw: torch.Tensor,
        batch_size: int = 500_000,
    ) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Preprocess and quantize one raw vector into per-plane bin indices."""
        x_prep, metadata = self.preprocess_x(x_raw)
        bins = self._encode_preprocessed(x_prep, batch_size)
        dtype = torch.uint8 if self.bins_per_plane < 2**8 else torch.uint16
        return bins.to(dtype), metadata

    def _encode_preprocessed(self, x_prep: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Encode normalized model inputs into per-plane bin indices."""
        def encode_batch(start: int, end: int) -> torch.Tensor:
            bins, _ = self.coding_model.encode(x_prep[start:end])
            return torch.stack(bins).cpu()

        bins = batch_loop(encode_batch, self.coding_model, x_prep.shape[0], batch_size, cat_dim=1)
        return bins

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
        side_info = self.side_info_tensor()
        recons = self._decode_preprocessed(bins, side_info, batch_size)
        return self.postprocess(recons, metadata)

    def _decode_preprocessed(
        self,
        bins: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Decode normalized reconstruction values from bins and normalized side information."""
        def decode_batch(start: int, end: int) -> torch.Tensor:
            codes = [
                F.one_hot(plane.long().to(self.device), num_classes=self.bins_per_plane).float()
                for plane in bins[:, start:end]
            ]
            return self.coding_model.decode(codes, side_info[start:end])[-1].cpu()

        recons = batch_loop(decode_batch, self.coding_model, bins.shape[1], batch_size)
        return recons.reshape(-1)

    def _get_posterior(
        self,
        x_raw: torch.Tensor,
        bins_vec_save_compute: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data_hash = PriorCalculator.get_hash(x_raw)
        cached_prior = self.prior_cache.get(data_hash)
        if isinstance(cached_prior, torch.Tensor):
            return cached_prior

        bins = self.encoding_process(x_raw)[0] if bins_vec_save_compute is None else bins_vec_save_compute
        side_info = self.side_info_tensor(for_prior=True)
        use_quantizer_prior = cached_prior == PRIOR_CACHE_NO_RETRAIN and not self.extra_si_for_prior
        prior_model = self.coding_model if use_quantizer_prior else PriorCalculator.train_prior_model(
            bins, side_info, self.num_planes, self.bins_per_plane, self.c_cfg
        )
        prior = PriorCalculator.compute_prior_from_network(prior_model, bins, side_info).to(torch.float16)
        self.prior_cache[data_hash] = prior
        return prior

    def preprocess_x(self, x_raw: torch.Tensor, skip_outliers: bool = False) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Normalize a raw vector per slice, optionally extract outliers, and shape it for the model."""
        if self.vector_size is None:
            self.vector_size = x_raw.numel()
        assert self.vector_size == x_raw.numel(), f"Expected vector size {self.vector_size}, got {x_raw.numel()}."

        x_prep = x_raw.clone()
        norm_factors: list[tuple[float, float]] = []
        for vector_slice in self.norm_slices:
            scale, center = normalization_params(x_prep[vector_slice])
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
                    range(0, recons.numel() + OUTLIER_CHUNK - 1, OUTLIER_CHUNK),
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

    def side_info_tensor(self, for_prior: bool = False) -> torch.Tensor:
        """Return side information in the normalized model input format."""
        assert self.vector_size is not None, "Vector size must be known before side information is formatted."
        assert self.side_info_list_used is not None, "Side information must be set before formatting."
        side_info_list = [] if self.side_info_list_used == "P" else list(self.side_info_list_used)
        if for_prior and self.extra_si_for_prior:
            # A zero placeholder carries no information, so the prior uses the extra side info alone.
            if side_info_list and torch.all(side_info_list[0] == 0):
                side_info_list = []
            side_info_list += self.extra_si_for_prior
        if not side_info_list:
            side_info_list = [torch.zeros(self.vector_size)]
        if not for_prior:
            assert len(side_info_list) == self.side_info_size

        tensors = [
            torch.zeros(self.vector_size, device=self.device) if torch.all(si == 0)
            else self.preprocess_x(si, skip_outliers=True)[0].squeeze(1)
            for si in side_info_list
        ]
        return torch.stack(tensors, dim=1).to(self.device, dtype=torch.float32).contiguous()


class SampledEncodingStats(NamedTuple):
    """Diagnostics from sampled-distance encoding, produced by the encoder for experiment logging."""

    vector_size: int
    sample_count: int
    inferred_count: int
    fallback_count: int
    distance_threshold: float


class DedupedDecodingStats(NamedTuple):
    """Diagnostics from deduplicated decoding, produced by the decoder for experiment logging."""

    vector_size: int
    unique_count: int
    reused_count: int
    collision_count: int


class DistanceSampledWZQuantizerCancer(WZQuantizerCancer):
    """WZ quantizer that encodes a sample exactly and infers safe remaining symbols by nearby sampled values.

    The sampling knobs are runtime-speed trade-offs rather than experiment parameters, so they
    are class constants instead of configuration.
    """

    sample_fraction: ClassVar[float] = 0.02
    neighbor_count: ClassVar[int] = 4
    distance_multiplier: ClassVar[float] = 64.0
    exact_encode_threshold: ClassVar[float] = 1.4
    assignment_batch_size: ClassVar[int] = 1_000_000
    sample_seed: ClassVar[int] = 0

    last_sampled_encoding_stats: SampledEncodingStats | None = None

    def _encode_preprocessed(
        self,
        x_prep: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode normalized model inputs from sampled anchors plus exact unsafe values."""
        vector_size = x_prep.shape[0]
        sample_count = min(
            vector_size,
            max(self.neighbor_count, int(np.ceil(vector_size * self.sample_fraction))),
        )
        if sample_count == vector_size:
            bins = super()._encode_preprocessed(x_prep, batch_size)
            self.last_sampled_encoding_stats = SampledEncodingStats(vector_size, sample_count, 0, 0, 0.0)
            return bins

        # The neural encoder is used only for sampled anchors and values rejected by sample-based assignment.
        generator = torch.Generator(device=self.device).manual_seed(self.sample_seed)
        sample_indices = torch.randperm(vector_size, device=self.device, generator=generator)[:sample_count]
        sample_bins = super()._encode_preprocessed(x_prep[sample_indices], batch_size).to(
            device=self.device, dtype=torch.long
        )

        sampled = torch.zeros(vector_size, dtype=torch.bool, device=self.device)
        sampled[sample_indices] = True
        remaining_indices = sampled.logical_not().nonzero(as_tuple=False).flatten()

        bins = torch.empty((self.num_planes, vector_size), dtype=torch.long, device=self.device)
        bins[:, sample_indices] = sample_bins

        # Pure 1D assignment: infer symbols from nearby sampled anchors and mark unsafe values for exact encoding.
        inferred_bins, inferred_mask, distance_threshold = self._assign_bins_from_samples(
            x_prep[sample_indices].squeeze(1),
            sample_bins,
            x_prep[remaining_indices].squeeze(1),
        )
        bins[:, remaining_indices[inferred_mask]] = inferred_bins[:, inferred_mask]

        # Unsafe values include contradictory neighborhoods, sparse neighborhoods, and optional normalized tails.
        fallback_indices = remaining_indices[~inferred_mask]
        if fallback_indices.numel():
            bins[:, fallback_indices] = super()._encode_preprocessed(
                x_prep[fallback_indices],
                batch_size,
            ).to(device=self.device, dtype=torch.long)

        self.last_sampled_encoding_stats = SampledEncodingStats(
            vector_size,
            sample_count,
            int(inferred_mask.sum().item()),
            int(fallback_indices.numel()),
            distance_threshold,
        )
        return bins.cpu()

    def _assign_bins_from_samples(
        self,
        sample_values: torch.Tensor,
        sample_bins: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Infer bins from sampled value-symbol pairs and mark unsafe values for exact encoding."""
        sort_order = sample_values.argsort()
        sorted_values = sample_values[sort_order]
        sorted_bins = sample_bins[:, sort_order]
        keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=sorted_values.device)
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
        sorted_values = sorted_values[keep].contiguous()
        sorted_bins = sorted_bins[:, keep].contiguous()

        gaps = sorted_values.diff().abs()
        positive_gaps = gaps[gaps > 0]
        distance_threshold = 0.0 if positive_gaps.numel() == 0 else float(
            positive_gaps.median().item() * self.distance_multiplier
        )
        interval_lowers, interval_uppers, interval_bins = self._symbol_intervals(
            sorted_values,
            sorted_bins,
            distance_threshold,
        )
        if interval_lowers.numel() == 0:
            return (
                torch.empty((self.num_planes, values.numel()), dtype=interval_bins.dtype, device=self.device),
                torch.zeros(values.numel(), dtype=torch.bool, device=self.device),
                float(distance_threshold),
            )

        inferred_bins_chunks: list[torch.Tensor] = []
        inferred_mask_chunks: list[torch.Tensor] = []
        for start in range(0, values.numel(), self.assignment_batch_size):
            value_batch = values[start:start + self.assignment_batch_size]
            interval_indices = torch.bucketize(value_batch, interval_lowers) - 1
            candidate_indices = interval_indices.clamp(0, interval_lowers.numel() - 1)
            inferred_mask = (interval_indices >= 0) & (value_batch <= interval_uppers[candidate_indices])
            inferred_mask &= value_batch.abs() <= self.exact_encode_threshold
            inferred_bins_chunks.append(interval_bins[:, candidate_indices])
            inferred_mask_chunks.append(inferred_mask)
        inferred_mask = torch.cat(inferred_mask_chunks)
        return (
            torch.cat(inferred_bins_chunks, dim=1),
            inferred_mask,
            float(distance_threshold),
        )

    def _symbol_intervals(
        self,
        sorted_values: torch.Tensor,
        sorted_bins: torch.Tensor,
        distance_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        changes = (sorted_bins[:, 1:] != sorted_bins[:, :-1]).any(dim=0)
        starts = torch.cat((
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.nonzero(changes, as_tuple=False).flatten() + 1,
        ))
        ends = torch.cat((starts[1:] - 1, starts.new_tensor([sorted_values.numel() - 1])))
        valid = ends - starts + 1 >= self.neighbor_count
        if not valid.any():
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.empty((self.num_planes, 0), dtype=sorted_bins.dtype, device=self.device)

        starts = starts[valid]
        ends = ends[valid]
        lower_positions = starts + self.neighbor_count - 1
        upper_positions = ends - self.neighbor_count + 1
        lower_bounds = sorted_values[lower_positions] - distance_threshold
        upper_bounds = sorted_values[upper_positions] + distance_threshold

        previous_conflicts = torch.full_like(lower_bounds, -torch.inf)
        has_previous = starts > 0
        previous_conflicts[has_previous] = sorted_values[starts[has_previous] - 1] + distance_threshold
        next_conflicts = torch.full_like(upper_bounds, torch.inf)
        has_next = ends < sorted_values.numel() - 1
        next_conflicts[has_next] = sorted_values[ends[has_next] + 1] - distance_threshold
        lower_bounds = torch.maximum(lower_bounds, previous_conflicts)
        upper_bounds = torch.minimum(upper_bounds, next_conflicts)
        valid_bounds = lower_bounds <= upper_bounds
        return lower_bounds[valid_bounds].contiguous(), upper_bounds[valid_bounds].contiguous(), sorted_bins[
            :,
            starts[valid_bounds],
        ].contiguous()
