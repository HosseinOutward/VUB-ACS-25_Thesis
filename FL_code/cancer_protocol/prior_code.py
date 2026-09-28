from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from .wz_quantizer import SideInfoMatrix, WZcfgQuant, _SideInfoRows


# Rows per full-vector inference pass. The layered RNN holds one hidden state per layer per
# batch row, so this directly sets peak VRAM; larger batches buy no throughput past this size.
# Defined here because wz_quantizer imports this module, never the reverse.
INFERENCE_BATCH = 100_000


@dataclass(frozen=True)
class PriorTrainingResult:
    """Fixed-symbol prior fit consumed by a round codec and its protocol warm pool."""

    model: EncoderDecoderLayeredRNN
    lineage_states: tuple[dict[str, torch.Tensor], ...]
    lineage_losses: tuple[float, ...]
    deployed_parent: int


class PriorCalculator:
    """Prior utilities used by NewQuant to estimate conditional symbol rates."""

    @staticmethod
    def compute_code_lengths(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> torch.Tensor:
        """Return every symbol's code length in bits, as a (num_planes, positions) tensor."""
        prior = prior.float()
        sample_idx = torch.arange(bins.shape[1], device=prior.device)
        return torch.stack([
            -torch.log2(
                prior[plane, sample_idx, bins[plane].to(prior.device, torch.long)].clamp(min=1e-8)
            )
            for plane in range(num_planes)
        ])

    @staticmethod
    def compute_plane_rates(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> tuple[float, ...]:
        """Compute each plane's mean code length from prior probabilities and realized bins."""
        return tuple(
            PriorCalculator.compute_code_lengths(prior, bins, num_planes).mean(dim=1).tolist())

    @staticmethod
    def compute_rate_from_prior_tensor(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> float:
        """Compute cumulative mean code length from prior probabilities and realized bins."""
        return sum(PriorCalculator.compute_plane_rates(prior, bins, num_planes))

    @staticmethod
    def compute_marginal_prior(bins_vec: torch.Tensor, bins_per_plane: int, num_planes: int) -> torch.Tensor:
        """Estimate one empirical marginal prior per plane and broadcast it over positions."""
        probs = torch.stack([
            torch.bincount(plane_bins.long(), minlength=bins_per_plane).float() / bins_vec.shape[1]
            for plane_bins in bins_vec
        ])
        return probs[:, None, :].expand(num_planes, bins_vec.shape[1], bins_per_plane)

    @staticmethod
    def compute_prior_from_network(
        model: EncoderDecoderLayeredRNN,
        bins_vec: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        batch_size: int = INFERENCE_BATCH,
    ) -> torch.Tensor:
        """Run a trained quantizer/prior network over all bins and return per-plane probabilities."""
        from .wz_quantizer import batch_loop

        def prior_batch(start: int, end: int) -> torch.Tensor:
            device = next(model.parameters()).device
            codes = [
                F.one_hot(plane.long().to(device), num_classes=model.bins_per_plane).float()
                for plane in bins_vec[:, start:end]
            ]
            return torch.stack(model.get_priors(codes=codes, y=side_info[start:end].to(device))).cpu()

        return batch_loop(prior_batch, model, bins_vec.shape[1], batch_size, cat_dim=1)

    @staticmethod
    def prior_loss(
        model: EncoderDecoderLayeredRNN,
        training_input: tuple[torch.Tensor, torch.Tensor],
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        epoch: int,
        c_cfg: WZcfgQuant,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute conditional code length for fixed transmitted bins."""
        from .wz_quantizer import _sample_mean

        bins = training_input[0].unbind(dim=1)
        context_codes = training_input[1].unbind(dim=1)
        tau = c_cfg.tau * np.exp(epoch / c_cfg.train_epochs * np.log(0.1 / c_cfg.tau))
        priors = model.get_priors(
            codes=context_codes,
            y=side_info,
            tau=tau,
            force_softmax=c_cfg.prior_probabilities == "categorical",
        )
        stage_mean = torch.stack([
            _sample_mean(
                -torch.log(prior.gather(1, plane_bins.long()[:, None]).squeeze(1) + 1e-12),
                sample_weights,
            )
            for plane_bins, prior in zip(bins, priors, strict=True)
        ]).sum()
        return stage_mean / c_cfg.num_planes

    @staticmethod
    def make_trained_prior_model(
        bins_vec: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        c_cfg: WZcfgQuant,
        source_model: EncoderDecoderLayeredRNN | None,
        batch_size: int = 50_000,
    ) -> EncoderDecoderLayeredRNN:
        """Fit fixed-symbol prior attempts and install the winner into the supplied model."""
        initial_states: tuple[Mapping[str, torch.Tensor] | None, ...] = (
            (None,) if source_model is None else (source_model.conditional_state_dict(),)
        )
        return PriorCalculator.fit_prior_model(
            bins_vec, soft_codes, side_info, c_cfg, source_model, initial_states,
            c_cfg.prior_train_repeats, batch_size,
        ).model

    @staticmethod
    def fit_prior_model(
        bins_vec: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        c_cfg: WZcfgQuant,
        target_model: EncoderDecoderLayeredRNN | None,
        initial_states: Sequence[Mapping[str, torch.Tensor] | None],
        attempts_per_parent: int,
        batch_size: int = 50_000,
    ) -> PriorTrainingResult:
        """Train each explicit prior lineage and install the best result into `target_model`."""
        assert bins_vec.shape == (c_cfg.num_planes, side_info.shape[0])
        assert soft_codes.shape == (
            c_cfg.num_planes, side_info.shape[0], c_cfg.bins_per_plane
        )
        assert initial_states and attempts_per_parent > 0
        if target_model is not None:
            assert target_model.num_planes == c_cfg.num_planes
            assert target_model.bins_per_plane == c_cfg.bins_per_plane
            assert target_model.marginal == c_cfg.marginal_loss, (
                "Fine-tuned prior configuration must match the quantizer's marginal architecture."
            )
            if not target_model.marginal:
                assert target_model.conditionalRNN.input_dim == (
                    c_cfg.bins_per_plane + side_info.shape[1]
                ), "Fine-tuned prior side-information width must match the quantizer prior architecture."
        lineage_models: list[EncoderDecoderLayeredRNN] = []
        lineage_losses: list[float] = []
        for parent_index, initial_state in enumerate(initial_states):
            model, loss = _train_prior_lineage(
                bins_vec, soft_codes, side_info,
                c_cfg.warm_retraining_config(c_cfg.prior_retrain_epochs),
                target_model, initial_state, attempts_per_parent, batch_size,
                f"parent {parent_index}",
            )
            lineage_models.append(model)
            lineage_losses.append(loss)
        deployed_parent = min(range(len(lineage_models)), key=lineage_losses.__getitem__)
        deployed_state = {
            name: value.detach().cpu().clone()
            for name, value in lineage_models[deployed_parent].state_dict().items()
        }
        model = target_model or _new_prior_model(c_cfg, side_info.shape[1])
        model.load_state_dict(deployed_state, strict=target_model is None)
        return PriorTrainingResult(
            model=model,
            lineage_states=tuple({
                name: value.detach().cpu().clone() for name, value in candidate.state_dict().items()
            } for candidate in lineage_models),
            lineage_losses=tuple(lineage_losses),
            deployed_parent=deployed_parent,
        )


def _new_prior_model(c_cfg: WZcfgQuant, side_info_size: int) -> EncoderDecoderLayeredRNN:
    """Construct the independent full-SI architecture used by scratch prior retraining."""
    from .wz_quantizer import new_rnn_model

    return new_rnn_model(
        c_cfg.num_planes,
        c_cfg.bins_per_plane,
        side_info_size,
        c_cfg.marginal_loss,
        c_cfg.hidden_dim,
        c_cfg.layers,
        c_cfg.shared_heads,
    ).copy_for_prior_retraining()


def _train_prior_lineage(
    bins_vec: torch.Tensor,
    soft_codes: torch.Tensor,
    side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
    c_cfg: WZcfgQuant,
    source_model: EncoderDecoderLayeredRNN | None,
    initial_state: Mapping[str, torch.Tensor] | None,
    repeats: int,
    batch_size: int,
    label: str,
) -> tuple[EncoderDecoderLayeredRNN, float]:
    from .wz_quantizer import wz_model_training_loop

    context_codes = (
        F.one_hot(bins_vec.T.long(), num_classes=c_cfg.bins_per_plane).float()
        if c_cfg.prior_context == "hard" else soft_codes.transpose(0, 1)
    )
    prior_input = (bins_vec.T.contiguous(), context_codes.contiguous())
    attempts: list[tuple[EncoderDecoderLayeredRNN, float]] = []
    for attempt_index in range(repeats + 2):
        if len(attempts) == repeats:
            break
        candidate = (
            _new_prior_model(c_cfg, side_info.shape[1])
            if source_model is None else source_model.copy_for_prior_retraining()
        )
        if initial_state is not None:
            candidate.load_state_dict(initial_state, strict=True)
        prior_heads = (
            candidate.conditionalPriors
            if not candidate.shared_priors else (candidate.conditionalPrior,))
        candidate.requires_grad_(False)
        prior_params = tuple(candidate.conditionalRNN.parameters()) + tuple(
            parameter for head in prior_heads for parameter in head.parameters())
        for parameter in prior_params:
            parameter.requires_grad_(True)
        loss = wz_model_training_loop(
            PriorCalculator.prior_loss, candidate, iter(prior_params),
            prior_input, side_info, c_cfg, batch_size,
            label=f"Prior {label} attempt {attempt_index + 1}",)
        candidate.requires_grad_(True)
        if np.isfinite(loss):
            attempts.append((candidate, loss))

    assert len(attempts) == repeats, (
        f"Created {len(attempts)}/{repeats} required {label} prior candidates."
    )
    return min(attempts, key=lambda attempt: attempt[1])


class DedupedPriorCalculator(PriorCalculator):
    """PriorCalculator that runs the prior network once per unique (bins, side info) row.

    Priors are a pure function of each position's bin symbols and side information, so duplicate
    rows reuse the representative's probabilities; with ``match_bits`` at 32 the result stays
    equal to the full pass, lower values trade exactness for more reuse.
    """

    @staticmethod
    def compute_prior_from_network(
        model: EncoderDecoderLayeredRNN,
        bins_vec: torch.Tensor,
        side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        batch_size: int = INFERENCE_BATCH,
    ) -> torch.Tensor:
        """Compute priors for unique (bins, side info) rows and broadcast them back to duplicates."""
        from .wz_quantizer import DedupedWZQuantizerCancer, select_si_rows, unique_row_groups

        bins_device = bins_vec.to(side_info.device, torch.long)
        representatives, inverse, collisions = unique_row_groups(
            (side_info, bins_device.T), DedupedWZQuantizerCancer.match_bits
        )
        prior = PriorCalculator.compute_prior_from_network(
            model, bins_device[:, representatives], select_si_rows(side_info, representatives), batch_size
        ).to(side_info.device)[:, inverse]
        if collisions.numel():
            prior[:, collisions] = PriorCalculator.compute_prior_from_network(
                model, bins_device[:, collisions], select_si_rows(side_info, collisions), batch_size
            ).to(side_info.device)
        return prior.cpu()
