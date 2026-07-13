from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from .brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from .wz_quantizer import WZcfgQuant


class PriorCalculator:
    """Prior utilities used by NewQuant to estimate conditional symbol rates."""

    @staticmethod
    def compute_plane_rates(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> tuple[float, ...]:
        """Compute each plane's mean code length from prior probabilities and realized bins."""
        prior = prior.float()
        sample_idx = torch.arange(bins.shape[1], device=prior.device)
        return tuple(
            -torch.log2(
                prior[plane, sample_idx, bins[plane].to(prior.device, torch.long)].clamp(min=1e-8)
            ).mean().item()
            for plane in range(num_planes)
        )

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
        return probs[:, None, :].expand(num_planes, bins_vec.shape[1], bins_per_plane).to(torch.float16)

    @staticmethod
    def compute_prior_from_network(
        model: EncoderDecoderLayeredRNN,
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int = 500_000,
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
        side_info: torch.Tensor,
        epoch: int,
        c_cfg: WZcfgQuant,
    ) -> torch.Tensor:
        """Compute categorical code length for fixed transmitted bins."""
        bins = training_input[0].unbind(dim=1)
        hard_codes = training_input[1].unbind(dim=1)
        priors = model.get_priors(codes=hard_codes, y=side_info, force_softmax=True)
        return torch.stack([
            -torch.log(prior.gather(1, plane_bins[:, None]) + 1e-12).mean()
            for plane_bins, prior in zip(bins, priors, strict=True)
        ]).mean()

    @staticmethod
    def make_trained_prior_model(
        bins_vec: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
        batch_size: int = 50_000,
    ) -> EncoderDecoderLayeredRNN:
        """Train a new conditional prior model from fixed quantizer bins and soft codes."""
        from .wz_quantizer import new_rnn_model, wz_model_training_loop

        assert bins_vec.shape == (c_cfg.num_planes, side_info.shape[0])
        assert soft_codes.shape == (
            c_cfg.num_planes, side_info.shape[0], c_cfg.bins_per_plane
        )
        hard_codes = F.one_hot(bins_vec.T.long(), num_classes=c_cfg.bins_per_plane).float()
        prior_input = (bins_vec.T.contiguous(), hard_codes.contiguous())
        attempts: list[tuple[EncoderDecoderLayeredRNN, float]] = []
        for attempt_index in range(c_cfg.prior_train_repeats + 2):
            if len(attempts) == c_cfg.prior_train_repeats:
                break
            candidate = new_rnn_model(
                c_cfg.num_planes,
                c_cfg.bins_per_plane,
                side_info.shape[1],
                c_cfg.marginal_loss,
            )
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
                label=f"Prior attempt {attempt_index + 1}",)
            candidate.requires_grad_(True)
            if np.isfinite(loss):
                attempts.append((candidate, loss))

        assert len(attempts) == c_cfg.prior_train_repeats, "Too many failed prior-retraining attempts."
        return min(attempts, key=lambda attempt: attempt[1])[0]


class DedupedPriorCalculator(PriorCalculator): # dont use / buggy
    """PriorCalculator that runs the prior network once per unique (bins, side info) row.

    Priors are a pure function of each position's bin symbols and side information, so duplicate
    rows reuse the representative's probabilities; with ``si_match_bits`` at 32 the result stays
    equal to the full pass, lower values trade exactness for more reuse.
    """

    si_match_bits: ClassVar[int] = 18

    @classmethod
    def compute_prior_from_network(
        cls,
        model: EncoderDecoderLayeredRNN,
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int = 500_000,
    ) -> torch.Tensor:
        """Compute priors for unique (bins, side info) rows and broadcast them back to duplicates."""
        from .wz_quantizer import unique_row_groups

        bins_device, representatives, inverse, collisions = unique_row_groups(bins_vec, side_info, cls.si_match_bits)
        prior = PriorCalculator.compute_prior_from_network(
            model, bins_device[:, representatives], side_info[representatives], batch_size
        ).to(side_info.device)[:, inverse]
        if collisions.numel():
            prior[:, collisions] = PriorCalculator.compute_prior_from_network(
                model, bins_device[:, collisions], side_info[collisions], batch_size
            ).to(side_info.device)
        return prior.cpu()
