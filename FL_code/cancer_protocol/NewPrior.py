from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from FL_code.FL_core.utils import create_training_progress_bar

from .brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from .NewCancer import NewCancerConfig


class PriorCalculator:
    """Prior utilities used by NewQuant to estimate conditional symbol rates."""

    @staticmethod
    def compute_rate_from_prior_tensor(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> float:
        """Compute mean per-symbol code length from prior probabilities and realized bins."""
        prior = prior.float()
        sample_idx = torch.arange(bins.shape[1], device=prior.device)
        return sum(
            -torch.log2(
                prior[plane, sample_idx, bins[plane].to(prior.device, torch.long)].clamp(min=1e-8)
            ).mean().item()
            for plane in range(num_planes)
        )

    @staticmethod
    def get_hash(x_vec: torch.Tensor, sample_size: int = 128) -> str:
        """Build the stable lightweight cache key used for prior reuse."""
        sample = x_vec[:sample_size * 3:3].cpu().numpy().round(decimals=1).astype(np.int32)
        return hashlib.md5(sample.tobytes()).hexdigest()

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
        from .NewQuant import batch_loop

        def prior_batch(start: int, end: int) -> torch.Tensor:
            device = next(model.parameters()).device
            codes = [
                F.one_hot(plane.long().to(device), num_classes=model.bins_per_plane).float()
                for plane in bins_vec[:, start:end]
            ]
            return torch.stack(model.get_priors(codes=codes, y=side_info[start:end].to(device))).cpu()

        return batch_loop(prior_batch, model, bins_vec.shape[1], batch_size, cat_dim=1)

    @staticmethod
    def train_prior_model(
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        num_planes: int,
        bins_per_plane: int,
        c_cfg: NewCancerConfig,
        batch_size: int = 50_000,
    ) -> EncoderDecoderLayeredRNN:
        """Train repeated conditional prior models and return the finite lowest-loss attempt."""
        attempts: list[tuple[EncoderDecoderLayeredRNN, float]] = []
        tries = 0
        while len(attempts) < c_cfg.prior_train_repeats:
            assert tries < c_cfg.prior_train_repeats * 5, "Too many failed prior-training attempts."
            tries += 1
            model, loss = PriorCalculator._train_prior_attempt(
                bins_vec, side_info, num_planes, bins_per_plane, c_cfg, batch_size
            )
            if np.isfinite(loss):
                attempts.append((model, loss))
        return min(attempts, key=lambda attempt: attempt[1])[0]

    @staticmethod
    def _train_prior_attempt(
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        num_planes: int,
        bins_per_plane: int,
        c_cfg: NewCancerConfig,
        batch_size: int,
    ) -> tuple[EncoderDecoderLayeredRNN, float]:
        from .NewQuant import new_rnn_model

        assert bins_vec.shape[0] == num_planes, "bins_vec first dimension must match num_planes."
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = new_rnn_model(num_planes, bins_per_plane, side_info.shape[1], False).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        total_samples = min(c_cfg.train_sample_size, bins_vec.shape[1])
        total_batches = (total_samples + batch_size - 1) // batch_size
        pbar = create_training_progress_bar(
            c_cfg.train_epochs * total_batches,
            disable=not c_cfg.training_progress_bar,
            desc="Prior Model",
        )

        epoch_loss = float("inf")
        for epoch in range(c_cfg.train_epochs):
            indices = torch.randint(bins_vec.shape[1], (total_samples,), dtype=torch.long)
            epoch_loss = 0.0
            for start in range(0, total_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                bins_batch = bins_vec[:, batch_indices].to(device, torch.long)
                si_batch = side_info[batch_indices].to(device)
                si_batch = si_batch + torch.randn_like(si_batch) * (1e-4 * si_batch.abs().mean())
                tau = c_cfg.tau * np.exp(epoch / (c_cfg.train_epochs + 1) * np.log(0.1 / c_cfg.tau))
                priors = torch.stack(model.get_priors(
                    codes=[F.one_hot(plane, num_classes=bins_per_plane).float() for plane in bins_batch],
                    y=si_batch,
                    tau=tau,
                ))
                selected_probs = priors.gather(2, bins_batch.long().unsqueeze(-1)).squeeze(-1)
                loss = -torch.log(selected_probs + 1e-12).mean(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if c_cfg.training_progress_bar:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    pbar.update(1)
            epoch_loss /= total_batches

        pbar.close()
        model.cpu().eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return model, epoch_loss


class DedupedPriorCalculator(PriorCalculator):
    """PriorCalculator that runs the prior network once per unique (bins, side info) row.

    Priors are a pure function of each position's bin symbols and side information, so duplicate
    rows reuse the representative's probabilities; with ``si_match_bits`` at 32 the result stays
    equal to the full pass, lower values trade exactness for more reuse.
    """

    si_match_bits: ClassVar[int] = 30

    @classmethod
    def compute_prior_from_network(
        cls,
        model: EncoderDecoderLayeredRNN,
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int = 500_000,
    ) -> torch.Tensor:
        """Compute priors for unique (bins, side info) rows and broadcast them back to duplicates."""
        from .NewQuant import unique_row_groups

        bins_device, representatives, inverse, collisions = unique_row_groups(bins_vec, side_info, cls.si_match_bits)
        prior = PriorCalculator.compute_prior_from_network(
            model, bins_device[:, representatives], side_info[representatives], batch_size
        ).to(side_info.device)[:, inverse]
        if collisions.numel():
            prior[:, collisions] = PriorCalculator.compute_prior_from_network(
                model, bins_device[:, collisions], side_info[collisions], batch_size
            ).to(side_info.device)
        return prior.cpu()
