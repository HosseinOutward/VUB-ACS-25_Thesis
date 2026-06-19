from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from run_fl import FLConfig
from brent_wz_models import EncoderDecoderLayeredRNN
from utils import create_training_progress_bar

class PriorCalculator:
    @staticmethod
    def compute_rate_from_prior_tensor(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> float:
        """Compute rate from prior probabilities for given bins."""
        n_samples = bins.shape[1]
        # Convert to float32 and clamp to avoid log(0) issues from float16 rounding
        prior = prior.float()
        min_prob = 1e-8  # Safe minimum for float32
        return sum(
            -torch.log2(prior[i, torch.arange(n_samples), bins[i].long()].clamp(min=min_prob)).mean().item()
            for i in range(num_planes)
        )

    @staticmethod
    def get_hash(x_vec: torch.Tensor, sample_size: int = 128) -> str:
        """Build a stable lightweight hash for prior-cache lookup."""
        sample = x_vec[:sample_size*3:3]
        sample = sample.cpu().numpy().round(decimals=1).astype(np.int32)
        hasher = hashlib.md5()
        hasher.update(sample.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def compute_marginal_prior(bins_vec: torch.Tensor, bins_per_plane: int, num_planes: int) -> torch.Tensor:
        """Estimate a per-plane marginal prior and broadcast it over all symbols."""
        vec_size = bins_vec.size(1)
        probs_per_plane = []
        for b_vec in bins_vec:
            counts = torch.bincount(b_vec.long(), minlength=bins_per_plane).float()
            probs = counts / vec_size
            probs_per_plane.append(probs)

        # Broadcast probabilities to all positions: (num_planes, N, bins_per_plane)
        probs_array = torch.stack(probs_per_plane)  # (num_planes, bins_per_plane)
        prior = torch.broadcast_to(
            probs_array[:, torch.newaxis, :],
            (num_planes, vec_size, bins_per_plane)
        ).to(torch.float16)

        return prior

    @staticmethod
    def _compute_prior_from_network(
        q_model: Any,
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        training_tau: float | bool = False,
        batch_size: int = 500_000
    ) -> torch.Tensor:
        from cancer_quantizer import WZQuantizerCancer

        training_mode = training_tau is not False

        bins_vec = bins_vec.to(torch.long)
        bins_per_plane = q_model.bins_per_plane

        def func(start_i: int, end_idx: int) -> torch.Tensor:
            codes = [F.one_hot(b, num_classes=bins_per_plane).cuda()
                     for b in bins_vec[:, start_i:end_idx]]
            side_info_batch = side_info[start_i:end_idx].cuda()

            tau = training_tau if training_mode else None

            priors = q_model.get_priors(codes=codes, y=side_info_batch, tau=tau)
            priors = torch.stack(priors)

            if training_mode:
                return priors
            return priors.cpu()

        priors = WZQuantizerCancer._batch_loop(func, q_model, bins_vec.size(1), batch_size, training_mode)

        return priors

    @staticmethod
    def train_prior_model(
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        num_planes: int,
        bins_per_plane: int,
        c_cfg: Any,
        batch_size: int = 50_000
    ) -> EncoderDecoderLayeredRNN:
        """Train repeated prior models and return the lowest-loss attempt."""
        train_attempts = []
        tries = 0
        while len(train_attempts) < c_cfg.prior_train_repeats:
            assert tries < c_cfg.prior_train_repeats * 5
            tries += 1

            q_model, q_loss = PriorCalculator._train_prior_model(
                bins_vec, side_info, num_planes, bins_per_plane,
                c_cfg, batch_size, return_loss=True)
            if torch.isnan(torch.tensor(q_loss)):
                continue
            train_attempts.append((q_model, q_loss))

        q_model, lowest_trained_rate = min(train_attempts, key=lambda x: x[1])
        return q_model

    @staticmethod
    def _train_prior_model(
        bins_vec: torch.Tensor,
        side_info: torch.Tensor,
        num_planes: int,
        bins_per_plane: int,
        c_cfg: Any,
        batch_size: int,
        return_loss: bool = False
    ) -> EncoderDecoderLayeredRNN | tuple[EncoderDecoderLayeredRNN, float]:
        assert bins_vec.size(0) == num_planes, "bins_vec first dimension must match num_planes"

        prior_model = EncoderDecoderLayeredRNN(
            num_planes=num_planes, bins_per_plane=bins_per_plane, side_info_size=side_info.size(1),
            input_dim=1, layers=3, hidden_dim=100, marginal=False
        )
        prior_model.cuda().train()

        optimizer = torch.optim.AdamW(prior_model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Convert to long once
        vec_size = bins_vec.size(1)
        total_samples = int(min(c_cfg.train_sample_size, vec_size))
        num_batches = (total_samples + batch_size - 1) // batch_size
        pbar = create_training_progress_bar(
            c_cfg.train_epochs * num_batches,
            disable=not FLConfig().training_progress_bar,
            desc="Prior Model")

        for epoch in range(c_cfg.train_epochs):
            indices = torch.randint(0, vec_size, (total_samples,), dtype=torch.long)
            bins_subset, si_subset = bins_vec[:, indices], side_info[indices]

            epoch_loss = 0.0
            for batch_idx, start_i in enumerate(range(0, total_samples, batch_size)):
                end_i = min(start_i + batch_size, total_samples)
                bins_batch, si_batch = bins_subset[:, start_i:end_i], si_subset[start_i:end_i]

                si_batch += torch.randn_like(si_batch) * (1e-4 * si_batch.abs().mean())

                training_prog = epoch / (c_cfg.train_epochs + 1)
                tau = c_cfg.tau * np.exp(training_prog * np.log(0.1 / c_cfg.tau))

                codes = [F.one_hot(b, num_classes=bins_per_plane).cuda()
                            for b in bins_vec[:, start_i:end_i].long()]

                prior_batch = prior_model.get_priors(codes=codes, y=si_batch, tau=tau)
                prior_batch = torch.stack(prior_batch)

                # Move to GPU for loss computation
                prior_batch = prior_batch.cuda()
                bins_batch = bins_batch.cuda()

                # Compute negative log-likelihood loss (cross-entropy)
                loss = 0.0
                for i in range(num_planes):
                    # Log-likelihood of actual bins under the predicted distribution
                    prior_slice = prior_batch[i][torch.arange(prior_batch[i].shape[0]), bins_batch[i].to(torch.long)]
                    loss += -torch.log(prior_slice + 1e-12).mean()

                loss = loss / num_planes
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
            epoch_loss /= batch_idx+1

        pbar.close()

        prior_model.cpu().eval()
        torch.cuda.empty_cache()

        return (prior_model, epoch_loss) if return_loss else prior_model
