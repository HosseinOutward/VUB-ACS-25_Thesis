import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
from utils import create_training_progress_bar

class PriorCalculator:
    @staticmethod
    def compute_rate_from_prior_tensor(prior: torch.Tensor, bins: torch.Tensor, num_planes: int) -> float:
        """Compute rate from prior probabilities for given bins."""
        n_samples = bins.shape[1]
        return sum(
            -torch.log2(prior[i, torch.arange(n_samples), bins[i].long()] + 1e-10).mean().item()
            for i in range(num_planes)
        )

    @staticmethod
    def get_hash(x_vec: torch.Tensor, sample_size: int = 128) -> str:
        sample = x_vec[:sample_size*3:3]
        sample = sample.cpu().numpy().round(decimals=1).astype(np.int32)
        hasher = hashlib.md5()
        hasher.update(sample.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def compute_marginal_prior(bins_vec: torch.Tensor, bins_per_plane, num_planes) -> torch.Tensor:
        vec_size = bins_vec.size(1)
        probs_per_plane = []
        for b_vec in bins_vec:
            counts = torch.bincount(b_vec, minlength=bins_per_plane)
            probs = counts / counts.sum()
            probs_per_plane.append(probs)

        # Broadcast probabilities to all positions: (num_planes, N, bins_per_plane)
        probs_array = torch.stack(probs_per_plane).to(torch.float16)  # (num_planes, bins_per_plane)
        prior = torch.broadcast_to(
            probs_array[:, torch.newaxis, :],
            (num_planes, vec_size, bins_per_plane)
        )

        return prior

    @staticmethod
    def _compute_prior_from_network(q_model, bins_vec, side_info, training_tau=False, batch_size=500_000):
        from cancer_quantizer import WZQuantizerCancer

        training_mode = training_tau is not False

        bins_vec = bins_vec.to(torch.long)
        bins_per_plane = q_model.bins_per_plane
        def func(start_i, end_idx):
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

        # Add small epsilon to actual bins and renormalize
        if not training_mode:
            for i in range(priors.shape[0]):
                priors[i, torch.arange(priors.shape[1]), bins_vec[i]] += 1e-6
                priors[i] /= priors[i].sum(dim=-1, keepdim=True)

        return priors

    @staticmethod
    def train_prior_model(bins_vec, side_info, num_planes, bins_per_plane, c_cfg = None,
                          train_sample_size=3e5, batch_size=500_000) -> EncoderDecoderLayeredRNN:
        from cancer_protocol import CancerConfig
        if c_cfg is None:
            c_cfg = CancerConfig()

        assert bins_vec.size(0) == num_planes, "bins_vec first dimension must match num_planes"

        prior_model = EncoderDecoderLayeredRNN(
            num_planes=num_planes, bins_per_plane=bins_per_plane, side_info_size=side_info.size(1),
            input_dim=1, layers=3, hidden_dim=100, marginal=False
        )
        prior_model.cuda().train()

        optimizer = torch.optim.AdamW(prior_model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Convert to long once
        vec_size = bins_vec.size(1)
        total_samples = int(min(train_sample_size, vec_size))
        num_batches = (total_samples + batch_size - 1) // batch_size
        pbar = create_training_progress_bar(
            c_cfg.train_epochs * num_batches,
            desc="Prior Model")
        for epoch in range(c_cfg.train_epochs):
            indices = torch.randint(0, vec_size, (total_samples,), dtype=torch.long)
            bins_subset, si_subset = bins_vec[:, indices], side_info[indices]

            epoch_loss = 0.0
            for start_i in range(0, total_samples, batch_size):
                end_i = min(start_i + batch_size, total_samples)
                bins_batch, si_batch = bins_subset[:, start_i:end_i], si_subset[start_i:end_i]

                si_batch += torch.randn_like(si_batch) * (1e-4 * si_batch.abs().mean())

                training_prog = epoch / (c_cfg.train_epochs + 1)
                tau = c_cfg.tau * np.exp(training_prog * np.log(0.1 / c_cfg.tau))

                # Get prior predictions
                prior_batch = PriorCalculator._compute_prior_from_network(
                    prior_model, bins_batch, si_batch,
                    training_tau=tau, batch_size=batch_size)

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


        pbar.close()

        prior_model.cpu().eval()
        torch.cuda.empty_cache()

        return prior_model