

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any

import torch

from FL_code.cancer_protocol import CancerCodec, CancerRecord


def _kmeans_clustering(
        data: np.ndarray,
        n_clusters: int,
        n_init: int = 5,
        max_iter: int = 50,
        rng: Optional[np.random.Generator] = None,
        batch_size: int = 65536,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means clustering, written to avoid huge (N,K,F) broadcasted temporaries.

    Uses squared Euclidean distance identity:
        ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    and computes assignments in batches.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_samples, n_features)")

    n_samples, n_features = data.shape
    n_clusters = int(min(max(1, n_clusters), n_samples))

    # Initialize centers
    if n_samples >= n_clusters:
        init_indices = rng.choice(n_samples, size=n_clusters, replace=False)
    else:
        init_indices = rng.integers(0, n_samples, size=n_clusters)
    centers = data[init_indices].copy()

    def assign_labels(x: np.ndarray, c: np.ndarray) -> np.ndarray:
        c_norm = (c * c).sum(axis=1)  # (K,)
        labels_out = np.empty(x.shape[0], dtype=np.int32)
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = x[start:end]                 # (B,F)
            x_norm = (xb * xb).sum(axis=1)    # (B,)
            d2 = x_norm[:, None] + c_norm[None, :] - 2.0 * (xb @ c.T)  # (B,K)
            labels_out[start:end] = d2.argmin(axis=1).astype(np.int32)
        return labels_out

    labels = assign_labels(data, centers)

    for _ in range(max_iter):
        counts = np.bincount(labels, minlength=n_clusters).astype(np.int64)
        sums = np.zeros((n_clusters, n_features), dtype=np.float32)
        np.add.at(sums, labels, data)

        new_centers = centers.copy()
        nonempty = counts > 0
        new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]

        empty = np.where(~nonempty)[0]
        if empty.size:
            refill_idx = rng.integers(0, n_samples, size=empty.size)
            new_centers[empty] = data[refill_idx]

        if np.allclose(new_centers, centers, rtol=1e-5, atol=1e-6):
            centers = new_centers
            break

        centers = new_centers
        labels = assign_labels(data, centers)

    return centers, labels



def _quantize_target(target: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize target signal into discrete bins using quantile-based binning.

    Args:
        target: Target signal to quantize
        n_bins: Number of quantization bins

    Returns:
        bin_values: Representative value for each bin (n_bins,)
        bin_indices: Bin assignment for each sample (n_samples,)
    """
    target = target.astype(float).ravel()
    n_samples = target.shape[0]

    # Ensure reasonable number of bins
    n_bins = int(min(max(4, n_bins), max(4, n_samples)))

    # Create quantile-based bin edges
    edges = np.quantile(target, np.linspace(0, 1, n_bins + 1))

    # Ensure edges are strictly increasing
    eps = 1e-12
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    # Assign samples to bins
    bin_indices = np.clip(
        np.searchsorted(edges, target, side="right") - 1,
        0,
        n_bins - 1
    )

    # Compute representative value for each bin (mean of samples in bin)
    bin_values = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_values[b] = target[mask].mean()
        else:
            # If bin is empty, use midpoint of bin edges
            bin_values[b] = 0.5 * (edges[b] + edges[b + 1])

    return bin_values, bin_indices


def _compute_conditional_pmf_soft(
        target_bin_indices: np.ndarray,
        soft_weights: np.ndarray,
        n_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute p(x|u) from soft cluster weights using Bayes' rule.

    For each cluster k and bin m:
    p(x=m|u=k) = sum_t [ w(k|y_t) * 1(x_t in bin m) ] / sum_t w(k|y_t)

    Args:
        target_bin_indices: Bin assignment for each sample (n_samples,)
        soft_weights: Soft cluster weights w(k|y_t) of shape (n_samples, n_clusters)
        n_bins: Number of target bins

    Returns:
        p_x_given_u: Conditional PMF p(x|u) of shape (n_clusters, n_bins)
        p_u: Marginal cluster probabilities p(u) of shape (n_clusters,)
    """
    n_samples, n_clusters = soft_weights.shape

    # Accumulate weighted counts for each (cluster, bin) pair
    counts = np.zeros((n_clusters, n_bins), dtype=np.float32)
    for t in range(n_samples):
        bin_idx = int(target_bin_indices[t])
        counts[:, bin_idx] += soft_weights[t, :]

    # Compute marginal cluster probabilities
    cluster_weights = soft_weights.sum(axis=0) + 1e-16
    p_u = cluster_weights / cluster_weights.sum()

    # Normalize to get conditional probabilities p(x|u)
    p_x_given_u = counts / cluster_weights[:, None]
    p_x_given_u = np.clip(p_x_given_u, 1e-16, 1.0)
    p_x_given_u = p_x_given_u / p_x_given_u.sum(axis=1, keepdims=True)

    return p_x_given_u, p_u


def _compute_conditional_pmf_hard(
        target_bin_indices: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        n_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute p(x|u) from hard cluster labels using empirical counts.

    Args:
        target_bin_indices: Bin assignment for each sample (n_samples,)
        cluster_labels: Hard cluster label for each sample (n_samples,)
        n_clusters: Number of clusters
        n_bins: Number of target bins

    Returns:
        p_x_given_u: Conditional PMF p(x|u) of shape (n_clusters, n_bins)
        p_u: Marginal cluster probabilities p(u) of shape (n_clusters,)
    """
    # Use bincount for efficient counting of (cluster, bin) pairs
    # Encode pair (k, m) as k * n_bins + m
    pair_indices = n_bins * cluster_labels + target_bin_indices
    counts = np.bincount(pair_indices, minlength=n_clusters * n_bins)
    counts = counts.reshape(n_clusters, n_bins).astype(float)

    # Compute marginal cluster probabilities
    cluster_counts = counts.sum(axis=1)
    p_u = cluster_counts / cluster_counts.sum()

    # Normalize to get conditional probabilities p(x|u)
    p_x_given_u = counts + 1e-16
    p_x_given_u = p_x_given_u / p_x_given_u.sum(axis=1, keepdims=True)

    return p_x_given_u, p_u


def _compute_mspe_distortion(
        target_values: np.ndarray,
        reconstruction_values: np.ndarray,
        epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute MSPE (Mean Squared Percentage Error) distortion matrix.

    MSPE(x, r) = [(x - r) / (|x| + ε)]²

    Args:
        target_values: Target alphabet values (n_bins,)
        reconstruction_values: Reconstruction alphabet values (n_recon,)
        epsilon: Small constant for numerical stability

    Returns:
        distortion_matrix: Distortion d(x, r) of shape (n_bins, n_recon)
    """
    # Reshape for broadcasting: (n_bins, 1) and (1, n_recon)
    x = target_values[:, None]
    r = reconstruction_values[None, :]

    # MSPE: [(x - r) / (|x| + ε)]²
    numerator = (x - r) ** 2
    denominator = (np.abs(x) + epsilon) ** 2

    return numerator / denominator


def _blahut_arimoto(
        source_pmf: np.ndarray,
        distortion_matrix: np.ndarray,
        slope: float,
        max_iter: int = 500,
        tol: float = 1e-7
) -> Tuple[np.ndarray, float, float]:
    """
    Blahut-Arimoto algorithm for rate-distortion computation.

    Iteratively computes the optimal reconstruction distribution and test channel
    for a given Lagrange multiplier (slope).

    Args:
        source_pmf: Source probability p(x) of shape (n_bins,)
        distortion_matrix: Distortion d(x, r) of shape (n_bins, n_recon)
        slope: Lagrange multiplier λ for R + λD optimization
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        recon_pmf: Reconstruction distribution p(r) of shape (n_recon,)
        distortion: Expected distortion D
        rate_bits: Rate R in bits
    """
    n_bins, n_recon = distortion_matrix.shape

    # Initialize reconstruction distribution uniformly
    recon_pmf = np.ones(n_recon) / n_recon

    # Initialize test channel
    exponent = np.exp(-slope * distortion_matrix) * recon_pmf[None, :]
    exponent_sum = exponent.sum(axis=1, keepdims=True)
    exponent_sum[exponent_sum == 0] = 1e-300
    test_channel = exponent / exponent_sum

    # Iterate Blahut-Arimoto updates
    for _ in range(max_iter):
        # Update test channel: q(r|x) ∝ p(r) * exp(-λ * d(x,r))
        exponent = np.exp(-slope * distortion_matrix) * recon_pmf[None, :]
        exponent_sum = exponent.sum(axis=1, keepdims=True)
        exponent_sum[exponent_sum == 0] = 1e-300  # Avoid division by zero
        test_channel = exponent / exponent_sum

        # Update reconstruction distribution: p(r) = sum_x p(x) * q(r|x)
        recon_pmf_new = (source_pmf[:, None] * test_channel).sum(axis=0)

        # Check convergence
        if np.linalg.norm(recon_pmf_new - recon_pmf, 1) < tol:
            recon_pmf = recon_pmf_new
            break
        recon_pmf = recon_pmf_new

    # Compute expected distortion: D = sum_{x,r} p(x) * q(r|x) * d(x,r)
    distortion = float((source_pmf[:, None] * test_channel * distortion_matrix).sum())

    # Compute rate: R = sum_{x,r} p(x) * q(r|x) * log[q(r|x) / p(r)]
    with np.errstate(divide='ignore'):
        log_ratio = np.log(test_channel + 1e-300) - np.log(recon_pmf[None, :] + 1e-300)
    rate_bits = float((source_pmf[:, None] * test_channel * log_ratio).sum()) / np.log(2.0)

    return recon_pmf, distortion, rate_bits


def wyner_ziv_bound(
        target: np.ndarray,
        side_info: Optional[np.ndarray] = None,
        soft_weights: Optional[np.ndarray] = None,
        hard_labels: Optional[np.ndarray] = None,
        n_clusters: int = 8,
        n_target_bins: int = 64,
        n_reconstruction_points: int = 64,
        slope_grid: Optional[np.ndarray] = None,
        distortion_targets: Optional[np.ndarray] = None,
        epsilon: float = 1e-6,
        max_iterations: int = 400,
        tolerance: float = 1e-6,
        random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute the Wyner-Ziv rate-distortion bound using conditional rate-distortion theory.

    The target signal is encoded with side information available only at the decoder.
    Uses MSPE (Mean Squared Percentage Error) as the distortion metric.

    Args:
        target: Target signal to encode (1D array)
        side_info: Side information available at decoder (1D or 2D array)
        soft_weights: Soft cluster assignments W[t,k] = p(cluster_k | side_info_t)
        hard_labels: Hard cluster assignments for each sample
        n_clusters: Number of clusters for side information (if clustering needed)
        n_target_bins: Number of quantization bins for target alphabet
        n_reconstruction_points: Number of reconstruction points
        slope_grid: Lagrange multiplier values for RD curve (default: logspace(-3, 3, 25))
        distortion_targets: Specific distortion values to interpolate rates at
        epsilon: Small constant for numerical stability
        max_iterations: Maximum iterations for Blahut-Arimoto algorithm
        tolerance: Convergence tolerance for Blahut-Arimoto
        random_seed: Random seed for k-means clustering

    Returns:
        Dictionary containing:
            - D: Distortion values
            - R: Rate values (bits)
            - slope_grid: Lagrange multipliers used
            - target_alphabet: Quantized target values
            - reconstruction_alphabet: Reconstruction point values
            - cluster_probs: Probability of each cluster p(u)
            - R_at_D: Interpolated rates at distortion_targets (if provided)
    """
    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Prepare target signal
    target = np.asarray(target, dtype=np.float32).ravel()
    n_samples = target.shape[0]

    # Set default slope grid if not provided
    if slope_grid is None:
        slope_grid = np.logspace(-3, 3, 25)
    slope_grid = np.asarray(slope_grid, dtype=np.float32)

    # Step 1: Quantize target signal into discrete alphabet
    target_alphabet, target_bin_indices = _quantize_target(target, n_target_bins)

    # Step 2: Create reconstruction alphabet (quantile-based)
    reconstruction_alphabet = np.quantile(target, np.linspace(0, 1, n_reconstruction_points)).astype(float)

    # Step 3: Compute MSPE distortion matrix d(x, r)
    distortion_matrix = _compute_mspe_distortion(target_alphabet, reconstruction_alphabet, epsilon)

    # Step 4: Compute conditional distribution p(x|u) based on side information clustering
    if soft_weights is not None:
        # Use provided soft cluster weights
        soft_weights = np.asarray(soft_weights, dtype=np.float32)
        # Normalize rows to sum to 1
        row_sums = soft_weights.sum(axis=1, keepdims=True) + 1e-16
        soft_weights = soft_weights / row_sums
        p_x_given_u, p_u = _compute_conditional_pmf_soft(target_bin_indices, soft_weights, len(target_alphabet))

    elif hard_labels is not None:
        # Use provided hard cluster labels
        hard_labels = np.asarray(hard_labels, dtype=int).ravel()
        assert hard_labels.shape[0] == n_samples, "hard_labels must have same length as target"
        n_clusters = int(hard_labels.max()) + 1
        p_x_given_u, p_u = _compute_conditional_pmf_hard(target_bin_indices, hard_labels, n_clusters,
                                                         len(target_alphabet))

    else:
        # Cluster side information using k-means
        if side_info is None:
            raise ValueError("Must provide one of: soft_weights, hard_labels, or side_info for clustering")

        side_info = np.asarray(side_info, dtype=np.float32)
        if side_info.ndim == 1:
            side_info = side_info[:, None]
        assert side_info.shape[0] == n_samples, "side_info must have same length as target"

        # Perform k-means clustering on side information
        centers, labels = _kmeans_clustering(side_info, n_clusters, rng=rng)
        n_clusters = centers.shape[0]
        p_x_given_u, p_u = _compute_conditional_pmf_hard(target_bin_indices, labels.astype(int), n_clusters,
                                                         len(target_alphabet))

    # Step 5: Compute Wyner-Ziv bound by sweeping over Lagrange multipliers
    # For each slope λ, compute conditional RD for each cluster and average
    rate_list = []
    distortion_list = []
    per_cluster_stats = []

    for slope in slope_grid:
        total_rate = 0.0
        total_distortion = 0.0
        cluster_stats = []

        for cluster_idx in range(p_x_given_u.shape[0]):
            # Skip clusters with negligible probability
            if p_u[cluster_idx] < 1e-14:
                cluster_stats.append({
                    'R': 0.0,
                    'D': 0.0,
                    'recon_pmf': np.full(reconstruction_alphabet.shape, np.nan)
                })
                continue

            # Run Blahut-Arimoto for this cluster
            recon_pmf, D_cluster, R_cluster = _blahut_arimoto(
                p_x_given_u[cluster_idx],
                distortion_matrix,
                slope,
                max_iter=max_iterations,
                tol=tolerance
            )

            # Weighted average by cluster probability
            total_rate += p_u[cluster_idx] * R_cluster
            total_distortion += p_u[cluster_idx] * D_cluster

            cluster_stats.append({
                'R': R_cluster,
                'D': D_cluster,
                'recon_pmf': recon_pmf
            })

        rate_list.append(total_rate)
        distortion_list.append(total_distortion)
        per_cluster_stats = cluster_stats  # Store stats from last slope for debugging

    # Convert to arrays
    rate = np.array(rate_list)
    distortion = np.array(distortion_list)

    # Prepare output dictionary
    result = {
        'D': distortion,
        'R': rate,
        'slope_grid': slope_grid,
        'target_alphabet': target_alphabet,
        'reconstruction_alphabet': reconstruction_alphabet,
        'cluster_probs': p_u,
        'per_cluster_stats': per_cluster_stats
    }

    # Step 6: Interpolate rates at specific distortion targets if requested
    if distortion_targets is not None:
        distortion_targets = np.asarray(distortion_targets, dtype=np.float32).ravel()

        # Sort by distortion in descending order and remove duplicates
        sort_indices = np.argsort(distortion)[::-1]
        D_sorted = distortion[sort_indices]
        R_sorted = rate[sort_indices]

        # Remove duplicate distortion values (keep first occurrence)
        _, unique_indices = np.unique(np.round(D_sorted, 12), return_index=True)
        D_unique = D_sorted[sorted(unique_indices)]
        R_unique = R_sorted[sorted(unique_indices)]

        # Sort in increasing order for interpolation
        increasing_order = np.argsort(D_unique)
        D_increasing = D_unique[increasing_order]
        R_increasing = R_unique[increasing_order]

        # Interpolate
        R_at_D = np.full(distortion_targets.shape, np.nan, dtype=np.float32)
        if D_increasing.size >= 2:
            for i, D_target in enumerate(distortion_targets):
                if D_increasing.min() <= D_target <= D_increasing.max():
                    R_at_D[i] = np.interp(D_target, D_increasing, R_increasing)

        result['R_at_D'] = R_at_D

    return result


class CancerWithBoundCalc(CancerCodec):
    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict:
        payload = super()._compress(delta_vec, record)

        bound_every = 3
        if (record.round_id % bound_every) != 0:
            return payload

        max_samples = 1_000_000
        folder_path = self.fl_cfg.records_dir + '/wz_bounds/'
        os.makedirs(folder_path, exist_ok=True)

        quantizer = self.frozen_quantizers[record.client_id]

        # Pull SI + target on-device, then SUBSAMPLE before moving to CPU/NumPy.
        si_t = quantizer.get_si_data(for_prior=True).cpu()
        target_t = delta_vec.cpu()

        max_samples = min(max_samples, target_t.numel())

        idx = torch.randperm(target_t.numel())[:max_samples]
        target = target_t[idx].float().numpy()
        si = si_t[idx].float().numpy()

        bound_dict = wyner_ziv_bound(
            target=target, side_info=si,
            slope_grid=np.logspace(-3, 3, 20, dtype=np.float32),
            distortion_targets=np.array([0.05, 0.4], dtype=np.float32),
            n_clusters=8, n_target_bins=56, n_reconstruction_points=56,
        )

        np.savez_compressed(
            folder_path + f'bound_rid{record.round_id}_cid{record.client_id}.npz',
            dist_v=bound_dict['D'], rate_v=bound_dict['R'],)

        return payload


# Example usage:
# target.shape == (n_samples,); side_info.shape == (n_samples, n_features);
# result = wyner_ziv_bound(
#     target=target,
#     side_info=side_info,
#     slope_grid=np.logspace(-3, 3, 20),
#     distortion_targets=np.array([0.05, 0.4]),
#     n_clusters=8,
#     n_target_bins=56,
#     n_reconstruction_points=56,
# )
# np.savez_compressed(f'wz_bound_agent-{agent_id}_round-{round_id}.npz',
#                     dist_v=result['D'], rate_v=result['R'])