
import numpy as np
from typing import Optional, Tuple, Dict, Any

def _kmeans(y: np.ndarray, n_clusters: int, n_init: int = 5, max_iter: int = 100, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means returning (centers, labels). y: (T, m)."""
    if rng is None:
        rng = np.random.default_rng(42)
    T = y.shape[0]
    # init centers by sampling without replacement if possible
    idx = rng.choice(T, size=n_clusters, replace=False) if T >= n_clusters else rng.integers(0, T, size=n_clusters)
    centers = y[idx].copy()
    for _ in range(max_iter):
        # assign
        d2 = ((y[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (T, K)
        labels = d2.argmin(axis=1)
        # update
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = y[mask].mean(axis=0)
            else:
                # re-seed empty cluster
                new_centers[k] = y[rng.integers(0, T)]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels

def _quantize_x(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize x to n_bins using quantile binning. 
    Returns (x_grid (M,), bin_index (T,))."""
    x = x.astype(float).ravel()
    # unique small number of bins if data smaller
    n_bins = int(min(max(4, n_bins), max(4, x.shape[0])))
    # bin edges by quantiles
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing (handle ties)
    eps = 1e-12
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    # bin index
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    # representative value per bin: mean of points in bin (fallback to mid-edge)
    x_grid = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if np.any(m):
            x_grid[b] = x[m].mean()
        else:
            x_grid[b] = 0.5 * (edges[b] + edges[b + 1])
    return x_grid, bin_idx

def _build_bucket_weights(y: Optional[np.ndarray], W: Optional[np.ndarray], n_buckets: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (weights W (T,K), p_u (K,)). If W is None, run k-means on y and make hard 1-hot weights."""
    if W is not None:
        W = np.asarray(W, dtype=float)
        # normalize rows
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        p_u = W.mean(axis=0)
        return W, p_u
    if y is None:
        raise ValueError("Either W (p(u|y)) or y must be provided.")
    centers, labels = _kmeans(y, n_buckets)
    T = y.shape[0]
    K = centers.shape[0]
    W = np.zeros((T, K))
    W[np.arange(T), labels] = 1.0
    p_u = W.mean(axis=0)
    return W, p_u

def _empirical_p_x_given_u(x_bin_idx: np.ndarray, W: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute p(x|u) for discrete x bins from sample weights.
    Returns (p_x_given_u: (K, M), p_u: (K,))."""
    T, K = W.shape
    M = n_bins
    counts = np.zeros((K, M), dtype=float)
    w_u = W.sum(axis=0) + 1e-16
    for t in range(T):
        b = int(x_bin_idx[t])
        counts[:, b] += W[t, :]
    p_x_given_u = counts / w_u[:, None]
    # avoid zeros (for stability); do not renormalize to keep true mass zero if any
    p_x_given_u = np.clip(p_x_given_u, 1e-16, 1.0)
    # renormalize each row
    p_x_given_u = p_x_given_u / p_x_given_u.sum(axis=1, keepdims=True)
    p_u = w_u / w_u.sum()
    return p_x_given_u, p_u

def _mspe_distances(x_grid: np.ndarray, r_grid: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return distortion matrix d(i,j) = ((x_i - r_j) / (|x_i| + eps))^2, shape (M, J)."""
    M = x_grid.shape[0]; J = r_grid.shape[0]
    num = (x_grid[:, None] - r_grid[None, :]) ** 2
    denom = (np.abs(x_grid)[:, None] + eps) ** 2
    return num / denom

def _ba_bucket(p_x: np.ndarray, dmat: np.ndarray, s: float, max_iter: int = 500, tol: float = 1e-7) -> Tuple[np.ndarray, float, float]:
    """Run BA for a single bucket (discrete x, fixed reconstruction grid).
    Inputs:
      p_x: (M,), dmat: (M,J), s >= 0 (slope).
    Returns:
      r: (J,) reproduction marginal,
      D: scalar distortion,
      R_bits: scalar mutual information in bits.
    """
    M, J = dmat.shape
    r = np.ones(J) / J  # init
    for _ in range(max_iter):
        # E-step: p(hat|x)
        a = np.exp(-s * dmat) * r[None, :]  # (M, J)
        a_sum = a.sum(axis=1, keepdims=True)
        a_sum[a_sum == 0] = 1e-300
        q = a / a_sum  # (M,J)
        # M-step: r
        r_new = (p_x[:, None] * q).sum(axis=0)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    # compute D and R
    D = float((p_x[:, None] * q * dmat).sum())
    # R in nats then to bits
    with np.errstate(divide='ignore'):
        log_term = np.log(q + 1e-300) - np.log(r[None, :] + 1e-300)
    R_nats = float((p_x[:, None] * q * log_term).sum())
    R_bits = R_nats / np.log(2.0)
    return r, D, R_bits

def conditional_rd_mspe(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    n_buckets: int = 8,
    n_x_bins: int = 64,
    n_recon: int = 64,
    s_grid: Optional[np.ndarray] = None,
    D_targets: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    max_iter: int = 400,
    tol: float = 1e-6,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """Compute an empirical approximation to R_{X|Y}(D) with MSPE distortion.
    
    Two modes:
      - If W is provided (shape T x K, rows sum to 1), uses soft buckets U with p(u|y_t)=W[t,k].
      - Else clusters Y into K=n_buckets hard buckets (k-means) and uses those as U.
    
    We quantize X into n_x_bins mass points (x_grid), choose a reconstruction grid r_grid of size n_recon
    (initialized to x_grid), and for each slope s in s_grid run BA per bucket with the *common* slope s.
    
    Args:
      x: (T,) real-valued target samples aligned with Y.
      y: (T, m) side information (only used if W is None).
      W: (T, K) soft weights p(U=k | Y=y_t). If provided, n_buckets is ignored.
      n_buckets: number of clusters if W is None.
      n_x_bins: number of quantization bins for X (discrete source alphabet size M).
      n_recon: number of reconstruction points J (codebook size).
      s_grid: slopes to sweep; if None, uses np.logspace(-3, 3, 25).
      D_targets: optional array of desired distortions to interpolate the rate at.
      eps: epsilon for MSPE denominator (avoid divide-by-zero).
      max_iter, tol: BA convergence parameters.
      rng_seed: for reproducible k-means initialization.
    
    Returns dict with:
      - 'D': (S,) distortions (MSPE) for each slope
      - 'R': (S,) rates (bits/sample) for each slope
      - 's_grid': (S,) slopes used
      - 'x_grid': (M,) source mass points
      - 'r_grid': (J,) reconstruction grid used
      - 'p_u': (K,) bucket priors
      - 'per_bucket': list of K dicts with last-iteration stats {'R': float, 'D': float, 'r': (J,)}
      - if D_targets provided: 'R_at_D': (len(D_targets),) via monotone interpolation (nan if outside range)
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, dtype=float).ravel()
    T = x.shape[0]
    if s_grid is None:
        s_grid = np.logspace(-3, 3, 25)
    s_grid = np.asarray(s_grid, dtype=float)
    # X quantization and reconstruction grid (start from same points)
    x_grid, x_bin_idx = _quantize_x(x, n_x_bins)
    # start reconstruction grid as copies of x_grid shrunk/expanded to n_recon via quantiles of x
    r_quant = np.quantile(x, np.linspace(0, 1, n_recon))
    r_grid = r_quant.copy().astype(float)
    # build bucket weights W and p_u
    if W is not None:
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-16)
        W = W_norm
        p_u = W.mean(axis=0)
        K = W.shape[1]
    else:
        if y is None:
            raise ValueError("Provide either W (p(u|y)) or y for automatic bucketing.")
        centers, labels = _kmeans(np.asarray(y, dtype=float), n_buckets, rng=rng)
        K = centers.shape[0]
        W = np.zeros((T, K))
        W[np.arange(T), labels] = 1.0
        p_u = W.mean(axis=0)
    # compute p(x|u)
    p_x_given_u, p_u_check = _empirical_p_x_given_u(x_bin_idx, W, x_grid.shape[0])
    # sanity normalize
    p_u = p_u_check
    # precompute distortion matrices for all buckets (same dmat because distortion only depends on x,r)
    dmat = _mspe_distances(x_grid, r_grid, eps=eps)  # (M,J)
    S = s_grid.shape[0]
    R_list = []
    D_list = []
    per_bucket_last = [None] * K
    for s in s_grid:
        R_sum = 0.0
        D_sum = 0.0
        per_bucket_stats = []
        for k in range(K):
            pk = p_u[k]
            if pk < 1e-14:
                per_bucket_stats.append({'R': 0.0, 'D': 0.0, 'r': np.full(r_grid.shape, np.nan)})
                continue
            p_x = p_x_given_u[k]  # (M,)
            # BA for this bucket
            r_k, D_k, R_k = _ba_bucket(p_x, dmat, s, max_iter=max_iter, tol=tol)
            R_sum += pk * R_k
            D_sum += pk * D_k
            per_bucket_stats.append({'R': R_k, 'D': D_k, 'r': r_k})
        R_list.append(R_sum)
        D_list.append(D_sum)
        per_bucket_last = per_bucket_stats  # keep last stats (for s final value)
    R = np.array(R_list)
    D = np.array(D_list)
    out = {
        'D': D,
        'R': R,
        's_grid': s_grid,
        'x_grid': x_grid,
        'r_grid': r_grid,
        'p_u': p_u,
        'per_bucket': per_bucket_last,
    }

    # Optionally interpolate R at specific D targets (monotone in s; R increases as D decreases)
    if D_targets is not None:
        D_targets = np.asarray(D_targets, dtype=float).ravel()
        # sort by increasing D (s small -> larger D)
        order = np.argsort(D)[::-1]  # high D to low D
        D_sorted = D[order]
        R_sorted = R[order]
        # Remove duplicates in D for interpolation
        # Use rounding to merge numerical duplicates
        _, uniq_idx = np.unique(np.round(D_sorted, decimals=12), return_index=True)
        Du = D_sorted[sorted(uniq_idx)]
        Ru = R_sorted[sorted(uniq_idx)]
        # Ensure monotone increasing in D for interpolation
        # Interpolate with numpy (expects increasing x)
        R_at = np.full(D_targets.shape, np.nan)
        if Du.size >= 2:
            # If not strictly increasing, enforce by argsort
            inc = np.argsort(Du)
            Du_inc = Du[inc]
            Ru_inc = Ru[inc]
            for i, Dt in enumerate(D_targets):
                if Dt < Du_inc.min() or Dt > Du_inc.max():
                    R_at[i] = np.nan
                else:
                    R_at[i] = np.interp(Dt, Du_inc, Ru_inc)
        out['R_at_D'] = R_at
    return out
