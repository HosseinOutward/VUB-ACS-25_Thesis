
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ----------------- utilities -----------------

def _kmeans(y: np.ndarray, n_clusters: int, n_init: int = 5, max_iter: int = 100, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means returning (centers, labels). y: (T, m)."""
    if rng is None:
        rng = np.random.default_rng(42)
    T = y.shape[0]
    idx = rng.choice(T, size=n_clusters, replace=False) if T >= n_clusters else rng.integers(0, T, size=n_clusters)
    centers = y[idx].copy()
    for _ in range(max_iter):
        d2 = ((y[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (T, K)
        labels = d2.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = y[mask].mean(axis=0)
            else:
                new_centers[k] = y[rng.integers(0, T)]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels

def _quantize_x(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize x to n_bins using quantile binning. Returns (x_grid (M,), bin_index (T,))."""
    x = x.astype(float).ravel()
    n_bins = int(min(max(4, n_bins), max(4, x.shape[0])))
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    eps = 1e-12
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    x_grid = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        if np.any(m):
            x_grid[b] = x[m].mean()
        else:
            x_grid[b] = 0.5 * (edges[b] + edges[b + 1])
    return x_grid, bin_idx

def _empirical_p_x_given_u_from_soft(x_bin_idx: np.ndarray, W: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute p(x|u) for discrete x bins from soft weights W(t,k)=p(u|y_t)."""
    T, K = W.shape
    M = n_bins
    counts = np.zeros((K, M), dtype=float)
    w_u = W.sum(axis=0) + 1e-16
    for t in range(T):
        b = int(x_bin_idx[t])
        counts[:, b] += W[t, :]
    p_x_given_u = counts / w_u[:, None]
    p_x_given_u = np.clip(p_x_given_u, 1e-16, 1.0)
    p_x_given_u = p_x_given_u / p_x_given_u.sum(axis=1, keepdims=True)
    p_u = w_u / w_u.sum()
    return p_x_given_u, p_u

def _empirical_p_x_given_u_from_hard(x_bin_idx: np.ndarray, u_idx: np.ndarray, K: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute p(x|u) using hard labels u_idx, without building a T×K matrix."""
    counts = np.bincount(M*u_idx + x_bin_idx, minlength=K*M).reshape(K, M).astype(float)
    p_u = counts.sum(axis=1); p_u /= p_u.sum()
    p_x_given_u = counts + 1e-16
    p_x_given_u /= p_x_given_u.sum(axis=1, keepdims=True)
    return p_x_given_u, p_u

def _distances(x_grid: np.ndarray, r_grid: np.ndarray, kind: str = "mspe", eps: float = 1e-6, mape_percent: bool = False) -> np.ndarray:
    """Return distortion matrix for 'mspe' or 'mape'."""
    x_col = x_grid[:, None]
    r_row = r_grid[None, :]
    if kind.lower() == "mspe":
        num = (x_col - r_row) ** 2
        den = (np.abs(x_col) + eps) ** 2
        return num / den
    elif kind.lower() == "mape":
        frac = np.abs(x_col - r_row) / (np.abs(x_col) + eps)
        return (100.0 * frac) if mape_percent else frac
    else:
        raise ValueError("Unknown distortion kind: %r" % kind)

def _ba_bucket(p_x: np.ndarray, dmat: np.ndarray, s: float, max_iter: int = 500, tol: float = 1e-7) -> Tuple[np.ndarray, float, float]:
    """Blahut–Arimoto for a single bucket (discrete source, fixed recon grid). Returns (r, D, R_bits)."""
    M, J = dmat.shape
    r = np.ones(J) / J
    for _ in range(max_iter):
        a = np.exp(-s * dmat) * r[None, :]
        a_sum = a.sum(axis=1, keepdims=True)
        a_sum[a_sum == 0] = 1e-300
        q = a / a_sum
        r_new = (p_x[:, None] * q).sum(axis=0)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    D = float((p_x[:, None] * q * dmat).sum())
    with np.errstate(divide='ignore'):
        log_term = np.log(q + 1e-300) - np.log(r[None, :] + 1e-300)
    R_bits = float((p_x[:, None] * q * log_term).sum()) / np.log(2.0)
    return r, D, R_bits

# ----------------- main APIs -----------------

def conditional_rd(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    u_idx: Optional[np.ndarray] = None,
    n_buckets: int = 8,
    n_x_bins: int = 64,
    n_recon: int = 64,
    s_grid: Optional[np.ndarray] = None,
    D_targets: Optional[np.ndarray] = None,
    dist: str = "mspe",
    mape_percent: bool = False,
    eps: float = 1e-6,
    max_iter: int = 400,
    tol: float = 1e-6,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """General conditional RD with choice of distortion ('mspe' or 'mape').
    Exactly one of {W, u_idx, y} must be provided to define buckets U:
      - W: soft weights p(u|y_t), shape (T,K)
      - u_idx: hard labels (T,), integers in [0..K-1]
      - y: will be clustered via k-means into n_buckets buckets (hard)
    Returns dict with arrays D (distortion) and R (bits/sample) across s_grid, plus metadata.
    """
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, dtype=float).ravel()
    T = x.shape[0]
    if s_grid is None:
        s_grid = np.logspace(-3, 3, 25)
    s_grid = np.asarray(s_grid, dtype=float)

    # X alphabet and recon grid
    x_grid, x_bin_idx = _quantize_x(x, n_x_bins)
    r_grid = np.quantile(x, np.linspace(0, 1, n_recon)).astype(float)
    dmat = _distances(x_grid, r_grid, kind=dist, eps=eps, mape_percent=mape_percent)

    # Buckets
    if W is not None:
        W = np.asarray(W, dtype=float)
        row_sums = W.sum(axis=1, keepdims=True) + 1e-16
        W = W / row_sums
        p_x_given_u, p_u = _empirical_p_x_given_u_from_soft(x_bin_idx, W, x_grid.shape[0])
    elif u_idx is not None:
        u_idx = np.asarray(u_idx, int).ravel()
        assert u_idx.shape[0] == T
        K = int(u_idx.max()) + 1
        p_x_given_u, p_u = _empirical_p_x_given_u_from_hard(x_bin_idx, u_idx, K, x_grid.shape[0])
    else:
        if y is None:
            raise ValueError("Provide one of: W (soft), u_idx (hard), or y (to cluster).")
        y = np.asarray(y, float)
        if y.ndim == 1:
            y = y[:, None]
        assert y.shape[0] == T
        centers, labels = _kmeans(y, n_buckets, rng=rng)
        K = centers.shape[0]
        p_x_given_u, p_u = _empirical_p_x_given_u_from_hard(x_bin_idx, labels.astype(int), K, x_grid.shape[0])

    # Sweep slopes
    R_list, D_list = [], []
    per_bucket_last = []
    for s in s_grid:
        R_sum = 0.0; D_sum = 0.0
        this_stats = []
        for k in range(p_x_given_u.shape[0]):
            if p_u[k] < 1e-14:
                this_stats.append({'R': 0.0, 'D': 0.0, 'r': np.full(r_grid.shape, np.nan)})
                continue
            r_k, D_k, R_k = _ba_bucket(p_x_given_u[k], dmat, s, max_iter=max_iter, tol=tol)
            R_sum += p_u[k]*R_k; D_sum += p_u[k]*D_k
            this_stats.append({'R': R_k, 'D': D_k, 'r': r_k})
        R_list.append(R_sum); D_list.append(D_sum); per_bucket_last = this_stats

    R = np.array(R_list); D = np.array(D_list)
    out = dict(D=D, R=R, s_grid=s_grid, x_grid=x_grid, r_grid=r_grid, p_u=p_u,
               per_bucket=per_bucket_last, dist=dist, mape_percent=mape_percent)

    if D_targets is not None:
        D_targets = np.asarray(D_targets, float).ravel()
        order = np.argsort(D)[::-1]
        D_sorted = D[order]; R_sorted = R[order]
        _, uniq_idx = np.unique(np.round(D_sorted, 12), return_index=True)
        Du = D_sorted[sorted(uniq_idx)]; Ru = R_sorted[sorted(uniq_idx)]
        inc = np.argsort(Du); Du_inc, Ru_inc = Du[inc], Ru[inc]
        R_at = np.full(D_targets.shape, np.nan, float)
        if Du_inc.size >= 2:
            for i, Dt in enumerate(D_targets):
                if Du_inc.min() <= Dt <= Du_inc.max():
                    R_at[i] = np.interp(Dt, Du_inc, Ru_inc)
        out["R_at_D"] = R_at

    return out

def conditional_rd_mspe(**kwargs) -> Dict[str, Any]:
    return conditional_rd(dist="mspe", **kwargs)

def conditional_rd_mape(mape_percent: bool = False, **kwargs) -> Dict[str, Any]:
    return conditional_rd(dist="mape", mape_percent=mape_percent, **kwargs)

def conditional_rd_hard_labels(x: np.ndarray, u_idx: np.ndarray, **kwargs) -> Dict[str, Any]:
    return conditional_rd(x=x, u_idx=u_idx, **kwargs)
