from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

# Label for `rd_curve_gap`'s output in analysis tables.
GAP_METRIC_NAME = "mean_interpolated_equal_distortion_gap"


def wmape_to_db(wmape_percent: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert positive linear WMAPE percentages to the repository's amplitude-ratio dB scale."""
    values = np.asarray(wmape_percent, dtype=np.float64)
    assert np.isfinite(values).all() and (values > 0).all(), "WMAPE must be finite and positive."
    return 20 * np.log10(values / 100)


def mean_wmape_db(wmape_percent: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average equal-weight WMAPE observations in linear space, then convert their curve to dB."""
    values = np.asarray(wmape_percent, dtype=np.float64)
    assert values.ndim >= 2, "Expected observations by stage before averaging WMAPE."
    return wmape_to_db(values.mean(axis=0))


def _mean_curve_gap(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ref_x: NDArray[np.float64],
    ref_y: NDArray[np.float64],
    name: str,
) -> tuple[float, float]:
    """Mean of y minus ref_y over the x span both curves cover, paired with the reference's mean y there.

    Both curves are read as piecewise-linear functions of x through their own points, so the caller
    picks which axis is matched by which way round it passes them.
    """
    low, high = max(float(x.min()), float(ref_x.min())), min(float(x.max()), float(ref_x.max()))
    if high <= low:
        logging.warning("%s: the two curves share no overlapping range, so their gap is undefined.", name)
        return float("nan"), float("nan")

    grid = np.unique(np.concatenate([x, ref_x, [low, high]]))
    grid = grid[(grid >= low) & (grid <= high)]
    at, ref_at = (
        np.interp(grid, axis[np.argsort(axis)], value[np.argsort(axis)])
        for axis, value in ((x, y), (ref_x, ref_y))
    )
    return (
        float(np.trapezoid(at - ref_at, grid)) / (high - low),
        float(np.trapezoid(ref_at, grid)) / (high - low),
    )


def rd_curve_gap(
    rates: NDArray[np.float64],
    distortions_db: NDArray[np.float64],
    ref_rates: NDArray[np.float64],
    ref_distortions_db: NDArray[np.float64],
    name: str = "curve vs reference",
) -> dict[str, float]:
    """Measure mean rate and distortion gaps between an RD curve and a reference curve.

    Each curve is read as its own piecewise-linear function. Positive values mean the candidate is
    worse: more rate at matched distortion or more distortion at matched rate. This is the
    `GAP_METRIC_NAME` interpolation gap.
    """
    rate_gap, mean_ref_rate = _mean_curve_gap(
        distortions_db, rates, ref_distortions_db, ref_rates, f"{name} rate"
    )
    distortion_gap, _ = _mean_curve_gap(
        rates, distortions_db, ref_rates, ref_distortions_db, f"{name} distortion"
    )
    return {
        "rate_gap_bits": rate_gap,
        "rate_gap_percent": 100.0 * rate_gap / mean_ref_rate,
        "distortion_gap_db": distortion_gap,
        "distortion_gap_percent": 100.0 * (10.0 ** (distortion_gap / 20.0) - 1.0),
    }
