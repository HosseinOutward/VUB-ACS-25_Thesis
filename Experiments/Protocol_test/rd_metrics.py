from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray


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
    worse: more rate at matched distortion or more distortion at matched rate.
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


def rd_stage_rate_gaps(
    rates: NDArray[np.float64],
    distortions_db: NDArray[np.float64],
    ref_rates: NDArray[np.float64],
    ref_distortions_db: NDArray[np.float64],
    name: str = "curve vs reference",
) -> NDArray[np.float64]:
    """Measure each stage's extra rate against the reference at matched distortion."""
    assert rates.size == ref_rates.size, (
        f"{name}: stage-wise gaps pair stage k with stage k, but the sides emit "
        f"{rates.size} and {ref_rates.size} stages."
    )
    order, ref_order = np.argsort(distortions_db), np.argsort(ref_distortions_db)
    at_reference = np.interp(
        distortions_db,
        ref_distortions_db[ref_order],
        ref_rates[ref_order],
        left=np.nan,
        right=np.nan,
    )
    at_curve = np.interp(
        ref_distortions_db,
        distortions_db[order],
        rates[order],
        left=np.nan,
        right=np.nan,
    )
    gaps = np.where(np.isfinite(at_reference), rates - at_reference, at_curve - ref_rates)
    if not np.isfinite(gaps).all():
        logging.warning(
            "%s: %d of %d stages sit outside the other curve's distortion range from both sides "
            "and have no gap.",
            name,
            int((~np.isfinite(gaps)).sum()),
            gaps.size,
        )
    return gaps
