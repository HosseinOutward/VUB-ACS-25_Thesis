"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols.

Find which coordinates gain from the extra Wyner-Ziv side information, and split the vector on it.

O and T code the same coordinates against different side information: O sees the whole server
ledger, T only the worker's own history. Their per-element dumps therefore say, coordinate by
coordinate, what the cross-worker columns bought. Averaging that over the most recent rounds gives
a stable partition, which a later round codec can train a separate quantizer on either side of.

Writes the partition mask plus a breakdown of who lands where.

TODO: This retained script emits the earlier binary majority mask. The generator that produced
the report's ``three_way_db.npy`` artifact was not retained and must be restored separately.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from compare_elements import RoundElements, load_round, model_layout  # noqa: E402


# Errors are stored divided by the vector's mean magnitude, so they are already WMAPE as a
# fraction. Below this the reconstruction is exact to storage precision and the decibel value is
# meaningless, so it is floored rather than sent to negative infinity.
WMAPE_FLOOR = 1e-6


def wmape_db(abs_error: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-coordinate distortion on the project's axis: WMAPE in dB relative to 100%."""
    return 20.0 * np.log10(np.maximum(abs_error, WMAPE_FLOOR))


def rate_at_distortion(
    rates: NDArray[np.float32], distortions: NDArray[np.float32], target: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Rate at which each coordinate's own curve first reaches ``target`` distortion.

    Rate is monotone by construction, being a cumulative sum of non-negative stage bits, but an
    individual coordinate's distortion is not: a stage that lowers the average can move one
    coordinate away from its target, which happens to 83% of them. So "the rate that buys this
    distortion" is the *first* stage to reach it, interpolated against the two real points that
    bracket the crossing. Nothing is reordered or replaced; a non-monotone curve is read as it is.

    Where a curve never reaches the target the result is its final rate, which understates what it
    would truly cost. ``unreached`` from ``curve_gap`` reports how often that happens.
    """
    reached = distortions <= target
    out = np.where(reached[0], rates[0], rates[-1])
    already = reached[0].copy()
    for stage in range(1, len(rates)):
        crossing = reached[stage] & ~already
        span = distortions[stage - 1] - distortions[stage]
        weight = np.where(span > 0, (distortions[stage - 1] - target) / np.maximum(span, 1e-12), 1.0)
        interpolated = rates[stage - 1] + np.clip(weight, 0.0, 1.0) * (rates[stage] - rates[stage - 1])
        out = np.where(crossing, interpolated, out)
        already |= reached[stage]
    return out


def client_curve_gap(
    oracle_dir: Path,
    temporal_dir: Path,
    rounds: tuple[int, ...],
    client_id: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compare one worker's two per-coordinate curves, averaged over rounds only.

    Each coordinate has its own three-stage rate-distortion curve on each side. The gap is the
    extra rate T's curve needs to reach each of the three distortions O's curve reaches, averaged
    over the three stages, exactly as a rate gap is read off a whole-vector curve. Rounds are
    averaged before comparing so the curve being compared is that coordinate's typical behaviour;
    nothing is pooled across coordinates or across workers.
    """
    sides: dict[str, list[RoundElements]] = {"o": [], "t": []}
    for round_id in rounds:
        sides["o"].append(load_round(oracle_dir, round_id, client_id))
        sides["t"].append(load_round(temporal_dir, round_id, client_id))
    o_rate = np.mean([r.cumulative_bits for r in sides["o"]], axis=0)
    o_dist = np.mean([r.abs_error for r in sides["o"]], axis=0)
    t_rate = np.mean([r.cumulative_bits for r in sides["t"]], axis=0)
    t_dist = np.mean([r.abs_error for r in sides["t"]], axis=0)
    del sides

    # Rounds are averaged in linear WMAPE and only then converted, which is how the whole-vector
    # curves in the shared protocol analysis are built, so the two are read on the same axis.
    o_dist, t_dist = wmape_db(o_dist), wmape_db(t_dist)

    # O's actual distortions are the targets; T's curve is asked what each would cost it.
    gap = np.mean([
        rate_at_distortion(t_rate, t_dist, o_dist[stage]) - o_rate[stage]
        for stage in range(len(o_rate))
    ], axis=0)
    unreached = float(np.mean(t_dist.min(axis=0) > o_dist.min(axis=0)))
    logging.info("worker %d: T never reaches O's best distortion on %.3f of coordinates.",
                 client_id, unreached)
    return gap.astype(np.float32), (t_rate - o_rate).astype(np.float32), (t_dist - o_dist).astype(np.float32)


def breakdown(
    gap: NDArray[np.float32],
    parameter_of: NDArray[np.int32],
    parameter_names: list[str],
    gains: NDArray[np.bool_],
) -> pd.DataFrame:
    """Where the gaining and losing coordinates sit, by parameter tensor."""
    rows = []
    for index, name in enumerate(parameter_names):
        in_tensor = parameter_of == index
        if not in_tensor.any():
            continue
        rows.append({
            "parameter": name,
            "count": int(in_tensor.sum()),
            "share_gaining": float(gains[in_tensor].mean()),
            "mean_gap_bits": float(gap[-1][in_tensor].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Measure the per-coordinate Wyner-Ziv benefit and write the split it implies."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=(16, 18, 20))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--fl-config", type=Path, default=EXPERIMENT_DIR / "replay_fl_config.json")
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "results" / "partitions")
    args = parser.parse_args()

    per_client = []
    for client_id in range(args.clients):
        gap, _, _ = client_curve_gap(
            args.oracle_run / "elements", args.temporal_run / "elements",
            tuple(args.rounds), client_id)
        per_client.append(gap)
        logging.info("client %d: %.4f of coordinates gain, mean gap %.4f bits",
                     client_id, float((gap > 0).mean()), float(gap.mean()))
    gaps = np.stack(per_client)
    parameter_of, parameter_names, _ = model_layout(args.fl_config)

    # A coordinate is only called a gainer if most workers independently agree, so the split is
    # not chasing one worker's training noise.
    votes = (gaps > 0).sum(axis=0)
    gains = votes > args.clients // 2
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "gains_from_wz.npy", gains)
    np.save(args.out / "gap_bits_per_client.npy", gaps.astype(np.float32))

    summary = {
        "method": "per-coordinate curve gap, rounds averaged, workers kept separate",
        "rounds": list(args.rounds),
        "clients": args.clients,
        "coordinates": int(gains.size),
        "share_gaining_per_client": [float((g > 0).mean()) for g in gaps],
        "mean_gap_per_client": [float(g.mean()) for g in gaps],
        "share_gaining_majority": float(gains.mean()),
        "share_all_workers_agree_gain": float((votes == args.clients).mean()),
        "share_all_workers_agree_loss": float((votes == 0).mean()),
        "mean_gap_gaining": float(gaps.mean(axis=0)[gains].mean()),
        "mean_gap_losing": float(gaps.mean(axis=0)[~gains].mean()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    breakdown(gaps.mean(axis=0)[None, :], parameter_of, parameter_names, gains).to_csv(
        args.out / "by_parameter.csv", index=False)
    logging.info("Wrote %s", args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
