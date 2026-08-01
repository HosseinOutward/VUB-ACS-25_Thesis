"""Reassign every coordinate to whichever of the three trained quantizers codes it best.

The first split was made before the three quantizers existed, from how the single O and T models
compared. Once they are trained, each of them can code the whole vector, so every coordinate has
three curves and can be given to the one that actually serves it. Retraining on the new assignment
and repeating is Lloyd's algorithm over a mixture of quantizers.

Reads the cross-evaluation arrays a ``cross_evaluate`` run wrote and emits the next partition.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
from numpy.typing import NDArray

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from build_wz_split import wmape_db  # noqa: E402


def load_cross(elements_dir: Path, rounds: tuple[int, ...], client_id: int) -> tuple[
    NDArray[np.float32], NDArray[np.float32]
]:
    """Average each quantizer's whole-vector rate and distortion over the given rounds."""
    bits, errors = [], []
    for round_id in rounds:
        path = elements_dir / f"round_{round_id}_client_{client_id}.npz"
        assert path.is_file(), f"Missing cross-evaluation dump: {path}."
        with np.load(path) as dump:
            assert "cross_bits" in dump.files, (
                f"{path} has no cross-evaluation arrays; the run needs cross_evaluate=True.")
            bits.append(np.cumsum(dump["cross_bits"].astype(np.float32), axis=1))
            errors.append(dump["cross_error"].astype(np.float32))
    return np.mean(bits, axis=0), np.mean(errors, axis=0)


def distortion_at_rate(
    rates: NDArray[np.float32], distortions: NDArray[np.float32], target: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Distortion each candidate reaches once it has spent ``target`` bits on this coordinate.

    Interpolation runs along the rate axis because that is the one that is monotone by
    construction. Matching on distortion instead would have to interpolate a non-monotone curve,
    and a candidate that simply never reaches the target would clamp to its final rate and so look
    cheap - failing to reconstruct would be rewarded.
    """
    out = np.where(target <= rates[0], distortions[0], distortions[-1])
    for stage in range(len(rates) - 1):
        span = rates[stage + 1] - rates[stage]
        inside = (target > rates[stage]) & (target <= rates[stage + 1])
        weight = np.where(span > 0, (target - rates[stage]) / np.maximum(span, 1e-12), 0.0)
        out = np.where(
            inside, distortions[stage] + weight * (distortions[stage + 1] - distortions[stage]), out)
    return out


def best_part(
    rates: NDArray[np.float32], distortions: NDArray[np.float32]
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Pick, per coordinate, the quantizer that reconstructs best for the same spend.

    Every candidate is read at the same three rates - the per-stage average across candidates - so
    none is judged at an operating point the others never visit. Distortion is WMAPE in dB.
    """
    parts, stages, _ = rates.shape
    reference = rates.mean(axis=0)
    scores = np.stack([
        np.mean([
            distortion_at_rate(rates[part], distortions[part], reference[stage])
            for stage in range(stages)
        ], axis=0)
        for part in range(parts)
    ])
    return scores.argmin(axis=0).astype(np.uint8), scores


def main() -> None:
    """Write the next partition from a cross-evaluation run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Records dir of the cross-eval run.")
    parser.add_argument("--rounds", type=int, nargs="+", default=(16, 18, 20))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--previous", type=Path, required=True, help="Partition that run used.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    previous = np.load(args.previous)
    labels, moved, summary = [], [], []
    for client_id in range(args.clients):
        rates, errors = load_cross(args.run / "elements", tuple(args.rounds), client_id)
        assignment, _ = best_part(rates, wmape_db(errors))
        labels.append(assignment)
        changed = float((assignment != previous[client_id]).mean())
        moved.append(changed)
        shares = [float((assignment == part).mean()) for part in range(rates.shape[0])]
        summary.append({"client": client_id, "shares": shares, "moved": changed})
        logging.info("client %d: shares %s, %.3f of coordinates moved", client_id,
                     [round(s, 3) for s in shares], changed)
        del rates, errors

    np.save(args.out, np.stack(labels))
    (args.out.with_suffix(".json")).write_text(json.dumps(
        {"rounds": list(args.rounds), "mean_moved": float(np.mean(moved)), "per_client": summary},
        indent=2))
    logging.info("Wrote %s; %.3f of coordinates changed hands on average.", args.out, np.mean(moved))


if __name__ == "__main__":
    main()
