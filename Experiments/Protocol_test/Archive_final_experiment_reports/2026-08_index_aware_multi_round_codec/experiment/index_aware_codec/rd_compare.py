"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Experiments.Protocol_test.analysis import (
    Curve,
    averaged_curve,
    parse_float_list,
    training_round_id,
    validate_columns,
    wmape_to_db,
)
from Experiments.Protocol_test.rd_metrics import rd_curve_gap


DEFAULT_RATE_TOLERANCE = 0.05
DEFAULT_DISTORTION_TOLERANCE_DB = 1.0


def _read_records(csv_path: Path) -> pd.DataFrame:
    assert csv_path.is_file(), f"Missing compression records: {csv_path}."
    records = pd.read_csv(csv_path)
    validate_columns(records, csv_path)
    return records


def _select_client_rounds(
    records: pd.DataFrame,
    csv_path: Path,
    round_id: int,
    client_ids: tuple[int, ...],
) -> pd.DataFrame:
    selected = records[
        records["round_id"].eq(round_id) & records["client_id"].isin(client_ids)
    ]
    counts = selected.groupby(["round_id", "client_id"]).size()
    assert len(selected) == len(client_ids) and counts.eq(1).all(), (
        f"Expected one row for round {round_id} and clients {client_ids} in {csv_path}; "
        f"found keys {counts.to_dict()}."
    )
    return selected.sort_values("client_id")


def _stage_arrays(records: pd.DataFrame, csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stage_rates = np.asarray([
        parse_float_list(str(value), csv_path, "prior_stage_rates")
        for value in records["prior_stage_rates"]
    ])
    stage_wmapes = np.asarray([
        parse_float_list(str(value), csv_path, "stage_wmapes")
        for value in records["stage_wmapes"]
    ])
    assert stage_rates.ndim == stage_wmapes.ndim == 2 and stage_rates.shape == stage_wmapes.shape
    return np.cumsum(stage_rates, axis=1), wmape_to_db(stage_wmapes)


def compare_baseline(
    candidate_csv: Path,
    reference_csv: Path,
    output_dir: Path,
    round_id: int = 12,
    client_ids: tuple[int, ...] = (0, 1),
    rate_tolerance: float = DEFAULT_RATE_TOLERANCE,
    distortion_tolerance_db: float = DEFAULT_DISTORTION_TOLERANCE_DB,
) -> dict[str, float | bool]:
    """Compare a two-client O replay with its stored reference and enforce the RD sanity gate."""
    candidate = _select_client_rounds(
        _read_records(candidate_csv), candidate_csv, round_id, client_ids
    )
    reference = _select_client_rounds(
        _read_records(reference_csv), reference_csv, round_id, client_ids
    )
    candidate_rates, candidate_db = _stage_arrays(candidate, candidate_csv)
    reference_rates, reference_db = _stage_arrays(reference, reference_csv)
    assert candidate_rates.shape == reference_rates.shape

    stage_rows = [
        {
            "client_id": client_id,
            "stage": stage + 1,
            "candidate_cumulative_rate": candidate_rates[client, stage],
            "reference_cumulative_rate": reference_rates[client, stage],
            "rate_delta": candidate_rates[client, stage] - reference_rates[client, stage],
            "candidate_wmape_db": candidate_db[client, stage],
            "reference_wmape_db": reference_db[client, stage],
            "distortion_delta_db": candidate_db[client, stage] - reference_db[client, stage],
        }
        for client, client_id in enumerate(client_ids)
        for stage in range(candidate_rates.shape[1])
    ]
    candidate_curve = Curve("baseline", "candidate", *averaged_curve(candidate, candidate_csv))
    reference_curve = Curve("baseline", "reference", *averaged_curve(reference, reference_csv))
    stage_rows += [
        {
            "client_id": "average",
            "stage": stage + 1,
            "candidate_cumulative_rate": candidate_curve.cumulative_prior_rates[stage],
            "reference_cumulative_rate": reference_curve.cumulative_prior_rates[stage],
            "rate_delta": (
                candidate_curve.cumulative_prior_rates[stage]
                - reference_curve.cumulative_prior_rates[stage]
            ),
            "candidate_wmape_db": candidate_curve.distortion_db[stage],
            "reference_wmape_db": reference_curve.distortion_db[stage],
            "distortion_delta_db": (
                candidate_curve.distortion_db[stage] - reference_curve.distortion_db[stage]
            ),
        }
        for stage in range(candidate_rates.shape[1])
    ]
    gap = rd_curve_gap(
        candidate_curve.cumulative_prior_rates,
        candidate_curve.distortion_db,
        reference_curve.cumulative_prior_rates,
        reference_curve.distortion_db,
        name="baseline sanity",
    )
    rate_gap_bits = gap["rate_gap_bits"]
    averaged_distortion_deltas = candidate_curve.distortion_db - reference_curve.distortion_db
    passed = (
        np.isfinite(rate_gap_bits)
        and abs(rate_gap_bits) <= rate_tolerance
        and bool((np.abs(averaged_distortion_deltas) <= distortion_tolerance_db).all())
    )
    summary: dict[str, float | bool] = {
        "passed": passed,
        **gap,
        "max_averaged_stage_distortion_delta_db": float(
            np.abs(averaged_distortion_deltas).max()
        ),
        "rate_tolerance": rate_tolerance,
        "distortion_tolerance_db": distortion_tolerance_db,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stage_rows).to_csv(output_dir / "baseline_stage_differences.csv", index=False)
    (output_dir / "baseline_sanity.json").write_text(json.dumps(summary, indent=2))
    assert passed, (
        f"Baseline RD sanity gate failed: |rate gap|={abs(rate_gap_bits):.6f} bit "
        f"(limit {rate_tolerance}), maximum averaged stage distortion difference="
        f"{summary['max_averaged_stage_distortion_delta_db']:.6f} dB "
        f"(limit {distortion_tolerance_db}). See {output_dir}."
    )
    return summary


def compare_experiment(
    candidate_csv: Path,
    reference_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Report matched-distortion rate gaps and matched-rate distortion gaps for training, frozen, and complete trajectories."""
    candidate = _read_records(candidate_csv)
    reference = _read_records(reference_csv)
    candidate_keys = set(zip(candidate["round_id"], candidate["client_id"], strict=True))
    reference_keys = set(zip(reference["round_id"], reference["client_id"], strict=True))
    assert candidate_keys == reference_keys, (
        f"Candidate/reference round-client coverage differs: "
        f"{candidate_keys - reference_keys=} {reference_keys - candidate_keys=}."
    )
    candidate_training = training_round_id(candidate, candidate_csv)
    reference_training = training_round_id(reference, reference_csv)
    assert candidate_training == reference_training
    selectors = {
        "training_round": lambda frame: frame[frame["round_id"].eq(candidate_training)],
        "frozen_rounds": lambda frame: frame[frame["round_id"].ne(candidate_training)],
        "full_trajectory": lambda frame: frame,
    }
    rows: list[dict[str, float | int | str]] = []
    for subset, selector in selectors.items():
        candidate_subset, reference_subset = selector(candidate), selector(reference)
        candidate_curve = Curve(subset, candidate_csv.parent.name, *averaged_curve(
            candidate_subset, candidate_csv
        ))
        reference_curve = Curve(subset, reference_csv.parent.name, *averaged_curve(
            reference_subset, reference_csv
        ))
        rows.append({
            "subset": subset,
            "rows": len(candidate_subset),
            **rd_curve_gap(
                candidate_curve.cumulative_prior_rates,
                candidate_curve.distortion_db,
                reference_curve.cumulative_prior_rates,
                reference_curve.distortion_db,
                name=f"{candidate_csv.parent.name} vs {reference_csv.parent.name} ({subset})",
            ),
        })
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare index-aware protocol RD records.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run either the baseline sanity gate or a trajectory RD comparison."""
    args = _parse_args()
    if args.baseline:
        compare_baseline(args.candidate, args.reference, args.output)
    else:
        compare_experiment(args.candidate, args.reference, args.output)


if __name__ == "__main__":
    main()
