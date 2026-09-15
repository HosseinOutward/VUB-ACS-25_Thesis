"""Plot staged RD and reuse drift from completed cases, with each rate definition kept explicit.

Reads one completed sparse-replay output root through `records.read_completed_case` (the shared
runner/analysis completion contract) and renders staged rate-distortion figures plus a comparison
gap table. Never guesses provenance a completed case did not record.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from Experiments.Protocol_test import records
from Experiments.Protocol_test.rd_metrics import GAP_METRIC_NAME, rd_curve_gap

RATE_FIELDS = ("prior_stage_rates", "coder_stage_rates", "final_bytes")
RATE_LABELS = ("Estimated conditional rate", "Entropy-coder rate", "Serialized payload rate (final-stage point)")
# Only staged rate sources give a multi-point curve; `final_bytes` is one point, not a curve.
_CURVE_RATE_FIELDS = RATE_FIELDS[:2]


def numeric_list(value: object, field: str, records_path: Path) -> NDArray[np.float64]:
    """Read a numeric list from a compression record without executing its contents."""
    parsed = ast.literal_eval(str(value))
    assert isinstance(parsed, list) and parsed, f"Empty or invalid {field}: {records_path}."
    values = np.asarray(parsed, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all(), f"Invalid {field}: {records_path}."
    return values


def load_completed_records(output_root: Path) -> pd.DataFrame:
    """Read every finished case under one output root via D's shared completion contract.

    Globs `cases/<codec_id>/<case_id>/` directly beneath `output_root`; a matched directory with
    no `complete.json` is unfinished output and fails loudly through `read_completed_case` rather
    than silently vanishing from the plot.
    """
    case_dirs = sorted(path for path in output_root.glob("cases/*/*") if path.is_dir())
    assert case_dirs, f"No case directories under {output_root}."
    frames = []
    for case_dir in case_dirs:
        frame = records.read_completed_case(case_dir)
        frame["records_path"] = str(case_dir / "compression_records.csv")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["experiment"] = output_root.name
    return combined


def round_curve(rows: pd.DataFrame, records_path: Path, rate_field: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Equal-worker mean WMAPE (then dB) and mean-then-cumulative rate for one round's measured workers.

    Each worker counts equally regardless of its gradient magnitude or model size, and rate fields
    are never substituted for one another: a partially populated `rate_field` across these rows is
    an invalid measurement and fails loudly rather than silently pooling fewer workers.
    """
    assert rate_field in RATE_FIELDS and rate_field in rows.columns and rows[rate_field].notna().all(), (
        f"Incomplete {rate_field}: {records_path}.")
    sizes = rows.model_size.to_numpy(dtype=float)
    assert (sizes == sizes[0]).all(), f"Aggregating workers with different model_size is unsupported: {records_path}."
    wmapes = np.stack([numeric_list(value, "stage_wmapes", records_path) for value in rows.stage_wmapes])
    mean_wmape = wmapes.mean(axis=0)
    assert (mean_wmape > 0).all(), f"Non-positive WMAPE: {records_path}."
    distortion = 20 * np.log10(mean_wmape / 100)
    if rate_field == "final_bytes":
        rate = rows.final_bytes.to_numpy(dtype=float).mean() * (2**20 * 8) / sizes[0]
        return np.array([rate]), distortion[-1:]
    rates = np.stack([numeric_list(value, rate_field, records_path) for value in rows[rate_field]])
    assert rates.shape == wmapes.shape, f"Rate/distortion stage counts differ: {records_path}."
    return np.cumsum(rates.mean(axis=0)), distortion


def stage_points(case_records: pd.DataFrame) -> pd.DataFrame:
    """Produce one explicit rate/distortion point per case, absolute round, rate source and stage.

    Every case/start-round/absolute-round stays independent (never pooled across start rounds or
    worker sets). Availability is decided per case: a rate source absent from the whole case is
    unavailable for that codec, while one missing from only some rounds fails in `round_curve`.
    """
    keys = ["experiment", "codec_id", "case_id", "start_round", "round_id"]
    meta_keys = ["relative_round", "phase", "client_ids", "trace_identity_digest"]
    points = []
    for _, case_rows in case_records.groupby(["experiment", "codec_id", "case_id"], sort=True):
        fields = [field for field in RATE_FIELDS if field in case_rows.columns and case_rows[field].notna().any()]
        for identity, rows in case_rows.groupby(keys, sort=True):
            meta = {key: rows[key].iloc[0] for key in meta_keys}
            for field in fields:
                rate, distortion = round_curve(rows, Path(rows.records_path.iloc[0]), field)
                points.extend(dict(zip(keys, identity, strict=True)) | meta | {
                    "rate_field": field, "stage": stage, "rate": x, "wmape_db": y,
                    "is_final_payload": field == "final_bytes"}
                    for stage, (x, y) in enumerate(zip(rate, distortion, strict=True), 1))
    return pd.DataFrame(points)


def plot_results(points: pd.DataFrame, output: Path) -> None:
    """Plot each rate source separately; lines connect quantizer stages within one round only."""
    assert not points.empty, "Records contain no stage RD metrics."
    output.mkdir(parents=True, exist_ok=True)
    points.to_csv(output / "stage_points.csv", index=False)
    experiments = list(points.experiment.unique())
    codecs = list(points.codec_id.unique())
    colors = dict(zip(codecs, matplotlib.colormaps["tab10"].resampled(len(codecs)).colors, strict=True))
    figure, axes = plt.subplots(len(experiments), 3, figsize=(15, 4 * len(experiments)), squeeze=False, layout="constrained")
    drift, drift_axes = plt.subplots(len(experiments), 3, figsize=(15, 4 * len(experiments)), squeeze=False, layout="constrained")
    for row, experiment in enumerate(experiments):
        experiment_points = points[points.experiment == experiment]
        experiment_codecs = sorted(experiment_points.codec_id.unique())
        for column, (field, label) in enumerate(zip(RATE_FIELDS, RATE_LABELS, strict=True)):
            axis, drift_axis = axes[row, column], drift_axes[row, column]
            subset = experiment_points[experiment_points.rate_field == field]
            missing = sorted(set(experiment_codecs) - set(subset.codec_id.unique()))
            title = experiment if not missing else f"{experiment}\n(unavailable for {', '.join(missing)})"
            for (codec, case, start), window in subset.groupby(["codec_id", "case_id", "start_round"]):
                for _, stages in window.groupby("round_id"):
                    stages = stages.sort_values("stage")
                    marker = "o" if stages.phase.iloc[0] == "trained" else "x"
                    axis.plot(stages.rate, stages.wmape_db, color=colors[codec], alpha=.65, marker=marker, linewidth=.8)
                last_stages = window.sort_values("stage").groupby("round_id").tail(1).sort_values("round_id")
                drift_axis.plot(last_stages.round_id, last_stages.rate, color=colors[codec], marker=".", label=f"{codec}: {case}")
            axis.set(xlabel=f"{label} (bits/coordinate)", ylabel="Mean worker WMAPE (dB rel. 100%)", title=title)
            drift_axis.set(xlabel="Absolute FL round", ylabel=f"{label} (bits/coordinate)", title=title)
            for ax in (axis, drift_axis):
                ax.grid(alpha=.25)
                ax.spines[["top", "right"]].set_visible(False)
            handles = [plt.Line2D([], [], color=colors[c], marker="o", label=c) for c in subset.codec_id.unique()]
            if handles:
                axis.legend(handles=handles, fontsize=7)
                drift_axis.legend(fontsize=6)
    figure.suptitle("Circle: trained-round quantizer; cross: frozen-round reuse. Oracle priors remain oracle assumptions.")
    figure.savefig(output / "rate_distortion.png", dpi=150)
    drift.savefig(output / "rates_by_round.png", dpi=150)
    plt.close(figure)
    plt.close(drift)


def comparison_gaps(case_records: pd.DataFrame, comparisons: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Compare declared candidate/reference codecs per matched placement, using the existing gap metric.

    A declared comparison pair is only evaluated where trace identity, case, start round, absolute
    round, and measured worker set all match; a mismatch there fails loudly because the comparison
    would be meaningless. A pair simply absent for some placement is skipped. Every evaluated
    (placement, rate field) gets a row: either the interpolation-based gap in `rd_metrics.rd_curve_gap`
    (not the repository's average-slope/perpendicular specification), or `gap_available=False` with a
    reason. Single-stage or non-monotone curves never abort plotting.
    """
    rows = []
    group_keys = ["experiment", "case_id", "start_round", "round_id"]
    for candidate, reference in comparisons:
        selected = case_records[case_records.codec_id.isin((candidate, reference))]
        for identity, group in selected.groupby(group_keys, sort=True):
            left, right = group[group.codec_id == candidate], group[group.codec_id == reference]
            if left.empty or right.empty:
                continue  # this placement was not run for both codecs; not a declared mismatch
            left_trace, right_trace = set(left.trace_identity_digest), set(right.trace_identity_digest)
            left_clients, right_clients = set(left.client_ids.iloc[0]), set(right.client_ids.iloc[0])
            assert len(left_trace) == 1 and left_trace == right_trace and left_clients == right_clients, (
                f"Declared comparison {candidate} vs {reference} at {identity} has mismatched trace "
                f"identity or measured worker set; this comparison would be meaningless.")
            for field in _CURVE_RATE_FIELDS:
                if field not in group.columns or left[field].isna().all() or right[field].isna().all():
                    continue  # an entire rate source missing on either codec: no comparison of it
                base = dict(zip(group_keys, identity, strict=True)) | {
                    "candidate": candidate, "reference": reference, "rate_field": field, "gap_metric": GAP_METRIC_NAME}
                curves = [round_curve(side, Path(side.records_path.iloc[0]), field) for side in (left, right)]
                reason = "single_stage" if any(len(rate) < 2 for rate, _ in curves) else None
                if reason is None and any(not (np.diff(rate) > 0).all() or not (np.diff(dist) < 0).all()
                                           for rate, dist in curves):
                    reason = "non_monotone"
                if reason is None:
                    gap = rd_curve_gap(*curves[0], *curves[1], name=f"{identity}: {candidate} vs {reference} ({field})")
                    reason = "non_overlapping" if any(np.isnan(value) for value in gap.values()) else None
                rows.append(base | ({"gap_available": False, "gap_unavailable_reason": reason} if reason
                                     else {"gap_available": True, "gap_unavailable_reason": None, **gap}))
    return pd.DataFrame(rows)


def main() -> None:
    """Render one completed sparse-replay output root through the shared analysis path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Output root holding cases/ and test_config.json.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    assert args.records.is_dir(), f"Expected a sparse-replay output root directory: {args.records}."
    case_records = load_completed_records(args.records)
    output = args.out or args.records / "analysis"
    plot_results(stage_points(case_records), output)
    manifest = json.loads((args.records / "test_config.json").read_text())
    comparisons = tuple(map(tuple, manifest["identity"]["comparisons"]))
    gaps = comparison_gaps(case_records, comparisons)
    gaps_path = output / "comparison_gaps.csv"
    if gaps.empty:
        gaps_path.unlink(missing_ok=True)
    else:
        gaps.to_csv(gaps_path, index=False)
    print(f"Wrote {output}; unavailable gaps (single_stage/non_overlapping/non_monotone) are rows, not omissions.")


if __name__ == "__main__":
    main()
