"""Plot averaged staged RD views from complete or explicitly incomplete sparse-replay cases.

Reads declared cases from one sparse-replay output root and renders overall, per-start, per-phase
and per-training-cycle RD figures, one directory per rate source, plus comparison and timing tables. Unfinished cases contribute only
their validated record prefix, with `INCOMPLETE` embedded in every plot filename and title.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from Experiments.Protocol_test import records
from Experiments.Protocol_test.rd_metrics import GAP_METRIC_NAME, rd_curve_gap

RATE_FIELDS = ("prior_stage_rates", "coder_stage_rates", "final_bytes")
RATE_LABELS = (
    "Cumulative estimated conditional rate",
    "Cumulative entropy-coder rate",
    "Serialized payload rate (final-stage point)",
)
# Only staged rate sources give a multi-point curve; `final_bytes` is one point, not a curve.
_CURVE_RATE_FIELDS = RATE_FIELDS[:2]


def numeric_list(value: object, field: str, records_path: Path) -> NDArray[np.float64]:
    """Read a numeric list from a compression record without executing its contents."""
    parsed = ast.literal_eval(str(value))
    assert isinstance(parsed, list) and parsed, f"Empty or invalid {field}: {records_path}."
    values = np.asarray(parsed, dtype=np.float64)
    assert values.ndim == 1 and np.isfinite(values).all(), f"Invalid {field}: {records_path}."
    return values


def load_available_records(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every available valid case prefix and report coverage of the declared plan."""
    manifest = json.loads((output_root / "test_config.json").read_text())["identity"]
    frames, coverage = [], []
    for case in manifest["cases"]:
        case_dir = output_root / "cases" / case["codec_id"] / case["case_id"]
        records_path, completion = case_dir / "compression_records.csv", case_dir / "complete.json"
        expected_rows = len(case["client_ids"]) * len(case["round_codecs"])
        if records_path.is_file():
            identity = {"trace": manifest["trace"], "case": case}
            frame = (records.read_completed_case(case_dir) if completion.is_file()
                     else records.read_available_case(case_dir, identity))
            frame["records_path"] = str(records_path)
            frames.append(frame)
            row_count = len(frame)
        else:
            row_count = 0
        coverage.append({
            "codec_id": case["codec_id"], "initial_si": case["initial_si"],
            "case_id": case["case_id"],
            "rows": row_count, "expected_rows": expected_rows,
            "complete": completion.is_file() and row_count == expected_rows,
        })
    assert frames, f"No case records under {output_root}."
    combined = pd.concat(frames, ignore_index=True)
    combined["experiment"] = output_root.name
    return combined, pd.DataFrame(coverage)


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
    # Records store WMAPE as a percentage; every curve here is dB of the plain ratio.
    wmapes = np.stack([numeric_list(value, "stage_wmapes", records_path) for value in rows.stage_wmapes]) / 100
    mean_wmape = wmapes.mean(axis=0)
    assert (mean_wmape > 0).all(), f"Non-positive WMAPE: {records_path}."
    distortion = 20 * np.log10(mean_wmape)
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
    keys = ["experiment", "codec_id", "initial_si", "case_id", "start_round", "round_id"]
    meta_keys = ["relative_round", "cycle", "phase", "client_ids", "trace_identity_digest"]
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


def averaged_curves(points: pd.DataFrame, group_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """Average staged curves over rows, converting WMAPE to dB only after aggregation."""
    curves = points[points.rate_field.isin(_CURVE_RATE_FIELDS)].copy()
    curves["wmape"] = 10 ** (curves.wmape_db / 20)
    keys = ["codec_id", "initial_si", *group_keys, "rate_field", "stage"]
    averaged = curves.groupby(keys, as_index=False).agg(rate=("rate", "mean"), wmape=("wmape", "mean"))
    averaged["wmape_db"] = 20 * np.log10(averaged.pop("wmape"))
    return averaged


def _tone(color: tuple[float, ...], index: int, count: int) -> tuple[float, ...]:
    """Fade one protocol color toward white by start-point order."""
    fade = 0 if count == 1 else .45 * index / (count - 1)
    rgb = np.asarray(color[:3]) * (1 - fade) + fade
    return (*rgb, 1.0)


def _draw_curves(
    axis: Axes, curves: pd.DataFrame, xlabel: str, title: str,
    starts: Sequence[Any], base_colors: Mapping[str, tuple[float, ...]], group_keys: tuple[str, ...],
) -> None:
    """Draw every protocol/start curve of one rate source onto one axis."""
    line_keys = ["codec_id", "initial_si", *group_keys]
    for identity, stages in curves.groupby(line_keys, sort=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        codec, initial_si = identity[0], identity[1]
        start = identity[line_keys.index("start_round")] if "start_round" in line_keys else None
        start_index = starts.index(start)
        color = _tone(base_colors[codec], start_index, len(starts))
        stages = stages.sort_values("stage")
        axis.plot(
            stages.rate, stages.wmape_db, color=color, alpha=.95 - .25 * start_index / max(len(starts) - 1, 1),
            marker="o" if initial_si == "raw" else "s", markeredgecolor=base_colors[codec],
            markerfacecolor="none" if initial_si == "raw" else color,
            markeredgewidth=.9, linewidth=1.8,
            linestyle="-" if initial_si == "raw" else "--",
            label=f"{codec} \u00b7 {initial_si}" + ("" if start is None else f" \u00b7 start {start}"),
        )
    axis.set(xlabel=f"{xlabel} (bits/coordinate)", ylabel="Mean WMAPE (dB)", title=title)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.grid(alpha=.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8)


def _plot_view(
    panels: Sequence[tuple[str, pd.DataFrame]], output: Path, name: str,
    group_keys: tuple[str, ...], suffix: str = "",
) -> None:
    """Write one figure per rate source, in that source's directory, one axis per named panel.

    Each rate source gets its own directory because it is a separate x axis: curves are only ever
    compared within one source, never across two of them.
    """
    curves = {title: averaged_curves(points, group_keys) for title, points in panels if not points.empty}
    curves = {title: frame for title, frame in curves.items() if not frame.empty}
    assert curves, f"No staged curves available for {name}."
    combined = pd.concat(curves.values())
    codecs = sorted(combined.codec_id.unique())
    base_colors = dict(zip(codecs, matplotlib.colormaps["tab10"].resampled(len(codecs)).colors, strict=True))
    starts = sorted(combined.start_round.unique()) if "start_round" in group_keys else [None]
    labels = dict(zip(RATE_FIELDS, RATE_LABELS, strict=True))
    for field in _CURVE_RATE_FIELDS:
        drawn = {title: frame[frame.rate_field == field] for title, frame in curves.items()}
        drawn = {title: frame for title, frame in drawn.items() if not frame.empty}
        if not drawn:
            continue
        figure, axes = plt.subplots(1, len(drawn), figsize=(6 * len(drawn), 4.5), squeeze=False,
                                    layout="constrained")
        for axis, (title, frame) in zip(axes[0], drawn.items(), strict=True):
            _draw_curves(axis, frame, labels[field], title, starts, base_colors, group_keys)
        (output / field).mkdir(parents=True, exist_ok=True)
        figure.savefig(output / field / f"rd_{name}{suffix}.png", dpi=180)
        plt.close(figure)


def plot_results(points: pd.DataFrame, output: Path, *, incomplete: bool = False) -> None:
    """Write stage data and the overall, per-start, per-phase, and per-cycle averaged RD views."""
    assert not points.empty, "Records contain no stage RD metrics."
    output.mkdir(parents=True, exist_ok=True)
    for stale_plot in output.glob("*/rd_*.png"):
        stale_plot.unlink()
    points.to_csv(output / "stage_points.csv", index=False)
    suffix = "_INCOMPLETE" if incomplete else ""
    title_suffix = " (incomplete run)" if incomplete else ""
    views: list[tuple[str, list[tuple[str, pd.DataFrame]], tuple[str, ...]]] = [
        ("overall", [(f"Overall protocol average{title_suffix}", points)], ()),
        ("by_start", [(f"Average by sparse start{title_suffix}", points)], ("start_round",)),
        # The trained round and its frozen reuse answer different questions, so they are read
        # side by side rather than pooled into one per-start average.
        ("by_phase", [
            (f"Trained round only{title_suffix}", points[points.phase == "trained"]),
            (f"Frozen-round average{title_suffix}", points[points.phase == "frozen"]),
        ], ("start_round",)),
    ]
    cycles = sorted(points.cycle.unique())
    if len(cycles) > 1:
        views.extend(
            (f"cycle_{cycle + 1}",
             [(f"Training cycle {cycle + 1}: trained round and frozen rounds{title_suffix}",
               points[points.cycle == cycle])],
             ("start_round",))
            for cycle in cycles
        )
    for name, panels, group_keys in views:
        _plot_view(panels, output, name, group_keys, suffix)


def timing_summary(case_records: pd.DataFrame) -> pd.DataFrame:
    """Summarize runtime separately for costly trained tokens and cheaper frozen reuse."""
    assert "total_seconds" in case_records and case_records.total_seconds.notna().all()
    timed = case_records.assign(round_token=case_records.round_name_full.str.split("|").str[0])
    return timed.groupby(["codec_id", "round_token", "phase"], as_index=False).agg(
        rows=("total_seconds", "size"), mean_seconds=("total_seconds", "mean"),
        median_seconds=("total_seconds", "median"), total_hours=("total_seconds", lambda values: values.sum() / 3600),
    )


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
    """Render the available validated rows from one sparse-replay output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Output root holding cases/ and test_config.json.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    assert args.records.is_dir(), f"Expected a sparse-replay output root directory: {args.records}."
    case_records, coverage = load_available_records(args.records)
    output = args.out or args.records / "analysis"
    incomplete = not coverage.complete.all()
    plot_results(stage_points(case_records), output, incomplete=incomplete)
    coverage.to_csv(output / "coverage.csv", index=False)
    timing_summary(case_records).to_csv(output / "timing_summary.csv", index=False)
    manifest = json.loads((args.records / "test_config.json").read_text())
    comparisons = tuple(map(tuple, manifest["identity"]["comparisons"]))
    gaps = comparison_gaps(case_records, comparisons)
    gaps_path = output / "comparison_gaps.csv"
    if gaps.empty:
        gaps_path.unlink(missing_ok=True)
    else:
        gaps.to_csv(gaps_path, index=False)
    print(f"Wrote {output} ({'INCOMPLETE' if incomplete else 'complete'}); unavailable gaps "
          "(single_stage/non_overlapping/non_monotone) are rows, not omissions.")


if __name__ == "__main__":
    main()
