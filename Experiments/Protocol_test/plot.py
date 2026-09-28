"""Plot averaged staged RD views from complete or explicitly incomplete sparse-replay cases.

Reads one sparse-replay output root and renders overall, per-start, per-phase and
per-training-cycle RD figures, one directory per rate source plus one holding both staged sources
on shared axes, alongside comparison and timing tables. With `--global`, it instead discovers every
output root under a container directory and draws them all into one `_global_analysis` beside them.
Unfinished cases contribute only their validated record prefix, with `INCOMPLETE` embedded in every
plot filename and title.
"""

from __future__ import annotations

import argparse
import ast
import colorsys
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from Experiments.Protocol_test import records
from Experiments.Protocol_test.rd_metrics import GAP_METRIC_NAME, mean_wmape_db, rd_curve_gap, wmape_to_db

RATE_FIELDS = ("prior_stage_rates", "coder_stage_rates", "final_bytes")
RATE_LABELS = (
    "Cumulative estimated conditional rate",
    "Cumulative entropy-coder rate",
    "Serialized payload rate (final-stage point)",
)
# Only staged rate sources give a multi-point curve; `final_bytes` is one point, not a curve.
_CURVE_RATE_FIELDS = RATE_FIELDS[:2]
_RATE_SHORT_LABELS = {"prior_stage_rates": "prior", "coder_stage_rates": "coder"}
# One figure directory per x axis, plus one drawing both staged sources against a shared x axis.
COMBINED_RATE_DIR = "combined_rate_sources"
GLOBAL_ANALYSIS_DIR = "_global_analysis"
_FIGURE_DIRECTORIES = {field: (field,) for field in _CURVE_RATE_FIELDS} | {COMBINED_RATE_DIR: _CURVE_RATE_FIELDS}
_DIRECTORY_LABELS = dict(zip(RATE_FIELDS, RATE_LABELS, strict=True)) | {COMBINED_RATE_DIR: "Cumulative staged rate"}
# Marker separates the two rate sources wherever they share an axis; dashes separate experiments.
_RATE_MARKERS = {"prior_stage_rates": "o", "coder_stage_rates": "s"}
_EXPERIMENT_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)))
# Fixed categorical hues, validated for colorblind and normal-vision separation on a light surface.
_CODEC_HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# These names are used across protocol studies. Keep their hues independent of which subset a
# figure happens to contain; unknown research codecs use the legacy sorted-palette fallback below.
_CODEC_COLOR_LEXICON = dict(zip(("O", "OR", "T", "TR"), _CODEC_HUES[:4], strict=True))
# Sparse starts are ordinal shades of their codec's hue, placed by target contrast against the white
# figure surface: the lightest still clears the 2:1 floor, the darkest stays readable ink.
_SHADE_CONTRAST_RANGE = (2.3, 7.5)
_LABEL_INK = "#52514e"


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
            "experiment": manifest["name"],
            "codec_id": case["codec_id"], "initial_si": case["initial_si"],
            "case_id": case["case_id"],
            "rows": row_count, "expected_rows": expected_rows,
            "complete": completion.is_file() and row_count == expected_rows,
        })
    assert frames, f"No case records under {output_root}."
    combined = pd.concat(frames, ignore_index=True)
    combined["experiment"] = manifest["name"]
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
    # Records store WMAPE as a percentage; conversion follows equal-worker averaging.
    wmapes = np.stack([numeric_list(value, "stage_wmapes", records_path) for value in rows.stage_wmapes])
    distortion = mean_wmape_db(wmapes)
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
    curves["wmape"] = 100 * 10 ** (curves.wmape_db / 20)
    keys = ["experiment", "codec_id", "initial_si", *group_keys, "rate_field", "stage"]
    averaged = curves.groupby(keys, as_index=False).agg(rate=("rate", "mean"), wmape=("wmape", "mean"))
    averaged["wmape_db"] = wmape_to_db(averaged.pop("wmape").to_numpy())
    return averaged


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of an sRGB triple."""
    red, green, blue = (channel / 12.92 if channel <= .04045 else ((channel + .055) / 1.055) ** 2.4
                        for channel in rgb)
    return .2126 * red + .7152 * green + .0722 * blue


def _shade(hue: str, index: int, count: int) -> tuple[float, float, float]:
    """Step `index` of one codec hue's ordinal ramp, lightest first; a sole start uses the hue itself.

    Steps are solved for a target contrast against the surface rather than offset by a fixed
    lightness, because one lightness step reads as a very different gap from one hue to the next,
    and the fade this replaces left same-codec starts indistinguishable.
    """
    if count == 1:
        return matplotlib.colors.to_rgb(hue)
    low, high = _SHADE_CONTRAST_RANGE
    target = 1.05 / (low * (high / low) ** (index / (count - 1))) - .05
    angle, _lightness, saturation = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(hue))
    lighter, darker = 0., 1.
    for _step in range(24):  # luminance rises monotonically with HLS lightness, so bisect on it
        middle = (lighter + darker) / 2
        if _relative_luminance(colorsys.hls_to_rgb(angle, middle, saturation)) < target:
            lighter = middle
        else:
            darker = middle
    return colorsys.hls_to_rgb(angle, (lighter + darker) / 2, saturation)


class CurveStyle(NamedTuple):
    """The visual channels one figure assigns: hue is the codec, its shade the sparse start.

    Built once per figure by `_curve_style` so every panel of that figure reads the same way.
    Dashes are the experiment and markers the rate source; `initial_si` is named in the legend
    only, and owns no visual channel.
    """

    hues: Mapping[str, str]
    linestyles: Mapping[str, Any]
    starts: Sequence[Any]
    name_experiment: bool


def _curve_style(curves: pd.DataFrame, group_keys: tuple[str, ...]) -> CurveStyle:
    """Assign one color per codec and one dash pattern per experiment across a whole figure."""
    codecs, experiments = sorted(curves.codec_id.unique()), sorted(curves.experiment.unique())
    assert len(experiments) <= len(_EXPERIMENT_LINESTYLES), (
        f"{len(experiments)} experiments exceed the {len(_EXPERIMENT_LINESTYLES)} distinguishable "
        f"dash patterns: {experiments}.")
    unknown = [codec for codec in codecs if codec not in _CODEC_COLOR_LEXICON]
    available = tuple(hue for hue in _CODEC_HUES if hue not in _CODEC_COLOR_LEXICON.values())
    assert len(unknown) <= len(available), (
        f"{len(unknown)} unknown codecs exceed the {len(available)} fallback hues: {unknown}.")
    hues = _CODEC_COLOR_LEXICON | dict(zip(unknown, available[:len(unknown)], strict=True))
    return CurveStyle(
        hues=hues,
        linestyles=dict(zip(experiments, _EXPERIMENT_LINESTYLES, strict=False)),
        starts=sorted(curves.start_round.unique()) if "start_round" in group_keys else [None],
        name_experiment=len(experiments) > 1,
    )


def _draw_curves(
    axis: Axes, curves: pd.DataFrame, xlabel: str, title: str,
    style: CurveStyle, group_keys: tuple[str, ...],
) -> None:
    """Draw every experiment/protocol/start/rate-source curve of one panel onto one axis."""
    line_keys = ["experiment", "codec_id", "initial_si", *group_keys, "rate_field"]
    name_rate_field = curves.rate_field.nunique() > 1
    # One start label per curve end, not one per rate source, or the two would print on top of each other.
    labelled_field = sorted(curves.rate_field.unique())[0]
    for identity, stages in curves.groupby(line_keys, sort=True):
        line = dict(zip(line_keys, identity, strict=True))
        start = line.get("start_round")
        color = _shade(style.hues[line["codec_id"]], style.starts.index(start), len(style.starts))
        marker = _RATE_MARKERS[line["rate_field"]]
        label = " \u00b7 ".join(part for part in (
            line["experiment"] if style.name_experiment else "",
            f"{line['codec_id']} ({line['initial_si']})",
            "" if start is None else f"start {start}",
            _RATE_SHORT_LABELS[line["rate_field"]] if name_rate_field else "",
        ) if part)
        stages = stages.sort_values("stage")
        axis.plot(
            stages.rate, stages.wmape_db, color=color, marker=marker, markeredgecolor=color,
            markerfacecolor="none" if marker == "o" else color,
            markeredgewidth=1.1, linewidth=2,
            linestyle=style.linestyles[line["experiment"]], label=label,
        )
        if start is not None and line["rate_field"] == labelled_field:
            axis.annotate(
                str(start), stages.iloc[-1][["rate", "wmape_db"]], textcoords="offset points",
                xytext=(6, 0), ha="left", va="center", fontsize=7, color=_LABEL_INK,
                # A leader keeps the label attached to its curve once `_space_end_labels` moves it.
                arrowprops={"arrowstyle": "-", "color": _LABEL_INK, "linewidth": .5,
                            "shrinkA": 0, "shrinkB": 3},
            )
    axis.set(xlabel=f"{xlabel} (bits/coordinate)", ylabel="Mean WMAPE (dB)", title=title)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    axis.grid(alpha=.25)
    axis.spines[["top", "right"]].set_visible(False)


def _space_end_labels(figure: Figure, axes: Sequence[Axes]) -> None:
    """Push overlapping start labels apart vertically, in points, after the layout is final.

    Curves of one codec often end within a few pixels of each other, where unspaced labels print on
    top of one another. Offsets are in points, so they survive `savefig`'s higher dpi.
    """
    figure.canvas.draw()  # constrained layout only fixes the axes box once the figure is drawn
    gap = 8 * figure.dpi / 72
    for axis in axes:
        occupied = float("-inf")
        for label in sorted(axis.texts, key=lambda text: axis.transData.transform(text.xy)[1]):
            position = axis.transData.transform(label.xy)[1]
            occupied = max(position, occupied + gap)
            label.set_position((5, (occupied - position) * 72 / figure.dpi))


def _legend_below(figure: Figure, axes: Sequence[Axes], width: float) -> None:
    """Give the whole figure one legend under it, growing the figure instead of shrinking the axes.

    Panels of one figure draw the same line identities, and the global view has enough of them to
    bury an in-axes legend, so the legend is deduplicated by label and placed outside.
    """
    entries = {label: handle for axis in axes
               for handle, label in zip(*axis.get_legend_handles_labels(), strict=True)}
    columns = max(1, min(3, int(width // 4.5)))
    rows = -(-len(entries) // columns)
    figure.set_size_inches(width, 4.5 + .26 * rows)
    figure.legend(entries.values(), entries.keys(), loc="outside lower center",
                  ncol=columns, fontsize=8, frameon=False)


def _plot_view(
    panels: Sequence[tuple[str, pd.DataFrame]], output: Path, name: str,
    group_keys: tuple[str, ...], suffix: str = "",
) -> None:
    """Write this view once per figure directory, one axis per named panel.

    Each single-source directory is a separate x axis, so its curves are only ever compared within
    that source. `COMBINED_RATE_DIR` deliberately overlays both staged sources on one axis: they
    share the bits/coordinate unit, and the prior rate is the estimate the coder rate is judged
    against, so seeing the pair together is the point.
    """
    curves = {title: averaged_curves(points, group_keys) for title, points in panels if not points.empty}
    curves = {title: frame for title, frame in curves.items() if not frame.empty}
    assert curves, f"No staged curves available for {name}."
    style = _curve_style(pd.concat(curves.values()), group_keys)
    for directory, fields in _FIGURE_DIRECTORIES.items():
        drawn = {title: frame[frame.rate_field.isin(fields)] for title, frame in curves.items()}
        drawn = {title: frame for title, frame in drawn.items() if not frame.empty}
        if not drawn:
            continue
        width = 6 * len(drawn)
        figure, axes = plt.subplots(1, len(drawn), figsize=(width, 4.5), squeeze=False,
                                    layout="constrained")
        for axis, (title, frame) in zip(axes[0], drawn.items(), strict=True):
            _draw_curves(axis, frame, _DIRECTORY_LABELS[directory], title, style, group_keys)
        _legend_below(figure, axes[0], width)
        _space_end_labels(figure, axes[0])
        (output / directory).mkdir(parents=True, exist_ok=True)
        figure.savefig(output / directory / f"rd_{name}{suffix}.png", dpi=180)
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


def discover_experiment_roots(container: Path) -> list[Path]:
    """Find every started sparse-replay output root under `container`, for the global view.

    A declared root that has not written a single case CSV yet is not an error: it is an experiment
    that has not started measuring, and the global view simply does not show it yet.
    """
    declared = sorted({path.parent for path in container.rglob("test_config.json")
                       if GLOBAL_ANALYSIS_DIR not in path.parts})
    roots = [root for root in declared if any(root.glob("cases/*/*/compression_records.csv"))]
    assert roots, f"No sparse-replay output root under {container} has written any case records yet."
    return roots


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
    round, and measured worker set all match; a completed mismatch there fails loudly because the
    comparison would be meaningless, while a live partial worker prefix is deferred. A pair simply
    absent for some placement is skipped. Every evaluated
    (placement, rate field) gets a row: either the interpolation-based gap in `rd_metrics.rd_curve_gap`
    (not the repository's average-slope/perpendicular specification), or `gap_available=False` with a
    reason. Single-stage or non-monotone curves never abort plotting.
    """
    rows = []
    group_keys = ["case_id", "start_round", "round_id"]
    for candidate, reference in comparisons:
        selected = case_records[case_records.codec_id.isin((candidate, reference))]
        for identity, group in selected.groupby(group_keys, sort=True):
            left, right = group[group.codec_id == candidate], group[group.codec_id == reference]
            if left.empty or right.empty:
                continue  # this placement was not run for both codecs; not a declared mismatch
            assert left.records_path.nunique() == right.records_path.nunique() == 1, (
                f"Declared comparison {candidate} vs {reference} at {identity} is ambiguous: "
                "each codec must come from exactly one case records file.")
            left_trace, right_trace = set(left.trace_identity_digest), set(right.trace_identity_digest)
            left_clients, right_clients = set(left.client_id), set(right.client_id)
            assert len(left_trace) == 1 and left_trace == right_trace, (
                f"Declared comparison {candidate} vs {reference} at {identity} has mismatched trace "
                "identity; this comparison would be meaningless.")
            if left_clients != right_clients and not (left.case_complete.all() and right.case_complete.all()):
                continue
            assert left_clients == right_clients, (
                f"Declared comparison {candidate} vs {reference} at {identity} has mismatched "
                "measured worker sets; this comparison would be meaningless.")
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


def render_global(container: Path, output: Path) -> bool:
    """Draw every discovered experiment under `container` into one set of figures; report completeness.

    Only figures and their stage points are written: coverage, timing and gap tables stay with the
    experiment that owns them. Experiment names must be unique across the discovered roots, or
    curves from different runs would be averaged into one line.
    """
    roots = discover_experiment_roots(container)
    loaded = [load_available_records(root) for root in roots]
    names = [frame.experiment.iloc[0] for frame, _coverage in loaded]
    assert len(set(names)) == len(names), (
        f"Experiment names repeat across {[str(root) for root in roots]}: {sorted(names)}; their "
        "curves would silently pool into one line.")
    coverage = pd.concat([coverage for _frame, coverage in loaded], ignore_index=True)
    incomplete = not coverage.complete.all()
    plot_results(stage_points(pd.concat([frame for frame, _coverage in loaded], ignore_index=True)),
                 output, incomplete=incomplete)
    return incomplete


def main() -> None:
    """Render one sparse-replay output root, or every root under a container with --global."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path,
                        help="Output root holding cases/ and test_config.json, or the directory holding "
                             "several such roots when --global is given.")
    parser.add_argument("--global", dest="global_view", action="store_true",
                        help=f"Draw every discovered experiment together into <records>/{GLOBAL_ANALYSIS_DIR}.")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--compare", nargs=2, action="append", default=[], metavar=("CANDIDATE", "REFERENCE"),
        help="Add a candidate/reference codec comparison within this experiment.",
    )
    args = parser.parse_args()
    assert args.records.is_dir(), f"Expected a directory: {args.records}."
    if args.global_view:
        assert not args.compare, "--compare measures one experiment's codecs; it is not part of the global view."
        output = args.out or args.records / GLOBAL_ANALYSIS_DIR
        incomplete = render_global(args.records, output)
        print(f"Wrote {output} ({'INCOMPLETE' if incomplete else 'complete'}).")
        return
    case_records, coverage = load_available_records(args.records)
    output = args.out or args.records / "analysis"
    incomplete = not coverage.complete.all()
    plot_results(stage_points(case_records), output, incomplete=incomplete)
    coverage.to_csv(output / "coverage.csv", index=False)
    timing_summary(case_records).to_csv(output / "timing_summary.csv", index=False)
    manifest = json.loads((args.records / "test_config.json").read_text())
    comparisons = tuple(dict.fromkeys(
        tuple(tuple(pair) for pair in manifest["identity"]["comparisons"]) + tuple(map(tuple, args.compare))))
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
