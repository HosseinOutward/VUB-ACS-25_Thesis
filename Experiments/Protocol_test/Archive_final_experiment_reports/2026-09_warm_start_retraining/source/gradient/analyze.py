"""Analyse reduced-epoch gradient warm starts using hard WMAPE and realized rANS rate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.Protocol_test.rd_metrics import GAP_METRIC_NAME, mean_wmape_db, rd_curve_gap
from Experiments.warm_start_gradient_retraining.experiment import StudyConfig, load_config, warm_conditions


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Read all existing parent and repeat CSV rows."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Atomically replace one derived CSV, retaining a header when empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def curves(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Group complete hard-symbol curves by condition and stochastic repeat."""
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["repeat"]))].append(row)
    return {key: sorted(group, key=lambda row: int(row["stage"])) for key, group in grouped.items()}


def gap_rows(config: StudyConfig, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare every round-22 curve with paired full scratch via repository interpolation."""
    grouped = curves(rows)
    results: list[dict[str, Any]] = []
    frozen = grouped.get(("frozen_parent", -1))
    references = [
        grouped[("scratch120", repeat)] for repeat in range(config.training_repeats)
        if len(grouped.get(("scratch120", repeat), ())) == config.num_planes
    ]
    if frozen is not None and references:
        gap = rd_curve_gap(
            np.array([float(row["coder_rate"]) for row in frozen]),
            np.array([float(row["wmape_db"]) for row in frozen]),
            np.mean([[float(row["coder_rate"]) for row in curve] for curve in references], axis=0),
            mean_wmape_db(np.array([
                [float(row["wmape"]) for row in curve] for curve in references
            ])),
            "frozen_parent vs mean scratch120",
        )
        results.append({
            "condition": "frozen_parent", "reference": "mean_scratch120", "repeat": -1,
            "metric": GAP_METRIC_NAME,
            "status": "ok" if np.isfinite(gap["rate_gap_bits"]) else "rejected_no_distortion_overlap",
            **gap,
        })
    for repeat in range(config.training_repeats):
        reference = grouped.get(("scratch120", repeat))
        for condition in ("scratch30", *warm_conditions(config)):
            candidate = grouped.get((condition, repeat))
            base = {
                "condition": condition, "reference": "scratch120", "repeat": repeat,
                "metric": GAP_METRIC_NAME,
            }
            if candidate is None or reference is None or any(
                len(curve) != config.num_planes for curve in (candidate, reference)
            ):
                results.append(base | {
                    "status": "rejected_incomplete", "rate_gap_bits": np.nan,
                    "rate_gap_percent": np.nan, "distortion_gap_db": np.nan,
                    "distortion_gap_percent": np.nan,
                })
                continue
            gap = rd_curve_gap(
                np.array([float(row["coder_rate"]) for row in candidate]),
                np.array([float(row["wmape_db"]) for row in candidate]),
                np.array([float(row["coder_rate"]) for row in reference]),
                np.array([float(row["wmape_db"]) for row in reference]),
                f"{condition} vs scratch120 repeat {repeat}",
            )
            status = "ok" if np.isfinite(gap["rate_gap_bits"]) else "rejected_no_distortion_overlap"
            results.append(base | {"status": status, **gap})
    return results


def paired_warm_scratch_gaps(config: StudyConfig, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare every warm-30 policy directly with scratch-30 for each paired seed."""
    grouped = curves(rows)
    results: list[dict[str, Any]] = []
    for repeat in range(config.training_repeats):
        scratch = grouped.get(("scratch30", repeat))
        for condition in warm_conditions(config):
            warm = grouped.get((condition, repeat))
            base = {
                "condition": condition, "reference": "scratch30", "repeat": repeat,
                "metric": GAP_METRIC_NAME,
            }
            if warm is None or scratch is None or any(
                len(curve) != config.num_planes for curve in (warm, scratch)
            ):
                results.append(base | {
                    "status": "rejected_incomplete", "rate_gap_bits": np.nan,
                    "rate_gap_percent": np.nan, "distortion_gap_db": np.nan,
                    "distortion_gap_percent": np.nan,
                })
                continue
            gap = rd_curve_gap(
                np.array([float(row["coder_rate"]) for row in warm]),
                np.array([float(row["wmape_db"]) for row in warm]),
                np.array([float(row["coder_rate"]) for row in scratch]),
                np.array([float(row["wmape_db"]) for row in scratch]),
                f"{condition} vs scratch30 repeat {repeat}",
            )
            results.append(base | {
                "status": "ok" if np.isfinite(gap["rate_gap_bits"]) else "rejected_no_distortion_overlap",
                **gap,
            })
    return results


def summary_rows(config: StudyConfig, rows: list[dict[str, str]], failures: list[dict[str, str]],
                 gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize convergence frequency, realized rANS RD gaps, and training time."""
    grouped = curves(rows)
    output: list[dict[str, Any]] = []
    for condition in ("frozen_parent", "scratch30", *warm_conditions(config), "scratch120"):
        repeats = (-1,) if condition == "frozen_parent" else range(config.training_repeats)
        complete = [
            grouped[(condition, repeat)] for repeat in repeats
            if len(grouped.get((condition, repeat), ())) == config.num_planes
        ]
        condition_gaps = [gap for gap in gaps if gap["condition"] == condition]
        valid = [float(gap["rate_gap_bits"]) for gap in condition_gaps if gap["status"] == "ok"]
        bad = sum(
            gap["status"] != "ok" or float(gap["rate_gap_bits"]) > config.bad_convergence_rate_gap_bits
            for gap in condition_gaps
        )
        times = [float(curve[0]["pre_encode_training_seconds"]) for curve in complete]
        output.append({
            "condition": condition, "requested_repeats": len(tuple(repeats)),
            "completed_repeats": len(complete),
            "failure_count": sum(row["condition"] == condition for row in failures),
            "rd_rejection_count": sum(gap["status"] != "ok" for gap in condition_gaps),
            "bad_convergence_count": bad,
            "bad_convergence_frequency": bad / len(condition_gaps) if condition_gaps else np.nan,
            "rate_gap_bits_mean": np.mean(valid) if valid else np.nan,
            "rate_gap_bits_median": np.median(valid) if valid else np.nan,
            "rate_gap_bits_std": np.std(valid, ddof=1) if len(valid) > 1 else np.nan,
            "training_seconds_mean": np.mean(times) if times else np.nan,
            "training_seconds_median": np.median(times) if times else np.nan,
            "total_seconds_median": np.median([
                float(curve[0]["total_seconds"]) for curve in complete]) if complete else np.nan,
        })
    return output


def endpoint_rows(config: StudyConfig, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize paired final-stage deltas without treating them as RD-curve evidence."""
    grouped = curves(rows)
    comparisons = (("scratch30", "scratch120"),) + tuple(
        (condition, reference)
        for condition in warm_conditions(config) for reference in ("scratch120", "scratch30")
    )
    output: list[dict[str, Any]] = []
    for candidate_name, reference_name in comparisons:
        distortion_deltas: list[float] = []
        rate_deltas: list[float] = []
        for repeat in range(config.training_repeats):
            candidate = grouped.get((candidate_name, repeat))
            reference = grouped.get((reference_name, repeat))
            if candidate is None or reference is None or any(
                len(curve) != config.num_planes for curve in (candidate, reference)
            ):
                continue
            distortion_deltas.append(float(candidate[-1]["wmape_db"]) - float(reference[-1]["wmape_db"]))
            rate_deltas.append(float(candidate[-1]["coder_rate"]) - float(reference[-1]["coder_rate"]))
        output.append({
            "candidate": candidate_name,
            "reference": reference_name,
            "paired_repeats": len(distortion_deltas),
            "wmape_db_delta_mean": np.mean(distortion_deltas) if distortion_deltas else np.nan,
            "wmape_db_delta_median": np.median(distortion_deltas) if distortion_deltas else np.nan,
            "wmape_db_delta_min": np.min(distortion_deltas) if distortion_deltas else np.nan,
            "wmape_db_delta_max": np.max(distortion_deltas) if distortion_deltas else np.nan,
            "lower_distortion_count": sum(value < 0 for value in distortion_deltas),
            "coder_rate_delta_mean": np.mean(rate_deltas) if rate_deltas else np.nan,
            "coder_rate_delta_median": np.median(rate_deltas) if rate_deltas else np.nan,
            "coder_rate_delta_min": np.min(rate_deltas) if rate_deltas else np.nan,
            "coder_rate_delta_max": np.max(rate_deltas) if rate_deltas else np.nan,
            "lower_rate_count": sum(value < 0 for value in rate_deltas),
        })
    return output


def plot_curves(config: StudyConfig, rows: list[dict[str, str]], path: Path) -> None:
    """Plot every repeat and condition using cumulative realized rANS rate."""
    grouped = curves(rows)
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = dict(zip(
        ("frozen_parent", "scratch30", *warm_conditions(config), "scratch120"),
        plt.get_cmap("tab20").colors, strict=False))
    for (condition, _repeat), curve in grouped.items():
        if condition == "parent_r12" or len(curve) != config.num_planes:
            continue
        axis.plot(
            [float(row["coder_rate"]) for row in curve],
            [float(row["wmape_db"]) for row in curve],
            "o-", color=colors[condition], alpha=0.22 if condition != "frozen_parent" else 0.9,
            linewidth=1.0, label=condition if condition not in axis.get_legend_handles_labels()[1] else None,
        )
    axis.set(
        xlabel="Cumulative realized rANS rate (bits/symbol)",
        ylabel="Hard-symbol stage WMAPE (dB)",
        title="Round 22: reduced-epoch warm start on seed-42069 gradients",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def write_report(config: StudyConfig, summaries: list[dict[str, Any]], paired_gaps: list[dict[str, Any]],
                 endpoints: list[dict[str, Any]], path: Path) -> None:
    """Write an explicitly incomplete or final concise experiment report."""
    completed = len(tuple(config.output_root.glob("cases/repeat_*/complete.json")))
    status = "FINAL" if completed == config.training_repeats else "INCOMPLETE — no conclusions"
    lines = [
        "# Reduced-epoch gradient warm-start report", "", f"Status: **{status}** ({completed}/{config.training_repeats} repeats).", "",
        "Seed-42069 SVHN/ResNet-18 trace with five total clients; only client 0 is compressed. The round-12 OR parent and round-22 candidates use ten raw SI vectors (all five clients from the preceding two rounds). Rounds 13–21 advance only in the raw FL trace and never call a codec.", "",
        "Rate is cumulative realized rANS `coder_stage_rates`; distortion is hard-symbol stage WMAPE in dB. The 30-epoch gaps are piecewise-linear equal-distortion rate differences versus paired 120-epoch scratch training; the single frozen parent is compared with the mean full-scratch curve. Positive is worse.", "",
        "| condition | complete | failures | rejects | bad frequency | valid-only median gap (bit) | median train s | median total s |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['completed_repeats']}/{row['requested_repeats']} | "
            f"{row['failure_count']} | {row['rd_rejection_count']} | "
            f"{row['bad_convergence_frequency']:.3f} | {row['rate_gap_bits_median']:+.4f} | "
            f"{row['training_seconds_median']:.1f} | {row['total_seconds_median']:.1f} |")
    if completed == config.training_repeats:
        by_condition = {row["condition"]: row for row in summaries}
        scratch = by_condition["scratch30"]
        full = by_condition["scratch120"]
        best = min(
            (by_condition[condition] for condition in warm_conditions(config)),
            key=lambda row: (row["bad_convergence_count"], row["rd_rejection_count"]),
        )
        lines.extend([
            "", "## Verdict", "",
            f"By the declared rejection-or->+{config.bad_convergence_rate_gap_bits:.2f}-bit rule, the lowest observed "
            f"bad-convergence count is {best['condition']} at {best['bad_convergence_count']}/{config.training_repeats}; "
            f"scratch-30 is {scratch['bad_convergence_count']}/{config.training_repeats}. Selection must also consider "
            "the valid paired RD gaps below; survivor-only medians are not evidence when overlap is sparse.",
            "", "## Direct warm versus scratch-30 RD", "",
            "| warm policy | valid overlaps | rejections | valid-only median gap (bit) | warm wins |",
            "|---|---:|---:|---:|---:|",
        ])
        for condition in warm_conditions(config):
            condition_gaps = [row for row in paired_gaps if row["condition"] == condition]
            valid = [float(row["rate_gap_bits"]) for row in condition_gaps if row["status"] == "ok"]
            lines.append(
                f"| {condition} | {len(valid)}/{config.training_repeats} | "
                f"{len(condition_gaps) - len(valid)} | "
                f"{np.median(valid):+.4f} | {sum(value < 0 for value in valid)}/{len(valid)} |"
                if valid else
                f"| {condition} | 0/{config.training_repeats} | {len(condition_gaps)} | n/a | 0/0 |"
            )
        lines.extend([
            "",
            f"All warm policies use round-12 model weights with a fresh optimizer and freshly trained conditional prior. "
            "The grid uses LR multipliers 1, 0.3, 0.1, and 0.03 and initial tau values 5.5, 1, 0.5, and 0.1. "
            "Every tau schedule anneals toward 0.1, so initial tau 0.1 is constant. Every round-22 fit uses one "
            "quantizer candidate.",
            "",
            f"The selected 30-epoch policy reduces median quantizer training time by "
            f"{full['training_seconds_median'] / best['training_seconds_median']:.2f}x and median end-to-end time by "
            f"{full['total_seconds_median'] / best['total_seconds_median']:.2f}x relative to scratch-120.",
            "", "## Final-stage diagnostic (not the RD decision metric)", "",
            "Negative deltas favor the candidate on the named endpoint metric.", "",
            "| candidate vs reference | pairs | median WMAPE delta (dB) | lower distortion | median rate delta (bit/symbol) | lower rate |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in endpoints:
            lines.append(
                f"| {row['candidate']} vs {row['reference']} | {row['paired_repeats']} | "
                f"{row['wmape_db_delta_median']:+.3f} | {row['lower_distortion_count']}/{row['paired_repeats']} | "
                f"{row['coder_rate_delta_median']:+.3f} | {row['lower_rate_count']}/{row['paired_repeats']} |")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def analyze(config_path: Path) -> None:
    """Refresh combined raw data, RD comparisons, timing summary, plot, and report."""
    config = load_config(config_path)
    paths = list(config.output_root.glob("cases/repeat_*/raw_stage_results.csv"))
    parent = config.output_root / "parent" / "raw_stage_results.csv"
    rows = read_rows(([parent] if parent.exists() else []) + paths)
    failures = read_rows(list(config.output_root.glob("cases/repeat_*/failures.csv")))
    gaps = gap_rows(config, rows)
    paired_gaps = paired_warm_scratch_gaps(config, rows)
    summaries = summary_rows(config, rows, failures, gaps)
    endpoints = endpoint_rows(config, rows)
    analysis = config.output_root / "analysis"
    if rows:
        write_csv(analysis / "raw_stage_results.csv", rows, list(rows[0]))
        plot_curves(config, rows, analysis / "rd_curves.png")
    write_csv(analysis / "rd_gaps.csv", gaps, list(gaps[0]) if gaps else [
        "condition", "reference", "repeat", "metric", "status", "rate_gap_bits",
        "rate_gap_percent", "distortion_gap_db", "distortion_gap_percent"])
    write_csv(analysis / "warm_variants_vs_scratch30_gaps.csv", paired_gaps, list(paired_gaps[0]))
    write_csv(analysis / "summary.csv", summaries, list(summaries[0]))
    write_csv(analysis / "endpoint_summary.csv", endpoints, list(endpoints[0]))
    write_report(config, summaries, paired_gaps, endpoints, analysis / "report.md")


def main() -> None:
    """Analyse one declared study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    analyze(args.config.resolve())


if __name__ == "__main__":
    main()
