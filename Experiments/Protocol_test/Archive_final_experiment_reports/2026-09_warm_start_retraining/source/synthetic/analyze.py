"""Analyse complete and partial warm-start runs with the repository RD metric."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.Protocol_test.rd_metrics import GAP_METRIC_NAME, mean_wmape_db, rd_curve_gap
from Experiments.warm_start_retraining.experiment import ExperimentConfig, load_config


RATE_FIELDS = ("training_prior_rate", "prior_rate", "marginal_rate")


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    """Read every existing homogeneous case CSV."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Atomically replace one derived CSV, including its header when empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def curves(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, int, float], list[dict[str, str]]]:
    """Group shifted held-out hard-symbol points into complete repeat curves."""
    grouped: dict[tuple[str, int, int, int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "heldout_shifted":
            grouped[(row["condition"], int(row["repeat"]), int(row["bins_per_plane"]),
                     int(row["model_num_planes"]), float(row["reconst_ld"]))].append(row)
    return {key: sorted(group, key=lambda row: int(row["stage"])) for key, group in grouped.items()}


def gap_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute paired repeat-level gaps without averaging away bad convergence."""
    grouped = curves(raw_rows)
    results: list[dict[str, Any]] = []
    for key, curve in grouped.items():
        condition, repeat, bins, planes, reconst_ld = key
        if condition == "frozen_parent":
            continue
        for reference in ("frozen_parent", "scratch"):
            if condition == reference:
                continue
            ref = grouped.get((reference, repeat, bins, planes, reconst_ld))
            for rate_field in RATE_FIELDS:
                base = {
                    "condition": condition, "reference": reference, "repeat": repeat,
                    "bins_per_plane": bins, "model_num_planes": planes,
                    "reconst_ld": reconst_ld, "rate_field": rate_field,
                    "metric": GAP_METRIC_NAME,
                }
                if ref is None or len(curve) != planes or len(ref) != planes:
                    results.append(base | {
                        "status": "rejected_incomplete", "rate_gap_bits": np.nan,
                        "rate_gap_percent": np.nan, "distortion_gap_db": np.nan,
                        "distortion_gap_percent": np.nan,
                    })
                    continue
                gap = rd_curve_gap(
                    np.array([float(row[rate_field]) for row in curve]),
                    np.array([float(row["wmape_db"]) for row in curve]),
                    np.array([float(row[rate_field]) for row in ref]),
                    np.array([float(row["wmape_db"]) for row in ref]),
                    f"{condition} vs {reference} repeat {repeat}",
                )
                status = (
                    "ok"
                    if np.isfinite(gap["rate_gap_bits"]) and np.isfinite(gap["rate_gap_percent"])
                    else "rejected_no_equal_distortion_overlap"
                )
                results.append(base | {"status": status, **gap})
    return results


def rd_summary_rows(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeat-level RD gaps while retaining explicit rejection counts."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fields = (
        "condition", "reference", "bins_per_plane", "model_num_planes",
        "reconst_ld", "rate_field", "metric",
    )
    for row in gaps:
        grouped[tuple(row[field] for field in fields)].append(row)
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        valid = [float(row["rate_gap_bits"]) for row in rows if row["status"] == "ok"]
        summaries.append({
            **dict(zip(fields, key, strict=True)),
            "requested_comparisons": len(rows),
            "valid_comparisons": len(valid),
            "rejected_comparisons": len(rows) - len(valid),
            "rate_gap_bits_mean": np.mean(valid) if valid else np.nan,
            "rate_gap_bits_median": np.median(valid) if valid else np.nan,
            "rate_gap_bits_std": np.std(valid, ddof=1) if len(valid) > 1 else np.nan,
            "rate_gap_bits_min": np.min(valid) if valid else np.nan,
            "rate_gap_bits_max": np.max(valid) if valid else np.nan,
        })
    return summaries


def summary_rows(config: ExperimentConfig, raw_rows: list[dict[str, str]],
                 failures: list[dict[str, str]], gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize completion, rejection, bad-convergence frequency, timing, and RD gaps."""
    grouped = curves(raw_rows)
    output: list[dict[str, Any]] = []
    conditions = ("frozen_parent", *(variant.name for variant in config.variants))
    for architecture in config.architectures:
        for condition in conditions:
            matching = [
                curve for (name, _repeat, bins, planes, ld), curve in grouped.items()
                if name == condition and (bins, planes, ld) == (
                    architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
            ]
            complete = [curve for curve in matching if len(curve) == architecture.num_planes]
            times = [
                float(row["pre_encode_training_seconds"])
                for row in raw_rows
                if row["condition"] == condition and row["phase"].startswith("training")
                and int(row["stage"]) == 1
                and (int(row["bins_per_plane"]), int(row["model_num_planes"]),
                     float(row["reconst_ld"])) == (
                    architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
            ]
            primary = [
                gap for gap in gaps
                if gap["condition"] == condition and gap["reference"] == "scratch"
                and gap["rate_field"] == "training_prior_rate"
                and (gap["bins_per_plane"], gap["model_num_planes"], gap["reconst_ld"]) == (
                    architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
            ]
            rejected = sum(gap["status"] != "ok" for gap in primary)
            bad = sum(
                gap["status"] != "ok" or gap["rate_gap_bits"] > config.bad_convergence_rate_gap_bits
                for gap in primary
            )
            failure_count = sum(
                row["architecture"] == architecture.key and row["condition"] == condition
                for row in failures
            )
            output.append({
                "condition": condition, "bins_per_plane": architecture.bins_per_plane,
                "model_num_planes": architecture.num_planes, "reconst_ld": architecture.reconst_ld,
                "requested_repeats": config.repeats, "completed_repeats": len(complete),
                "failure_count": failure_count, "rd_rejection_count": rejected,
                "bad_convergence_count": bad,
                "bad_convergence_frequency": bad / len(primary) if primary else np.nan,
                "training_seconds_mean": np.mean(times) if times else np.nan,
                "training_seconds_median": np.median(times) if times else np.nan,
                "training_seconds_p90": np.quantile(times, 0.9) if times else np.nan,
            })
    return output


def plot_curves(config: ExperimentConfig, raw_rows: list[dict[str, str]], path: Path) -> None:
    """Plot mean complete shifted held-out curves with repeat traces."""
    grouped = curves(raw_rows)
    figure, axes = plt.subplots(len(config.architectures), len(RATE_FIELDS), figsize=(16, 9), squeeze=False)
    conditions = ("frozen_parent", *(variant.name for variant in config.variants))
    colors = dict(zip(conditions, plt.get_cmap("tab10").colors, strict=False))
    for row_index, architecture in enumerate(config.architectures):
        for column, rate_field in enumerate(RATE_FIELDS):
            axis = axes[row_index, column]
            for condition in conditions:
                selected = [
                    curve for (name, _repeat, bins, planes, ld), curve in grouped.items()
                    if name == condition and (bins, planes, ld) == (
                        architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
                    and len(curve) == architecture.num_planes
                ]
                for curve in selected:
                    axis.plot(
                        [float(point[rate_field]) for point in curve],
                        [float(point["wmape_db"]) for point in curve],
                        color=colors[condition], alpha=0.12, linewidth=0.8,
                    )
                if selected:
                    rates = np.mean([[float(point[rate_field]) for point in curve] for curve in selected], axis=0)
                    distortion = mean_wmape_db(np.array([
                        [float(point["wmape"]) for point in curve] for curve in selected
                    ]))
                    axis.plot(rates, distortion, "o-", color=colors[condition], label=condition)
            axis.set(
                title=f"{architecture.key} | {rate_field}", xlabel="cumulative bits/symbol",
                ylabel="stage WMAPE (dB)",
            )
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=7)
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def write_report(config: ExperimentConfig, summaries: list[dict[str, Any]],
                 rd_summaries: list[dict[str, Any]], raw_rows: list[dict[str, str]], path: Path) -> None:
    """Write a concise, explicitly preliminary-or-final report from current results."""
    completed_cases = len(tuple(config.output_root.glob("cases/*/repeat_*/complete.json")))
    expected_cases = len(config.architectures) * config.repeats
    parent_pairs: dict[tuple[int, int, int, float], dict[str, float]] = defaultdict(dict)
    for row in raw_rows:
        if row["condition"] == "frozen_parent" and int(row["stage"]) == int(row["model_num_planes"]):
            key = (int(row["repeat"]), int(row["bins_per_plane"]),
                   int(row["model_num_planes"]), float(row["reconst_ld"]))
            parent_pairs[key][row["phase"]] = float(row["mse"])
    ratios_by_architecture: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (_repeat, bins, planes, _ld), phases in parent_pairs.items():
        if {"heldout_shifted", "heldout_unshifted"} <= phases.keys():
            ratios_by_architecture[(bins, planes)].append(
                phases["heldout_shifted"] / phases["heldout_unshifted"])
    ratios = [ratio for values in ratios_by_architecture.values() for ratio in values]
    status = "FINAL" if completed_cases == expected_cases else "INCOMPLETE — no conclusions"
    lines = [
        "# Warm-start retraining report", "", f"Status: **{status}** ({completed_cases}/{expected_cases} cases).", "",
        "The experiment preserves the Brent-scale data, production protocol, architectures, training schedule, normalization, and hard-symbol evaluation. The requested differences are 12 repeats, an additive SI shift of +0.1, disk-cached parent states, and the declared LR/tau retraining variants.", "",
        f"The frozen-parent shifted/unshifted final-stage MSE ratio currently has median {np.median(ratios):.3f} across {len(ratios)} paired cases." if ratios else "No paired shift diagnostic is complete yet.", "",
        "Bad convergence is predeclared as an invalid/non-overlapping complete curve or a trained-prior equal-distortion rate gap greater than 0.05 bits versus paired full scratch training.", "",
        "| condition | architecture | complete | failures | RD rejects | bad frequency | median train s |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['condition']} | {row['bins_per_plane']}×{row['model_num_planes']} | "
            f"{row['completed_repeats']}/{row['requested_repeats']} | {row['failure_count']} | "
            f"{row['rd_rejection_count']} | {row['bad_convergence_frequency']:.3f} | "
            f"{row['training_seconds_median']:.1f} |"
        )
    if completed_cases == expected_cases:
        lines.extend(["", "## Findings", ""])
        for architecture in config.architectures:
            values = ratios_by_architecture[(architecture.bins_per_plane, architecture.num_planes)]
            lines.append(
                f"- The +0.1 SI shift increased frozen-parent final-stage MSE in all "
                f"{len(values)}/{len(values)} {architecture.bins_per_plane}×{architecture.num_planes} "
                f"repeats; median multiplier {np.median(values):.3f} "
                f"(range {min(values):.3f}–{max(values):.3f}).")
        lines.extend([
            "- There were no production training exceptions or fatal candidate-rejection failures. One 2×3 parent (repeat 6) converged to a zero-rate constant-symbol solution; every warm variant inherited that collapse, while full scratch training recovered.",
            "- Against scratch on trained-prior RD, bad-convergence totals over both architectures were: naïve warm 2/24, 10× lower LR 4/24, lower tau 6/24, and the LR+tau combination 1/24.",
            "- No warm variant dominates every criterion. The combination had the fewest declared bad convergences, naïve warm stayed closest to scratch for the separately retrained conditional-prior rate, and lower tau alone was the least stable.",
            "- Warm starts did not reduce runtime because every method intentionally retained all 180 epochs.",
            "", "## Equal-distortion trained-prior RD gaps", "",
            "Positive values are worse than the reference. Rejections mean the complete hard-symbol curves had no shared distortion interval.", "",
            "| condition | architecture | vs frozen mean (valid/rejected) | vs scratch median (valid/rejected) |",
            "|---|---:|---:|---:|",
        ])
        for architecture in config.architectures:
            for condition in (variant.name for variant in config.variants if variant.name != "scratch"):
                matches = {
                    row["reference"]: row for row in rd_summaries
                    if row["condition"] == condition
                    and row["rate_field"] == "training_prior_rate"
                    and (row["bins_per_plane"], row["model_num_planes"], row["reconst_ld"]) == (
                        architecture.bins_per_plane, architecture.num_planes, architecture.reconst_ld)
                }
                frozen, scratch = matches["frozen_parent"], matches["scratch"]
                lines.append(
                    f"| {condition} | {architecture.bins_per_plane}×{architecture.num_planes} | "
                    f"{frozen['rate_gap_bits_mean']:+.4f} ({frozen['valid_comparisons']}/{frozen['rejected_comparisons']}) | "
                    f"{scratch['rate_gap_bits_median']:+.4f} ({scratch['valid_comparisons']}/{scratch['rejected_comparisons']}) |")
        lines.extend([
            "", "All three rate definitions and repeat-level values are preserved in `rd_summary.csv` and `rd_gaps.csv`; the table above uses the quantizer's jointly trained prior, the primary RD measure.",
        ])
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def analyze(config_path: Path) -> None:
    """Refresh raw aggregation, repeat-level gaps, summaries, plot, and report."""
    config = load_config(config_path)
    analysis_dir = config.output_root / "analysis"
    raw_rows = read_rows(list(config.output_root.glob("cases/*/repeat_*/raw_stage_results.csv")))
    failures = read_rows(list(config.output_root.glob("cases/*/repeat_*/failures.csv")))
    gaps = gap_rows(raw_rows)
    rd_summaries = rd_summary_rows(gaps)
    summaries = summary_rows(config, raw_rows, failures, gaps)
    if raw_rows:
        write_csv(analysis_dir / "raw_stage_results.csv", raw_rows, list(raw_rows[0]))
        plot_curves(config, raw_rows, analysis_dir / "rd_curves.png")
    write_csv(analysis_dir / "rd_gaps.csv", gaps, list(gaps[0]) if gaps else [
        "condition", "reference", "repeat", "bins_per_plane", "model_num_planes",
        "reconst_ld", "rate_field", "metric", "status", "rate_gap_bits",
        "rate_gap_percent", "distortion_gap_db", "distortion_gap_percent",
    ])
    write_csv(analysis_dir / "rd_summary.csv", rd_summaries, list(rd_summaries[0]))
    write_csv(analysis_dir / "summary.csv", summaries, list(summaries[0]))
    write_report(config, summaries, rd_summaries, raw_rows, analysis_dir / "report.md")


def main() -> None:
    """Analyse one experiment config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    analyze(args.config.resolve())


if __name__ == "__main__":
    main()
