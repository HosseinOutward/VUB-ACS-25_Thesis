"""Publish matched static-gradient seed medians and independently directed RD comparisons."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db
from FL_code.FL_core.explicit_cache import content_hash


def median_points(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cumulative rates and raw WMAPE before computing curve gaps or converting to dB."""
    points = {key: (np.median([np.cumsum(record[key]) for record in records], axis=0).tolist()
                    if all(record.get(key) is not None for record in records) else None)
              for key in ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
                          "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")}
    points["stage_wmapes"] = np.median([record["stage_wmapes"] for record in records], axis=0).tolist()
    points["wmape_db"] = wmape_to_db(np.array(points["stage_wmapes"])).tolist()
    points["marginal_rate"] = float(np.median([record["marginal_rate"] for record in records]))
    points["observations"] = len(records)
    return points


def curve_gap(row: dict[str, Any], column: dict[str, Any], field: str) -> float:
    """Compare valid monotone curves over their shared distortion range without extrapolation."""
    if any(np.any(np.diff(point[field]) <= 0) or np.any(np.diff(point["stage_wmapes"]) >= 0)
           for point in (row, column)):
        return float("nan")
    return rd_curve_gap(np.array(row[field]), np.array(row["wmape_db"]),
                        np.array(column[field]), np.array(column["wmape_db"]))["rate_gap_percent"]


def matrix_text(points: dict[str, dict[str, Any]], field: str) -> str:
    """Render the complete square, calculating each direction using its own denominator."""
    labels = tuple(points)
    lines = ["            " + " ".join(f"{label:>10}" for label in labels)]
    for row in labels:
        cells = []
        for column in labels:
            gap = curve_gap(points[row], points[column], field)
            cells.append(f"{gap:+.3f}" if np.isfinite(gap) else "—")
        lines.append(f"{row:>11} " + " ".join(f"{cell:>10}" for cell in cells))
    return "\n".join(lines)


def publish(reference_root: Path, roots: tuple[Path, ...], destination: Path) -> None:
    """Read completed three-seed fits, validate their matching provenance, and publish raw tables."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in roots]
    for key in ("baseline_state_hash", "reference_manifest"):
        assert all(manifest[key] == manifests[0][key] for manifest in manifests), f"Unmatched {key}."
    assert all(Path(manifest["plan"]["reference_root"]).resolve() == reference_root.resolve() for manifest in manifests)
    assert all(manifest["numerical_runtime"] == manifests[0]["numerical_runtime"] for manifest in manifests)
    paths = [path for root in roots for path in sorted((root / "cases").glob("*/*/complete.json"))]
    runs = [json.loads(path.read_text()) for path in paths]
    for path in paths:
        weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
        assert all(torch.equal(value, weights["native"][key]) for key, value in weights["joint"].items()
                   if not key.startswith("conditional")), f"Prior fitting changed encoder/decoder weights: {path}"
    grouped = {method: sorted((run for run in runs if run["method"] == method), key=lambda run: run["fit_seed"])
               for method in ("O", "O_tau025", "S_tau025", "T") if any(run["method"] == method for run in runs)}
    assert {"O", "S_tau025", "T"} <= grouped.keys(), "Missing matched O/S/T controls."
    assert all([run["fit_seed"] for run in group] == [43, 44, 45] for group in grouped.values()), "Incomplete three-seed coverage."
    expected_identities = {manifest["identity"] for manifest in manifests}
    assert all(run["identity"] in expected_identities for run in runs)
    frozen = json.loads((reference_root / "cases/F/target/complete.json").read_text())
    assert frozen["weights_unchanged"] and frozen["quantizer_fits"] == frozen["prior_fits"] == 0
    baseline = torch.load(manifests[0]["baseline_artifact"], map_location="cpu", weights_only=True)["state"]
    assert content_hash(baseline) == manifests[0]["baseline_state_hash"], "Retained starting weights changed."
    ordinary = next(path for path, run in zip(paths, runs, strict=True) if run["method"] == "O")
    deployed = torch.load(ordinary.with_name("weights.pt"), map_location="cpu", weights_only=True)["joint"]
    prefix = {key: baseline[key] for key in deployed}
    # F hashed Torch's ordered state container; the fitter recorded an ordinary dictionary.
    assert content_hash(OrderedDict(prefix)) == frozen["frozen_state_hash"], "F starting prefix differs from the retained baseline."
    assert all(run["initial_state_hash"] == content_hash(prefix) for run in runs if run["method"] != "T"), "O/S starting weights differ from F."
    records = {method: [run["record"] for run in grouped[method]] for method in ("O", "O_tau025") if method in grouped}
    records |= {"F*": [frozen["record"]], "S_tau025": [run["record"] for run in grouped["S_tau025"]],
                "T": [run["record"] for run in grouped["T"]]}
    points = {method: median_points(values) for method, values in records.items()}
    diagnostics = []
    for run in runs:
        record = run["record"]
        assert record["symbol_error_rate"] == 0.0
        for field in ("temporal_si_prior", "server_si_prior"):
            if record.get(f"{field}_stage_rates") is not None:
                assert np.isclose(sum(record[f"{field}_stage_rates"]), record[f"{field}_rate"], rtol=2e-6)
        flags = []
        if np.any(np.diff(record["stage_wmapes"]) >= 0):
            flags.append("nonmonotone distortion")
        if record["quantizer_prior_rate"] > record["marginal_rate"]:
            flags.append("joint prior exceeds marginal on full target")
        for index, rate in enumerate(record["coder_stage_rates"], 1):
            if rate < 0.01 or rate > 1.05:
                flags.append(f"stage {index} native stream growth {rate:.6f} bpf")
        diagnostics.append(dict(method=run["method"], seed=run["fit_seed"], flags=flags))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "median_stage_points.json").write_text(json.dumps(points, indent=2))
    (destination / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    lines = ["### Matched median stage RD measurements", "",
             "Cumulative bits/float; medians over fit seeds 43–45. F* is one fixed seed-43 starting model.",
             "Fine-tuned priors are directly measured at every stage; marginal is measured only as a final total.",
             "F*'s joint-prior column is frozen CE, and its refitted-prior columns are unavailable.", "",
             "| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Refitted full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    staged_rows = []
    for method, point in points.items():
        for stage in range(3):
            values = [point["marginal_rate"] if stage == 2 else None]
            values.extend(point[field][stage] if point[field] is not None else None
                          for field in ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
                                        "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates"))
            cells = ["—" if value is None else f"{value:.6f}" for value in values]
            lines.append(f"| {method} | {stage + 1} | " + " | ".join(cells) +
                         f" | {point['wmape_db'][stage]:.6f} ({point['stage_wmapes'][stage]:.6f}%) |")
            staged_rows.append([method, stage + 1, *values, point["wmape_db"][stage], point["stage_wmapes"][stage]])
    with (destination / "median_stage_points.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "stage", "marginal", "joint_prior", "temporal_prior", "full_si_prior", "native_rans", "temporal_rans", "full_si_rans", "wmape_db", "wmape_percent"])
        writer.writerows(staged_rows)
    lines.extend(["", "### Equal-distortion rate changes (%)", "",
                  "Row versus column; negative favors the row. Both directions are recomputed.",
                  f"Coverage: one client/round, all three fit seeds for {'/'.join(grouped)}; F* fixed. No clients excluded.",
                  "With/without symmetric whole-window client exclusions is identical (1/1 target retained).", "",
                  "Full-SI rANS, all measured clients (client 0):", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
                  "Full-SI rANS, with symmetric client exclusions (none):", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
                  "Native transport (T uses temporal SI), all clients and no-exclusion retained population:", "", "```text", matrix_text(points, "coder_stage_rates"), "```", "",
                  "### Individual fit-seed raw metrics", "",
                  "| Method | Seed | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Full-SI rANS | WMAPE % | Quantizer fit s | Prior-fit/encode s | Decode s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for method, group in grouped.items():
        for run in group:
            r = run["record"]
            values = [r.get(key) for key in ("marginal_rate", "quantizer_prior_rate", "temporal_si_prior_rate", "server_si_prior_rate", "coder_rate", "server_si_rans_rate", "wmape", "pre_encode_training_seconds", "encode_seconds", "decode_seconds")]
            lines.append(f"| {method} | {run['fit_seed']} | " + " | ".join("—" if value is None else f"{value:.6f}" for value in values) + " |")
    lines.extend(["", "### Paired-seed comparisons", "",
                  "Directed full-SI rANS gaps versus the same-seed reference; seed-gap medians are separate from median-curve gaps.", "",
                  "| Method | Reference | Seed 43 | Seed 44 | Seed 45 | Valid pairs | Median paired gap % | Directed median-curve gap % |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    comparisons = [("S_tau025", "O"), ("T", "O")]
    if "O_tau025" in grouped:
        comparisons = [("O_tau025", "O"), ("S_tau025", "O"), ("S_tau025", "O_tau025"), ("T", "O")]
    for method, reference in comparisons:
        gaps = [curve_gap(median_points([run["record"]]), median_points([control["record"]]), "server_si_rans_stage_rates")
                for run, control in zip(grouped[method], grouped[reference], strict=True)]
        valid = np.array(gaps)[np.isfinite(gaps)]
        gap = curve_gap(points[method], points[reference], "server_si_rans_stage_rates")
        cells = [f"{value:+.6f}" if np.isfinite(value) else "—" for value in gaps]
        paired = f"{np.median(valid):+.6f}" if valid.size else "—"
        aggregate = f"{gap:+.6f}" if np.isfinite(gap) else "—"
        lines.append(f"| {method} | {reference} | " + " | ".join(cells) + f" | {valid.size}/3 | {paired} | {aggregate} |")
    lines.extend(["", "Diagnostics (no measurement is silently removed):", "",
                  *[f"- {item['method']} / seed {item['seed']}: {', '.join(item['flags']) or 'monotone stages; no flagged rate growth or full-target prior failure'}." for item in diagnostics],
                  "", "All native rANS symbol errors are zero. Per-stage marginals were not measured; symbol-stage death is not inferred.", ""])
    (destination / "tables.md").write_text("\n".join(lines))
    matrix_rows = [dict(rate_source=field, row=row, column=column, rate_change_percent=curve_gap(at, ref, field))
                   for field in ("server_si_rans_stage_rates", "coder_stage_rates")
                   for row, at in points.items() for column, ref in points.items()]
    with (destination / "comparison_gaps.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, field, title in zip(axes, ("server_si_rans_stage_rates", "coder_stage_rates"),
                                  ("Full-SI rANS", "Native transport; T uses temporal SI"), strict=True):
        for method, point in points.items():
            axis.plot(point[field], point["wmape_db"], "o-", label=method)
        axis.set(xlabel="Cumulative bits/float", title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("WMAPE (dB)")
    figure.suptitle("Fixed SVHN gradient: seed-43 starting weights; fit-seed medians 43–45")
    figure.tight_layout()
    figure.savefig(destination / "matched_rd.png", dpi=180)
    figure.savefig(destination / "matched_rd.pdf")
    plt.close(figure)
    print(f"Published matched tables: {destination}")


def main() -> None:
    """Build report tables from the retained F artifact and two completed comparison outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    publish(args.reference_root, tuple(args.roots), args.out)


if __name__ == "__main__":
    main()
