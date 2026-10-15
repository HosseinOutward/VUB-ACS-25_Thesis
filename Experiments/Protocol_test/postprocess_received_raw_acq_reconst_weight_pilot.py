"""Publish the matched seed-43 acquisition-loss weight comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text
from FL_code.FL_core.explicit_cache import content_hash

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
COLDFIT = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
NORMALIZED = EXPERIMENTS / "s_received_mae_five_seed_20261012"
RAW_ACQUISITION = EXPERIMENTS / "s_received_raw_mae_pilot_20261013"
RAW_LD05 = EXPERIMENTS / "s_received_raw_mae_acq_ld05_seed43_20261013"
RAW_LD075 = EXPERIMENTS / "s_received_raw_mae_acq_ld075_seed43_20261013"
OUTPUT = EXPERIMENTS / "s_received_raw_mae_reconst_weight_seed43_20261013" / "final_analysis"
SEED = 43
METHOD_ROOTS = (
    ("O", COLDFIT, "O"),
    ("S_norm_MAE", NORMALIZED, "current_e1_mae_received_prefix_bins"),
    ("S_raw_MAE_ld1", RAW_ACQUISITION, "current_e1_raw_mae_received_prefix_bins"),
    ("S_raw_MAE_ld05", RAW_LD05, "current_e1_raw_mae_received_prefix_bins"),
    ("S_raw_MAE_ld075", RAW_LD075, "current_e1_raw_mae_received_prefix_bins"),
)


def read_run(root: Path, method: str) -> tuple[dict[str, Any], Path]:
    """Load one completed seed-43 record and its output path."""
    path = root / "cases" / f"{method}_seed{SEED}" / "target" / "complete.json"
    run = json.loads(path.read_text())
    assert run["fit_seed"] == SEED and run["record"]["symbol_error_rate"] == 0, path
    return run, path


def encoder_hash(path: Path) -> str:
    """Hash the deployed full-target encoder and binners saved for one method."""
    weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
    state = {key: value for key, value in weights["native"].items()
             if key.startswith(("encoder.", "binner", "binners."))}
    return content_hash(state)


def common_span(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float] | None:
    """Return the overlap of two WMAPE-dB ranges, or None when they do not overlap."""
    low = max(min(first["wmape_db"]), min(second["wmape_db"]))
    high = min(max(first["wmape_db"]), max(second["wmape_db"]))
    return (low, high) if high > low else None


def metadata_complete_record(run: dict[str, Any]) -> dict[str, Any]:
    """Add measured decoder-required metadata once to full-SI rANS stage one."""
    record = run["record"].copy()
    metadata_mib = record["meta_data_size"] + (record.get("sample_metadata_size") or 0.0)
    metadata_bpf = metadata_mib * (1024 ** 2) * 8 / record["model_size"]
    stages = list(record["server_si_rans_stage_rates"])
    stages[0] += metadata_bpf
    record["server_si_rans_stage_rates"] = stages
    record["server_si_rans_rate"] = sum(stages)
    record["metadata_complete_cost_bpf"] = metadata_bpf
    return record


def make_report() -> str:
    """Validate the fixed seed and write full-SI, native, stage, and sample comparisons."""
    runs: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    manifests = []
    for label, root, method in METHOD_ROOTS:
        runs[label], paths[label] = read_run(root, method)
        manifests.append(json.loads((root / "test_config.json").read_text()))
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(item[field] == manifests[0][field] for item in manifests), f"Mismatched {field}."
    assert all(Path(item["plan"]["reference_root"]).resolve() == Path(manifests[0]["plan"]["reference_root"]).resolve()
               for item in manifests), "Mismatched retained target and starting-state reference."

    corrected_records = {label: metadata_complete_record(run) for label, run in runs.items()}
    points = {label: mean_points([corrected_records[label]]) for label in runs}
    coded_points = {label: mean_points([run["record"]]) for label, run in runs.items()}
    for label, run in runs.items():
        assert np.allclose(run["record"]["coder_stage_rates"],
                           run["record"]["server_si_rans_stage_rates"], rtol=0, atol=1e-12), (
                               f"Native and original full-SI symbol rates differ for {label}.")
    for label, point in points.items():
        assert np.all(np.diff(point["wmape_db"]) < 0), f"Nonmonotone WMAPE curve: {label}"
        assert np.all(np.diff(point["server_si_rans_stage_rates"]) > 0), f"Nonmonotone rANS curve: {label}"
    sample_methods = tuple(label for label in points if label != "O")
    index_hashes = {runs[label]["sample_indices_hash"] for label in sample_methods}
    coordinate_seeds = {runs[label]["sample_coordinate_seed"] for label in sample_methods}
    target_hashes = {runs[label]["refiner_training_target_hash"] for label in sample_methods}
    encoders = {encoder_hash(paths[label]) for label in sample_methods}
    assert len(index_hashes) == len(coordinate_seeds) == len(target_hashes) == len(encoders) == 1, (
        "Sample methods do not share indices, source target, or frozen base encoder.")
    for label, multiplier in (("S_raw_MAE_ld05", .5), ("S_raw_MAE_ld075", .75)):
        root = next(root for candidate, root, _method in METHOD_ROOTS if candidate == label)
        manifest = json.loads((root / "test_config.json").read_text())
        assert manifest["plan"]["refiner_fit_reconst_multiplier"] == multiplier
        assert runs[label]["refiner_fit_reconst_multiplier"] == multiplier
        assert runs[label]["refiner_fit_reconst_ld_effective"] == (
            runs[label]["refiner_fit_reconst_ld_base"] * multiplier)
        assert runs[label]["sample_fit_reconst_ld"] == runs[label]["refiner_fit_reconst_ld_base"]
    assert runs["S_raw_MAE_ld05"]["acquisition_state_hash"] != runs["S_raw_MAE_ld1"]["acquisition_state_hash"]
    assert runs["S_raw_MAE_ld075"]["acquisition_state_hash"] != runs["S_raw_MAE_ld1"]["acquisition_state_hash"]

    labels = tuple(points)
    matrix = matrix_text(points, "server_si_rans_stage_rates")
    native_matrix = matrix_text(points, "coder_stage_rates")
    span_rows = []
    for index, row in enumerate(labels):
        for column in labels[index + 1:]:
            span = common_span(points[row], points[column])
            if span is None:
                span_rows.append(f"| {row} | {column} | — | — | — | — |")
                continue
            low, high = span
            low_pct, high_pct = 100 * 10 ** (low / 20), 100 * 10 ** (high / 20)
            forward = curve_gap(points[row], points[column], "server_si_rans_stage_rates")
            reverse = curve_gap(points[column], points[row], "server_si_rans_stage_rates")
            span_rows.append(f"| {row} | {column} | {low:.6f} to {high:.6f} dB | "
                             f"{low_pct:.6f}% to {high_pct:.6f}% | {forward:+.6f}% | {reverse:+.6f}% |")

    stage_rows = []
    for label in labels:
        point, record = points[label], runs[label]["record"]
        for stage in range(len(point["stage_wmapes"])):
            index = stage + 1
            marginal = f"{point['marginal_rate']:.6f}" if index == len(point["stage_wmapes"]) else "—"
            prior = point["quantizer_prior_stage_rates"][stage]
            full_prior = point["server_si_prior_stage_rates"][stage]
            native = point["coder_stage_rates"][stage]
            full = point["server_si_rans_stage_rates"][stage]
            coded_full = coded_points[label]["server_si_rans_stage_rates"][stage]
            stage_rows.append(
                f"| {label} | {index} | {marginal} | {prior:.6f} | — | {full_prior:.6f} | "
                f"{native:.6f} | — | {coded_full:.6f} | {full:.6f} | {point['wmape_db'][stage]:.6f} "
                f"({record['stage_wmapes'][stage]:.6f}%) |"
            )

    sample_rows, sample_stage_rows, metadata_rows = [], [], []
    for label in labels:
        record = runs[label]["record"]
        base_metadata = record["meta_data_size"]
        sample_metadata = record.get("sample_metadata_size") or 0.0
        corrected_stage1 = corrected_records[label]["server_si_rans_stage_rates"][0]
        original_stage1 = record["server_si_rans_stage_rates"][0]
        metadata_rows.append(
            f"| {label} | {base_metadata:.9f} MiB | {sample_metadata:.9f} MiB | "
            f"{corrected_records[label]['metadata_complete_cost_bpf']:.9f} | "
            f"{original_stage1:.9f} | {corrected_stage1:.9f} |"
        )
    for label in sample_methods:
        record = runs[label]["record"]
        sample_rows.append(
            f"| {label} | {record['sample_count']} | {record['sample_wmape']:.6f}% | "
            f"{record['sample_prior_rate']:.9f} | {record['sample_coder_rate']:.9f} | "
            f"{record['sample_coder_overhead_rate']:.9f} | {record['meta_data_size']:.9f} MiB | "
            f"{record['sample_metadata_size']:.9f} MiB | "
            f"{corrected_records[label]['metadata_complete_cost_bpf']:.9f} | "
            f"{runs[label]['sample_coordinate_seed']} | `{runs[label]['sample_indices_hash']}` |"
        )
        for stage, (wmape, entropy, prior) in enumerate(zip(
            record["sample_stage_wmape"], record["sample_stage_marginal_entropy_bps"],
            record["sample_stage_prior_rate"], strict=True,
        ), start=1):
            sample_stage_rows.append(f"| {label} | {stage} | {wmape:.6f}% | {entropy:.6f} | {prior:.9f} |")

    return "\n".join((
        "# Seed-43 acquisition raw-MAE reconstruction-weight probe",
        "",
        "This is a fixed, single-seed matched comparison; no seeds are trimmed. The causal coefficient "
        "comparison is the same raw-domain acquisition-MAE path with `reconst_ld` multipliers 1.0, 0.5, "
        "and 0.75. The normalized-MAE practical S and O curves provide end-to-end context. O is read-only. "
        "Every method uses the same retained target/SI and baseline encoder; all sample methods use the same "
        "coordinates and frozen full-target encoder. The primary is a metadata-adjusted measured-size estimate: "
        "the coded full-SI rANS rate plus independently measured gzip/pickle sizes for decoder metadata, charged "
        "once at stage one as `8 * 2^20 * (meta_data_size + sample_metadata_size) / model_size` bits/float. O "
        "uses base metadata only; S also includes sample metadata. These compressed-size measurements are proxies, "
        "not exact additive fragments of nested payload serialization. Sample rANS bytes already include header "
        "and lane flush; diagnostic overhead is not added again. Full-payload `final_bytes` is distinct and is not "
        "substituted for this estimate. Original coded full-SI rates and exact native transport remain separate. "
        "Negative matrix entries favor the row. No symbol errors or nonmonotone curves occur.",
        "",
        "For the multiplier runs, only the local acquisition refiner's `reconst_ld` was scaled. Raw-MAE "
        "acquisition distortion, baseline-derived scales/denominator, sample packet format, received-prefix "
        "final fit, and final-fit/base `reconst_ld` stayed fixed. Provenance records the requested multiplier, "
        "base value, and effective acquisition value. The normalized-MAE comparator changes the acquisition "
        "loss as a separate end-to-end reference.",
        "",
        "Seed-43 result: λ=0.5 is 0.419% better than normalized-MAE and 5.374% better than raw-MAE λ=1 "
        "on the metadata-adjusted measured-size full-SI estimate; λ=0.75 is 1.939% worse than normalized-MAE "
        "and 3.137% better than raw-MAE λ=1 on that estimate. "
        "These are single-seed probe results, pending the preauthorized fresh seeds 44–47 confirmation. "
        "The prior untrimmed seed-43–45 final-S raw-MAE objective pilot was 1.561% worse than its matched "
        "normalized-MAE final-MSE control.",
        "",
        "## Metadata-adjusted measured-size full-SI estimate: directed matrix (%)",
        "",
        "```text",
        matrix,
        "```",
        "",
        "## Original coded full-SI rANS directed matrix (%)",

        "This is the measured coded rANS symbol rate before the metadata-size estimate.",

        "```text",
        matrix_text(coded_points, "server_si_rans_stage_rates"),
        "```",
        "",
        "## Native transport directed matrix (%)",
        "",
        "This is the exact measured native symbol transport, kept separate from the metadata-adjusted estimate. "
        "In this cohort, it equals the unadjusted full-SI rANS symbols.",
        "",
        "```text",
        native_matrix,
        "```",
        "",
        "## Pairwise shared distortion spans and directed gaps",
        "",
        "Gaps use the metadata-adjusted measured-size full-SI estimate. The shared WMAPE range is the interval both curves cover; each direction "
        "is independently recomputed with its own denominator.",
        "",
        "| Row | Column | Shared WMAPE dB | Shared raw WMAPE | Row vs column (%) | Column vs row (%) |",
        "|---|---|---:|---:|---:|---:|",
        *span_rows,
        "",
        "## Full-vector staged RD points",
        "",
        "Rates are cumulative bits/float through each stage. Full-SI estimate includes the metadata-size proxy "
        "at stage one; coded full-SI and native transport remain unadjusted. Scalar marginal rate is shown at the final stage only.",
        "",
        "| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Coded full-SI rANS | Metadata-adjusted estimate | WMAPE dB (raw %) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *stage_rows,
        "",
        "### Independently measured metadata-size proxies",
        "",
        "Base decoder metadata is measured for O and S. Sample-coordinate/normalization metadata is measured only "
        "for S. These standalone gzip/pickle compressed-size measurements are proxies, not exact additive wire "
        "fragments. Their b/float estimate uses `record.model_size`; neither nested full-payload `final_bytes` nor "
        "an alternative wrapper serialization is substituted.",
        "",
        "| Method | Base metadata | Sample metadata | Added estimate b/float | Coded full-SI stage one | Metadata-adjusted estimate stage one |",
        "|---|---:|---:|---:|---:|---:|",
        *metadata_rows,
        "",
        "## Sample cost and reconstruction diagnostics (FYI)",
        "",
        "Sample WMAPE is a diagnostic only; the primary objective is the whole-vector curve above.",
        "",
        "| Method | Samples | Final sample WMAPE | Sample prior rate | Sample rANS | rANS overhead | Base metadata MiB | Sample metadata MiB | Total metadata b/float | Coordinate seed | Index hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *sample_rows,
        "",
        "### Acquisition-stage sample values",
        "",
        "| Method | Acquired stage | Sample WMAPE | Marginal entropy b/float | Joint prior b/float |",
        "|---|---:|---:|---:|---:|",
        *sample_stage_rows,
        "",
        "## Acquisition weight provenance",
        "",
        "| Method | Acquisition multiplier | Base `reconst_ld` | Effective acquisition `reconst_ld` | Final sample-fit `reconst_ld` | Acquisition state hash |",
        "|---|---:|---:|---:|---:|---|",
        *[f"| {label} | {runs[label].get('refiner_fit_reconst_multiplier', 1.0):.2f} | "
          f"{runs[label].get('refiner_fit_reconst_ld_base', '—')} | "
          f"{runs[label].get('refiner_fit_reconst_ld_effective', '—')} | "
          f"{runs[label].get('sample_fit_reconst_ld', '—')} | `{runs[label].get('acquisition_state_hash', '—')}` |"
          for label in sample_methods],
        "",
        "The 0.5 and 0.75 fits are independent fresh seed-43 acquisition/final fits. Neither uses a cached fit.",
        "",
    ))


def main() -> None:
    """Write the validated comparison to its dated analysis directory."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "matched_rd_comparison.md").write_text(make_report())
    print(OUTPUT / "matched_rd_comparison.md")


if __name__ == "__main__":
    main()
