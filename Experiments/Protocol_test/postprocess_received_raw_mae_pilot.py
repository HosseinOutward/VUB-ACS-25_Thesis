"""Compare the received-prefix raw-MAE pilot with matched read-only controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
COLDFIT = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
NORMALIZED = EXPERIMENTS / "s_received_mae_five_seed_20261012"
PILOT = EXPERIMENTS / "s_received_raw_mae_pilot_20261013"
SEEDS = (43, 44, 45)
METHODS = ("O", "S_e1", "S_received_MAE", "S_received_raw_MAE")


def read_case(root: Path, method: str, seed: int) -> dict[str, Any]:
    """Read a completed, zero-symbol-error record for one named method and fit seed."""
    path = root / "cases" / f"{method}_seed{seed}" / "target" / "complete.json"
    run = json.loads(path.read_text())
    assert run["record"]["symbol_error_rate"] == 0, f"Symbol errors in {path}"
    return run


def check_manifests(roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    """Require the controls and candidate to share target, starting state, source, SI, and runtime."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in roots]
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(manifest[field] == manifests[0][field] for manifest in manifests), f"Mismatched {field}."
    assert all(Path(item["plan"]["reference_root"]).resolve() == Path(manifests[0]["plan"]["reference_root"]).resolve()
               for item in manifests), "Mismatched retained reference root."
    return manifests


def load_groups() -> dict[str, list[dict[str, Any]]]:
    """Load matched seed 43–45 controls and fresh raw-MAE fits, checking the controlled setup."""
    manifests = check_manifests((COLDFIT, NORMALIZED, PILOT))
    controls = manifests[0]["plan"]
    normalized, raw = manifests[1]["plan"], manifests[2]["plan"]
    shared_settings = ("sample_size", "sample_fit_epochs", "sample_fit_lr", "sample_fit_tau",
                       "sample_fit_batch_size", "refiner_fit_epochs", "refiner_fit_lr", "refiner_fit_tau",
                       "refiner_fit_lr_step", "refiner_fit_batch_size", "sample_refiner",
                       "sample_refiner_residual_anchor", "sample_train_encoder", "extra_planes")
    assert all(normalized[field] == raw[field] for field in shared_settings), "Pilot changed a non-loss setting."
    assert raw["seeds"] == list(SEEDS) and raw["variant_modes"] == ["current_e1_raw_mae_received_prefix_bins"]
    assert controls["seeds"][:3] == list(SEEDS) and normalized["seeds"][:3] == list(SEEDS)

    groups = {
        "O": [read_case(COLDFIT, "O", seed) for seed in SEEDS],
        "S_e1": [read_case(COLDFIT, "S_e1", seed) for seed in SEEDS],
        "S_received_MAE": [read_case(NORMALIZED, f"current_e1_mae_received_prefix_bins", seed) for seed in SEEDS],
        "S_received_raw_MAE": [read_case(PILOT, "current_e1_raw_mae_received_prefix_bins", seed) for seed in SEEDS],
    }
    for method, runs in groups.items():
        assert [run["fit_seed"] for run in runs] == list(SEEDS), f"Incomplete seed coverage for {method}."
    for normalized_run, raw_run in zip(groups["S_received_MAE"], groups["S_received_raw_MAE"], strict=True):
        for field in ("sample_coordinate_seed", "sample_indices_hash"):
            assert normalized_run[field] == raw_run[field], f"Paired sample coordinates differ at {field}."
        for field in ("sample_count",):
            assert normalized_run["record"][field] == raw_run["record"][field] == 3_600_000
        assert normalized_run["record"]["sample_fit_mode"] == raw_run["record"]["sample_fit_mode"] == "received_prefix_bins"
        assert normalized_run["record"]["sample_training_encoder"] is False
        assert normalized_run["sample_acquisition_distortion"] == "mae"
        assert raw_run["sample_acquisition_distortion"] == "raw_mae"
        assert raw_run["record"]["sample_acquisition_weight_source"] == "baseline_fit_target_norm_factors_and_raw_mean_abs"
        assert raw_run["record"]["sample_training_encoder"] is False
    source_hashes = {run["refiner_training_target_hash"] for method in ("S_e1", "S_received_MAE", "S_received_raw_MAE")
                     for run in groups[method]}
    assert len(source_hashes) == 1, "Acquisition fits used different source/baseline targets."
    assert next(iter(source_hashes)) != manifests[1]["reference_manifest"]["target_hash"], (
        "Acquisition training source matches the held-out current target.")
    return groups


def stage_table(points: dict[str, dict[str, Any]]) -> list[str]:
    """Render the averaged cumulative RD stages, converting mean raw WMAPE to dB afterward."""
    fields = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
              "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    headings = ("Joint/pre-fit prior", "Temporal prior", "Full-SI prior", "Native rANS", "Temporal rANS", "Full-SI rANS")
    lines = ["| Method | Stage | Marginal | " + " | ".join(headings) + " | WMAPE dB (raw %) |",
             "|---|---:|---:|" + "---:|" * len(headings) + "---|"]
    for method, point in points.items():
        for index in range(3):
            rates = [point[field][index] if point[field] is not None else None for field in fields]
            marginal = point["marginal_rate"] if index == 2 else None
            cells = ["—" if value is None else f"{value:.6f}" for value in (marginal, *rates)]
            lines.append(f"| {method} | {index + 1} | " + " | ".join(cells) +
                         f" | {point['wmape_db'][index]:.6f} ({point['stage_wmapes'][index]:.6f}%) |")
    return lines


def sample_table(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Render per-seed and cohort sample diagnostics without adding them again to whole-vector rates."""
    methods = ("S_e1", "S_received_MAE", "S_received_raw_MAE")
    lines = ["| Method | Seed | Sample WMAPE % | Sample prior/rANS b/float | rANS overhead b/float | Metadata MiB | Full-vector/full-SI rANS b/float |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for method in methods:
        for run in groups[method]:
            record = run["record"]
            cumulative = np.cumsum(record["server_si_rans_stage_rates"])
            lines.append(f"| {method} | {run['fit_seed']} | {record['sample_wmape']:.6f} | "
                         f"{record['sample_prior_rate']:.6f}/{record['sample_coder_rate']:.6f} | "
                         f"{record['sample_coder_overhead_rate']:.6f} | {record['sample_metadata_size']:.6f} | "
                         f"{'/'.join(f'{value:.6f}' for value in cumulative)} |")
        records = [run["record"] for run in groups[method]]
        lines.append(f"| {method} mean | 43–45 | {np.mean([r['sample_wmape'] for r in records]):.6f} | "
                     f"{np.mean([r['sample_prior_rate'] for r in records]):.6f}/"
                     f"{np.mean([r['sample_coder_rate'] for r in records]):.6f} | "
                     f"{np.mean([r['sample_coder_overhead_rate'] for r in records]):.6f} | "
                     f"{np.mean([r['sample_metadata_size'] for r in records]):.6f} | — |")
    return lines


def paired_table(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Report each raw-MAE seed's directed RD gaps against the same-seed controls."""
    lines = ["| Seed | Comparator | Raw-MAE vs comparator (%) | Comparator vs raw-MAE (%) |", "|---:|---|---:|---:|"]
    for seed_index, seed in enumerate(SEEDS):
        candidate = mean_points([groups["S_received_raw_MAE"][seed_index]["record"]])
        for method in ("O", "S_e1", "S_received_MAE"):
            comparator = mean_points([groups[method][seed_index]["record"]])
            forward = curve_gap(candidate, comparator, "server_si_rans_stage_rates")
            reverse = curve_gap(comparator, candidate, "server_si_rans_stage_rates")
            display = lambda value: f"{value:+.3f}" if np.isfinite(value) else "—"
            lines.append(f"| {seed} | {method} | {display(forward)} | {display(reverse)} |")
    return lines


def build_report(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Build the untrimmed four-method aggregate and paired-seed RD report."""
    points = {method: mean_points([run["record"] for run in runs]) for method, runs in groups.items()}
    nonmonotone = [(method, run["fit_seed"]) for method, runs in groups.items() for run in runs
                   if np.any(np.diff(run["record"]["stage_wmapes"]) >= 0)]
    return "\n".join([
        "# Received-prefix raw-MAE acquisition pilot",
        "",
        "This is an untrimmed, fixed seed 43–45 comparison. The objective-only comparison is raw-MAE versus normalized-MAE: acquisition/refiner and final-S schedules, 3.6M sample size, received-prefix fitting, frozen final-S encoder, target, starting weights, SI, and sample coordinates are matched; only the acquisition loss changes. O and S_e1/MSE are read-only end-to-end context from the cold-fit five-seed study, not causal controls: S_e1 uses reencode-teacher final fitting with a trainable encoder. Normalized-MAE records are read-only from the received-prefix five-seed confirmation. No control fits or sample payloads were reused.",
        "",
        "For each active plane, raw-MAE distortion is `(2 * reconst_ld / max(mean(abs(baseline_raw_target)), float32.tiny)) * mean_i(scale_i * mean_coordinate(abs(xnorm_hat_i - xnorm_i)))`, plus the unchanged KL rate; plane losses are then averaged. Each `scale_i` is the exact per-slice scale from `PreprocessMetadata.norm_factors` for that baseline fit target. The scale-normalized denominator makes positive uniform changes of raw units cancel, including for tiny source values. No current-target labels or current-target-derived scale weights enter acquisition training. The final-S fit consumes the actually received ordinary prefix sample, and the reported full-vector/full-SI rANS curve includes all charged sample symbols, overhead, and metadata. Sample WMAPE is diagnostic only.",
        "",
        "Cohort curves average cumulative rates and raw WMAPE over the same three seeds before WMAPE-to-dB conversion and equal-distortion comparison. Each matrix direction is recomputed; negative favors the row. No trimming or client exclusions are applied.",
        "",
        "## Full-vector stage RD means",
        "",
        *stage_table(points),
        "",
        "## Full-SI rANS directed matrix (%)",
        "",
        "```text",
        matrix_text(points, "server_si_rans_stage_rates"),
        "```",
        "",
        "## Paired seed comparisons",
        "",
        "Each gap uses that pair's shared WMAPE-dB interval; entries are diagnostic and are not averaged for the cohort result. Raw-MAE versus normalized-MAE isolates the acquisition objective. O and old MSE comparisons show end-to-end context because their final-S fitting paths differ.",
        "",
        *paired_table(groups),
        "",
        "## Sample diagnostics (FYI)",
        "",
        *sample_table(groups),
        "",
        f"All 12 curves have zero symbol errors. Nonmonotone seed curves: {nonmonotone or 'none'}.",
        "",
    ])


def main() -> None:
    """Write the report after all three raw-MAE pilot cases complete."""
    groups = load_groups()
    destination = PILOT / "final_analysis" / "matched_rd_comparison.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(groups))
    print(destination)


if __name__ == "__main__":
    main()
