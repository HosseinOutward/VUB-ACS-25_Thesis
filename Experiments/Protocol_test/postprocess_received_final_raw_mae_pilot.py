"""Compare received-prefix final raw-MAE fitting with its matched normalized-MAE control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text
from FL_code.FL_core.explicit_cache import content_hash
from FL_code.FL_core.utils import decompress_data_list

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
COLDFIT = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
NORMALIZED = EXPERIMENTS / "s_received_mae_five_seed_20261012"
PILOT = EXPERIMENTS / "s_received_final_raw_mae_pilot_20261013"
SEEDS = (43, 44, 45)
FINAL_RAW_VARIANT = "current_e1_mae_received_prefix_bins_final_raw_mae"
METHODS = ("O", "S_e1", "S_received_MAE", "S_received_final_raw_MAE")


def read_case(root: Path, method: str, seed: int) -> dict[str, Any]:
    """Read a completed, zero-symbol-error record and retain its case path for payload checks."""
    path = root / "cases" / f"{method}_seed{seed}" / "target" / "complete.json"
    run = json.loads(path.read_text())
    assert run["record"]["symbol_error_rate"] == 0, f"Symbol errors in {path}"
    run["_case_path"] = path
    return run


def sample_packet(run: dict[str, Any]) -> tuple[Any, bytes]:
    """Read the acquired sample object and retain its serialized wrapper bytes."""
    payload = decompress_data_list(run["_case_path"].with_name("payload.bin").read_bytes())
    serialized = payload["sample_payload"]
    return decompress_data_list(serialized), serialized


def encoder_hash(run: dict[str, Any]) -> str:
    """Hash the deployed base encoder and binners saved with one completed case."""
    weights = torch.load(run["_case_path"].with_name("weights.pt"), map_location="cpu", weights_only=True)
    state = {key: value for key, value in weights["native"].items()
             if key.startswith(("encoder.", "binner", "binners."))}
    return content_hash(state)


def check_manifests(roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    """Require shared baseline state, reference target/SI, and numerical runtime."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in roots]
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(manifest[field] == manifests[0][field] for manifest in manifests), f"Mismatched {field}."
    assert all(Path(item["plan"]["reference_root"]).resolve() == Path(manifests[0]["plan"]["reference_root"]).resolve()
               for item in manifests), "Mismatched retained reference root."
    return manifests


def load_groups() -> dict[str, list[dict[str, Any]]]:
    """Load predeclared seeds and prove final-loss variants received identical acquisition packets."""
    manifests = check_manifests((COLDFIT, NORMALIZED, PILOT))
    cold, normalized, final_raw = (item["plan"] for item in manifests)
    shared = ("sample_size", "sample_fit_epochs", "sample_fit_lr", "sample_fit_tau", "sample_fit_batch_size",
              "refiner_fit_epochs", "refiner_fit_lr", "refiner_fit_tau", "refiner_fit_lr_step",
              "refiner_fit_batch_size", "sample_refiner", "sample_refiner_residual_anchor",
              "sample_train_encoder", "extra_planes")
    assert all(normalized[field] == final_raw[field] for field in shared), "Final-loss pilot changed fit settings."
    assert final_raw["seeds"] == list(SEEDS) and final_raw["variant_modes"] == [FINAL_RAW_VARIANT]
    assert final_raw["sample_acquisition_distortion"] == "mse"  # the explicit variant selects normalized MAE.
    assert final_raw["sample_final_distortion"] == "raw_mae"
    assert normalized.get("sample_final_distortion", "mse") == "mse"
    assert cold["seeds"][:3] == normalized["seeds"][:3] == list(SEEDS)

    groups = {
        "O": [read_case(COLDFIT, "O", seed) for seed in SEEDS],
        "S_e1": [read_case(COLDFIT, "S_e1", seed) for seed in SEEDS],
        "S_received_MAE": [read_case(NORMALIZED, "current_e1_mae_received_prefix_bins", seed) for seed in SEEDS],
        "S_received_final_raw_MAE": [read_case(PILOT, FINAL_RAW_VARIANT, seed) for seed in SEEDS],
    }
    for method, runs in groups.items():
        assert [run["fit_seed"] for run in runs] == list(SEEDS), f"Incomplete seed coverage for {method}."
    for normalized_run, candidate in zip(groups["S_received_MAE"], groups["S_received_final_raw_MAE"], strict=True):
        assert normalized_run["acquisition_state_hash"] == candidate["acquisition_state_hash"], (
            f"Acquisition weights changed at seed {candidate['fit_seed']}")
        assert normalized_run["refiner_state_hash"] == candidate["refiner_state_hash"]
        control_packet, control_wrapper = sample_packet(normalized_run)
        final_packet, final_wrapper = sample_packet(candidate)
        assert len(control_wrapper) == len(final_wrapper), (
            f"Acquisition packet wire length changed at seed {candidate['fit_seed']}")
        for field in ("vector_size", "sample_size", "seed", "format"):
            assert getattr(control_packet, field) == getattr(final_packet, field), (
                f"Acquisition packet {field} changed at seed {candidate['fit_seed']}")
        assert control_packet.symbols == final_packet.symbols, (
            f"Acquired symbol bytes changed at seed {candidate['fit_seed']}")
        assert torch.equal(control_packet.indices(), final_packet.indices()), (
            f"Regenerated sample coordinates changed at seed {candidate['fit_seed']}")
        assert torch.equal(control_packet.metadata.norm_factors, final_packet.metadata.norm_factors), (
            f"Sample normalization factors changed at seed {candidate['fit_seed']}")
        control_outliers, final_outliers = control_packet.metadata.outliers, final_packet.metadata.outliers
        assert len(control_outliers.positions) == len(final_outliers.positions) and all(
            np.array_equal(left, right) for left, right in zip(control_outliers.positions, final_outliers.positions, strict=True)
        ), f"Sample outlier positions changed at seed {candidate['fit_seed']}"
        assert (control_outliers.scale is None and final_outliers.scale is None) or np.array_equal(
            control_outliers.scale, final_outliers.scale), f"Sample outlier scale changed at seed {candidate['fit_seed']}"
        assert np.array_equal(control_outliers.signs, final_outliers.signs), (
            f"Sample outlier signs changed at seed {candidate['fit_seed']}")
        assert candidate.get("sample_payload_hash") == content_hash(final_wrapper)
        control_wrapper = decompress_data_list(
            normalized_run["_case_path"].with_name("payload.bin").read_bytes())["sample_payload"]
        assert len(control_wrapper) == len(final_wrapper), f"Sample packet size changed at seed {candidate['fit_seed']}"
        for field in ("sample_coordinate_seed", "sample_indices_hash", "acquisition_state_hash", "refiner_state_hash",
                      "refiner_training_target_hash"):
            assert normalized_run[field] == candidate[field], f"Sample coordinates changed at seed {candidate['fit_seed']}"
        for field in ("sample_count", "sample_wmape", "sample_prior_rate", "sample_coder_rate",
                      "sample_coder_overhead_rate", "sample_metadata_size", "sample_stage_wmape",
                      "sample_stage_prior_rate", "sample_stage_occupied_bins", "sample_stage_marginal_entropy_bps",
                      "sample_num_planes", "sample_extra_prior_rate", "sample_extra_coder_rate",
                      "sample_extra_prefix_conditional_entropy_bps", "sample_extra_final_prefix_shuffle_wmape",
                      "sample_extra_final_prior_mode_wmape", "sample_extra_permuted_wmape", "sample_mape",
                      "sample_fit_epochs", "sample_fit_lr", "sample_fit_tau", "sample_training_batch_size",
                      "sample_training_planes", "sample_training_encoder", "sample_training_prior",
                      "sample_training_backbones", "sample_training_hard_inputs", "sample_noise_std"):
            left, right = normalized_run["record"].get(field), candidate["record"].get(field)
            assert left == right, f"Received acquisition metric {field} changed at seed {candidate['fit_seed']}"
        assert candidate["record"]["sample_count"] == 3_600_000
        assert candidate["record"]["sample_num_planes"] == normalized_run["record"]["sample_num_planes"] == 4
        assert candidate["record"]["sample_fit_mode"] == "received_prefix_bins"
        assert candidate["record"]["sample_acquisition_distortion"] == "mae"
        assert candidate["record"]["sample_final_distortion"] == "raw_mae"
        assert candidate["refiner_fit_distortion"] == normalized_run["sample_acquisition_distortion"] == "mae"
        assert candidate["sample_final_distortion"] == "raw_mae"
        assert candidate["record"]["sample_final_distortion_weight_source"] == (
            "received_norm_factors_and_decoded_teacher_raw_mean_abs")
        assert normalized_run["record"].get("sample_final_distortion") in (None, "mse"), (
            "The historical normalized-MAE control is not the default MSE-final path.")
        assert candidate["refiner_training_target_hash"] == normalized_run["refiner_training_target_hash"]
        assert candidate["refiner_training_target_hash"] != manifests[1]["reference_manifest"]["target_hash"], (
            "Acquisition training source matches the held-out current target.")
        assert candidate["record"]["sample_training_encoder"] is False
        assert encoder_hash(normalized_run) == encoder_hash(candidate), (
            f"Full-S encoder state changed at seed {candidate['fit_seed']}")
    return groups


def stage_table(points: dict[str, dict[str, Any]]) -> list[str]:
    """Render cumulative stage rates and mean raw WMAPE before dB conversion."""
    fields = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
              "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    headings = ("Joint/pre-fit prior", "Temporal prior", "Full-SI prior", "Native rANS", "Temporal rANS", "Full-SI rANS")
    lines = ["| Method | Stage | Marginal | " + " | ".join(headings) + " | WMAPE dB (raw %) |",
             "|---|---:|---:|" + "---:|" * len(headings) + "---|"]
    for method, point in points.items():
        for index in range(3):
            rates = [point[field][index] if point[field] is not None else None for field in fields]
            values = [point["marginal_rate"] if index == 2 else None, *rates]
            cells = ["—" if value is None else f"{value:.6f}" for value in values]
            lines.append(f"| {method} | {index + 1} | " + " | ".join(cells) +
                         f" | {point['wmape_db'][index]:.6f} ({point['stage_wmapes'][index]:.6f}%) |")
    return lines


def sample_table(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Show sample costs as diagnostics; they already remain charged in whole-vector rates."""
    methods = ("S_e1", "S_received_MAE", "S_received_final_raw_MAE")
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
    """Report independently calculated, same-seed directed gaps for each comparator."""
    candidate_method = "S_received_final_raw_MAE"
    lines = ["| Seed | Comparator | Final raw-MAE vs comparator (%) | Comparator vs final raw-MAE (%) |",
             "|---:|---|---:|---:|"]
    for index, seed in enumerate(SEEDS):
        candidate = mean_points([groups[candidate_method][index]["record"]])
        for method in ("O", "S_e1", "S_received_MAE"):
            comparator = mean_points([groups[method][index]["record"]])
            forward = curve_gap(candidate, comparator, "server_si_rans_stage_rates")
            reverse = curve_gap(comparator, candidate, "server_si_rans_stage_rates")
            display = lambda value: f"{value:+.3f}" if np.isfinite(value) else "—"
            lines.append(f"| {seed} | {method} | {display(forward)} | {display(reverse)} |")
    return lines


def acquisition_parity_table(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Show per-seed acquired state, coordinates, exact symbol bytes, and stream cost."""
    lines = ["| Seed | Acquisition/refiner state hash | Index hash | Symbol-payload hash | Encoder hash | Wrapper bytes | Sample rANS b/float |",
             "|---:|---|---|---|---|---:|---:|"]
    for candidate in groups["S_received_final_raw_MAE"]:
        packet, wrapper = sample_packet(candidate)
        lines.append(f"| {candidate['fit_seed']} | `{candidate['acquisition_state_hash']}` | "
                     f"`{candidate['sample_indices_hash']}` | `{content_hash(packet.symbols)}` | "
                     f"`{encoder_hash(candidate)}` | "
                     f"{len(wrapper)} | {candidate['record']['sample_coder_rate']:.9f} |")
    return lines


def build_report(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Build the untrimmed matched report, with normalized-MAE final fitting as the causal control."""
    points = {method: mean_points([run["record"] for run in runs]) for method, runs in groups.items()}
    nonmonotone = [(method, run["fit_seed"]) for method, runs in groups.items() for run in runs
                   if np.any(np.diff(run["record"]["stage_wmapes"]) >= 0)]
    symbol_errors = [(method, run["fit_seed"]) for method, runs in groups.items() for run in runs
                     if run["record"]["symbol_error_rate"] != 0]
    return "\n".join([
        "# Received-prefix final raw-MAE pilot", "",
        "This is an untrimmed, predeclared seed 43–45 comparison. The causal objective-only comparison is final-S raw-MAE versus final-S MSE under the existing normalized-MAE acquisition fit. Acquisition and final-fit schedule, 3.6M sample size, received-prefix fitting, frozen final-S encoder, source target, starting weights, decoder SI, sample coordinates, acquired symbol bytes and metadata are paired exactly. The only intended difference is the final-S distortion objective.",
        "",
        "The selected sample is decoded before fitting the replacement S model. Final raw-MAE uses the exact per-slice scales from the received packet's normalization metadata and denominator `max(mean(abs(decoded_teacher_raw_sample)), float32.tiny)`. This uses only the acquired reconstruction and packet metadata, not current-target labels. The previous/acquisition fit remains MSE at its final-distortion boundary. The full-vector/full-SI rANS curve charges all acquired symbols, overhead, and metadata; sample WMAPE is diagnostic only.",
        "",
        "O and old S_e1/MSE are read-only end-to-end context, not causal controls: old S_e1 uses a trainable encoder and reencode-teacher final fitting. The normalized-MAE control is read-only from the matched received-prefix study. No baseline or reference fits were rerun.",
        "",
        "Cohort points average cumulative rates and raw WMAPE over all three fixed seeds before dB conversion and equal-distortion comparison. Each matrix direction is recomputed; negative favors the row. No seed trimming or client exclusions are applied. Nonmonotone curves or pairs without shared distortion support are shown as unavailable. The gzip/pickle wrappers differ because their nested PyTorch tensor pickle stores a storage-key string that changes with the allocated storage (and gzip headers record the serialization time); decoded symbols, coordinates, normalization factors, outlier metadata, wrapper lengths, and charged rates are checked directly.",
        "", "## Full-vector stage RD means", "", *stage_table(points),
        "", "## Full-SI rANS directed matrix (%)", "", "Row versus column; negative favors row.", "", "```text",
        matrix_text(points, "server_si_rans_stage_rates"), "```",
        "", "## Paired seed comparisons", "",
        "Each gap uses that pair's shared WMAPE-dB interval; it is a seed-level diagnostic, not an aggregate estimate.",
        "", *paired_table(groups),
        "", "## Acquisition packet and state parity", "",
        "Decoded acquisition symbols and metadata match the normalized-MAE control exactly. The serialized wrapper is not byte-identical, but has equal length and equal charged rates.",
        "", *acquisition_parity_table(groups),
        "", "## Sample diagnostics (FYI)", "", *sample_table(groups),
        "", f"Nonmonotone seed curves: {nonmonotone or 'none'}.",
        f"Symbol-error records: {symbol_errors or 'none'}.", "",
    ])


def main() -> None:
    """Write the report after all three final raw-MAE cases complete."""
    groups = load_groups()
    destination = PILOT / "final_analysis" / "matched_rd_comparison.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(groups))
    print(destination)


if __name__ == "__main__":
    main()
