"""Publish matched RD and provenance comparisons for the validated hard-input pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text


def _load(path: Path) -> dict[str, Any]:
    """Load a completed case and name the unreadable artifact on failure."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {path}: {error}") from error


def publish(output_root: Path, soft_root: Path, oracle_root: Path, best_root: Path,
            destination: Path, categorical_root: Path | None = None,
            hard_transport_root: Path | None = None) -> None:
    """Compare validated FP16 pilots with matched oracle and current-best controls."""
    manifest = _load(output_root / "test_config.json")
    plan = manifest["plan"]
    soft_manifest = _load(soft_root / "test_config.json")
    categorical_manifest = _load(categorical_root / "test_config.json") if categorical_root else None
    hard_transport_manifest = _load(hard_transport_root / "test_config.json") if hard_transport_root else None
    oracle_manifest = _load(oracle_root / "test_config.json")
    best_manifest = _load(best_root / "test_config.json")
    for other in (soft_manifest, categorical_manifest, hard_transport_manifest, oracle_manifest, best_manifest):
        if other is None:
            continue
        assert other["reference_manifest"]["target_hash"] == manifest["reference_manifest"]["target_hash"]
        assert other["baseline_state_hash"] == manifest["baseline_state_hash"]
    seed = plan["seeds"][0]
    cases = {
        "O (read-only oracle)": _load(oracle_root / "cases" / f"O_seed{seed}" / "target" / "complete.json"),
        "S current best (MAE prefix)": _load(best_root / "cases" / f"current_e1_mae_received_prefix_bins_seed{seed}" / "target" / "complete.json"),
    }
    for size in plan["budget_sample_sizes"]:
        hard_method = f"received_fp16_training_message_validated_split_hard_inputs_n{size}"
        soft_method = f"received_fp16_training_message_validated_split_n{size}"
        hard = _load(output_root / "cases" / f"{hard_method}_seed{seed}" / "target" / "complete.json")
        soft = _load(soft_root / "cases" / f"{soft_method}_seed{seed}" / "target" / "complete.json")
        categorical = None
        if categorical_root:
            categorical_method = f"received_fp16_training_message_validated_split_hard_categorical_prior_n{size}"
            categorical = _load(categorical_root / "cases" / f"{categorical_method}_seed{seed}" / "target" / "complete.json")
        hard_transport = None
        if hard_transport_root:
            transport_method = f"received_fp16_training_message_validated_split_hard_transport_nll_n{size}"
            hard_transport = _load(hard_transport_root / "cases" / f"{transport_method}_seed{seed}" / "target" / "complete.json")
        for run in (hard, soft, categorical, hard_transport):
            if run is None:
                continue
            record = run["record"]
            assert run["fit_seed"] == seed and record["symbol_error_rate"] == 0
            assert record["sample_full_target_prior_refit_rows"] == 0
            assert record["sample_prior_refit_target"] == "received_fp16_training_message_only"
            assert record["sample_validation_train_rows"] + record["sample_validation_rows"] == size
            assert record["sample_validation_seed"] == plan["sample_validation_seed"]
            assert record["sample_prior_state_hash_before_full_eval"] == record["sample_prior_state_hash_after_full_eval"]
        assert hard["record"]["sample_training_hard_inputs"] is True
        assert soft["record"]["sample_training_hard_inputs"] is False
        assert hard["record"]["sample_prior_fit_context"] == "received_hard_symbols"
        assert soft["record"].get("sample_prior_fit_context") in (None, "received_hard_symbols")
        assert hard["record"]["sample_validation_split_hash"] == soft["record"]["sample_validation_split_hash"]
        cases[f"S-{size:,} soft inputs"] = soft
        cases[f"S-{size:,} hard inputs"] = hard
        if categorical is not None:
            record = categorical["record"]
            assert record["sample_training_hard_inputs"] is True
            assert record["sample_fit_prior_distribution"] == "categorical_softmax"
            assert record["sample_prior_fit_context"] == "received_hard_symbols"
            assert record["sample_validation_split_hash"] == hard["record"]["sample_validation_split_hash"]
            cases[f"S-{size:,} hard categorical p"] = categorical
        if hard_transport is not None:
            record = hard_transport["record"]
            assert record["sample_training_hard_inputs"] is True
            assert record["sample_fit_prior_distribution"] == "categorical_softmax"
            assert record["sample_fit_rate_objective"] == "hard_transport_nll_with_expected_ce_encoder_gradient"
            assert record["sample_prior_fit_context"] == "received_hard_symbols"
            assert record["sample_validation_split_hash"] == hard["record"]["sample_validation_split_hash"]
            cases[f"S-{size:,} hard transport NLL"] = hard_transport

    points = {label: mean_points([run["record"]]) for label, run in cases.items()}
    metadata_bpf = {label: run["record"]["meta_data_size"] * (1 << 20) * 8 / run["record"]["model_size"]
                    for label, run in cases.items()}
    adjusted = {}
    for label, point in points.items():
        rates = list(point["server_si_rans_stage_rates"])
        rates[0] += metadata_bpf[label]
        adjusted[label] = point | {"server_si_rans_stage_rates": rates}

    destination.mkdir(parents=True, exist_ok=True)
    rate_fields = ("quantizer_prior_stage_rates", "server_si_prior_stage_rates", "server_si_rans_stage_rates")
    lines = [
        "# Received FP16 validated-split hard-input and hard-transport pilots", "",
        "Seed-43 budgets use a common coordinate seed and deterministic stratified 80/20 received-sample split across the soft-input, hard-input, categorical-p, and hard-transport-NLL variants. Hard mode uses straight-through one-hot decoder/prior inputs in joint fitting. Categorical p uses deterministic softmax probabilities. Hard-transport NLL trains deterministic encoder softmax probabilities with the measured hard-symbol transport NLL plus a zero-valued expected-cross-entropy encoder-gradient surrogate; its distortion coefficient and categorical p remain unchanged. All N current symbols are coded, the sample packet is charged once at stage 1, and the sample-trained prior is frozen during full-target evaluation. These remain two-budget single-seed pilots.", "",
        "The read-only comparison set includes matched O, the validated soft/hard/categorical controls, and the current-best `current_e1_mae_received_prefix_bins` seed-43 run. The current-best reference uses its MAE received-prefix pipeline and is shown as a distinct benchmark.", "",
        "Primary rates are full-SI rANS. Ordinary codec metadata is shown as a separate stage-1 proxy for every method, not added to the exact sample-packet cost a second time. The current-best and O sample acquisition costs follow their original protocol records.", "",
        "## Stage RD measurements", "",
        "Cumulative bits/float; marginal rate is the final total only. Distortion is raw WMAPE percent and its dB conversion.", "",
        "| Method | Stage | Marginal | Joint / pre-fit prior | Full-SI prior | Full-SI rANS | WMAPE dB (raw %) |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, point in points.items():
        for stage in range(3):
            vals = [point["marginal_rate"] if stage == 2 else None,
                    *(point[field][stage] if point.get(field) is not None else None for field in rate_fields)]
            lines.append(f"| {label} | {stage + 1} | " + " | ".join(
                "—" if value is None else f"{value:.7f}" for value in vals)
                + f" | {point['wmape_db'][stage]:.6f} ({point['stage_wmapes'][stage]:.6f}%) |")

    lines += ["", "## Directed equal-distortion rate changes", "",
              "Full square matrices compare row against column; negative favors the row. Each direction is independently recomputed over the shared distortion interval.", "",
              "Primary full-SI rANS:", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
              "With ordinary metadata proxy:", "", "```text", matrix_text(adjusted, "server_si_rans_stage_rates"), "```", "",
              "## Received-sample provenance and fit selection", "",
              "| Rows | Input mode | Train / validation | Split hash | Main draw cap / updates per accepted candidate | Main selected epoch / score | Prior selected epoch / CE (nats) | Sample packet bytes / bpf | Full-target prior refit rows |", "|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    for size in plan["budget_sample_sizes"]:
        for kind, label in (("soft", f"S-{size:,} soft inputs"),
                            ("hard", f"S-{size:,} hard inputs"),
                            ("categorical", f"S-{size:,} hard categorical p"),
                            ("hard transport NLL", f"S-{size:,} hard transport NLL")):
            if label not in cases:
                continue
            name = label
            record = cases[name]["record"]
            lines.append(f"| {size:,} | {kind} | {record['sample_validation_train_rows']:,} / "
                         f"{record['sample_validation_rows']:,} | {record['sample_validation_split_hash']} | "
                         f"{record['sample_fit_draws_per_epoch']:,} / {record['sample_fit_updates_per_accepted_candidate']:,} | "
                         f"{record['sample_fit_validation_best_epoch']} / {record['sample_fit_validation_best_score']:.6f} | "
                         f"{record['sample_prior_validation_best_epoch']} / {record['sample_prior_validation_best_score']:.6f} | "
                         f"{record['sample_training_message_bytes']:,} / {record['sample_training_message_rate']:.7f} | "
                         f"{record['sample_full_target_prior_refit_rows']} |")
    lines += ["", "The hard-input flag is recorded as `sample_training_hard_inputs=true`. The separate prior trainer uses hard symbol contexts in all modes: new cases record `sample_prior_fit_context=received_hard_symbols`, and the soft pilot predates that provenance field while its fitter converts received bins to one-hot contexts internally. Categorical p changes the joint quantizer's in-model prior distribution to deterministic softmax probabilities; the hard-transport NLL variant keeps that p and changes the encoder-side rate objective to the actual hard-symbol NLL with the zero-valued expected-CE gradient surrogate. Coordinate/split hashes match across variants; measured packet byte counts are reported separately. Quantizer and prior fits use only train rows; validation scores are receiver-only and no prior fit uses full-target rows.", "",
              "## Model distribution", "",
              "Encoder-only size is the worker downlink; decoder and prior weights remain server-side.", "",
              "| Method | Ordinary metadata MiB | Metadata proxy bpf | Encoder-only MiB | Decoder/server MiB |", "|---|---:|---:|---:|---:|"]
    for label, run in cases.items():
        record = run["record"]
        lines.append(f"| {label} | {record['meta_data_size']:.6f} | {metadata_bpf[label]:.7f} | "
                     f"{record['encoder_size']:.6f} | {record['decoder_size']:.6f} |")
    for label, point in points.items():
        point["ordinary_metadata_proxy_bpf"] = metadata_bpf[label]
    (destination / "points.json").write_text(json.dumps({"primary": points, "ordinary_metadata_proxy": adjusted}, indent=2))
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Published read-only analysis: {destination}")


def main() -> None:
    """Build CPU-only matched RD tables for the completed hard-input pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--soft-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--best-root", type=Path, required=True)
    parser.add_argument("--categorical-root", type=Path)
    parser.add_argument("--hard-transport-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output_root, args.soft_root, args.oracle_root, args.best_root, args.out,
            args.categorical_root, args.hard_transport_root)


if __name__ == "__main__":
    main()
