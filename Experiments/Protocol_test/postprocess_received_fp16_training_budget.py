"""Build a read-only RD and wire-cost report for the received-FP16 budget pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text


def _load(path: Path) -> dict[str, Any]:
    """Load a completed protocol case and identify its source on failure."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read completed case {path}: {error}") from error


def publish(output_root: Path, destination: Path) -> None:
    """Compare completed sample-only fits with their read-only O case and publish wire costs."""
    manifest = _load(output_root / "test_config.json")
    plan = manifest["plan"]
    comparison_root = Path(plan["comparison_root"])
    comparison_manifest = _load(comparison_root / "test_config.json")
    assert comparison_manifest["reference_manifest"]["target_hash"] == manifest["reference_manifest"]["target_hash"]
    assert comparison_manifest["baseline_state_hash"] == manifest["baseline_state_hash"]
    seed = plan["seeds"][0]
    oracle_path = comparison_root / "cases" / f"O_seed{seed}" / "target" / "complete.json"
    oracle = _load(oracle_path)
    assert oracle["method"] == "O" and oracle["fit_seed"] == seed
    assert oracle["identity"] == comparison_manifest["identity"]
    runs = {}
    mode = plan["variant_modes"][0]
    for size in plan["budget_sample_sizes"]:
        name = f"{mode}_n{size}_seed{seed}"
        path = output_root / "cases" / name / "target" / "complete.json"
        run = _load(path)
        assert run["identity"] == manifest["identity"] and run["record"]["symbol_error_rate"] == 0
        record = run["record"]
        validated_split = run["method"].startswith("received_fp16_training_message_validated_split")
        if validated_split:
            assert record["sample_validation_train_rows"] + record["sample_validation_rows"] == size
            assert record["sample_fit_training_pool_rows"] == record["sample_validation_train_rows"]
            assert record["sample_prior_refit_rows"] == record["sample_validation_train_rows"]
            assert record["sample_validation_seed"] == plan["sample_validation_seed"]
            assert record["sample_validation_split_hash"]
        else:
            assert record["sample_fit_training_pool_rows"] == size
            assert record["sample_prior_refit_rows"] == size
        assert record["sample_full_target_prior_refit_rows"] == 0
        assert record["sample_prior_refit_target"] == "received_fp16_training_message_only"
        assert run["record"]["sample_prior_state_hash_before_full_eval"] == run["record"]["sample_prior_state_hash_after_full_eval"]
        assert run["record"]["sample_training_message_bytes"] > 0
        runs[size] = run
    validated_split = all(
        run["method"].startswith("received_fp16_training_message_validated_split") for run in runs.values())

    points = {"O": mean_points([oracle["record"]]), **{
        f"S-{size:,} received FP16": mean_points([run["record"]]) for size, run in runs.items()}}
    ordinary_metadata_bpf = {
        label: record["meta_data_size"] * (1 << 20) * 8 / record["model_size"]
        for label, record in [("O", oracle["record"]),
                              *((f"S-{size:,} received FP16", run["record"]) for size, run in runs.items())]
    }
    metadata_points = {}
    for label, point in points.items():
        adjusted = dict(point)
        adjusted["server_si_rans_stage_rates"] = list(point["server_si_rans_stage_rates"])
        adjusted["server_si_rans_stage_rates"][0] += ordinary_metadata_bpf[label]
        metadata_points[label] = adjusted

    destination.mkdir(parents=True, exist_ok=True)
    fields = ("quantizer_prior_stage_rates", "server_si_prior_stage_rates", "server_si_rans_stage_rates")
    lines = [
        "# Received FP16 training-message budget pilot", "",
        ("One fixed SVHN client-0 round-5 target, trace seed 43, fit seed 43. Both S models start from the same projected three-plane baseline and use O's warm 35-epoch optimizer settings. A validated-split candidate trains on 80% of received rows and caps draws at that training partition, so its update count per accepted quantizer candidate is 105 updates at 20k rows and 455 at 100k; it does not match O's 3,080-update budget." if validated_split else
         "One fixed SVHN client-0 round-5 target, trace seed 43, fit seed 43. Both S models start from the same projected three-plane baseline and use O's warm 35-epoch optimizer settings. Each fit repeats the received sample pool to O's 550k draws per epoch (88 batches, 3,080 updates per accepted quantizer candidate).") + " This is a two-case exploratory pilot, not a best-three result.", "",
        "O is imported read-only from the matched cold-fit study; its target and starting-state hashes match. O's full-target quantizer and prior are the intended oracle reference. S fits its quantizer and prior from received FP16 sample values/SI only; the prior is frozen during full-target evaluation. Each S transmits symbols for all N coordinates, with no sample-position restoration.", "",
        "Primary full-SI rANS curves include the complete compressed sample packet exactly once in stage 1. The sample packet includes transmitted sample coordinates and FP16 labels. Ordinary codec metadata is shown only as a separate proxy adjustment; it is not added to the sample packet cost a second time. The metadata proxy adds the measured ordinary metadata size at stage 1 for both O and S.", "",
        "Encoder feedback is reported separately as the encoder-only downlink. Decoder and prior weights remain at the server and are excluded from that size. The full-SI prior column for S was trained on sample-only bins/SI; O's corresponding prior uses full-target access.", "",
        "## Stage RD measurements", "",
        "Cumulative bits/float; — marks unmeasured values. Distortion averages are raw WMAPE before dB conversion. Marginal rates are final totals only.", "",
        "| Method | Stage | Marginal | Joint / pre-fit prior | Sample-trained full-SI prior | Full-SI rANS primary | WMAPE dB (raw %) |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, point in points.items():
        for stage in range(3):
            vals = [point["marginal_rate"] if stage == 2 else None,
                    *(point[field][stage] if point.get(field) is not None else None for field in fields)]
            lines.append(f"| {label} | {stage + 1} | " + " | ".join("—" if value is None else f"{value:.7f}" for value in vals)
                         + f" | {point['wmape_db'][stage]:.6f} ({point['stage_wmapes'][stage]:.6f}%) |")
    for label, point in metadata_points.items():
        point["ordinary_metadata_proxy_bpf"] = ordinary_metadata_bpf[label]
    (destination / "points.json").write_text(json.dumps({"primary": points, "ordinary_metadata_proxy": metadata_points}, indent=2))
    lines += ["", "## Full-SI rANS equal-distortion rate changes", "",
              "Complete directed matrices; row versus column, negative favors the row. Each reverse entry is independently recomputed over the shared distortion span.", "",
              "Exact sample-packet costs are present in the primary S curves; this comparison adds ordinary metadata proxy costs to stage 1.", "",
              "Primary:", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
              "With ordinary metadata proxy:", "", "```text", matrix_text(metadata_points, "server_si_rans_stage_rates"), "```", "",
              "## Packet and model distribution", "",
              "| Method | Sample rows | Sample packet bytes | Packet bpf, charged once | Ordinary metadata MiB | Ordinary metadata proxy bpf | Encoder-only downlink MiB | Decoder MiB (server-side) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = [("O", oracle["record"], None), *((f"S-{size:,} received FP16", run["record"], size) for size, run in runs.items())]
    for label, record, size in rows:
        lines.append(f"| {label} | {'—' if size is None else f'{size:,}'} | "
                     f"{'—' if size is None else f'{record['sample_training_message_bytes']:,}'} | "
                     f"{'—' if size is None else f'{record['sample_training_message_rate']:.7f}'} | "
                     f"{record['meta_data_size']:.6f} | {ordinary_metadata_bpf[label]:.7f} | "
                     f"{record['encoder_size']:.6f} | {record['decoder_size']:.6f} |")
    lines += ["", "## Update-budget, validation selection, and sample-only prior checks", "",
              "Validation partitions are deterministic per normalization slice and shared between main quantizer and prior selection. Training and prior fits see only the 80% train partition; all current N coordinates are still coded. Main validation score is hard-symbol normalized MSE RD plus conditional CE in nats; prior validation score is hard conditional CE in nats. `updates / candidate` is the full epoch cap; selected best epochs may stop before it. Validation-attempt update totals count every scored attempt that ran.", "",
              "| Sample rows | Train rows | Validation rows | Split hash | Main draws / epoch | Main batches / epoch | Main updates / candidate | Main accepted candidates | Main best epoch / score | Main attempts / scored updates | Prior draws / epoch | Prior updates / candidate | Prior candidates | Prior best epoch / CE nats | Prior attempts / scored updates | Full-target prior refit rows |", "|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|"]
    for size, run in runs.items():
        record = run["record"]
        def show(value: Any, digits: int = 6) -> str:
            return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)

        lines.append(f"| {size:,} | {show(record.get('sample_validation_train_rows'))} | "
                     f"{show(record.get('sample_validation_rows'))} | {record.get('sample_validation_split_hash', '—')} | "
                     f"{record['sample_fit_draws_per_epoch']:,} | {record['sample_fit_batches_per_epoch']} | "
                     f"{record['sample_fit_updates_per_accepted_candidate']:,} | {record['sample_fit_required_accepted_candidates']} | "
                     f"{show(record.get('sample_fit_validation_best_epoch'))} / {show(record.get('sample_fit_validation_best_score'))} | "
                     f"{show(record.get('sample_fit_validation_attempts'))} / {show(record.get('sample_fit_validation_updates'))} | "
                     f"{record['sample_prior_draws_per_epoch']:,} | {record['sample_prior_updates_per_candidate']:,} | "
                     f"{record['sample_prior_required_candidates']} | "
                     f"{show(record.get('sample_prior_validation_best_epoch'))} / {show(record.get('sample_prior_validation_best_score'))} | "
                     f"{show(record.get('sample_prior_validation_attempts'))} / {show(record.get('sample_prior_validation_updates'))} | "
                     f"{record['sample_full_target_prior_refit_rows']} |")
    lines += ["", "## Diagnostics", ""]
    for label, run in [("O", oracle), *((f"S-{size:,} received FP16", run) for size, run in runs.items())]:
        record = run["record"]
        flags = []
        if np.any(np.diff(record["stage_wmapes"]) >= 0):
            flags.append("nonmonotone distortion")
        if record["quantizer_prior_rate"] > record["marginal_rate"]:
            flags.append("joint prior exceeds marginal")
        flags.extend(f"stage {stage} native stream growth {rate:.6f} bpf"
                     for stage, rate in enumerate(record["coder_stage_rates"], 1)
                     if rate < 0.01 or rate > 1.05)
        lines.append(f"- {label}: {', '.join(flags) or 'monotone distortion; no flagged prior/rate growth'}; symbol error rate {record['symbol_error_rate']:.1%}.")
    lines += ["", "No frozen evaluation window is included. The old S study's full-target prior-refit rate is not used as a sample-only prior result. Any differences from this pilot also reflect the stricter prior-information restriction and should not be attributed solely to sample size.", ""]
    (destination / "report.md").write_text("\n".join(lines))
    print(f"Published read-only analysis: {destination}")


def main() -> None:
    """Run the CPU-only report builder on a completed pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    publish(args.output_root, args.out)


if __name__ == "__main__":
    main()
