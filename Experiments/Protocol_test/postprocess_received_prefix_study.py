"""Merge received-prefix pilot and five-seed confirmation records with named read-only controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.sampled_plane_sweep import summarize
from Experiments.Protocol_test.starting_model_tables import matrix_text
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
PILOT = EXPERIMENTS / "s_received_bins_pilot_20261012"
SOURCE_PILOT = EXPERIMENTS / "s_acquisition_loss_residual_anchor_pilot_20261012"
COLDFIT = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
CONFIRMATION = EXPERIMENTS / "s_received_mae_five_seed_20261012"


def read_case(root: Path, case: str) -> dict[str, Any]:
    """Read one completed case and fail on missing completion or symbol errors."""
    path = root / "cases" / case / "target" / "complete.json"
    result = json.loads(path.read_text())
    assert result["record"]["symbol_error_rate"] == 0, f"Symbol errors in {path}"
    return result


def check_reference_manifests(*roots: Path) -> None:
    """Require identical retained target, starting weights, side information, and runtime provenance."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in roots]
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(manifest[field] == manifests[0][field] for manifest in manifests), f"Mismatched {field}."


def pilot_report() -> Path:
    """Write the seed-43 full-curve comparison against explicit acquisition-pilot references."""
    check_reference_manifests(PILOT, SOURCE_PILOT)
    names = {
        "O": (SOURCE_PILOT, "O_seed43"),
        "MSE_ref": (SOURCE_PILOT, "current_e1_mse_seed43"),
        "MAE_ref": (SOURCE_PILOT, "current_e1_mae_seed43"),
        "MSE_received": (PILOT, "current_e1_mse_received_prefix_bins_seed43"),
        "MAE_received": (PILOT, "current_e1_mae_received_prefix_bins_seed43"),
        "MSE_reencode_frozen": (PILOT, "current_e1_mse_reencode_teacher_frozen_control_seed43"),
    }
    runs = {label: read_case(root, case) for label, (root, case) in names.items()}
    for label, reference in (("MSE_received", "MSE_ref"), ("MAE_received", "MAE_ref"),
                             ("MSE_reencode_frozen", "MSE_ref")):
        result, prior = runs[label], runs[reference]
        for field in ("initial_state_hash", "acquisition_state_hash", "sample_coordinate_seed", "sample_indices_hash"):
            assert result.get(field) == prior.get(field), f"{label} and {reference} differ at {field}."
        for field in ("sample_count", "sample_wmape", "sample_prior_rate", "sample_coder_rate"):
            assert result["record"].get(field) == prior["record"].get(field), f"{label} and {reference} differ at {field}."

    points = {label: mean_points([run["record"]]) for label, run in runs.items()}
    labels = tuple(names)
    spans = []
    for index, row in enumerate(labels):
        for column in labels[index + 1:]:
            row_db = np.array(points[row]["wmape_db"])
            column_db = np.array(points[column]["wmape_db"])
            low, high = max(row_db.min(), column_db.min()), min(row_db.max(), column_db.max())
            spans.append(f"{row}/{column}: [{low:.6f}, {high:.6f}] dB")

    rows = ["# Received-prefix pilot: full-vector full-SI comparison", "",
            "Primary rate is cumulative full-vector full-SI rANS, including the complete acquired-sample payload and metadata. “MAE_ref” is retained for context but its stage WMAPE is nonmonotone, so every primary matrix comparison involving it is unrankable. Sample WMAPE is FYI.", "",
            "References are read-only records from `s_acquisition_loss_residual_anchor_pilot_20261012`; no fit state or sample packet was reused. The pilot and reference manifests match in target, SI, baseline, source, and numerical runtime. Exact acquisition and coordinate/index parity checks are recorded in the completion JSONs.", "",
            "| Method | WMAPE % by stage | WMAPE dB by stage | Full-SI rANS increment | Full-SI rANS cumulative | SI-prior cumulative | Joint-prior cumulative | Marginal b/float | Sample WMAPE % | Sample prior/rANS b/float | Sample overhead b/float | Metadata MiB |", "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for label in labels:
        result, record, point = runs[label], runs[label]["record"], points[label]
        vec = lambda values: "/".join(f"{float(value):.6f}" for value in values)
        prior = record.get("server_si_prior_stage_rates")
        joint = record.get("quantizer_prior_stage_rates")
        sample = "—" if record.get("sample_wmape") is None else f"{record['sample_wmape']:.4f}"
        sample_rates = "—" if record.get("sample_prior_rate") is None else f"{record['sample_prior_rate']:.6f}/{record['sample_coder_rate']:.6f}"
        overhead = "—" if record.get("sample_coder_overhead_rate") is None else f"{record['sample_coder_overhead_rate']:.6f}"
        metadata = "—" if record.get("sample_metadata_size") is None else f"{record['sample_metadata_size']:.6f}"
        rows.append(f"| {label} | {vec(record['stage_wmapes'])} | {vec(point['wmape_db'])} | {vec(record['server_si_rans_stage_rates'])} | {vec(point['server_si_rans_stage_rates'])} | {vec(np.cumsum(prior)) if prior else '—'} | {vec(np.cumsum(joint)) if joint else '—'} | {record.get('marginal_rate', float('nan')):.6f} | {sample} | {sample_rates} | {overhead} | {metadata} |")
    rows += ["", "Full directed equal-distortion gap matrix, row versus column (negative favors row; nonmonotone primary curves remain `—`):", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
             "Pairwise shared WMAPE-dB spans:", "", *[f"- {span}" for span in spans], "",
             "All valid monotone curves share [-8.520415, -5.225069] dB. On that identical interval, the fresh O, prior current-e1 MSE, new received-bin MSE, new received-bin MAE, and frozen re-encode MSE gaps versus O are +0.000%, +11.350%, +11.301%, +8.947%, and +11.350%, respectively. The new received-bin MAE is -2.159% versus current-e1 MSE on that same interval. MAE_ref is excluded because its WMAPE increases at stage 2.", "",
             "The new MSE received-bin and frozen re-encode MSE acquisition hashes both equal the prior current-e1 MSE hash; their coordinate/index hashes and sample costs match exactly. New MAE received-bin acquisition and sample hashes/costs match the prior current-e1 MAE reference. Thus the final-fit mode is the controlled difference. The frozen re-encode control uses bins regenerated from decoded teacher values and may not match deployment bins; evaluation uses unchanged deployed symbols and charges the same acquired payload.", ""]
    destination = PILOT / "final_analysis" / "merged_reference_comparison.md"
    destination.write_text("\n".join(rows))
    return destination


def confirmation_report() -> Path:
    """Merge the fresh five-seed candidate with matched saved O/S_e1 controls and rank best-three cohorts."""
    check_reference_manifests(COLDFIT, CONFIRMATION)
    groups = {
        "O": [read_case(COLDFIT, f"O_seed{seed}") for seed in (43, 44, 45, 46, 47)],
        "S_e1": [read_case(COLDFIT, f"S_e1_seed{seed}") for seed in (43, 44, 45, 46, 47)],
        "S_received_MAE": [read_case(CONFIRMATION, f"current_e1_mae_received_prefix_bins_seed{seed}")
                           for seed in (43, 44, 45, 46, 47)],
    }
    assert groups["O"][0]["record"]["coder_stage_rates"] == read_case(SOURCE_PILOT, "O_seed43")["record"]["coder_stage_rates"]
    assert groups["S_e1"][0]["record"]["coder_stage_rates"] == read_case(SOURCE_PILOT, "current_e1_mse_seed43")["record"]["coder_stage_rates"]
    for method, runs in groups.items():
        assert [run["fit_seed"] for run in runs] == [43, 44, 45, 46, 47], f"Incomplete {method} seed coverage."
    for candidate in groups["S_received_MAE"]:
        reference = read_case(SOURCE_PILOT, "current_e1_mae_seed43") if candidate["fit_seed"] == 43 else None
        assert candidate["record"]["sample_fit_mode"] == "received_prefix_bins"
        assert candidate["record"]["sample_training_encoder"] is False
        assert candidate["record"]["sample_count"] == 3_600_000
        if reference is not None:
            assert candidate["acquisition_state_hash"] == reference["acquisition_state_hash"]
            assert candidate["sample_indices_hash"] == reference["sample_indices_hash"]

    result = summarize(groups, retain=3)
    cohorts = result["cohorts"]
    report = ["# Received-prefix MAE five-seed confirmation", "",
              "The candidate was freshly fitted for seeds 43–47. It uses normalized-space MAE for acquisition and trains final S from the received ordinary prefix bins with the encoder frozen; normalized MAE is a WMAPE-aligned surrogate, not the exact raw WMAPE objective. Evaluation uses ordinary deployed full-vector coding, and cumulative full-SI rANS charges all sample payload and metadata costs. Sample WMAPE is FYI. O and current-e1 MSE are explicit read-only controls from `s_dedicated_refiner_coldfit_five_seed_20261011`; seed43 O/S_e1 curves match the prior pilot controls exactly. No control fit or sample packet was reused.", "",
              "Best-three selection is separate per method, scored against the all-five O mean full-SI rANS curve on each method's shared monotone WMAPE-dB span, with no extrapolation. Cohort means average cumulative rates and raw WMAPE before dB conversion. Unrankable curves remain excluded from primary comparisons.", "",
              f"Ranking spans: `{json.dumps(result['ranking_distortion_db'], sort_keys=True)}`", "",
              f"Seed selection: `{json.dumps(result['selection'], sort_keys=True)}`", ""]
    for cohort_name, points in cohorts.items():
        report += [f"## {cohort_name} directed matrix", "", "```text", matrix_text(points, "server_si_rans_stage_rates"), "```", "",
                   "| Method | WMAPE % mean by stage | Full-SI rANS cumulative b/float | SI-prior cumulative b/float | Joint-prior cumulative b/float | Marginal b/float |", "|---|---|---|---|---|---:|"]
        for method, point in points.items():
            def display(key: str) -> str:
                values = point[key]
                return "—" if values is None else "/".join(f"{float(value):.6f}" for value in values)
            report.append(f"| {method} | {display('stage_wmapes')} | {display('server_si_rans_stage_rates')} | {display('server_si_prior_stage_rates')} | {display('quantizer_prior_stage_rates')} | {point['marginal_rate']:.6f} |")
        report.append("")
    report += ["## Candidate sample diagnostics", "", "| Seed | Sample WMAPE % | Sample prior/rANS b/float | Sample overhead b/float | Metadata MiB | Full-SI rANS cumulative b/float | WMAPE % by stage |", "|---:|---:|---:|---:|---:|---|---|"]
    for run in groups["S_received_MAE"]:
        record = run["record"]
        report.append(f"| {run['fit_seed']} | {record['sample_wmape']:.6f} | {record['sample_prior_rate']:.6f}/{record['sample_coder_rate']:.6f} | {record['sample_coder_overhead_rate']:.6f} | {record['sample_metadata_size']:.6f} | {'/'.join(f'{x:.6f}' for x in np.cumsum(record['server_si_rans_stage_rates']))} | {'/'.join(f'{x:.6f}' for x in record['stage_wmapes'])} |")
    selected_seeds = {item["seed"] for item in result["selection"]
                      if item["method"] == "S_received_MAE" and item["retained"]}
    for cohort, values in (("all five", groups["S_received_MAE"]),
                           ("retained best three", [run for run in groups["S_received_MAE"]
                                                     if run["fit_seed"] in selected_seeds])):
        report.append(f"\n{cohort} sample means: WMAPE {np.mean([run['record']['sample_wmape'] for run in values]):.6f}%; prior/rANS {np.mean([run['record']['sample_prior_rate'] for run in values]):.6f}/{np.mean([run['record']['sample_coder_rate'] for run in values]):.6f} b/float; overhead {np.mean([run['record']['sample_coder_overhead_rate'] for run in values]):.6f} b/float; metadata {np.mean([run['record']['sample_metadata_size'] for run in values]):.6f} MiB.")
    report.append("")
    destination = CONFIRMATION / "final_analysis" / "merged_confirmation.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(report))
    return destination


def main() -> None:
    """Run the selected, explicitly scoped postprocessor after its cases complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("pilot", "confirmation"))
    kind = parser.parse_args().kind
    print(pilot_report() if kind == "pilot" else confirmation_report())


if __name__ == "__main__":
    main()
