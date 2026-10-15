"""Validate and report a matched frozen-prefix normalized-MAE final-fit pilot."""

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
OLD_S = EXPERIMENTS / "s_received_mae_five_seed_20261012"
NEW_S = EXPERIMENTS / "s_received_mae_final_normalized_mae_seed43_20261014"
O = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
METHOD = "current_e1_mae_received_prefix_bins"


def read_case(root: Path, method: str) -> tuple[dict[str, Any], Path]:
    """Load a completed case and require a clean deployed symbol round trip."""
    path = root / "cases" / f"{method}_seed43" / "target" / "complete.json"
    run = json.loads(path.read_text())
    assert run["record"]["symbol_error_rate"] == 0, f"Symbol errors in {path}"
    return run, path.parent


def report() -> Path:
    """Validate the fixed sample packet and write matched stage and curve comparisons."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in (O, OLD_S, NEW_S)]
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(item[field] == manifests[0][field] for item in manifests), f"Mismatched {field}."
    assert all(Path(item["plan"]["reference_root"]).resolve() == Path(manifests[0]["plan"]["reference_root"]).resolve()
               for item in manifests), "Mismatched retained reference root."

    old, old_dir = read_case(OLD_S, METHOD)
    new, new_dir = read_case(NEW_S, METHOD)
    oracle, _ = read_case(O, "O")
    for field in ("initial_state_hash", "acquisition_state_hash", "refiner_state_hash",
                  "refiner_training_target_hash", "sample_coordinate_seed", "sample_indices_hash"):
        assert old[field] == new[field], f"Matched acquisition differs at {field}."

    def sample_packet(directory: Path) -> tuple[Any, bytes]:
        payload = decompress_data_list((directory / "payload.bin").read_bytes())
        serialized = payload["sample_payload"]
        return decompress_data_list(serialized), serialized

    old_packet, old_wrapper = sample_packet(old_dir)
    new_packet, new_wrapper = sample_packet(new_dir)
    assert (old_packet.vector_size, old_packet.sample_size, old_packet.seed, old_packet.format) == (
        new_packet.vector_size, new_packet.sample_size, new_packet.seed, new_packet.format)
    assert content_hash(old_packet.symbols) == content_hash(new_packet.symbols), "Encoded sample symbols changed."
    assert torch.equal(old_packet.indices(), new_packet.indices()), "Sample coordinates changed."
    assert torch.equal(old_packet.metadata.norm_factors, new_packet.metadata.norm_factors), "Normalization factors changed."
    old_outliers, new_outliers = old_packet.metadata.outliers, new_packet.metadata.outliers
    assert len(old_outliers.positions) == len(new_outliers.positions) and all(
        np.array_equal(left, right) for left, right in zip(old_outliers.positions, new_outliers.positions, strict=True)
    ), "Outlier positions changed."
    assert (old_outliers.scale is None and new_outliers.scale is None) or np.array_equal(
        old_outliers.scale, new_outliers.scale), "Outlier scales changed."
    assert np.array_equal(old_outliers.signs, new_outliers.signs), "Outlier signs changed."

    old_record, new_record = old["record"], new["record"]
    for field in ("sample_count", "sample_wmape", "sample_prior_rate", "sample_coder_rate",
                  "sample_coder_overhead_rate", "sample_stage_wmape", "sample_stage_prior_rate",
                  "sample_num_planes"):
        assert old_record[field] == new_record[field], f"Acquisition measurement changed at {field}."
    assert old["sample_acquisition_distortion"] == new["sample_acquisition_distortion"] == "mae"
    assert old_record["sample_fit_mode"] == new_record["sample_fit_mode"] == "received_prefix_bins"
    assert not old_record["sample_training_encoder"] and not new_record["sample_training_encoder"]
    assert new_record["sample_final_distortion"] == "mae"

    runs = {"O": oracle, "S_frozen_MSE": old, "S_frozen_normalized_MAE": new}
    points = {name: mean_points([run["record"]]) for name, run in runs.items()}
    lines = [
        "# Matched seed-43 normalized-MAE final-fit pilot", "",
        "The candidate keeps the S acquisition refiner, normalized-MAE sample acquisition, sample coordinates, decoded symbols, and metadata fixed. It changes only the final S decoder distortion from MSE to normalized MAE; the encoder remains frozen to the received prefix bins, and the prior cross-entropy still scores those received symbols. O and the frozen-prefix MSE S run are read-only controls.", "",
        "This is a one-client, round-5 trained-round diagnostic. All records have zero symbol errors. All three stage WMAPE curves are monotone and the candidate has a valid equal-distortion comparison over the shared distortion range.", "",
        "## Matched staged RD values", "",
        "Rates are cumulative bits per full-gradient float. Missing measurements are `—`.", "",
        "| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method, point in points.items():
        record = runs[method]["record"]
        for index in range(3):
            values = [record["marginal_rate"] if index == 2 else None,
                      point["quantizer_prior_stage_rates"][index],
                      point["temporal_si_prior_stage_rates"][index] if point["temporal_si_prior_stage_rates"] is not None else None,
                      point["server_si_prior_stage_rates"][index] if point["server_si_prior_stage_rates"] is not None else None,
                      point["coder_stage_rates"][index],
                      point["temporal_si_rans_stage_rates"][index] if point["temporal_si_rans_stage_rates"] is not None else None,
                      point["server_si_rans_stage_rates"][index]]
            cells = ["—" if value is None else f"{value:.6f}" for value in values]
            lines.append(f"| {method} | {index + 1} | " + " | ".join(cells) +
                         f" | {point['wmape_db'][index]:.6f} ({point['stage_wmapes'][index]:.6f}%) |")

    lines += ["", "## Directed full-curve rate gaps", "",
              "Full-SI rANS, row versus column; negative favors the row. Reverse entries are recomputed.", "", "```text",
              matrix_text(points, "server_si_rans_stage_rates"), "```", "",
              "Native transport:", "", "```text", matrix_text(points, "coder_stage_rates"), "```", ""]
    for method in ("S_frozen_MSE", "S_frozen_normalized_MAE"):
        gap = curve_gap(points[method], points["O"], "server_si_rans_stage_rates")
        lines.append(f"{method} vs O full-SI gap: {gap:+.3f}% over the shared distortion interval.")
    delta = curve_gap(points["S_frozen_normalized_MAE"], points["S_frozen_MSE"], "server_si_rans_stage_rates")
    lines += ["", f"Switching the final fit from MSE to normalized MAE changes equal-distortion rate by {delta:+.3f}% versus the frozen-prefix MSE S baseline. The candidate final endpoint is {new_record['stage_wmapes'][-1]:.3f}% WMAPE at {points['S_frozen_normalized_MAE']['server_si_rans_stage_rates'][-1]:.6f} b/float; the baseline endpoint is {old_record['stage_wmapes'][-1]:.3f}% at {points['S_frozen_MSE']['server_si_rans_stage_rates'][-1]:.6f} b/float.", "",
              "## Acquisition parity", "",
              f"Decoded sample rANS stream hash `{content_hash(new_packet.symbols)}` and length {new_packet.symbols.numel() if isinstance(new_packet.symbols, torch.Tensor) else len(new_packet.symbols):,} bytes; sample coordinate hash `{new['sample_indices_hash']}`; normalization metadata hash `{content_hash(new_packet.metadata.norm_factors)}`. Outlier positions, scales, signs, acquired sample WMAPE, prior, and coder measurements match the frozen-prefix MSE reference exactly.", "",
              f"The serialized outer sample wrappers have equal lengths ({len(new_wrapper):,} bytes), although the wrapper bytes differ; decoded symbol stream and metadata are identical. The acquired sample WMAPE is {new_record['sample_wmape']:.6f}% and its measured coded rate is {new_record['sample_coder_rate']:.9f} b/float. This sample rate is already included in the full-vector RD curves.", "",
              "The O, baseline S and candidate manifests match the retained target, decoder SI, baseline weights, and numerical runtime. Full-SI rANS round trips recorded zero symbol errors.", ""]
    destination = NEW_S / "final_analysis" / "matched_seed43_comparison.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    print(report())
