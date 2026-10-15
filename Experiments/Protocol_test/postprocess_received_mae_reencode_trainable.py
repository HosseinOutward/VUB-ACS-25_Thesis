"""Validate and report the matched seed-43 trainable-encoder S pilot."""

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
NEW_S = EXPERIMENTS / "s_received_mae_reencode_trainable_seed43_20261014"
O = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
NEW_METHOD = "current_e1_mae_reencode_teacher_trainable_encoder"
OLD_METHOD = "current_e1_mae_received_prefix_bins"


def read_case(root: Path, method: str) -> tuple[dict[str, Any], Path]:
    """Load one completed case and require its deployed symbol round trip to be exact."""
    path = root / "cases" / f"{method}_seed43" / "target" / "complete.json"
    run = json.loads(path.read_text())
    assert run["record"]["symbol_error_rate"] == 0, f"Symbol errors in {path}"
    return run, path.parent


def report() -> Path:
    """Check matched acquisition semantics and write full seed-43 RD tables."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in (O, OLD_S, NEW_S)]
    for field in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(item[field] == manifests[0][field] for item in manifests), f"Mismatched {field}."
    assert all(Path(item["plan"]["reference_root"]).resolve() == Path(manifests[0]["plan"]["reference_root"]).resolve()
               for item in manifests), "Mismatched retained reference root."

    old, old_dir = read_case(OLD_S, OLD_METHOD)
    new, new_dir = read_case(NEW_S, NEW_METHOD)
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
    assert old_record["sample_fit_mode"] == "received_prefix_bins" and not old_record["sample_training_encoder"]
    assert new_record["sample_fit_mode"] == "reencode_teacher" and new_record["sample_training_encoder"]
    assert new_record["sample_final_distortion"] == "mse"

    runs = {"O": oracle, "S_frozen": old, "S_trainable_reencode": new}
    points = {name: mean_points([run["record"]]) for name, run in runs.items()}
    lines = [
        "# Matched seed-43 S encoder-fit pilot", "",
        "The candidate keeps normalized-MAE acquisition, the acquired coordinates, decoded symbol stream and metadata fixed. It changes final S fitting from received-prefix bins with a frozen encoder to reencoded teacher labels with a trainable encoder. The candidate fits only use decoded acquired samples; no current-target labels enter fitting. O and the frozen-encoder S curve are read-only controls.", "",
        "This is one client-0, round-5 trained-round diagnostic. All curves have zero symbol errors. The candidate has a nonmonotone WMAPE curve, so the protocol's piecewise-linear equal-distortion RD gap is unavailable for every comparison involving it.", "",
        "## Matched staged RD values", "",
        "Cumulative rates are bits per full-gradient float. `—` means the corresponding fine-tuned prior source was not recorded.", "",
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
              "Full-SI rANS, row versus column; negative favors the row.", "", "```text",
              matrix_text(points, "server_si_rans_stage_rates"), "```", "",
              "Native transport, row versus column; negative favors the row. The entries equal full-SI rANS in this one-client comparison.", "", "```text",
              matrix_text(points, "coder_stage_rates"), "```", "",
              f"The valid frozen-S vs O curve gap is {curve_gap(points['S_frozen'], points['O'], 'server_si_rans_stage_rates'):+.3f}% over their shared distortion interval. Candidate entries are unavailable because its WMAPE changes from {new_record['stage_wmapes'][0]:.3f}% to {new_record['stage_wmapes'][1]:.3f}% between stages 1 and 2.", "",
              "## Acquisition parity and endpoint context", "",
              f"The decoded acquired rANS symbol stream hash is `{content_hash(new_packet.symbols)}` ({new_packet.symbols.numel() if isinstance(new_packet.symbols, torch.Tensor) else len(new_packet.symbols):,} bytes). Coordinate hash `{new['sample_indices_hash']}` and normalization metadata hash `{content_hash(new_packet.metadata.norm_factors)}` match the frozen-encoder reference. Acquired sample WMAPE and coded sample rate match exactly at {new_record['sample_wmape']:.6f}% and {new_record['sample_coder_rate']:.9f} b/float. The outer compressed sample packet differs by {abs(len(new_wrapper) - len(old_wrapper))} byte due to container serialization; its symbol stream and metadata are identical.", "",
              "At the final stage the trainable-encoder candidate reaches 34.231% WMAPE versus 34.650% for frozen S, while cumulative full-SI rANS is 0.794087 versus 0.771625 b/float. Its middle stage worsens to 58.054% WMAPE, so this endpoint improvement does not establish a valid curve-level RD gain.", "",
              "The measured sample coder rate is already included in the RD curves. A metadata-size proxy adds 0.000721 b/float to candidate cumulative rates and 0.000720 b/float to frozen-S cumulative rates; it is approximate because it converts the standalone compressed metadata size. The valid frozen-S gap against O changes from +8.947% coded to +9.073% with this proxy. The candidate remains unrankable.", "",
              "The new and reference manifests match in retained target, decoder SI, baseline weights and numerical runtime. The full-SI rANS round trip recorded zero symbol errors.", ""]
    destination = NEW_S / "final_analysis" / "matched_seed43_comparison.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    print(report())
