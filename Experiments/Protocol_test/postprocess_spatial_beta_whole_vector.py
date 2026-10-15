"""Report the matched seed-43 full-vector RD effect of a decoder-local spatial blend."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import re
from typing import Any

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
OUTPUT = EXPERIMENTS / "s_setup_spatial_refiner_20261014"
O_PATH = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011/cases/O_seed43/target/complete.json"
S_PATH = EXPERIMENTS / "s_received_mae_five_seed_20261012/cases/current_e1_mae_received_prefix_bins_seed43/target/complete.json"


def _mean(record: dict[str, Any]) -> dict[str, Any]:
    return mean_points([record])


def report() -> Path:
    """Validate unchanged bit rates and write exact and metadata-proxy matched curves."""
    score = list(csv.DictReader((OUTPUT / "whole_vector_beta_score.csv").open()))
    assert len(score) == 3 and [int(row["stage"]) for row in score] == [1, 2, 3]
    probe_report = (OUTPUT / "sample_fitted_beta_report.md").read_text()
    provenance_text = re.search(r"Artifact provenance:\n```json\n(.*?)\n```", probe_report, re.S)
    assert provenance_text is not None, "Missing frozen spatial-beta provenance."
    provenance = json.loads(provenance_text.group(1))
    oracle = json.loads(O_PATH.read_text())
    baseline = json.loads(S_PATH.read_text())
    assert all(row["sample_coordinates_unchanged"] == "True" for row in score)
    assert len({row["received_symbol_digest"] for row in score}) == 1
    assert all(float(row["full_si_rans_bits_per_float_unchanged"]) == float(baseline["record"]["server_si_rans_rate"])
               for row in score)

    spatial_record = copy.deepcopy(baseline["record"])
    spatial_record["stage_wmapes"] = [float(row["spatial_blend_wmape_percent"]) for row in score]
    cases = {"O": oracle["record"], "S_original": baseline["record"], "S_spatial": spatial_record}
    points = {name: _mean(record) for name, record in cases.items()}
    assert points["S_spatial"]["server_si_rans_stage_rates"] == points["S_original"]["server_si_rans_stage_rates"]
    for row, original in zip(score, baseline["record"]["stage_wmapes"], strict=True):
        assert abs(float(row["original_wmape_percent"]) - original) < 0.02

    metadata_adjusted: dict[str, dict[str, Any]] = {}
    metadata_proxy: dict[str, float] = {}
    for name, source_case in (("O", oracle), ("S_original", baseline), ("S_spatial", baseline)):
        record = copy.deepcopy(cases[name])
        source_record = source_case["record"]
        metadata_proxy[name] = 8 * 2**20 * (
            float(source_record.get("meta_data_size") or 0)
            + float(source_record.get("sample_metadata_size") or 0)
        ) / float(source_record["model_size"])
        record["server_si_rans_stage_rates"] = list(record["server_si_rans_stage_rates"])
        record["server_si_rans_stage_rates"][0] += metadata_proxy[name]
        metadata_adjusted[name] = _mean(record)

    lines = [
        "# Seed-43 local spatial beta: whole-vector RD score", "",
        "This is a receiver-side post-decoder diagnostic on the retained best S seed-43 packet. For each 3×3 convolution-weight tensor and decoded stage, one beta was fit by weighted-median L1 on received decoded sample labels only. The predictor averages all received labels among the available other taps in each finite 3×3 kernel (3 neighbors at corners, 5 at edges, 8 in the interior); its center tap is excluded. Fit and validation labels were split, while all received sample labels stayed in neighbor context. Unsampled coordinates with no received neighbor and all sampled coordinates are unchanged.", "",
        "No model or prior was retrained, no raw target label entered beta fitting, and the packet/coder/prior are unchanged. The raw target was loaded only after beta freeze for full-vector scoring and exact regenerated ordinary-symbol digest verification. The S full-vector and sample rANS rates below are the retained bit costs and remain identical for the spatial result.", "",
        "This is one client-0, round-5 trained-round diagnostic. Each WMAPE curve is averaged as raw percent before conversion to dB. Rates are cumulative bits per full-gradient float; the primary gap is full-SI rANS over pairwise shared distortion ranges.", "",
        "## Stage outputs", "",
        "| Curve | Stage | Full-SI rANS (b/float) | WMAPE dB (raw %) | Whole-vector gain vs original S (points) | Ordinary WMAPE raw (%) | Sample WMAPE raw (%) | Modified ordinary coefficients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, (record, row) in enumerate(zip(points["S_spatial"]["server_si_rans_stage_rates"], score, strict=True), 1):
        db = points["S_spatial"]["wmape_db"][stage - 1]
        lines.append(f"| S spatial beta | {stage} | {record:.9f} | {db:.6f} ({float(row['spatial_blend_wmape_percent']):.6f}%) | "
                     f"{float(row['whole_vector_gain_points']):+.6f} | {float(row['ordinary_blend_wmape_percent']):.6f} | "
                     f"{float(row['sample_blend_wmape_percent']):.6f} | {int(row['modified_unsampled_coordinate_count']):,} |")
    for method in ("O", "S_original"):
        point = points[method]
        for stage in range(3):
            db = point["wmape_db"][stage]
            lines.append(f"| {method} | {stage + 1} | {point['server_si_rans_stage_rates'][stage]:.9f} | "
                         f"{db:.6f} ({point['stage_wmapes'][stage]:.6f}%) | — | — | — | — |")

    lines += ["", "## Exact directed equal-distortion gaps", "",
              "Full-SI rANS, row versus column; negative favors the row. Reverse entries are recomputed.", "", "```text",
              matrix_text(points, "server_si_rans_stage_rates"), "```", "",
              "Native transport is full-SI rANS for this one-client comparison:", "", "```text",
              matrix_text(points, "coder_stage_rates"), "```", ""]
    for name in ("S_original", "S_spatial"):
        lines.append(f"{name} vs O: {curve_gap(points[name], points['O'], 'server_si_rans_stage_rates'):+.3f}%.")
    lines.append(f"Spatial beta vs original S: {curve_gap(points['S_spatial'], points['S_original'], 'server_si_rans_stage_rates'):+.3f}%.")

    lines += ["", "## Metadata-size proxy", "",
              "The proxy adds serialized `meta_data_size` plus `sample_metadata_size` to stage-1 rate as 8 MiB-scaled bytes per model float. It is approximate; standalone compressed metadata is not an exact additive wire fragment. Spatial beta adds no metadata, so its proxy rate equals the original S rate.", "",
              "| Curve | Added metadata proxy (b/float) | Full-SI gap vs O |", "|---|---:|---:|"]
    for name in ("S_original", "S_spatial"):
        lines.append(f"| {name} | {metadata_proxy[name]:.9f} | "
                     f"{curve_gap(metadata_adjusted[name], metadata_adjusted['O'], 'server_si_rans_stage_rates'):+.3f}% |")
    lines.append(f"| O | {metadata_proxy['O']:.9f} | — |")

    row0 = score[0]
    lines += ["", "## Cost and provenance", "",
              f"Ordinary rANS digest: `{row0['received_symbol_digest']}` (header and regenerated stream match). Sample coordinates are unchanged; the source sample-bin hash is `{provenance['sample_bins_hash']}`, sample-coordinate hash `{provenance['sample_indices_hash']}`, beta artifact hash `{row0['beta_artifact_hash']}`, and raw target vector hash `{provenance['source_vector_hash']}`.", "",
              f"The measured sample coder cost remains {row0['sample_coder_rate_bits_per_float_unchanged']} b/float; sample prior estimate {row0['sample_prior_rate_bits_per_float_unchanged']} and coder overhead {row0['sample_coder_overhead_rate_bits_per_float_unchanged']} b/float are unchanged. The beta adjusts {int(row0['modified_unsampled_coordinate_count']):,} ordinary coefficients at stages 1–2; stage 3 has zero fitted beta and no changed coordinates. All stages retain zero symbol errors.", "",
              "The spatial beta improves seed-43 curve gap by less than one percentage point while leaving the final endpoint and all coded bits unchanged. Multi-seed repeat is not inferred from this one-seed diagnostic.", ""]
    destination = OUTPUT / "whole_vector_rd_report.md"
    destination.write_text("\n".join(lines))
    return destination


if __name__ == "__main__":
    print(report())
