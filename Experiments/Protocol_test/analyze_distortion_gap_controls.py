"""Report-only sample-stream accounting for the retained O and received-prefix S curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "_/experiments/s_distortion_audit_20261014/controls"
S_PATH = ROOT / "_/experiments/s_received_mae_five_seed_20261012/final_analysis/raw_records.json"
O_PATH = ROOT / "_/experiments/s_dedicated_refiner_coldfit_five_seed_20261011/final_analysis/raw_records.json"
S_KEY = "current_e1_mae_received_prefix_bins"


def mean_curve(records: list[dict[str, Any]], *, remove_sample: bool = False,
               metadata_adjusted: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average cumulative full-SI rates and raw WMAPE, optionally removing S acquisition cost."""
    rates, errors = [], []
    for item in records:
        record = item["record"]
        rate = np.cumsum(np.asarray(record["server_si_rans_stage_rates"], dtype=np.float64))
        if remove_sample:
            rate = rate - float(record["sample_coder_rate"])
            if metadata_adjusted:
                rate = rate - 8 * 2**20 * float(record["sample_metadata_size"]) / float(record["model_size"])
        if metadata_adjusted:
            rate = rate + 8 * 2**20 * (float(record["meta_data_size"])
                                        + (0.0 if remove_sample else float(record.get("sample_metadata_size") or 0.0))) \
                / float(record["model_size"])
        rates.append(rate)
        errors.append(np.asarray(record["stage_wmapes"], dtype=np.float64))
    return np.mean(rates, axis=0), wmape_to_db(np.mean(errors, axis=0))


def report() -> dict[str, Any]:
    """Compare measured curves with an explicitly non-achievable free-acquisition diagnostic."""
    s_records = json.loads(S_PATH.read_text())[S_KEY]
    o_records = json.loads(O_PATH.read_text())["O"]
    assert sorted(item["fit_seed"] for item in s_records) == sorted(item["fit_seed"] for item in o_records)
    outputs: dict[str, Any] = {"seed_count": len(s_records), "controls": {}}
    for label, remove_sample, metadata_adjusted in (
        ("measured_coded_full_si", False, False),
        ("measured_metadata_adjusted_proxy", False, True),
        ("sample_coder_free_counterfactual", True, False),
        ("sample_coder_and_sample_metadata_free_counterfactual", True, True),
    ):
        o_rate, o_db = mean_curve(o_records, metadata_adjusted=metadata_adjusted)
        s_rate, s_db = mean_curve(s_records, remove_sample=remove_sample,
                                  metadata_adjusted=metadata_adjusted)
        result = rd_curve_gap(s_rate, s_db, o_rate, o_db, name=label)
        result["shared_distortion_db"] = [float(max(s_db.min(), o_db.min())),
                                          float(min(s_db.max(), o_db.max()))]
        outputs["controls"][label] = {
            "o_rate_bits_per_float": o_rate.tolist(), "o_wmape_db": o_db.tolist(),
            "s_rate_bits_per_float": s_rate.tolist(), "s_wmape_db": s_db.tolist(),
            **result,
        }
    outputs["sample_cost"] = {
        "mean_sample_coder_bits_per_float": float(np.mean([
            item["record"]["sample_coder_rate"] for item in s_records])),
        "mean_sample_prior_bits_per_float": float(np.mean([
            item["record"]["sample_prior_rate"] for item in s_records])),
        "mean_sample_metadata_bits_per_float_proxy": float(np.mean([
            8 * 2**20 * item["record"]["sample_metadata_size"] / item["record"]["model_size"]
            for item in s_records])),
        "mean_sample_wmape_percent": float(np.mean([
            item["record"]["sample_wmape"] for item in s_records])),
        "mean_full_vector_wmape_percent": float(np.mean([
            item["record"]["wmape"] for item in s_records])),
    }
    return outputs


def main() -> None:
    """Write reproducible JSON and a concise interpretation of the rate accounting."""
    result = report()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "sample_rate_counterfactual.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# O/S sample-rate counterfactual (report-only)", "",
        "Five-seed normalized-MAE received-prefix S and retained O curves. The two counterfactuals",
        "subtract the measured sample coder stream, with the second also removing the sample metadata",
        "proxy. They hold S's trained model and full-vector distortions fixed, so they are bookkeeping",
        "diagnostics, not deployable S performance or evidence that free raw labels are available.", "",
        "Rates are averaged per stage and raw WMAPE is averaged before dB conversion. Gap percentages",
        "are recomputed from piecewise-linear equal-distortion curves; component percentages are not",
        "added. The metadata-adjusted column remains an estimate because metadata sizes are standalone",
        "gzip/pickle proxies, not exact additive payload fragments.", "",
        "| Rate view | Mean rate gap (b/float) | Gap vs O (%) | Shared WMAPE dB |", "|---|---:|---:|---:|",
    ]
    for name, data in result["controls"].items():
        low, high = data["shared_distortion_db"]
        span = f"{low:.3f} to {high:.3f} dB"
        lines.append(f"| {name} | {data['rate_gap_bits']:.6f} | {data['rate_gap_percent']:+.3f}% | {span} |")
    sample = result["sample_cost"]
    lines += ["", f"Mean sample coder cost is {sample['mean_sample_coder_bits_per_float']:.6f} b/float; "
              f"sample prior estimate is {sample['mean_sample_prior_bits_per_float']:.6f}; "
              f"sample metadata proxy is {sample['mean_sample_metadata_bits_per_float_proxy']:.6f}. "
              f"Mean sample WMAPE is {sample['mean_sample_wmape_percent']:.3f}% versus "
              f"{sample['mean_full_vector_wmape_percent']:.3f}% at the full-vector final stage.", ""]
    (OUTPUT / "sample_rate_counterfactual.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
