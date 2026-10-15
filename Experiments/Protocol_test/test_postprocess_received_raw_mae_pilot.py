from __future__ import annotations

import numpy as np

from Experiments.Protocol_test.postprocess_received_raw_mae_pilot import build_report


def test_report_uses_raw_wmape_means_and_keeps_all_paired_seeds() -> None:
    groups = {}
    for method_index, method in enumerate(("O", "S_e1", "S_received_MAE", "S_received_raw_MAE")):
        runs = []
        for seed_index, seed in enumerate((43, 44, 45)):
            shift = method_index * 3 + seed_index
            record = {
                "stage_wmapes": [70.0 + shift, 50.0 + shift, 30.0 + shift],
                "marginal_rate": 1.0 + shift * .01,
                "quantizer_prior_stage_rates": [.1, .1, .1],
                "temporal_si_prior_stage_rates": [.1, .1, .1],
                "server_si_prior_stage_rates": [.1, .1, .1],
                "coder_stage_rates": [.11, .11, .11],
                "temporal_si_rans_stage_rates": [.12, .12, .12],
                "server_si_rans_stage_rates": [.13, .13, .13],
                "symbol_error_rate": 0.0,
                "sample_wmape": 33.0 + shift,
                "sample_prior_rate": .25 + shift * .001,
                "sample_coder_rate": .26 + shift * .001,
                "sample_coder_overhead_rate": .005,
                "sample_metadata_size": .001,
            }
            runs.append({"fit_seed": seed, "record": record})
        groups[method] = runs

    report = build_report(groups)

    raw_mean = np.mean([70.0, 71.0, 72.0])
    mean_db = 20 * np.log10(raw_mean / 100)
    assert f"{mean_db:.6f} ({raw_mean:.6f}%)" in report
    assert "S_received_raw_MAE" in report
    assert "| 43 | S_received_MAE |" in report
    assert "| 44 | S_e1 |" in report
    assert "| 45 | O |" in report
    assert "No trimming or client exclusions are applied." in report
