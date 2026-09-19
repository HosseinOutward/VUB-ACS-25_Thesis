from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Experiments.Protocol_test import records as record_io
from Experiments.Protocol_test.plot import (
    averaged_curves,
    comparison_gaps,
    load_available_records,
    numeric_list,
    round_curve,
    stage_points,
    timing_summary,
)


def _write_case(
    root: Path,
    codec_id: str,
    case_id: str,
    *,
    start_round: int,
    client_ids: list[int],
    round_codecs: list[str],
    rows: list[dict[str, object]],
    trace: dict[str, object] | None = None,
) -> Path:
    """Write one completed case directory in the real `complete.json` shape (identity/provenance),
    matching `exp_records/refactor_smoke/cases/T/start5/complete.json`, so tests exercise the real
    `records.read_completed_case` contract instead of a parallel fixture format."""
    directory = root / "cases" / codec_id / case_id
    directory.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(directory / "compression_records.csv", index=False)
    payload = {
        "identity": {
            "trace": trace or {"trace_version": 1, "fl_config": {"num_clients": 3}},
            "case": {
                "protocol": "wz_cancer", "start_round": start_round,
                "initial_si": "raw", "client_ids": client_ids, "round_codecs": round_codecs,
            },
        },
        "provenance": {},
    }
    (directory / "complete.json").write_text(json.dumps(payload))
    return directory


def _rows(round_id: int, round_name: str, client_wmapes: dict[int, str], **extra_columns: list[str]) -> list[dict[str, object]]:
    """Build compression_records.csv rows for one round across the given clients' stage_wmapes."""
    rows = []
    for index, (client_id, wmapes) in enumerate(client_wmapes.items()):
        row: dict[str, object] = {
            "round_id": round_id, "client_id": client_id, "round_name_full": round_name,
            "stage_wmapes": wmapes,
        }
        for column, values in extra_columns.items():
            row[column] = values[index]
        rows.append(row)
    return rows


def _records() -> pd.DataFrame:
    """Two measured workers (0, 2) at one round, with unequal per-worker WMAPE, for direct
    `round_curve`/`stage_points` tests that don't need to go through a case directory."""
    return pd.DataFrame({
        "experiment": ["data"] * 2, "codec_id": ["T"] * 2, "case_id": ["start5"] * 2,
        "initial_si": ["raw"] * 2,
        "start_round": [5] * 2, "round_id": [5] * 2, "relative_round": [0] * 2,
        "cycle": [0] * 2, "phase": ["trained"] * 2,
        "client_ids": [(0, 2)] * 2, "trace_identity_digest": ["abc"] * 2,
        "model_size": [100, 100],
        "stage_wmapes": ["[10, 5]", "[30, 10]"], "prior_stage_rates": ["[1, 1]", "[2, 2]"],
        "coder_stage_rates": ["[2, 2]", "[3, 3]"], "final_bytes": [0.01, 0.02],
        "records_path": ["records.csv"] * 2,
    })


def test_round_curve_uses_equal_worker_mean_and_cumulative_stage_rates() -> None:
    records = _records()
    rate, distortion = round_curve(records, Path("records.csv"), "prior_stage_rates")
    # Mean incremental rates are [1.5, 1.5], so the plotted stage positions are cumulative.
    np.testing.assert_allclose(rate, [1.5, 3.])
    # Equal-worker mean WMAPE per stage: [(10+30)/2, (5+10)/2] = [20, 7.5], then dB.
    np.testing.assert_allclose(distortion, 20 * np.log10(np.array([20., 7.5]) / 100))
    points = stage_points(records)
    assert len(points) == 5  # 2 stages each for prior/coder rates, plus 1 final-payload point
    assert set(points.rate_field) == {"prior_stage_rates", "coder_stage_rates", "final_bytes"}
    payload = points[points.rate_field == "final_bytes"]
    assert payload.rate.iloc[0] == pytest.approx(.015 * 2**20 * 8 / 100)
    assert payload.wmape_db.iloc[0] == pytest.approx(distortion[-1])
    assert payload.is_final_payload.iloc[0]
    assert not points[points.rate_field != "final_bytes"].is_final_payload.any()


def test_curve_averaging_happens_in_linear_wmape_space() -> None:
    points = stage_points(_records())
    later = points.assign(round_id=6, wmape_db=20 * np.log10(40 / 100))
    averaged = averaged_curves(pd.concat([points, later]))
    prior_stage_one = averaged[
        (averaged.rate_field == "prior_stage_rates") & (averaged.stage == 1)
    ].iloc[0]
    expected_wmape = (20 + 40) / 2
    assert prior_stage_one.wmape_db == pytest.approx(20 * np.log10(expected_wmape / 100))


def test_curve_averaging_can_preserve_sparse_starts() -> None:
    points = stage_points(_records())
    other_start = points.assign(start_round=10, case_id="start10")
    averaged = averaged_curves(pd.concat([points, other_start]), ("start_round",))
    assert set(averaged.start_round) == {5, 10}


def test_runtime_summary_does_not_extrapolate_trained_rows_over_frozen_rows() -> None:
    # Runtime forecasts must use these protocol-wide strata separately: T/O fits are expensive,
    # while F reuses are cheaper, so early trained rows cannot represent the full schedule.
    # Initial uncached 3-plane estimates (2026-09-18): O ~= 760 s/row, T ~= 145 s/row, F ~= 61
    # s/row. These are provisional until timing_summary.csv contains complete strata.
    records = pd.DataFrame({
        "codec_id": ["T"] * 4,
        "round_name_full": ["T|training_seed=43", "T|training_seed=43", "F", "F"],
        "phase": ["trained", "trained", "frozen", "frozen"],
        "total_seconds": [100.0, 120.0, 10.0, 20.0],
    })
    summary = timing_summary(records).set_index("round_token")
    assert summary.loc["T", "mean_seconds"] == 110
    assert summary.loc["F", "mean_seconds"] == 15


def test_partial_coder_rates_do_not_fall_back_to_prior() -> None:
    records = _records()
    records.loc[0, "coder_stage_rates"] = None
    with pytest.raises(AssertionError, match="Incomplete coder_stage_rates"):
        stage_points(records)


def test_rate_source_missing_from_one_round_of_a_case_fails_loudly() -> None:
    later = _records().assign(round_id=6, relative_round=1, phase="frozen", prior_stage_rates=None)
    with pytest.raises(AssertionError, match="Incomplete prior_stage_rates"):
        stage_points(pd.concat([_records(), later], ignore_index=True))


def test_codec_missing_an_entire_rate_source_is_skipped_not_an_error() -> None:
    records = _records()
    records["prior_stage_rates"] = None  # whole source unavailable for this codec, not partial
    points = stage_points(records)
    assert set(points.rate_field) == {"coder_stage_rates", "final_bytes"}


def test_csv_list_parser_does_not_execute_code(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        numeric_list("__import__('pathlib').Path('unexpected').touch()", "stage_wmapes", tmp_path)


def test_available_records_include_unfinished_prefix_and_report_missing_cases(tmp_path: Path) -> None:
    case = {
        "protocol": "wz_cancer", "case_id": "start5", "codec_id": "O", "start_round": 5,
        "initial_si": "raw", "client_ids": [0, 2], "round_codecs": ["O", "F"],
    }
    missing = case | {"case_id": "start10", "start_round": 10}
    manifest = {"identity": {"trace": {"trace_version": 1}, "cases": [case, missing]}}
    (tmp_path / "test_config.json").write_text(json.dumps(manifest))
    directory = tmp_path / "cases" / "O" / "start5"
    directory.mkdir(parents=True)
    pd.DataFrame(_rows(5, "O", {0: "[10]"}, model_size=[100])).to_csv(
        directory / "compression_records.csv", index=False)

    case_records, coverage = load_available_records(tmp_path)

    assert case_records.client_id.tolist() == [0]
    assert case_records.case_complete.tolist() == [False]
    assert coverage[["rows", "expected_rows", "complete"]].values.tolist() == [
        [1, 4, False], [0, 4, False]]


def test_mismatched_trace_identity_in_declared_comparison_fails_loudly(tmp_path: Path) -> None:
    rows = _rows(5, "X", {0: "[10]", 2: "[30]"}, model_size=[100, 100],
                  prior_stage_rates=["[1]", "[2]"], coder_stage_rates=["[1]", "[2]"])
    case_dirs = [
        _write_case(tmp_path, "T", "start5", start_round=5, client_ids=[0, 2], round_codecs=["X"],
                    rows=rows, trace={"trace_version": 1}),
        _write_case(tmp_path, "O", "start5", start_round=5, client_ids=[0, 2], round_codecs=["X"],
                    rows=rows, trace={"trace_version": 2}),  # different trace identity
    ]
    frames = []
    for case_dir in case_dirs:
        frame = record_io.read_completed_case(case_dir)
        frame["records_path"] = str(case_dir / "compression_records.csv")
        frames.append(frame)
    case_records = pd.concat(frames, ignore_index=True)
    case_records["experiment"] = "test"
    with pytest.raises(AssertionError, match="mismatched trace identity"):
        comparison_gaps(case_records, (("T", "O"),))


def test_final_bytes_point_sits_at_final_stage_distortion() -> None:
    records = _records()
    points = stage_points(records)
    payload = points[points.rate_field == "final_bytes"].iloc[0]
    prior_last_stage = points[(points.rate_field == "prior_stage_rates") & (points.stage == 2)].iloc[0]
    assert payload.wmape_db == pytest.approx(prior_last_stage.wmape_db)


def test_single_stage_comparison_is_unavailable_not_omitted() -> None:
    left = _records()
    left["stage_wmapes"], left["prior_stage_rates"], left["coder_stage_rates"] = "[10]", "[1]", "[2]"
    right = left.assign(codec_id="O")
    gaps = comparison_gaps(pd.concat([left, right]), (("T", "O"),))
    assert not gaps.empty
    assert (gaps.gap_available == False).all()  # noqa: E712
    assert set(gaps.gap_unavailable_reason) == {"single_stage"}


def test_non_monotone_curve_is_unavailable_not_a_crash() -> None:
    left = _records()
    # Distortion increases with rate on both sides: never strictly improving, so no envelope
    # or sort should be invented to force a gap.
    left["stage_wmapes"] = ["[10, 30]", "[30, 60]"]
    right = left.assign(codec_id="O")
    gaps = comparison_gaps(pd.concat([left, right]), (("T", "O"),))
    assert not gaps.empty
    assert set(gaps.gap_unavailable_reason) == {"non_monotone"}
