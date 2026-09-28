import json
from pathlib import Path

import pandas as pd
import pytest

from Experiments.Protocol_test.records import read_available_case, read_completed_case, validate_case_records


def _metadata() -> dict[str, object]:
    return {
        "start_round": 5,
        "protocol": "wz_cancer_warm|warm_retrain_repeats=2",
        "warm_retrain_repeats": 2,
        "initial_si": "raw",
        "client_ids": [0, 1],
        "round_codecs": ["O|bins_per_plane=2", "F"],
    }


def test_records_accept_protocol_owned_suffixes_on_trained_rounds(tmp_path: Path) -> None:
    path = tmp_path / "compression_records.csv"
    pd.DataFrame([
        (5, 0, "O|bins_per_plane=2"),
        (5, 1, "O|bins_per_plane=2|warm_training|warm_retrain_repeats=2"),
        (6, 0, "F"),
        (6, 1, "F"),
    ], columns=("round_id", "client_id", "round_name_full")).to_csv(path, index=False)

    assert len(validate_case_records(path, _metadata())) == 4


def test_records_reject_suffixes_on_frozen_rounds(tmp_path: Path) -> None:
    path = tmp_path / "compression_records.csv"
    pd.DataFrame([
        (5, 0, "O|bins_per_plane=2"),
        (5, 1, "O|bins_per_plane=2"),
        (6, 0, "F"),
        (6, 1, "F|unexpected"),
    ], columns=("round_id", "client_id", "round_name_full")).to_csv(path, index=False)

    with pytest.raises(AssertionError, match="Incomplete, duplicated, or reordered"):
        validate_case_records(path, _metadata())


def test_records_reject_unknown_or_reordered_warm_suffix(tmp_path: Path) -> None:
    path = tmp_path / "compression_records.csv"
    pd.DataFrame([
        (5, 0, "O|bins_per_plane=2|warm_retrain_repeats=2|warm_training"),
        (5, 1, "O|bins_per_plane=2|warm_training|warm_retrain_repeats=3"),
        (6, 0, "F"),
        (6, 1, "F"),
    ], columns=("round_id", "client_id", "round_name_full")).to_csv(path, index=False)

    with pytest.raises(AssertionError, match="Incomplete, duplicated, or reordered"):
        validate_case_records(path, _metadata())


def test_available_records_accept_warm_suffix_in_valid_prefix(tmp_path: Path) -> None:
    case_dir = tmp_path / "O" / "start5"
    case_dir.mkdir(parents=True)
    pd.DataFrame([
        (5, 0, "O|bins_per_plane=2"),
        (5, 1, "O|bins_per_plane=2|warm_training|warm_retrain_repeats=2"),
    ], columns=("round_id", "client_id", "round_name_full")).to_csv(
        case_dir / "compression_records.csv", index=False,
    )

    records = read_available_case(case_dir, {"trace": {}, "case": _metadata()})

    assert len(records) == 2


def test_completed_records_cross_check_warm_repeats_with_protocol_snapshot(tmp_path: Path) -> None:
    """Stored protocol configuration remains authoritative for completed warm records."""
    case_dir = tmp_path / "O" / "start5"
    case_dir.mkdir(parents=True)
    pd.DataFrame([
        (5, 0, "O|bins_per_plane=2"),
        (5, 1, "O|bins_per_plane=2|warm_training|warm_retrain_repeats=2"),
        (6, 0, "F"),
        (6, 1, "F"),
    ], columns=("round_id", "client_id", "round_name_full")).to_csv(
        case_dir / "compression_records.csv", index=False,
    )
    case_metadata = _metadata()
    del case_metadata["warm_retrain_repeats"]
    identity = {
        "trace": {},
        "case": case_metadata,
        "protocol_snapshot": {"warm_quantizer_pool": {"warm_retrain_repeats": 2}},
    }
    (case_dir / "complete.json").write_text(json.dumps({"identity": identity}))

    assert len(read_completed_case(case_dir)) == 4
