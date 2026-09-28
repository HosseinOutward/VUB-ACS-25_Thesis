"""The one completion contract shared by the sparse runner and the analysis path.

`validate_case_records` checks that a case's CSV holds exactly its scheduled rows in order.
`read_completed_case` reads one finished case's `complete.json`, optionally checks its identity
against an expected value, validates its rows, and enriches them with placement metadata (codec,
case, initial-SI mode, start round, relative round, phase, worker set, trace identity digest) so neither the runner
nor the plotter re-derives it. No imports from the runner or the plotter: both import this module,
not each other.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from FL_code.FL_core.cache import content_hash
from FL_code.FL_core.utils import parse_configurable_name

_ROW_COLUMNS = ("round_id", "client_id", "round_name_full")


def _expected_rows(case_metadata: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Return the scheduled round/client/token row order for one case."""
    return [
        (round_id, client_id, token)
        for round_id, token in enumerate(case_metadata["round_codecs"], case_metadata["start_round"])
        for client_id in case_metadata["client_ids"]
    ]


def _matches_schedule(
    actual: tuple[int, int, str], expected: tuple[int, int, str], case_metadata: Mapping[str, Any],
) -> bool:
    """Match a recorded token to its schedule and the one suffix owned by the warm protocol."""
    actual_round, actual_client, actual_token = actual
    expected_round, expected_client, expected_token = expected
    allowed_tokens = {expected_token}
    protocol = parse_configurable_name(case_metadata["protocol"], "protocol").name
    if protocol == "wz_cancer_warm" and expected_token != "F":
        assert "warm_retrain_repeats" in case_metadata, (
            "Warm case identity must record its effective warm_retrain_repeats."
        )
        allowed_tokens.add(
            expected_token + f"|warm_training|warm_retrain_repeats={case_metadata['warm_retrain_repeats']}"
        )
    return (
        (actual_round, actual_client) == (expected_round, expected_client)
        and actual_token in allowed_tokens
    )


def identity_mismatches(expected: Mapping[str, Any], stored: Mapping[str, Any]) -> list[str]:
    """Return the top-level identity keys whose values differ or exist on only one side."""
    return sorted(key for key in expected.keys() | stored.keys() if expected.get(key) != stored.get(key))


def _validation_case_metadata(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve and cross-check protocol-owned record fields from one stored identity."""
    case_metadata = identity["case"]
    protocol = parse_configurable_name(case_metadata["protocol"], "protocol").name
    if protocol != "wz_cancer_warm" or "protocol_snapshot" not in identity:
        return case_metadata
    snapshot_repeats = identity["protocol_snapshot"]["warm_quantizer_pool"]["warm_retrain_repeats"]
    stored_repeats = case_metadata.get("warm_retrain_repeats", snapshot_repeats)
    assert stored_repeats == snapshot_repeats, (
        "Case and protocol snapshot disagree on warm_retrain_repeats."
    )
    return case_metadata | {"warm_retrain_repeats": snapshot_repeats}


def validate_case_records(path: Path, case_metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Assert `path` holds exactly the ordered product of scheduled rounds x selected workers.

    `case_metadata` supplies `start_round`, `round_codecs`, and `client_ids`: the schedule that
    determines the expected row order. Missing, extra, duplicated, or reordered rows fail loudly,
    naming `path`, rather than silently accepting a partial or corrupted case.
    """
    assert path.is_file(), f"Missing case records: {path}."
    records = pd.read_csv(path)
    assert set(_ROW_COLUMNS) <= set(records.columns), (
        f"Case records at {path} are missing required columns {_ROW_COLUMNS}.")
    actual = list(records[list(_ROW_COLUMNS)].itertuples(index=False, name=None))
    expected = _expected_rows(case_metadata)
    assert len(actual) == len(expected) and all(
        _matches_schedule(actual_row, expected_row, case_metadata)
        for actual_row, expected_row in zip(actual, expected, strict=True)
    ), (
        f"Incomplete, duplicated, or reordered case records: {path}.")
    return records


def read_completed_case(case_dir: Path, expected_identity: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Read one finished case, validate it, and enrich its rows for downstream analysis.

    A missing `complete.json` means unfinished output, not a valid empty case, and fails loudly.
    When `expected_identity` is supplied, it is compared for exact equality against the stored
    identity (the part of `complete.json` that decides reuse, never its provenance); a mismatch
    fails loudly, naming `case_dir` and the differing top-level keys.
    """
    completion = case_dir / "complete.json"
    assert completion.is_file(), f"Case has no complete.json, so it is unfinished output: {case_dir}."
    payload = json.loads(completion.read_text())
    identity = payload["identity"]
    if expected_identity is not None:
        mismatched = identity_mismatches(expected_identity, identity)
        assert not mismatched, f"Case identity differs from expected at {case_dir}: {mismatched}."

    case_metadata = _validation_case_metadata(identity)
    records = validate_case_records(case_dir / "compression_records.csv", case_metadata).copy()
    return _enrich(records, case_dir, identity, complete=True)


def read_available_case(case_dir: Path, identity: Mapping[str, Any]) -> pd.DataFrame:
    """Read the valid ordered prefix of an unfinished case for explicitly incomplete previews."""
    path = case_dir / "compression_records.csv"
    assert path.is_file(), f"Missing case records: {path}."
    case_metadata = _validation_case_metadata(identity)
    case_records = pd.read_csv(path)
    assert set(_ROW_COLUMNS) <= set(case_records.columns), (
        f"Case records at {path} are missing required columns {_ROW_COLUMNS}.")
    actual = list(case_records[list(_ROW_COLUMNS)].itertuples(index=False, name=None))
    expected = _expected_rows(case_metadata)
    assert len(actual) <= len(expected) and all(
        _matches_schedule(actual_row, expected_row, case_metadata)
        for actual_row, expected_row in zip(actual, expected, strict=False)
    ), f"Invalid unfinished case record prefix: {path}."
    return _enrich(case_records.copy(), case_dir, identity, complete=False)


def _enrich(
    case_records: pd.DataFrame, case_dir: Path, identity: Mapping[str, Any], *, complete: bool,
) -> pd.DataFrame:
    """Attach case placement and completion metadata to validated record rows."""
    case_metadata = identity["case"]
    start_round = case_metadata["start_round"]
    phases = {
        round_id: "frozen" if token == "F" else "trained"
        for round_id, token in enumerate(case_metadata["round_codecs"], start_round)
    }
    cycles: dict[int, int] = {}
    cycle = -1
    for round_id, token in enumerate(case_metadata["round_codecs"], start_round):
        cycle += token != "F"
        cycles[round_id] = cycle
    case_records.insert(0, "case_complete", complete)
    case_records.insert(0, "phase", [phases[round_id] for round_id in case_records["round_id"]])
    case_records.insert(0, "cycle", [cycles[round_id] for round_id in case_records["round_id"]])
    case_records.insert(0, "relative_round", case_records["round_id"] - start_round)
    case_records.insert(0, "client_ids", [tuple(case_metadata["client_ids"])] * len(case_records))
    case_records.insert(0, "start_round", start_round)
    case_records.insert(0, "initial_si", case_metadata["initial_si"])
    case_records.insert(0, "case_id", case_dir.name)
    case_records.insert(0, "codec_id", case_dir.parent.name)
    case_records["trace_identity_digest"] = content_hash((identity["trace"],), {})[:16]
    return case_records
