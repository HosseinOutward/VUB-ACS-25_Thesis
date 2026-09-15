"""The one completion contract shared by the sparse runner and the analysis path.

`validate_case_records` checks that a case's CSV holds exactly its scheduled rows in order.
`read_completed_case` reads one finished case's `complete.json`, optionally checks its identity
against an expected value, validates its rows, and enriches them with placement metadata (codec,
case, start round, relative round, phase, worker set, trace identity digest) so neither the runner
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

_ROW_COLUMNS = ("round_id", "client_id", "round_name_full")


def identity_mismatches(expected: Mapping[str, Any], stored: Mapping[str, Any]) -> list[str]:
    """Return the top-level identity keys whose values differ or exist on only one side."""
    return sorted(key for key in expected.keys() | stored.keys() if expected.get(key) != stored.get(key))


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
    expected = [
        (round_id, client_id, token)
        for round_id, token in enumerate(case_metadata["round_codecs"], case_metadata["start_round"])
        for client_id in case_metadata["client_ids"]
    ]
    actual = list(records[list(_ROW_COLUMNS)].itertuples(index=False, name=None))
    assert actual == expected, f"Incomplete, duplicated, or reordered case records: {path}."
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

    case_metadata = identity["case"]
    records = validate_case_records(case_dir / "compression_records.csv", case_metadata).copy()
    start_round = case_metadata["start_round"]
    records.insert(0, "phase", ["trained" if round_id == start_round else "frozen"
                                 for round_id in records["round_id"]])
    records.insert(0, "relative_round", records["round_id"] - start_round)
    records.insert(0, "client_ids", [tuple(case_metadata["client_ids"])] * len(records))
    records.insert(0, "start_round", start_round)
    records.insert(0, "case_id", case_dir.name)
    records.insert(0, "codec_id", case_dir.parent.name)
    records["trace_identity_digest"] = content_hash((identity["trace"],), {})[:16]
    return records
