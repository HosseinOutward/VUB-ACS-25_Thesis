"""Lightweight replay-plan contracts shared by launchers, monitors, and the numerical runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReplayCase(BaseModel):
    """A protocol window; the runner measures every client of the trace in each of its rounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    codec_id: str
    protocol: str = "wz_cancer"
    start_round: int = Field(ge=0)
    initial_si: Literal["raw", "nsplit"] = "raw"
    round_codecs: tuple[str, ...]

    @field_validator("case_id", "codec_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Keep case output beneath its experiment directory."""
        assert value not in {"", ".", ".."} and Path(value).name == value
        return value

    @field_validator("round_codecs", mode="before")
    @classmethod
    def expand_schedule(cls, value: str) -> tuple[str, ...]:
        """Expand a space-separated schedule whose tokens optionally end in `*positive_integer`."""
        schedule: list[str] = []
        for item in value.split():
            match = re.fullmatch(r"([^*]+)(?:\*([1-9][0-9]*))?", item)
            assert match is not None, (
                f"Invalid round_codecs item {item!r}; expected token or token*positive_integer."
            )
            token, repeats = match.groups()
            schedule += [token] * int(repeats or 1)
        return tuple(schedule)

    @model_validator(mode="after")
    def validate_window(self) -> ReplayCase:
        """Require an explicit train/freeze protocol window."""
        assert len(self.round_codecs) >= 2, "round_codecs must contain at least two rounds."
        assert self.round_codecs[0] != "F", "The first schedule token must train a codec, not 'F'."
        assert "F" in self.round_codecs, "round_codecs must contain at least one frozen round."
        return self


class ReplayPlan(BaseModel):
    """One raw FL trace shared by independent sparse protocol cases and their outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    description: str | None = None
    fl_config: Path
    trace_device: str = "cuda:0"
    dataset_revision: str
    output_root: Path
    nsplit_bins: int = Field(default=2, ge=2, le=256)
    cases: tuple[ReplayCase, ...]
    comparisons: tuple[tuple[str, str], ...] = ()
    seed: int = Field(default=43, ge=0)
    gpu_ids: tuple[int, ...] = ()
    max_parallel_cases: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_cases(self) -> ReplayPlan:
        """Require unique case identities, valid comparisons, and the plan's seed in every case."""
        assert self.name and self.cases
        identities = [(case.codec_id, case.case_id) for case in self.cases]
        assert len(set(identities)) == len(identities), "Cases must have unique (codec_id, case_id) identities."
        assert len(set(self.gpu_ids)) == len(self.gpu_ids) and all(gpu >= 0 for gpu in self.gpu_ids)
        assert self.max_parallel_cases <= max(1, len(self.gpu_ids)), (
            "max_parallel_cases cannot exceed the number of configured GPUs."
        )
        codecs = {case.codec_id for case in self.cases}
        for candidate, reference in self.comparisons:
            assert candidate != reference and {candidate, reference} <= codecs, "Invalid comparison."
        return self


def load_plan(path: Path) -> ReplayPlan:
    """Read one plan, resolving its paths relative to the JSON file."""
    payload = json.loads(path.read_text())
    for field in ("fl_config", "output_root"):
        payload[field] = (path.parent / payload[field]).resolve()
    return ReplayPlan.model_validate(payload)
