"""Small plan contract for a fixed-history quantizer training diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Plan(BaseModel):
    """Fixed decoder history and training phases consumed by the runner and supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    fl_config: Path
    dataset_revision: str
    output_root: Path
    trace_device: str = "cuda:0"
    gpu_id: int = Field(default=1, ge=0)
    seed: int = Field(default=43, ge=0)
    target_round: int = Field(default=7, ge=2)
    cases: tuple[int, ...] = (0, 1, 2, 3, 4)
    phases: tuple[Literal["initial_joint", "prior", "quantizer", "joint"], ...] = (
        "initial_joint", "prior", "quantizer", "prior", "joint", "prior",
    )
    quantizer: dict[str, Any] = Field(default_factory=lambda: {
        "bins_per_plane": 2, "num_planes": 3, "training_log_every_epochs": 10,
    })
    min_available_ram_gib: float = Field(default=12, gt=0)
    min_free_gpu_gib: float = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_schedule(self) -> Plan:
        """Keep this diagnostic on all five clients and one initial fit followed by continuations."""
        assert self.cases == (0, 1, 2, 3, 4), "This experiment requires exactly clients 0–4."
        assert self.phases and self.phases[0] == "initial_joint"
        assert "initial_joint" not in self.phases[1:]
        assert "norm_slices" not in self.quantizer, "Model normalization slices come from the FL model."
        return self


def load_plan(path: Path) -> Plan:
    """Read the experiment plan, resolving filesystem paths relative to its JSON file."""
    payload = json.loads(path.read_text())
    for field in ("fl_config", "output_root"):
        payload[field] = (path.parent / payload[field]).resolve()
    return Plan.model_validate(payload)
