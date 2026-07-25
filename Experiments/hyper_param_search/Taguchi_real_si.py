"""Small reusable Taguchi API for three-level hyperparameter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


Level: TypeAlias = bool | int | float | str | None


@dataclass(frozen=True, slots=True)
class Factor:
    """One three-level control factor consumed by an L27 experiment."""

    name: str
    levels: tuple[Level, Level, Level]

    def __post_init__(self) -> None:
        assert self.name, "Taguchi factor names must not be empty."
        assert len(set(self.levels)) == 3, (
            f"Taguchi factor {self.name!r} must have three distinct levels."
        )


@dataclass(frozen=True, slots=True)
class Trial:
    """One L27 row expanded into factor names, level indices, and concrete values."""

    number: int
    level_indices: tuple[int, ...]
    values: tuple[tuple[str, Level], ...]

    def as_dict(self) -> dict[str, Level]:
        """Return the concrete factor values keyed by factor name."""
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class FactorEffect:
    """Mean smaller-is-better response at each level of one control factor."""

    factor: str
    levels: tuple[Level, Level, Level]
    mean_responses: tuple[float, float, float]
    selected_level: Level
    effect_range: float


def l27_array(factor_count: int) -> NDArray[np.int64]:
    """Return a strength-two L27 orthogonal array for at most thirteen factors."""
    assert 1 <= factor_count <= 13, "L27 supports between 1 and 13 three-level factors."
    run_coordinates = np.array(
        [(first, second, third) for first in range(3) for second in range(3) for third in range(3)],
        dtype=np.int64,
    )
    column_vectors = np.array(
        (
            *((1, second, third) for second in range(3) for third in range(3)),
            *((0, 1, third) for third in range(3)),
            (0, 0, 1),
        ),
        dtype=np.int64,
    )
    return (run_coordinates @ column_vectors[:factor_count].T) % 3


def build_trials(factors: tuple[Factor, ...]) -> tuple[Trial, ...]:
    """Expand factors through L27 and return deterministic numbered trials."""
    assert factors, "At least one Taguchi factor is required."
    names = tuple(factor.name for factor in factors)
    assert len(set(names)) == len(names), f"Taguchi factor names must be unique: {names}."
    return tuple(
        Trial(
            number=number,
            level_indices=tuple(int(index) for index in row),
            values=tuple(
                (factor.name, factor.levels[int(level_index)])
                for factor, level_index in zip(factors, row, strict=True)
            ),
        )
        for number, row in enumerate(l27_array(len(factors)), start=1)
    )


def main_effects(
    factors: tuple[Factor, ...],
    trials: tuple[Trial, ...],
    responses: dict[int, float],
) -> tuple[FactorEffect, ...]:
    """Estimate factor main effects and select the lowest-response level of each factor."""
    assert set(responses) == {trial.number for trial in trials}, (
        "Responses must contain exactly one value for every Taguchi trial."
    )
    effects: list[FactorEffect] = []
    for factor_index, factor in enumerate(factors):
        level_means = tuple(
            float(np.mean([
                responses[trial.number]
                for trial in trials
                if trial.level_indices[factor_index] == level_index
            ]))
            for level_index in range(3)
        )
        selected_index = int(np.argmin(level_means))
        effects.append(FactorEffect(
            factor=factor.name,
            levels=factor.levels,
            mean_responses=level_means,
            selected_level=factor.levels[selected_index],
            effect_range=max(level_means) - min(level_means),
        ))
    return tuple(effects)


def recommended_values(effects: tuple[FactorEffect, ...]) -> dict[str, Level]:
    """Return the smaller-is-better main-effect recommendation."""
    return {effect.factor: effect.selected_level for effect in effects}
