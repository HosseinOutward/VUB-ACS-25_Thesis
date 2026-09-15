"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols.

Structural tests for the reusable three-level Taguchi API."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from Experiments.hyper_param_search.Taguchi_real_si import (
    Factor,
    build_trials,
    l27_array,
    main_effects,
    recommended_values,
)
from Experiments.hyper_param_search.Taguchi_oracle_search import (
    compare_actual_polylines,
    quantizer_config,
    reference_curve,
)
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant


def test_l27_is_balanced_and_pairwise_orthogonal() -> None:
    """Every L27 column and pair of columns has the expected balanced counts."""
    array = l27_array(13)
    assert array.shape == (27, 13)
    for column in array.T:
        assert np.array_equal(np.bincount(column), np.full(3, 9))
    for left, right in combinations(array.T, 2):
        pairs = left * 3 + right
        assert np.array_equal(np.bincount(pairs), np.full(9, 3))


def test_main_effect_recommendation_uses_lowest_mean_response() -> None:
    """The API selects the smaller-is-better level independently per factor."""
    factors = (
        Factor("first", (10, 20, 30)),
        Factor("second", ("low", "middle", "high")),
    )
    trials = build_trials(factors)
    responses = {
        trial.number: float(
            (trial.as_dict()["first"] != 20) + 2 * (trial.as_dict()["second"] != "high")
        )
        for trial in trials
    }
    assert recommended_values(main_effects(factors, trials, responses)) == {
        "first": 20,
        "second": "high",
    }


def test_non_searched_parameters_keep_protocol_defaults() -> None:
    """Only searched values, stage count, slices, schedule, and cache policy differ from defaults."""
    values = {
        "reconst_ld": 900.0,
        "tau": 3.0,
        "lr": 9e-3,
        "train_epochs": 120,
        "train_sample_size": 600_000,
        "train_batch_size": 100_000,
        "lr_step_fraction": 0.25,
        "hidden_dim": 150,
        "layers": 4,
        "source_perturbation": "add",
    }
    config = quantizer_config(values, (slice(0, 10),))
    defaults = WZcfgQuant(bins_per_plane=3, num_planes=3)
    searched = set(values) - {"lr_step_fraction"}
    intentionally_resolved = searched | {
        "bins_per_plane", "num_planes", "training_match_bits",
        "norm_slices", "lr_step", "cache_dir",
    }
    assert {
        name: getattr(config, name)
        for name in type(config).model_fields
        if name not in intentionally_resolved
    } == {
        name: getattr(defaults, name)
        for name in type(defaults).model_fields
        if name not in intentionally_resolved
    }
    assert config.quantizer_train_repeats == config.prior_train_repeats == 2
    assert config.bins_per_plane == config.num_planes == 3
    assert config.training_match_bits == 26
    assert config.cache_dir is None


def test_actual_polyline_gap_is_zero_for_reference_and_tracks_rate_shift() -> None:
    """The new objective compares actual polylines and reports horizontal rate change."""
    rates, distortions = reference_curve((0,))
    identical = compare_actual_polylines(rates, distortions, rates, distortions)
    shifted = compare_actual_polylines(
        rates,
        distortions,
        tuple(rate - 0.02 for rate in rates),
        distortions,
    )
    assert identical.mean_rate_gap == 0
    assert identical.reference_overlap_fraction == 1
    assert np.isclose(shifted.mean_rate_gap, -0.02)
