"""Protect raw-space aggregation and common-span seed ranking semantics."""

import numpy as np
import pytest

from Experiments.Protocol_test.five_seed_tables import mean_points, mean_rate


def test_mean_points_averages_raw_error_before_db() -> None:
    """The aggregate curve uses arithmetic raw WMAPE and cumulative rate means."""
    records = [dict(quantizer_prior_stage_rates=[.2, .3, .4], stage_wmapes=[80., 60., 40.], marginal_rate=1.2),
               dict(quantizer_prior_stage_rates=[.4, .1, .2], stage_wmapes=[60., 40., 20.], marginal_rate=1.6)]
    point = mean_points(records)
    np.testing.assert_allclose(point["quantizer_prior_stage_rates"], [.3, .5, .8])
    np.testing.assert_allclose(point["stage_wmapes"], [70., 50., 30.])
    np.testing.assert_allclose(point["wmape_db"], 20 * np.log10(np.array([.7, .5, .3])))
    assert point["server_si_rans_stage_rates"] is None
    assert point["marginal_rate"] == pytest.approx(1.4)


def test_mean_rate_integrates_internal_stage_knot() -> None:
    """A stage inside the common span changes its integrated score despite equal endpoints."""
    point = dict(wmape_db=[-2., -4., -8.], server_si_rans_stage_rates=[.2, .6, .8])
    assert mean_rate(point, -6., -3.) == pytest.approx(.6)


def test_mean_rate_rejects_extrapolation() -> None:
    """Selection cannot score a codec outside its measured distortion range."""
    point = dict(wmape_db=[-2., -4., -8.], server_si_rans_stage_rates=[.2, .6, .8])
    with pytest.raises(AssertionError, match="extrapolate"):
        mean_rate(point, -9., -3.)
