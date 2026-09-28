import numpy as np

from Experiments.Protocol_test.rd_metrics import mean_wmape_db


def test_mean_wmape_is_taken_before_db_conversion() -> None:
    """Mean curves must preserve the arithmetic mean in linear WMAPE space."""
    observations = np.array([[10.0, 40.0], [30.0, 80.0]])

    result = mean_wmape_db(observations)

    np.testing.assert_allclose(result, 20 * np.log10(np.array([20.0, 60.0]) / 100))
    assert not np.allclose(result, np.mean(20 * np.log10(observations / 100), axis=0))
