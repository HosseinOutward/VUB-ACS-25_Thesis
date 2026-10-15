"""Check zero-sensitive metrics and the decoder's history restoration contract."""

import numpy as np
import pytest
import torch

from Experiments.history_distortion.run import RoundNormalizedHistory, metrics, variants
from Experiments.history_distortion.histogram import Distribution
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.utils import normalization_params
from FL_code.cancer_protocol.history import CompressedHistoryVector, CompressedReconstructionHistory, float_prefix
from FL_code.cancer_protocol.wz_quantizer import SideInfoMatrix


def test_metrics_do_not_hide_small_values_or_zero_errors() -> None:
    result = metrics(np.array([0.0, 1e-9, 1.0]), np.array([1e-5, 0.0, 1.0]))
    assert result["mape_pct"] == 50.0
    assert result["wmape_pct"] == pytest.approx(100 * (1e-5 + 1e-9) / (1 + 1e-9))
    assert result["lost_nonzero"] == 1 and result["changed_zero"] == 1
    assert metrics(np.zeros(2), np.zeros(2))["mape_pct"] is None


def test_centering_error_reaches_decoder_rows() -> None:
    source = torch.tensor([-1.0, 0.0, 1e-9, 1.0])
    slices = [slice(0, 4)]
    history = CompressedHistoryVector(source, slices, torch.tensor([[1.0, 0.1]]))
    restored = history.materialize()
    matrix = SideInfoMatrix([history], 4, torch.tensor([[1.0, 0.0]]), slices, torch.device("cpu"))
    assert torch.equal(matrix[torch.tensor([2, 0, 3, 1])].flatten(), restored[[2, 0, 3, 1]])
    assert abs(float(restored[2])) > 1e-5
    assert restored[1] != 0


def test_scaling_prevents_float16_underflow_and_preserves_prefix_error() -> None:
    source = torch.tensor([-1e-9, 0.0, 1e-9, 2e-9])
    slices = [slice(0, 4)]
    results = {name: restored for name, restored, _, _ in variants(source, slices, CompressedReconstructionHistory(1, slices), 0, 0)}
    assert torch.count_nonzero(results["float16_raw"]) == 0
    assert torch.count_nonzero(results["float16_pow2"]) == 3
    assert torch.equal(results["prefix_raw"], float_prefix(source))
    assert torch.equal(results["prefix_pow2"], results["prefix_raw"])


def test_round_factors_refresh_on_submission_count_and_old_values_stay_decodable() -> None:
    """Check exact quantizer factors, first-item fitting, shared reuse, and retained inverse metadata."""
    source = torch.tensor([-1.0, 0.0, 0.01, 2.0, -0.03, 0.0, 0.01, 0.02])
    slices = [slice(0, 4), slice(4, 8)]
    history = RoundNormalizedHistory(3, slices)
    history.commit(source, CompressionRecord(0, 0, "test"), Access.TEMPORAL_TOO)
    first = history.view(Access.TEMPORAL_TOO, 0)[0][-1].value
    assert isinstance(first, CompressedHistoryVector)
    original = first.materialize()
    expected = torch.tensor([normalization_params(source[layer]) for layer in slices])
    assert torch.equal(first.normalization_factors, expected)
    for client in (1, 2):
        history.commit(source * 8 + 1, CompressionRecord(0, client, "test"), Access.TEMPORAL_TOO)
        stored = history.view(Access.TEMPORAL_TOO, client)[client][-1].value
        assert isinstance(stored, CompressedHistoryVector)
        assert torch.equal(stored.normalization_factors, expected)
    history.commit(source * 8 + 1, CompressionRecord(1, 0, "test"), Access.TEMPORAL_TOO)
    assert history.submissions == 4 and not torch.equal(history.factors, expected)
    assert torch.equal(first.materialize(), original)
    assert torch.equal(first.normalization_factors, expected)


def test_histogram_counts_zero_changes_without_inventing_percentage_errors() -> None:
    """Check affected-coordinate denominators and an exact zero's exclusion from the APE histogram."""
    distribution = Distribution(np.r_[0.0, np.geomspace(1e-6, 1e6, 97), np.inf])
    distribution.add(torch.tensor([[0.0, 1.0], [2.0, -4.0]]), torch.tensor([[0.01, 1.5], [2.0, -8.0]]))
    summary = distribution.summary()
    assert summary["coordinates"] == 4 and summary["nonzero_reference"] == 3
    assert summary["changed_pct_all"] == 75 and summary["changed_zero_pct_all"] == 25
    assert summary["ape_gt_1_pct_nonzero"] == pytest.approx(200 / 3)
    assert summary["ape_gt_100_pct_nonzero"] == 0
    assert distribution.bins.sum() == 3 and distribution.bins[0] == 1
