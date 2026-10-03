"""Protect exact SI embedding and experimental warm-row validation."""

import torch
import pytest

from Experiments.Protocol_test.records import _matches_schedule
from Experiments.Protocol_test.training_diagnostic import embed_temporal, evaluate, fit_prior, fit_quantizer
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, new_rnn_model
from FL_code.cancer_protocol.prior_code import PriorCalculator


def test_temporal_embedding_is_exact() -> None:
    """Extra oracle SI initially changes neither symbols, reconstruction, nor priors."""
    torch.manual_seed(43)
    temporal = new_rnn_model(3, 2, 5, False, 128, 4).eval()
    columns = torch.arange(55, 60)
    oracle = embed_temporal(temporal, columns, 150)
    x, si = torch.randn(32, 1), torch.randn(32, 150)
    with torch.no_grad():
        left, right = temporal(x, si[:, columns]), oracle(x, si)
    for temporal_values, oracle_values in zip(left, right, strict=True):
        for temporal_value, oracle_value in zip(temporal_values, oracle_values, strict=True):
            torch.testing.assert_close(temporal_value, oracle_value, atol=2e-6, rtol=2e-6)


def test_multi_round_warm_suffix_is_valid() -> None:
    """Multi-round warm rows satisfy the same scheduled token contract as ordinary warm rows."""
    metadata = {"protocol": "wz_cancer_multi_warm|train_offsets=(1,)"}
    assert _matches_schedule((6, 5, "OMR|warm_training"), (6, 5, "OMR"), metadata)
    assert not _matches_schedule((7, 5, "F|warm_training"), (7, 5, "F"), metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Diagnostic fitting requires CUDA")
def test_diagnostic_joint_multi_prior_and_both_coders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the final-encoder historical fit and both SI coders on a tiny real training loop."""
    torch.manual_seed(43)
    # This two-epoch fixture checks data/coder flow, not convergence of a fitted prior.
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si, weights=None: (1.0, 2.0)))
    problems = [{"x": torch.randn(32, 1), "si": torch.randn(32, 150),
                 "columns": torch.arange(25, 30), "factors": torch.tensor([[1., 0.]]).repeat(32, 1)}
                for _ in range(2)]
    for problem in problems:
        problem["raw"] = problem["x"].flatten()
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3, train_epochs=2, prior_retrain_epochs=1,
                     train_batch_size=32, quantizer_train_repeats=1, prior_train_repeats=1)
    joint, loss = fit_quantizer(cfg, problems, True, 43)
    assert loss == loss
    priors = {source: fit_prior(cfg, joint, problems, source == "temporal", True, 1043)
              for source in ("temporal", "full")}
    rows = evaluate(joint, problems[0], True, priors, False)
    assert len(rows) == 3
    assert all(row["temporal_prior"] > 0 and row["full_prior"] > 0 for row in rows)
    assert all(row["temporal_rans"] > 0 and row["full_rans"] > 0 for row in rows)
