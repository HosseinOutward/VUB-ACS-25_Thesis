"""Check retry boundaries without running expensive quantizer or prior fits."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import torch

from Experiments.Protocol_test.sparse_replay import case_protocol, load_plan
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import (
    WZcfgQuant, WZQuantizerCancer, new_rnn_model, wz_model_training_loop,
)


@pytest.mark.parametrize("kind", ["quantizer", "prior"])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("quota,rejections,limit", [(3, 5, 8), (3, 6, 8), (2, 6, 8), (2, 7, 8), (3, 0, 8), (2, 2, 4)])
def test_training_attempt_budget(
    kind: Literal["quantizer", "prior"], warm: bool, quota: int, rejections: int,
    limit: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold and inherited fits accept late candidates, stop at quota, and enforce the total cap."""
    if kind == "quantizer" and not torch.cuda.is_available():
        pytest.skip("Quantizer candidate setup uses CUDA RNG")
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1, warm_training=warm)
    assert cfg.max_training_attempts == cfg.warm_retraining_config().max_training_attempts == 8
    cfg.max_training_attempts = limit
    calls: list[EncoderDecoderLayeredRNN] = []

    def train(
        loss_fn: object, model: EncoderDecoderLayeredRNN, parameters: Iterator[torch.nn.Parameter],
        *args: object, **kwargs: object,
    ) -> float:
        calls.append(model)
        with torch.no_grad():
            model.conditionalRNN.input_layers[0].weight.fill_(len(calls))
        return float(len(calls))

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(
        lambda *args: (1.2 if len(calls) <= rejections else 0.8, 1.0)))
    bins, si = torch.tensor([[0, 1, 0, 1]]), torch.zeros(4, 1)
    if kind == "quantizer":
        quantizer = WZQuantizerCancer(c_cfg=cfg, si_size=1)
        quantizer.vector_size = 4
        monkeypatch.setattr(quantizer, "encode_prepared", lambda *args: (bins, torch.zeros(1, 4, 2)))
        monkeypatch.setattr(quantizer, "decode_prepared", lambda *args: torch.zeros(4))
        monkeypatch.setattr(quantizer, "_candidate_is_usable", lambda *args: True)
        source = quantizer.coding_model
    else:
        source = new_rnn_model(1, 2, 1, False, 4, 1) if warm else None
    parent = source.state_dict() if warm else None

    def fit() -> EncoderDecoderLayeredRNN:
        if kind == "quantizer":
            return quantizer._train_finite_candidate(torch.zeros(4, 1), si, 1.0, parent, quota, "test")[0]
        return PriorCalculator.fit_prior_model(bins, torch.zeros(1, 4, 2), si, cfg, source, (parent,), quota).model

    if rejections + quota > limit:
        with pytest.raises(AssertionError, match=f"after {limit} attempts"):
            fit()
        assert len(calls) == limit
    else:
        result = fit()
        assert len(calls) == rejections + quota
        assert torch.all(result.conditionalRNN.input_layers[0].weight == rejections + 1)


def test_attempt_limit_is_recorded_in_protocol_identity() -> None:
    """Replay identities distinguish new attempt limits from historical training policies."""
    plan = load_plan(Path(__file__).with_name("plans") / "or_tr_svhn_30c_f9_warm_seed44_20261004.json")
    for case in plan.cases:
        assert case_protocol(plan, case).config_snapshot()["quantizer_defaults"]["max_training_attempts"] == 8


def test_validation_selection_restores_best_initial_checkpoint() -> None:
    """An opt-in heldout metric scores epoch zero and restores its winner without gradients."""
    if not torch.cuda.is_available():
        pytest.skip("The shared training loop is CUDA-only")
    cfg = WZcfgQuant(
        bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1,
        train_epochs=1, train_sample_size=4, lr=0.01, fused_optimizer=False,
    )
    model = new_rnn_model(1, 2, 1, False, 4, 1)
    parameter = model.conditionalRNN.input_layers[0].weight
    initial = parameter.detach().clone()
    validation_calls: list[tuple[int, bool, bool]] = []

    def loss(
        current: EncoderDecoderLayeredRNN, training_inputs: tuple[torch.Tensor, ...],
        side_info: torch.Tensor, epoch: int, config: WZcfgQuant,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        return -current.conditionalRNN.input_layers[0].weight.mean()

    def score(current: EncoderDecoderLayeredRNN, epoch: int) -> float:
        validation_calls.append((epoch, current.training, torch.is_grad_enabled()))
        return float(current.conditionalRNN.input_layers[0].weight.mean())

    result = wz_model_training_loop(
        loss, model, iter(model.parameters()), (torch.zeros(4, 1),), torch.zeros(4, 1),
        cfg, batch_size=4, selection_metric=score,
    )

    assert validation_calls == [(-1, False, False), (0, False, False)]
    assert model.training_selection_epoch == -1
    assert model.training_selection_score == pytest.approx(float(initial.mean()), rel=1e-5, abs=1e-7)
    assert result == pytest.approx(float(initial.mean()), rel=1e-5, abs=1e-7)
    assert torch.equal(model.conditionalRNN.input_layers[0].weight, initial.cpu())
    assert model.training
