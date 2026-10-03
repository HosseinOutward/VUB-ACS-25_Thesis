"""Check same-symbol marginal gating before cold/warm quantizer and prior selection."""

from collections.abc import Iterator

import pytest
import torch

from FL_code.FL_core.codec import create_protocol
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer, new_rnn_model
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN


def test_candidate_rates_use_same_hard_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong prior loses to the empirical marginal; equality is acceptable."""
    model = new_rnn_model(1, 2, 1, False, 4, 1)
    bins = torch.tensor([[0, 0, 0, 1]])
    si = torch.zeros(4, 1)
    bad = torch.tensor([[[.25, .75]]]).expand(1, 4, 2)
    monkeypatch.setattr(PriorCalculator, "compute_prior_from_network", staticmethod(lambda model, bins, si: bad))
    prior, marginal = PriorCalculator.candidate_rates(model, bins, si)
    assert prior > marginal
    assert marginal == pytest.approx(0.8112781)
    good = PriorCalculator.compute_marginal_prior(bins, 2, 1)
    monkeypatch.setattr(PriorCalculator, "compute_prior_from_network", staticmethod(lambda model, bins, si: good))
    prior, marginal = PriorCalculator.candidate_rates(model, bins, si)
    assert prior == marginal


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantizer candidate setup uses CUDA RNG")
def test_quantizer_argmin_excludes_bad_low_loss(warm: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lowest-loss rejected model cannot beat the two accepted cold/warm candidates."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1, warm_training=warm)
    quantizer = WZQuantizerCancer(c_cfg=cfg, si_size=1)
    quantizer.vector_size = 4
    losses = iter([.01, 5., 3.])
    rates = iter([(1.2, 1.), (.8, 1.), (.9, 1.)])
    calls: list[EncoderDecoderLayeredRNN] = []

    def train(loss_fn: object, model: EncoderDecoderLayeredRNN, parameters: Iterator[torch.nn.Parameter], *args: object, **kwargs: object) -> float:
        calls.append(model)
        return next(losses)

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(quantizer, "encode_prepared", lambda x, batch: (torch.tensor([[0, 1, 0, 1]]), torch.zeros(1, 4, 2)))
    monkeypatch.setattr(quantizer, "decode_prepared", lambda bins, si, batch: torch.zeros(4))
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si: next(rates)))
    parent = quantizer.coding_model.state_dict() if warm else None
    model, loss = quantizer._train_finite_candidate(torch.zeros(4, 1), torch.zeros(4, 1), 1., parent, 2, "test")
    assert loss == 3 and model is calls[2] and len(calls) == 3


@pytest.mark.parametrize("warm", [False, True])
def test_prior_argmin_excludes_bad_low_loss(warm: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both scratch and inherited priors reject low-loss candidates whose code length is worse."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1)
    source = new_rnn_model(1, 2, 1, False, 4, 1) if warm else None
    state = source.copy_for_prior_retraining().state_dict() if source is not None else None
    losses = iter([.01, 5., 3.])
    rates = iter([(1.2, 1.), (.8, 1.), (.9, 1.)])
    calls: list[EncoderDecoderLayeredRNN] = []

    def train(loss_fn: object, model: EncoderDecoderLayeredRNN, parameters: Iterator[torch.nn.Parameter], *args: object, **kwargs: object) -> float:
        calls.append(model)
        with torch.no_grad():
            model.conditionalRNN.input_layers[0].weight.fill_(len(calls))
        return next(losses)

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si: next(rates)))
    result = PriorCalculator.fit_prior_model(torch.tensor([[0, 1, 0, 1]]), torch.zeros(1, 4, 2), torch.zeros(4, 1),
                                             cfg, source, (state,), 2)
    assert len(calls) == 3
    assert torch.all(result.model.conditionalRNN.input_layers[0].weight == 3)


@pytest.mark.parametrize("kind", ["quantizer", "prior"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantizer candidate setup uses CUDA RNG")
def test_all_candidates_rejected_fails_clearly(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exhausted retry budget reports the offending rates instead of returning a bad model."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1)
    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", lambda *args, **kwargs: .01)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si: (1.2, 1.)))
    with pytest.raises(AssertionError, match="prior rate 1.2 exceeds marginal 1"):
        if kind == "prior":
            PriorCalculator.fit_prior_model(torch.tensor([[0, 1]]), torch.zeros(1, 2, 2), torch.zeros(2, 1), cfg, None, (None,), 1)
        else:
            quantizer = WZQuantizerCancer(c_cfg=cfg, si_size=1)
            quantizer.vector_size = 2
            monkeypatch.setattr(quantizer, "encode_prepared", lambda x, batch: (torch.tensor([[0, 1]]), torch.zeros(1, 2, 2)))
            monkeypatch.setattr(quantizer, "decode_prepared", lambda bins, si, batch: torch.zeros(2))
            quantizer._train_finite_candidate(torch.zeros(2, 1), torch.zeros(2, 1), 1., None, 1, "test")


@pytest.mark.parametrize("name", ["wz_cancer", "wz_cancer_warm"])
def test_acceptance_policy_is_part_of_replay_identity(name: str) -> None:
    """Cold and warm identities prevent reuse of records produced without marginal gating."""
    snapshot = create_protocol(name).config_snapshot()
    assert snapshot["candidate_acceptance"] == "finite_usable_and_total_hard_prior_le_same_symbol_empirical_marginal_v1"
