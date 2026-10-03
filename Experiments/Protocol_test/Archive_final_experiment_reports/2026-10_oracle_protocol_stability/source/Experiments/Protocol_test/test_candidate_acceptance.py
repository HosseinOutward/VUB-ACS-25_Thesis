"""Check same-symbol marginal gating before cold/warm argmin and weighted historical sampling."""

from collections.abc import Iterator

import pytest
import torch

from Experiments.Protocol_test.experimental_codecs import OM_RoundCodec, TM_RoundCodec
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol
from FL_code.cancer_protocol.history import ReconstructionHistory
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer, _stacked_problem_sampler, new_rnn_model
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


def test_prior_gate_uses_weighted_paired_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance uses the declared mixture, not uniform pooling or a cross-round marginal."""
    model = new_rnn_model(1, 2, 1, False, 4, 1)
    bins = torch.tensor([[0, 1, 0, 0, 1, 1]])
    si = torch.tensor([[0.], [0.], [1.], [1.], [2.], [2.]])
    seen: list[int] = []

    def prior(model: EncoderDecoderLayeredRNN, symbols: torch.Tensor, paired_si: torch.Tensor) -> torch.Tensor:
        block = int(paired_si[0, 0])
        seen.append(block)
        return torch.tensor([[[.5, .5]], [[.8, .2]], [[.1, .9]]])[block:block + 1].expand(1, 2, 2)

    monkeypatch.setattr(PriorCalculator, "compute_prior_from_network", staticmethod(prior))
    rate, marginal = PriorCalculator.candidate_rates(model, bins, si, (.5, .3, .2))
    assert seen == [0, 1, 2]
    assert rate == pytest.approx(.5 + .3 * -torch.log2(torch.tensor(.8)).item()
                                 + .2 * -torch.log2(torch.tensor(.9)).item())
    assert marginal == .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Weighted prior sampling uses CUDA")
def test_prior_training_samples_declared_round_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scratch and fine-tuned prior lineages receive the same 50/30/20 sampler."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1)
    counts: list[torch.Tensor] = []

    def train(*args: object, **kwargs: object) -> float:
        rows = kwargs["index_sampler"](100_000)
        counts.append(torch.bincount(rows // 4, minlength=3).cpu() / rows.numel())
        return 1.

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda *args: (.8, 1.)))
    for model in (None, new_rnn_model(1, 2, 1, False, 4, 1)):
        PriorCalculator.fit_prior_model(torch.tensor([[0, 1] * 6]), torch.zeros(1, 12, 2),
            torch.zeros(12, 1), cfg, model, (None,), 1, problem_weights=(.5, .3, .2))
    assert len(counts) == 2
    for count in counts:
        torch.testing.assert_close(count, torch.tensor([.5, .3, .2]), atol=.01, rtol=0)


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
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si, weights=None: next(rates)))
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
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si, weights=None: next(rates)))
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
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si, weights=None: (1.2, 1.)))
    with pytest.raises(AssertionError, match="prior rate 1.2 exceeds marginal 1"):
        if kind == "prior":
            PriorCalculator.fit_prior_model(torch.tensor([[0, 1]]), torch.zeros(1, 2, 2), torch.zeros(2, 1), cfg, None, (None,), 1)
        else:
            quantizer = WZQuantizerCancer(c_cfg=cfg, si_size=1)
            quantizer.vector_size = 2
            monkeypatch.setattr(quantizer, "encode_prepared", lambda x, batch: (torch.tensor([[0, 1]]), torch.zeros(1, 2, 2)))
            monkeypatch.setattr(quantizer, "decode_prepared", lambda bins, si, batch: torch.zeros(2))
            quantizer._train_finite_candidate(torch.zeros(2, 1), torch.zeros(2, 1), 1., None, 1, "test")


@pytest.mark.parametrize("codec_class", [OM_RoundCodec, TM_RoundCodec])
def test_two_previous_targets_match_their_own_si(codec_class: type[OM_RoundCodec | TM_RoundCodec]) -> None:
    """Default lag-1/lag-2 targets keep capture-time SI and the declared 50/30/20 shares."""
    protocol = create_protocol("wz_cancer_multi")
    protocol.set_schedule(7, (codec_class.round_name + "|bins_per_plane=2|num_planes=3", "F"))
    protocol._recons_history = ReconstructionHistory(5)
    for round_id in range(7):
        for client in range(2):
            raw = torch.tensor([float(100 * client + round_id)])
            protocol.remember_target(raw, round_id, client)
            protocol._recons_history.commit(raw + 1000, CompressionRecord(round_id, client, "I"), Access.TEMPORAL_TOO)
    codec = codec_class({"bins_per_plane": 2, "num_planes": 3, "sd_slices": None}, codec_class.round_name)
    protocol.training_problem(codec, 7, 1)
    assert [target.item() for target, _ in codec.extra_training_problems()] == [106, 105]
    for lag, (_, si) in zip((1, 2), codec.extra_training_problems(), strict=True):
        temporal = [float(1100 + r) for r in range(7 - lag - 5, 7 - lag)]
        server = [float(1000 + r) for r in range(7 - lag - 4, 8 - lag)]
        expected = temporal if codec_class is TM_RoundCodec else server + temporal
        assert [value.materialize().item() for value in si] == expected
    assert codec.training_problem_weights() == (.5, .3, .2)
    assert protocol.history_seed_depth == 7
    snapshot = protocol.config_snapshot()
    assert snapshot["multi_round_training"]["train_weights"] == (.5, .3, .2)
    assert "prior_le_same_symbol_empirical_marginal" in snapshot["candidate_acceptance"]


def test_sampling_uses_503020_shares() -> None:
    """The existing weighted row sampler actually draws the requested three-block mixture."""
    torch.manual_seed(43)
    sampler = _stacked_problem_sampler((.5, .3, .2), 128, torch.device("cpu"))
    counts = torch.bincount(sampler(100_000) // 128, minlength=3).float() / 100_000
    torch.testing.assert_close(counts, torch.tensor([.5, .3, .2]), atol=.005, rtol=0)
