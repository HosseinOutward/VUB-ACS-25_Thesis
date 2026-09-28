from types import SimpleNamespace
from typing import Any

import pytest
import torch

from FL_code.FL_core.codec import Access

from . import wz_quantizer as wz_quantizer_module
from .bin_coder.coded_rounds import OracleRansRoundCodec, TRansRoundCodec
from .brent_wz_models import EncoderDecoderLayeredRNN
from .prior_code import PriorCalculator
from .round_codecs import Oracle_RoundCodec, ServerSIPriorMixin, T_RoundCodec
from .wz_quantizer import WZcfgQuant


def _model(side_info_size: int = 1) -> EncoderDecoderLayeredRNN:
    return EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=side_info_size, bins_per_plane=2,
        num_planes=2, layers=1, hidden_dim=4,
    )


def _problem(side_info_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((2, 3), dtype=torch.long),
        torch.zeros((2, 3, 2)),
        torch.zeros((3, side_info_size)),
    )


def test_prior_copy_contains_only_conditional_parameters() -> None:
    """A prior candidate releases the unrelated encoder and reconstruction modules."""
    prior = _model().copy_for_prior_retraining()
    assert prior.encoder is prior.decoder is prior.binners is prior.reconstructors is None
    assert all(name.startswith("conditional") for name in prior.state_dict())


def test_native_finetuning_repeats_from_one_state_and_mutates_the_actual_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent candidates share one parent and the winner is installed in place."""
    config = WZcfgQuant(
        bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1,
        prior_train_repeats=2, prior_retrain_epochs=1,
    )
    source = _model()
    initial = source.conditional_state_dict()
    encoder = {name: value.clone() for name, value in source.encoder_state_dict().items()}
    starts: list[dict[str, torch.Tensor]] = []
    losses = iter((2.0, 1.0))

    def fake_loop(
        loss_fn: Any, model: EncoderDecoderLayeredRNN, parameters: Any,
        *args: Any, **kwargs: Any,
    ) -> float:
        starts.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        with torch.no_grad():
            next(model.parameters()).add_(len(starts))
        return next(losses)

    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", fake_loop)
    bins, soft_codes, side_info = _problem()
    result = PriorCalculator.fit_prior_model(
        bins, soft_codes, side_info, config, source, (initial,), 2,
    )

    assert result.model is source
    assert len(starts) == 2
    assert all(all(torch.equal(state[name], initial[name]) for name in initial) for state in starts)
    assert any(not torch.equal(source.state_dict()[name], value) for name, value in initial.items())
    assert all(torch.equal(source.state_dict()[name], value) for name, value in encoder.items())


def test_prior_fitting_uses_stabilized_warm_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed-symbol fitting reduces LR and temperature and uses the prior epoch budget."""
    config = WZcfgQuant(
        bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1,
        lr=0.02, tau=0.9, prior_retrain_epochs=30, prior_train_repeats=1,
    )
    observed: list[WZcfgQuant] = []

    def fake_loop(*args: Any, **kwargs: Any) -> float:
        observed.append(args[5])
        return 0.0

    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", fake_loop)
    bins, soft_codes, side_info = _problem()
    PriorCalculator.make_trained_prior_model(bins, soft_codes, side_info, config, _model())
    assert observed[0].train_epochs == 30
    assert observed[0].lr == pytest.approx(0.006)
    assert observed[0].tau == pytest.approx(0.5)


def test_native_prior_rejects_incompatible_side_information_width() -> None:
    """Native fine-tuning cannot silently reinterpret a temporal model as a server model."""
    bins, soft_codes, side_info = _problem(2)
    source = _model(1)
    with pytest.raises(AssertionError, match="side-information width"):
        PriorCalculator.fit_prior_model(
            bins, soft_codes, side_info,
            WZcfgQuant(bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1),
            source, (source.conditional_state_dict(),), 1,
        )


def test_explicit_cold_server_si_prior_uses_full_si_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """A protocol-requested cold lineage constructs a prior for the supplied server SI."""
    config = WZcfgQuant(
        bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1,
        prior_train_repeats=1, prior_retrain_epochs=1,
    )
    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", lambda *args, **kwargs: 0.0)
    bins, soft_codes, side_info = _problem(2)
    result = PriorCalculator.fit_prior_model(
        bins, soft_codes, side_info, config, None, (None,), 1,
    )
    assert result.model.conditionalRNN.input_dim == 4
    assert all(name.startswith("conditional") for name in result.model.state_dict())


def test_history_access_drives_native_prior_identity() -> None:
    """T is the temporal exception; ordinary WZ/O codecs use full server history."""
    assert T_RoundCodec.history_access is Access.TEMPORAL_TOO
    assert Oracle_RoundCodec.history_access is Access.SERVER_ONLY
    assert issubclass(T_RoundCodec, ServerSIPriorMixin)
    assert not issubclass(Oracle_RoundCodec, ServerSIPriorMixin)
    assert T_RoundCodec.codec_provenance()["native_prior_si"] == "temporal_too"
    assert Oracle_RoundCodec.codec_provenance()["native_prior_si"] == "server_only"


def test_rans_transport_access_is_explicit() -> None:
    """TR transports its temporal report while OR transports its server-SI report."""
    assert TRansRoundCodec.transport_prior_access is Access.TEMPORAL_TOO
    assert OracleRansRoundCodec.transport_prior_access is Access.SERVER_ONLY
    assert TRansRoundCodec.codec_provenance()["transport_prior_si"] == "temporal_too"
    assert OracleRansRoundCodec.codec_provenance()["transport_prior_si"] == "server_only"


def test_prior_finetuning_rejects_marginal_architecture() -> None:
    """Conditional fixed-symbol fitting cannot mutate a marginal quantizer architecture."""
    bins, soft_codes, side_info = _problem()
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2,
        layers=1, hidden_dim=4, marginal=True,
    )
    with pytest.raises(AssertionError, match="marginal architecture"):
        PriorCalculator.fit_prior_model(
            bins, soft_codes, side_info,
            WZcfgQuant(bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1),
            source, (source.conditional_state_dict(),), 1,
        )


def test_server_si_prior_mixin_requires_protocol_plan() -> None:
    """A trained T codec cannot silently choose a cold server-SI-prior initialization."""
    codec = object.__new__(T_RoundCodec)
    codec.round_name_full = "T"
    codec._server_si_prior_training_plan = None
    codec._server_si_prior_side_information = [object()]
    codec.quantizer = SimpleNamespace(normalize_side_info_tensor=lambda *args: torch.zeros((3, 1)))
    codec.c_cfg = WZcfgQuant(bins_per_plane=2, num_planes=2)
    with pytest.raises(AssertionError, match="protocol server-SI-prior plan"):
        codec.fit_additional_priors(
            torch.zeros((2, 3), dtype=torch.long), torch.zeros((2, 3, 2)),
            SimpleNamespace(), SimpleNamespace(),
        )
