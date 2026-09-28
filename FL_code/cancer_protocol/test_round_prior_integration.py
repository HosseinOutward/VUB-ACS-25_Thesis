from types import SimpleNamespace
from typing import Any

import pytest
import torch

from . import prior_code as prior_code_module
from . import wz_quantizer as wz_quantizer_module
from .bin_coder.coded_rounds import OracleRansRoundCodec, TRansRoundCodec
from .prior_code import PriorCalculator, _prior_training_hash
from .brent_wz_models import EncoderDecoderLayeredRNN
from .round_codecs import Oracle_RoundCodec, T_RoundCodec
from .wz_quantizer import WZcfgQuant


def test_retrained_prior_is_initialized_from_quantizer_and_reused_by_frozen_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A training round passes its coding model to prior fitting and F reuses the result."""
    source_model = object()
    fitted_prior = object()
    calls: list[dict[str, object]] = []

    def fake_train(**kwargs: object) -> object:
        calls.append(kwargs)
        assert kwargs["source_model"] is source_model
        return fitted_prior

    class FakePriorCalculator:
        @staticmethod
        def compute_prior_from_network(*args: object, **kwargs: object) -> torch.Tensor:
            return torch.full((2, 3, 2), 0.5)

    monkeypatch.setattr(PriorCalculator, "make_trained_prior_model", fake_train)
    monkeypatch.setattr(
        PriorCalculator,
        "compute_code_lengths",
        staticmethod(lambda prior, bins, num_planes: torch.zeros((num_planes, bins.shape[1]))),
    )

    codec = Oracle_RoundCodec(None, "O")
    codec.c_cfg = WZcfgQuant(bins_per_plane=2, num_planes=2)
    codec._retrained_prior_model = None
    codec.quantizer = SimpleNamespace(
        coding_model=source_model,
        prior_calculator=FakePriorCalculator,
    )
    bins = torch.zeros((2, 3), dtype=torch.long)
    soft_codes = torch.zeros((2, 3, 2))
    side_info = torch.zeros((3, 1))

    first_lengths, first_model = codec.conditional_code_lengths(
        codec.quantizer, bins, soft_codes, side_info, codec.c_cfg)
    assert first_model is fitted_prior
    assert first_lengths.shape == (2, 3)
    assert len(calls) == 1

    codec.round_name_full = "F"
    second_lengths, second_model = codec.conditional_code_lengths(
        codec.quantizer, bins, soft_codes, side_info, codec.c_cfg)
    assert second_model is fitted_prior
    assert torch.equal(second_lengths, first_lengths)
    assert len(calls) == 1


def test_prior_copy_keeps_only_independent_conditional_parameters() -> None:
    """Prior fine-tuning copies conditional weights and releases coding/reconstruction modules."""
    source = EncoderDecoderLayeredRNN(
        input_dim=1,
        side_info_size=1,
        bins_per_plane=2,
        num_planes=2,
        layers=1,
        hidden_dim=4,
    )
    prior = source.copy_for_prior_retraining()

    assert prior.encoder is None
    assert prior.decoder is None
    assert prior.binners is None
    assert prior.reconstructors is None
    assert all(name.startswith("conditional") for name in prior.state_dict())
    prior.eval()
    priors = prior.get_priors(
        codes=[torch.nn.functional.one_hot(torch.zeros(3, dtype=torch.long), 2).float()] * 2,
        y=torch.zeros(3, 1),
    )
    assert len(priors) == 2
    assert all(probabilities.shape == (3, 2) for probabilities in priors)
    prior.conditionalRNN.input_layers[0].weight.data.add_(1)
    assert not torch.equal(
        prior.conditionalRNN.input_layers[0].weight,
        source.conditionalRNN.input_layers[0].weight,
    )


def test_prior_retraining_uses_short_effective_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prior fitting overrides epochs, learning rate, and tau without changing the base config."""
    config = WZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        hidden_dim=4,
        layers=1,
        lr=0.02,
        tau=0.9,
        prior_retrain_epochs=30,
    )
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2, layers=1, hidden_dim=4)
    observed: dict[str, object] = {}

    def fake_train(
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
        source_model: EncoderDecoderLayeredRNN,
        batch_size: int,
    ) -> EncoderDecoderLayeredRNN:
        observed.update(c_cfg=c_cfg, source_model=source_model)
        return source_model.copy_for_prior_retraining()

    monkeypatch.setattr(prior_code_module, "_train_prior_model", fake_train)
    bins = torch.zeros((2, 3), dtype=torch.long)
    soft_codes = torch.zeros((2, 3, 2))
    side_info = torch.zeros((3, 1))
    PriorCalculator.make_trained_prior_model(bins, soft_codes, side_info, config, source)

    effective = observed["c_cfg"]
    assert effective.train_epochs == 30
    assert effective.lr == pytest.approx(0.006)
    assert effective.tau == 0.5
    assert config.train_epochs != effective.train_epochs
    assert config.lr == pytest.approx(0.02)
    assert config.tau == pytest.approx(0.9)


def test_prior_retraining_repeats_start_from_the_same_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each prior-training repeat clones the untouched quantizer prior independently."""
    config = WZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        hidden_dim=4,
        layers=1,
        prior_train_repeats=2,
        prior_retrain_epochs=1,
    )
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2, layers=1, hidden_dim=4)
    initial_states: list[dict[str, torch.Tensor]] = []

    def fake_loop(
        loss_fn: Any,
        model: EncoderDecoderLayeredRNN,
        parameters: Any,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        initial_states.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return 0.0

    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", fake_loop)
    bins = torch.zeros((2, 3), dtype=torch.long)
    soft_codes = torch.zeros((2, 3, 2))
    side_info = torch.zeros((3, 1))
    prior = PriorCalculator.make_trained_prior_model(bins, soft_codes, side_info, config, source)

    expected = source.conditional_state_dict()
    assert len(initial_states) == 2
    assert all(
        all(torch.equal(state[name], expected[name]) for name in expected)
        for state in initial_states
    )
    assert all(torch.equal(source.state_dict()[name], value) for name, value in expected.items())
    assert all(name.startswith("conditional") for name in prior.state_dict())


def test_prior_retraining_rejects_incompatible_side_information_width() -> None:
    """A copied conditional prior cannot accept a wider or narrower SI tensor."""
    config = WZcfgQuant(bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1)
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2, layers=1, hidden_dim=4)
    bins = torch.zeros((2, 3), dtype=torch.long)
    soft_codes = torch.zeros((2, 3, 2))
    with pytest.raises(AssertionError, match="side-information width"):
        PriorCalculator.make_trained_prior_model(bins, soft_codes, torch.zeros(3, 2), config, source)


def test_scratch_prior_retraining_builds_for_full_side_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T's independent retrained prior is constructed for its wider server-side SI."""
    config = WZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        hidden_dim=4,
        layers=1,
        prior_train_repeats=1,
        prior_retrain_epochs=1,
    )
    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", lambda *args, **kwargs: 0.0)

    prior = PriorCalculator.make_trained_prior_model(
        torch.zeros((2, 3), dtype=torch.long),
        torch.zeros((2, 3, 2)),
        torch.zeros((3, 2)),
        config,
        source_model=None,
    )

    assert prior.conditionalRNN.input_dim == 4
    assert all(name.startswith("conditional") for name in prior.state_dict())


def test_t_retrained_prior_starts_from_scratch_while_oracle_copies_quantizer() -> None:
    """Only temporal quantizers need a new architecture for the server-wide prior."""
    assert T_RoundCodec.retrained_prior_from_scratch is True
    assert Oracle_RoundCodec.retrained_prior_from_scratch is False


def test_rans_temporal_and_oracle_codecs_use_an_out_of_band_oracle_prior() -> None:
    """TR and OR code symbols with the study prior, whose grant is outside payload bytes."""
    for codec_class in (TRansRoundCodec, OracleRansRoundCodec):
        codec = object.__new__(codec_class)
        retrained_prior, quantizer_prior = object(), object()
        codec._oracle_prior = retrained_prior
        codec.quantizer = SimpleNamespace(coding_model=quantizer_prior)
        provenance = codec_class.codec_provenance()
        assert codec_class.code_with_oracle_prior is True
        assert codec.coder_prior_model() is retrained_prior
        assert provenance["oracle_prior_granted"] is True
        assert provenance["coder_prior_si"] == "decoder_legal_exact_oracle_grant"
        assert provenance["prior_stage_rates_si"] == "decoder_legal_exact_oracle_grant"
        assert provenance["oracle_prior_grant_channel"] == "out_of_band_not_in_final_bytes"


def test_prior_retraining_rejects_incompatible_marginal_architecture() -> None:
    """Prior retraining cannot change a copied model between marginal and conditional input."""
    config = WZcfgQuant(bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1)
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2,
        layers=1, hidden_dim=4, marginal=True,
    )
    bins = torch.zeros((2, 3), dtype=torch.long)
    soft_codes = torch.zeros((2, 3, 2))

    with pytest.raises(AssertionError, match="marginal architecture"):
        PriorCalculator.make_trained_prior_model(
            bins, soft_codes, torch.zeros(3, 1), config, source)


def test_round_prior_retraining_rejects_marginal_quantizers_explicitly() -> None:
    """Round codecs reject the currently unsupported marginal-to-conditional prior transfer."""
    codec = T_RoundCodec(None, "T")
    config = WZcfgQuant(bins_per_plane=2, num_planes=2, marginal_loss=True)

    with pytest.raises(AssertionError, match="do not currently support marginal quantizers"):
        codec.conditional_code_lengths(
            SimpleNamespace(),
            torch.zeros((2, 3), dtype=torch.long),
            torch.zeros((2, 3, 2)),
            torch.zeros((3, 1)),
            config,
        )


def test_prior_parent_state_changes_only_the_prior_cache_key() -> None:
    """Parent priors distinguish retraining cache entries without invalidating quantizer caches."""
    config = WZcfgQuant(bins_per_plane=2, num_planes=2, hidden_dim=4, layers=1)
    source = EncoderDecoderLayeredRNN(
        input_dim=1, side_info_size=1, bins_per_plane=2, num_planes=2, layers=1, hidden_dim=4)
    first_state = source.conditional_state_dict()
    second_state = {name: value.clone() for name, value in first_state.items()}
    next(iter(second_state.values())).add_(1)
    common = {
        "bins_vec": torch.zeros((2, 3), dtype=torch.long),
        "soft_codes": torch.zeros((2, 3, 2)),
        "side_info": torch.zeros((3, 1)),
        "c_cfg": config.model_copy(update={"train_epochs": 30, "lr": config.lr * 0.3, "tau": 0.5}),
        "source_model": source,
        "batch_size": 3,
    }

    assert _prior_training_hash((), common | {"source_prior_state": first_state}) != (
        _prior_training_hash((), common | {"source_prior_state": second_state})
    )
    assert "prior_retrain_epochs" not in config.training_cache_payload()
