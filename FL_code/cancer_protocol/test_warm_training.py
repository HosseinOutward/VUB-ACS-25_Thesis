from __future__ import annotations

from pathlib import Path

import pytest
import torch

from FL_code.FL_core.cache import model_training_hash
from FL_code.FL_core.codec import create_protocol
from FL_code.run_fl import FLConfig

from . import wz_quantizer as wz_quantizer_module
from .cancer_protocol import WarmWZCancerProtocol, WZCancerProtocol
from .history import HistoryVector
from .round_codecs import Oracle_RoundCodec, T_RoundCodec
from .wz_quantizer import WZQuantizerCancer, WZcfgQuant


def _options(*, warm_training: bool = False) -> dict[str, object]:
    """Return a small quantizer configuration suitable for unit tests."""
    return {
        "bins_per_plane": 2,
        "num_planes": 2,
        "hidden_dim": 4,
        "layers": 1,
        "fused_optimizer": False,
        "warm_training": warm_training,
    }


def _config(*, warm_training: bool = False) -> WZcfgQuant:
    return WZcfgQuant(**_options(warm_training=warm_training))


def _codec(codec_class: type[T_RoundCodec | Oracle_RoundCodec], *, warm_training: bool) -> T_RoundCodec | Oracle_RoundCodec:
    codec = codec_class(
        _options(warm_training=warm_training) | {"sd_slices": None},
        codec_class.round_name,
    )
    codec.quantizer = WZQuantizerCancer(codec.c_cfg, si_size=1)
    return codec


def test_warm_protocol_and_three_cold_attempts_are_the_production_defaults() -> None:
    """Default FL uses the pool protocol while explicit cold training keeps its legacy name."""
    config = _config()

    assert FLConfig().protocol == WarmWZCancerProtocol.protocol_name == "wz_cancer_warm"
    assert isinstance(create_protocol(FLConfig().protocol), WarmWZCancerProtocol)
    assert isinstance(create_protocol("wz_cancer"), WZCancerProtocol)
    assert config.quantizer_train_repeats == config.prior_train_repeats == 3
    assert config.warm_retrain_repeats == 2
    assert config.training_cache_payload()["quantizer_train_repeats"] == 3
    assert config.training_cache_payload()["prior_train_repeats"] == 3
    historical_payload = config.model_copy(
        update={"quantizer_train_repeats": 2, "prior_train_repeats": 2}
    ).training_cache_payload()
    assert "quantizer_train_repeats" not in historical_payload
    assert historical_payload["prior_train_repeats"] == 2


def test_warm_start_state_is_loaded_once_and_copied() -> None:
    """A warm quantizer retains one independent copy of its initial parameters."""
    quantizer = WZQuantizerCancer(_config(warm_training=True), si_size=1)
    state = {name: value.detach().clone() for name, value in quantizer.coding_model.state_dict().items()}

    quantizer.load_warm_start_state(state)
    state[next(iter(state))].add_(1)

    assert len(quantizer._warm_start_states) == 1
    assert torch.equal(
        quantizer._warm_start_states[0][next(iter(state))],
        quantizer.coding_model.state_dict()[next(iter(state))],
    )
    assert quantizer._warm_start_states[0][next(iter(state))].data_ptr() != state[next(iter(state))].data_ptr()
    with pytest.raises(AssertionError, match="only be loaded once"):
        quantizer.load_warm_start_state(state)


def test_warm_training_requires_loaded_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm training refuses to start until the protocol supplies initial parameters."""
    quantizer = WZQuantizerCancer(_config(warm_training=True), si_size=1)
    monkeypatch.setattr(quantizer, "_train_model_uncached", lambda *args, **kwargs: None)

    with pytest.raises(AssertionError, match="requires load_warm_start_states"):
        quantizer.train_model(torch.arange(8, dtype=torch.float32), [HistoryVector(torch.arange(8.0))])


def test_warm_training_reduces_effective_epochs_lr_and_tau() -> None:
    """Warm fitting uses one quarter of the epochs and lower rate and temperature."""
    config = _config(warm_training=True)
    quantizer = WZQuantizerCancer(config, si_size=1)
    effective = quantizer.training_config()

    assert effective.train_epochs == max(1, round(config.train_epochs * 0.25))
    assert effective.lr == config.lr * 0.3
    assert effective.tau == 0.5
    assert config.train_epochs == WZcfgQuant.model_fields["train_epochs"].default
    assert config.lr == WZcfgQuant.model_fields["lr"].default
    assert config.tau == WZcfgQuant.model_fields["tau"].default
    assert config.training_cache_payload()["warm_training"] is True
    assert config.training_cache_payload()["warm_retrain_repeats"] == 2
    assert "warm_training" not in config.model_copy(update={"warm_training": False}).training_cache_payload()
    assert "warm_retrain_repeats" not in config.model_copy(
        update={"warm_training": False}
    ).training_cache_payload()


def test_warm_training_epoch_reduction_has_a_one_epoch_minimum() -> None:
    """Very short base fits still produce a valid warm-training configuration."""
    config = _config(warm_training=True).model_copy(update={"train_epochs": 1})

    assert config.warm_retraining_config().train_epochs == 1


def test_warm_parent_state_is_part_of_training_cache_key() -> None:
    """Different supplied parent states cannot reuse a warm-training cache entry."""
    config = _config(warm_training=True)
    quantizer = WZQuantizerCancer(config, si_size=1)
    first_state = {name: value.detach().clone() for name, value in quantizer.coding_model.state_dict().items()}
    second_state = {name: value.clone() for name, value in first_state.items()}
    next(iter(second_state.values())).add_(1)
    common = {
        "quantizer_class": WZQuantizerCancer,
        "c_cfg": config,
        "x_raw": torch.arange(8, dtype=torch.float32),
        "side_information": (HistoryVector(torch.arange(8, dtype=torch.float32)),),
    }

    assert model_training_hash((), common | {"warm_start_states": (first_state, second_state)}) != model_training_hash(
        (), common | {"warm_start_states": (second_state, first_state)}
    )


def _populate_pool(protocol: WarmWZCancerProtocol, si_size: int = 1) -> None:
    """Install five distinguishable compatible CPU states in the shared protocol pool."""
    states = []
    for value in range(protocol.pool_size):
        quantizer = WZQuantizerCancer(_config(), si_size=si_size)
        with torch.no_grad():
            next(quantizer.coding_model.parameters()).fill_(value)
        states.append({
            name: tensor.detach().clone()
            for name, tensor in quantizer.coding_model.state_dict().items()
        })
    protocol.quantizer_pool = states
    protocol.quantizer_pool_si_size = si_size


def test_warm_protocol_selects_two_shared_pool_parents_for_any_client() -> None:
    """Warm selection uses one ordered pool regardless of the current codec or client."""
    protocol = WarmWZCancerProtocol()
    _populate_pool(protocol)

    for codec_class in (T_RoundCodec, Oracle_RoundCodec):
        for client_id in (0, 1):
            codec = _codec(codec_class, warm_training=True)
            states = protocol.quantizer_initial_states(codec, client_id, si_size=1)
            assert codec.warm_parent_slots == (0, 1)
            assert len(states) == 2
            parameter_name = next(iter(states[0]))
            assert [float(state[parameter_name].flatten()[0]) for state in states] == [0, 1]


def test_protocol_uses_cold_training_until_shared_pool_is_full_then_asserts_si() -> None:
    """Pool fullness alone switches training mode; warm entry only asserts prepared SI width."""
    protocol = WarmWZCancerProtocol()
    assert protocol.quantizer_config_overrides(5, 7) == {}

    _populate_pool(protocol)
    assert protocol.quantizer_config_overrides(5, 7) == {
        "warm_training": True, "warm_retrain_repeats": 2}
    warm_codec = _codec(T_RoundCodec, warm_training=True)
    with pytest.raises(AssertionError, match="expected 1 side-information vectors, got 2"):
        protocol.quantizer_initial_states(warm_codec, 0, si_size=2)


def test_first_five_completed_training_calls_populate_one_shared_pool() -> None:
    """Base states come from arbitrary clients and rounds, then the next call becomes warm."""
    protocol = WarmWZCancerProtocol()
    calls = ((0, 5), (1, 5), (0, 10), (2, 10), (1, 15))
    for expected_size, (client_id, round_id) in enumerate(calls, start=1):
        codec = _codec(Oracle_RoundCodec, warm_training=False)
        codec.quantizer.quantizer_prior_stage_rates = (0.0, 0.0)
        codec.joint_quantizer_state = codec.quantizer.coding_model.state_dict()
        protocol.after_decode(
            torch.zeros(8, dtype=torch.float32), codec.create_r_record(round_id, client_id), codec)
        assert len(protocol.quantizer_pool) == expected_size

    assert protocol.quantizer_config_overrides(15, 99)["warm_training"] is True


def test_warm_parent_rotation_is_balanced() -> None:
    """Five refreshes select every pool slot exactly twice."""
    protocol = WarmWZCancerProtocol()
    _populate_pool(protocol)
    selections = []
    for refresh in range(5):
        protocol.warm_refresh_count = refresh
        codec = _codec(T_RoundCodec, warm_training=True)
        protocol.quantizer_initial_states(codec, refresh, si_size=1)
        selections.append(codec.warm_parent_slots)

    assert selections == [(0, 1), (2, 3), (4, 0), (1, 2), (3, 4)]
    assert [sum(slot in selection for selection in selections) for slot in range(5)] == [2] * 5


def test_t_server_si_prior_uses_the_quantizer_parent_slots() -> None:
    """The T-only server-SI-prior pool follows the quantizer's static lineage selection."""
    protocol = WarmWZCancerProtocol()
    _populate_pool(protocol)
    protocol.server_si_prior_pool = [
        {"marker": torch.tensor(float(slot))} for slot in range(protocol.pool_size)
    ]
    codec = _codec(T_RoundCodec, warm_training=True)

    protocol.quantizer_initial_states(codec, 0, si_size=1)
    plan = protocol.server_si_prior_training_plan(codec)

    assert codec.warm_parent_slots == plan.parent_slots == (0, 1)
    assert plan.attempts_per_parent == protocol.warm_retrain_repeats
    assert [float(state["marker"]) for state in plan.initial_states] == [0.0, 1.0]


def test_frozen_prior_refitting_is_opt_in() -> None:
    """Frozen codecs reuse fitted priors unless the protocol explicitly enables adaptation."""
    assert WZCancerProtocol().refit_priors_on_frozen is False
    assert WZCancerProtocol({"refit_priors_on_frozen": True}).refit_priors_on_frozen is True


def test_warm_training_keeps_one_winner_per_parent_and_deploys_the_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lineages use the warm repeat count and retain both lineage and global winners."""
    config = _config(warm_training=True).model_copy(update={"warm_retrain_repeats": 3})
    quantizer = WZQuantizerCancer(config, si_size=1)
    parents = []
    for value in (1.0, 2.0):
        state = {name: tensor.detach().clone() for name, tensor in quantizer.coding_model.state_dict().items()}
        next(iter(state.values())).fill_(value)
        parents.append(state)
    quantizer.load_warm_start_states(parents)

    observed_repeats = []

    def fake_train(
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        wmspe_denom: float,
        initial_state: dict[str, torch.Tensor],
        repeats: int,
        label: str,
    ) -> tuple[object, float]:
        observed_repeats.append(repeats)
        model = quantizer._new_coding_model(quantizer.side_info_size)
        model.load_state_dict(initial_state)
        parent_value = float(next(iter(initial_state.values())).flatten()[0])
        return model, {1.0: 3.0, 2.0: 1.0}[parent_value]

    monkeypatch.setattr(quantizer, "_train_finite_candidate", fake_train)
    winner = quantizer._train_finite_candidate_and_retrieve_best(
        torch.zeros((1, 1)), torch.zeros((1, 1)), 1.0
    )

    assert observed_repeats == [3, 3]
    assert quantizer.warm_retrained_losses == (3.0, 1.0)
    assert quantizer.warm_deployed_parent == 1
    parameter_name = next(iter(parents[1]))
    assert torch.equal(winner.state_dict()[parameter_name], parents[1][parameter_name])


def test_completed_warm_round_replaces_only_selected_pool_slots() -> None:
    """Pool mutation occurs after decode and preserves both unselected lineages."""
    protocol = WarmWZCancerProtocol()
    _populate_pool(protocol)
    codec = _codec(Oracle_RoundCodec, warm_training=True)
    protocol.quantizer_initial_states(codec, 0, si_size=1)
    replacements = tuple({
        name: tensor.detach().clone() + 10
        for name, tensor in protocol.quantizer_pool[slot].items()
    } for slot in codec.warm_parent_slots)
    codec.quantizer.warm_retrained_states = replacements
    untouched = {
        slot: {name: tensor.clone() for name, tensor in protocol.quantizer_pool[slot].items()}
        for slot in (2, 3, 4)
    }

    protocol.after_decode(
        torch.zeros(8, dtype=torch.float32), codec.create_r_record(30, 0), codec
    )

    assert protocol.warm_refresh_count == 1
    for slot, replacement in zip(codec.warm_parent_slots, replacements, strict=True):
        assert all(torch.equal(protocol.quantizer_pool[slot][name], value)
                   for name, value in replacement.items())
    for slot, state in untouched.items():
        assert all(torch.equal(protocol.quantizer_pool[slot][name], value)
                   for name, value in state.items())


def test_cache_results_restore_legacy_cold_and_complete_warm_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Cold cache entries keep two items while warm hits restore every lineage result."""
    target = torch.arange(8, dtype=torch.float32)
    side_information = [HistoryVector(target)]
    cold = WZQuantizerCancer(_config().model_copy(update={"cache_dir": tmp_path}), si_size=1)
    cold_state = {name: value.detach().clone() for name, value in cold.coding_model.state_dict().items()}
    monkeypatch.setattr(wz_quantizer_module, "cache_call", lambda *args, **kwargs: (cold_state, (0.4, 0.5)))
    cold.train_model(target, side_information)
    assert cold.quantizer_prior_stage_rates == (0.4, 0.5)
    assert cold.warm_retrained_states == ()

    warm = WZQuantizerCancer(
        _config(warm_training=True).model_copy(update={"cache_dir": tmp_path}), si_size=1)
    parents = tuple({name: value.clone() for name, value in cold_state.items()} for _ in range(2))
    warm.load_warm_start_states(parents)
    replacements = tuple({name: value.clone() + index for name, value in cold_state.items()}
                         for index in range(2))
    monkeypatch.setattr(
        wz_quantizer_module,
        "cache_call",
        lambda *args, **kwargs: (replacements[1], (0.2, 0.3), replacements, (3.0, 1.0), 1),
    )
    warm.train_model(target, side_information)

    assert warm.warm_retrained_losses == (3.0, 1.0)
    assert warm.warm_deployed_parent == 1
    assert all(torch.equal(warm.coding_model.state_dict()[name], value)
               for name, value in replacements[1].items())


def test_warm_protocol_preserves_the_cold_schedule_before_compatible_warm_rounds() -> None:
    """Warmup only prepares history; routine calls populate and then refresh the shared pool."""
    assert WarmWZCancerProtocol.warmup_round_codecs == WarmWZCancerProtocol.history_prep_round_codecs
    assert WarmWZCancerProtocol.history_prep_round_codecs[-1] == "I"
    assert WarmWZCancerProtocol.routine_round_codecs == WZCancerProtocol.routine_round_codecs

    configured = WarmWZCancerProtocol({"warm_retrain_repeats": 4})
    _populate_pool(configured)
    assert configured.quantizer_config_overrides(5, 0)["warm_retrain_repeats"] == 4


def test_ordinary_wz_protocol_does_not_select_warm_parent() -> None:
    """Parent selection belongs to the warm protocol mixin, not the ordinary protocol."""
    protocol = WZCancerProtocol()
    codec = _codec(T_RoundCodec, warm_training=True)

    assert protocol.quantizer_initial_states(codec, 0, si_size=1) == ()
