from __future__ import annotations

import pytest
import torch

from FL_code.FL_core.cache import model_training_hash

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


def test_warm_start_state_is_loaded_once_and_copied() -> None:
    """A warm quantizer retains one independent copy of its initial parameters."""
    quantizer = WZQuantizerCancer(_config(warm_training=True), si_size=1)
    state = {name: value.detach().clone() for name, value in quantizer.coding_model.state_dict().items()}

    quantizer.load_warm_start_state(state)
    state[next(iter(state))].add_(1)

    assert quantizer._warm_start_state is not None
    assert torch.equal(
        quantizer._warm_start_state[next(iter(state))],
        quantizer.coding_model.state_dict()[next(iter(state))],
    )
    assert quantizer._warm_start_state[next(iter(state))].data_ptr() != state[next(iter(state))].data_ptr()
    with pytest.raises(AssertionError, match="only be loaded once"):
        quantizer.load_warm_start_state(state)


def test_warm_training_requires_loaded_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm training refuses to start until the protocol supplies initial parameters."""
    quantizer = WZQuantizerCancer(_config(warm_training=True), si_size=1)
    monkeypatch.setattr(quantizer, "_train_model_uncached", lambda *args, **kwargs: None)

    with pytest.raises(AssertionError, match="requires load_warm_start_state"):
        quantizer.train_model(torch.arange(8, dtype=torch.float32), [HistoryVector(torch.arange(8.0))])


def test_warm_training_overrides_only_effective_lr_and_tau() -> None:
    """Warm fitting changes only the effective optimizer rate and temperature."""
    config = _config(warm_training=True)
    quantizer = WZQuantizerCancer(config, si_size=1)
    effective = quantizer.training_config()

    assert effective.lr == config.lr * 0.3
    assert effective.tau == 0.5
    assert config.lr == WZcfgQuant.model_fields["lr"].default
    assert config.tau == WZcfgQuant.model_fields["tau"].default
    assert config.training_cache_payload()["warm_training"] is True
    assert "warm_training" not in config.model_copy(update={"warm_training": False}).training_cache_payload()


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

    assert model_training_hash((), common | {"warm_start_state": first_state}) != model_training_hash(
        (), common | {"warm_start_state": second_state}
    )


def test_latest_warm_start_mixin_selects_state_per_client_for_t_and_o() -> None:
    """The warm protocol supplies each client its own latest completed quantizer state."""
    protocol = WarmWZCancerProtocol()
    parents = {client_id: _codec(T_RoundCodec, warm_training=False) for client_id in (0, 1)}
    for client_id, parent in parents.items():
        parent.quantizer.training_prior_stage_rates = (0.0, 0.0)
        with torch.no_grad():
            next(parent.quantizer.coding_model.parameters()).fill_(client_id + 1)
        protocol.last_trained_codec[client_id] = parent

    for codec_class in (T_RoundCodec, Oracle_RoundCodec):
        codec = _codec(codec_class, warm_training=True)
        for client_id, parent in parents.items():
            state = protocol.quantizer_initial_state(codec, client_id)
            assert state is not None
            parameter_name = next(iter(state))
            assert torch.equal(state[parameter_name], parent.quantizer.coding_model.state_dict()[parameter_name])


def test_warm_protocol_requires_a_completed_client_parent() -> None:
    """The latest-state policy rejects missing and not-yet-trained parents."""
    protocol = WarmWZCancerProtocol()
    warm_codec = _codec(T_RoundCodec, warm_training=True)

    with pytest.raises(AssertionError, match="previously trained quantizer for client 0"):
        protocol.quantizer_initial_state(warm_codec, 0)

    protocol.last_trained_codec[0] = _codec(T_RoundCodec, warm_training=False)
    with pytest.raises(AssertionError, match="has not completed training"):
        protocol.quantizer_initial_state(warm_codec, 0)


def test_warm_protocol_preserves_the_cold_schedule_before_compatible_warm_rounds() -> None:
    """The default warm protocol keeps one full cold 3x3 cycle before warming that architecture."""
    assert WarmWZCancerProtocol.warmup_round_codecs == (
        *WZCancerProtocol.warmup_round_codecs,
        *WZCancerProtocol.routine_round_codecs,
    )
    assert WarmWZCancerProtocol.routine_round_codecs == (
        f"{WZCancerProtocol.routine_round_codecs[0]}|warm_training",
        *WZCancerProtocol.routine_round_codecs[1:],
    )


def test_ordinary_wz_protocol_does_not_select_warm_parent() -> None:
    """Parent selection belongs to the warm protocol mixin, not the ordinary protocol."""
    protocol = WZCancerProtocol()
    codec = _codec(T_RoundCodec, warm_training=True)

    assert protocol.quantizer_initial_state(codec, 0) is None
