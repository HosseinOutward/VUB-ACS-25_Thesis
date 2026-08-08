from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from FL_code.FL_core.codec import (
    Access, CompressionRecord, ReconstructionHistory)
from .sparse_encoder_history_queue import (
    CONFIGS,
    M_TRAINING_PROBLEM_WEIGHTS,
)

from .encoder_history_quantizer import (
    EncoderHistoryDedupedWZQuantizer,
    EncoderHistoryWZcfgQuant,
)
from .round_codecs import OM_FE_RoundCodec, TM_FE_RoundCodec
from .run_replay import RunMode, build_protocol


def _options() -> dict[str, Any]:
    return {
        "bins_per_plane": 3,
        "num_planes": 2,
        "sd_slices": [slice(None)],
        "hidden_dim": 4,
        "layers": 1,
        "train_epochs": 1,
        "train_sample_size": 4,
        "train_batch_size": 2,
        "quantizer_train_repeats": 1,
        "prior_train_repeats": 1,
        "fused_optimizer": False,
    }


def _history() -> ReconstructionHistory:
    history = ReconstructionHistory(
        max_per_client=3,
        routine_accesses=(Access.SERVER_ONLY, Access.TEMPORAL_TOO),
    )
    for round_id in range(3):
        for client_id in (0, 1):
            history.commit(
                torch.tensor([100 * client_id + round_id], dtype=torch.float32),
                CompressionRecord(round_id, client_id, "raw_history"),
                Access.TEMPORAL_TOO,
            )
    return history


@pytest.mark.parametrize(
    ("mode", "codec_class"),
    [
        (RunMode.OM_FE, OM_FE_RoundCodec),
        (RunMode.TM_FE, TM_FE_RoundCodec),
    ],
)
def test_protocol_discovers_encoder_history_variants(
    tmp_path: Path,
    mode: RunMode,
    codec_class: type[OM_FE_RoundCodec | TM_FE_RoundCodec],
) -> None:
    protocol = build_protocol(
        mode, 6, {}, [slice(None)], tmp_path, train_offsets=(1,)
    )
    assert protocol._get_curr_round_codec_name(6)[0] is codec_class
    assert protocol._get_curr_round_codec_name(7)[2] == "F"


@pytest.mark.parametrize(
    ("codec_class", "expected_side_info", "expected_indices"),
    [
        (OM_FE_RoundCodec, [1, 2, 101, 102], (2, 3)),
        (TM_FE_RoundCodec, [101, 102], (0, 1)),
    ],
)
def test_encoder_history_variants_select_only_requesting_clients_columns(
    codec_class: type[OM_FE_RoundCodec | TM_FE_RoundCodec],
    expected_side_info: list[int],
    expected_indices: tuple[int, ...],
) -> None:
    codec = codec_class(
        _options(), codec_class.round_name, train_offsets=(1,), history_window=2
    )
    target, side_info = codec.get_training_data(_history(), client_id=1)

    assert target is None
    assert [int(value.item()) for value in side_info] == expected_side_info
    assert codec.c_cfg.encoder_side_information_indices == expected_indices
    assert isinstance(codec.c_cfg, EncoderHistoryWZcfgQuant)
    assert codec.quantizer_class is EncoderHistoryDedupedWZQuantizer


def test_quantizer_adds_selected_history_only_to_encoder_and_cache() -> None:
    config_options = _options()
    del config_options["sd_slices"]
    config = EncoderHistoryWZcfgQuant(
        **(config_options | {
            "encoder_side_information_indices": (1, 3),
            "norm_slices": [slice(None)],
        })
    )
    quantizer = EncoderHistoryDedupedWZQuantizer(config, si_size=4)
    source = torch.tensor([[1.0], [2.0]])
    side_info = torch.tensor([
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
    ])

    encoder_input, decoder_side_info = quantizer._stack_training_pairs(
        source, side_info, ()
    )

    assert torch.equal(
        encoder_input,
        torch.tensor([[1.0, 11.0, 13.0], [2.0, 21.0, 23.0]]),
    )
    assert decoder_side_info is side_info
    assert quantizer.coding_model.encoder.input_dim == 3
    assert quantizer.coding_model.decoder.input_dim == config.bins_per_plane + 4
    assert quantizer.coding_model.conditionalRNN.input_dim == config.bins_per_plane + 4
    assert config.training_cache_payload()["encoder_side_information_indices"] == (1, 3)


def test_sparse_queue_uses_the_existing_m_settings_for_encoder_history_codecs() -> None:
    configs = {
        config.mode: config for config in CONFIGS if config.mode in {RunMode.OM_FE, RunMode.TM_FE}
    }

    assert set(configs) == {RunMode.OM_FE, RunMode.TM_FE}
    assert all(
        config.training_problem_weights == M_TRAINING_PROBLEM_WEIGHTS
        for config in configs.values()
    )
