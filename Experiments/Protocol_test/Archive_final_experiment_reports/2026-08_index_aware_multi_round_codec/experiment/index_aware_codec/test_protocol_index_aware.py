"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from FL_code.FL_core.codec import Access
from FL_code.cancer_protocol.prior_code import PriorCalculator

from .index_quantizer import IndexAwareDedupedWZQuantizer, IndexAwareWZcfgQuant
from .rd_compare import compare_baseline, compare_experiment
from .round_codecs import OI_RoundCodec, TI_RoundCodec
from .run_replay import RunMode, _requested_keys, build_protocol


def _config(**updates: Any) -> IndexAwareWZcfgQuant:
    defaults: dict[str, Any] = {
        "index_cache_version": "index-side-information/v1",
    }
    return IndexAwareWZcfgQuant(
        bins_per_plane=3,
        num_planes=2,
        hash_table_count=1,
        hash_dim=2,
        route_count=3,
        embedding_dim=2,
        hidden_dim=4,
        layers=1,
        train_epochs=1,
        train_sample_size=4,
        train_batch_size=2,
        quantizer_train_repeats=1,
        prior_train_repeats=1,
        fused_optimizer=False,
        **(defaults | updates),
    )


@pytest.mark.parametrize(
    ("mode", "codec_class", "access"),
    [
        (RunMode.OI, OI_RoundCodec, Access.SERVER_ONLY),
        (RunMode.TI, TI_RoundCodec, Access.TEMPORAL_TOO),
    ],
)
def test_protocol_discovers_index_rounds_and_freezes_after_training(
    tmp_path: Path,
    mode: RunMode,
    codec_class: type[OI_RoundCodec | TI_RoundCodec],
    access: Access,
) -> None:
    protocol = build_protocol(mode, 12, {}, [slice(0, 4)], tmp_path)
    round_class, parsed, _ = protocol._get_curr_round_codec_name(12)
    assert round_class is codec_class
    assert round_class.can_decode_where is access
    assert isinstance(parsed.options, dict)
    assert protocol.routine_round_codecs == ("F",)


def test_index_round_config_selects_experimental_config() -> None:
    codec = OI_RoundCodec(
        {
            "bins_per_plane": 3,
            "num_planes": 2,
            "sd_slices": [slice(0, 4)],
            "hash_table_count": 1,
            "hidden_dim": 90,
        },
        "OI|bins_per_plane=3|num_planes=2",
    )
    assert isinstance(codec.c_cfg, IndexAwareWZcfgQuant)
    assert codec.c_cfg.hidden_dim == 90
    assert codec.quantizer_class is IndexAwareDedupedWZQuantizer


def test_prior_factory_reuses_conditioner_with_wider_server_si(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(cache_dir=Path("unused"))
    codec = OI_RoundCodec(
        {
            "bins_per_plane": cfg.bins_per_plane,
            "num_planes": cfg.num_planes,
            "sd_slices": [slice(0, 4)],
            **cfg.model_dump(exclude={"bins_per_plane", "num_planes", "norm_slices"}),
        },
        "OI",
    )
    quantizer = IndexAwareDedupedWZQuantizer(cfg, si_size=1)
    quantizer.vector_size = 4
    quantizer.coding_model.set_vector_size(4)
    side_info = torch.column_stack((
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.arange(4, dtype=torch.float32),
    ))
    bins = torch.randint(3, (2, 4))
    soft_codes = torch.softmax(torch.randn(2, 4, 3), dim=-1)
    captured: dict[str, Any] = {}

    def fake_train(
        bins_vec: torch.Tensor,
        codes: torch.Tensor,
        indexed_si: torch.Tensor,
        prior_cfg: IndexAwareWZcfgQuant,
        batch_size: int = 50_000,
        *,
        model_factory: Any = None,
    ) -> torch.nn.Module:
        captured["cache_dir"] = prior_cfg.cache_dir
        captured["batch_size"] = batch_size
        model = model_factory(indexed_si.shape[1])
        captured["model"] = model
        return model

    monkeypatch.setattr(PriorCalculator, "make_trained_prior_model", fake_train)
    codec.quantizer = quantizer
    codec.round_name_full = "F"
    code_lengths, _ = codec.conditional_code_lengths(
        quantizer, bins, soft_codes, side_info, cfg
    )
    prior_model = captured["model"]
    assert captured["cache_dir"] is None
    assert prior_model.ordinary_side_info_size == 3
    assert code_lengths.shape == bins.shape
    for key, value in quantizer.coding_model.index_conditioner.state_dict().items():
        assert torch.equal(prior_model.index_conditioner.state_dict()[key], value)


def _records(path: Path, rate_offset: float = 0.0, wmape_scale: float = 1.0) -> Path:
    rows = [
        {
            "round_id": round_id,
            "client_id": client_id,
            "round_name_full": "O|bins_per_plane=3|num_planes=3" if round_id == 12 else "F",
            "prior_stage_rates": str([
                0.2 + rate_offset,
                0.3 + rate_offset,
                0.4 + rate_offset,
            ]),
            "stage_wmapes": str([
                30.0 * wmape_scale,
                20.0 * wmape_scale,
                10.0 * wmape_scale,
            ]),
            "protocol_name_full": "wz_cancer|train_at_round=12",
        }
        for round_id in (12, 13)
        for client_id in (0, 1)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_baseline_gate_reports_clients_and_average(tmp_path: Path) -> None:
    reference = _records(tmp_path / "reference.csv")
    candidate = _records(tmp_path / "candidate.csv")
    output = tmp_path / "comparison"
    summary = compare_baseline(candidate, reference, output)
    differences = pd.read_csv(output / "baseline_stage_differences.csv")
    assert summary["passed"] is True
    assert len(differences) == 9
    assert set(differences["client_id"].astype(str)) == {"0", "1", "average"}


def test_baseline_gate_fails_large_rate_shift(tmp_path: Path) -> None:
    reference = _records(tmp_path / "reference.csv")
    candidate = _records(tmp_path / "candidate.csv", rate_offset=0.2)
    with pytest.raises(AssertionError, match="Baseline RD sanity gate failed"):
        compare_baseline(candidate, reference, tmp_path / "comparison")


def test_experiment_comparison_reports_three_subsets(tmp_path: Path) -> None:
    reference = _records(tmp_path / "reference.csv")
    candidate = _records(tmp_path / "candidate.csv", rate_offset=0.01)
    result = compare_experiment(candidate, reference, tmp_path / "rd.csv")
    assert result["subset"].tolist() == [
        "training_round",
        "frozen_rounds",
        "full_trajectory",
    ]
    assert np.isfinite(result["rate_gap_bits"]).all()


def test_full_trajectory_key_count() -> None:
    assert len(_requested_keys(RunMode.OI, 12, 20, tuple(range(5)))) == 45
