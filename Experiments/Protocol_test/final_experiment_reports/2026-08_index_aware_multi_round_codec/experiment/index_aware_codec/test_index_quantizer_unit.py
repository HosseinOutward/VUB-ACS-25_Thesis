from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from FL_code.cancer_protocol import wz_quantizer
from FL_code.cancer_protocol.wz_quantizer import unique_row_groups

from .index_quantizer import (
    IndexAwareDedupedWZQuantizer,
    IndexAwareWZcfgQuant,
)


def _config(**updates: object) -> IndexAwareWZcfgQuant:
    defaults: dict[str, object] = {
        "index_cache_version": "index-side-information/v1",
    }
    return IndexAwareWZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        norm_slices=[slice(None)],
        hidden_dim=8,
        layers=2,
        hash_table_count=2,
        hash_bucket_count=65_536,
        hash_dim=3,
        route_count=5,
        embedding_dim=4,
        fused_optimizer=False,
        **(defaults | updates),
    )


def _problem(size: int = 8) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.linspace(-2, 3, size), [torch.linspace(4, -1, size)]


def test_index_config_cache_payload_covers_semantics_only(tmp_path: Path) -> None:
    baseline = _config()
    changed_seed = _config(hash_seed=3)
    changed_training = _config(training_seed=44, training_match_bits=25)
    runtime_change = _config(cache_dir=tmp_path, training_progress_bar=True)

    assert baseline.training_cache_payload() != changed_seed.training_cache_payload()
    assert baseline.training_cache_payload() != changed_training.training_cache_payload()
    assert baseline.training_cache_payload() == runtime_change.training_cache_payload()
    assert baseline.training_cache_payload()["index_cache_version"] == "index-side-information/v1"


def test_v1_cache_payload_is_unchanged_by_v2_only_options() -> None:
    baseline = _config()
    changed = _config(
        raw_index_scale=0.02,
        routing_temperature_start=2.5,
        routing_temperature_end=0.7,
        routing_dense_warmup_fraction=0.2,
        routing_top_k=3,
        routing_balance_loss_weight=2e-3,
        routing_z_loss_weight=2e-3,
        embedding_diversity_loss_weight=2e-4,
        index_dropout=0.1,
        routed_scale_initial=0.02,
    )

    assert baseline.training_cache_payload() == changed.training_cache_payload()
    assert not IndexAwareWZcfgQuant._V2_CACHE_FIELDS & baseline.training_cache_payload().keys()


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("raw_index_scale", 0.02),
        ("routing_temperature_start", 2.5),
        ("routing_temperature_end", 0.7),
        ("routing_dense_warmup_fraction", 0.2),
        ("routing_top_k", 3),
        ("routing_balance_loss_weight", 2e-3),
        ("routing_z_loss_weight", 2e-3),
        ("embedding_diversity_loss_weight", 2e-4),
        ("index_dropout", 0.1),
        ("routed_scale_initial", 0.02),
        ("routed_scale_max", 0.08),
        ("content_routing", True),
        ("current_problem_sampling_fraction", 0.8),
    ),
)
def test_v2_cache_payload_covers_each_trainable_semantic(
    field: str, changed_value: object
) -> None:
    baseline = _config(index_cache_version="index-side-information/v2")
    changed = _config(
        index_cache_version="index-side-information/v2", **{field: changed_value}
    )

    assert baseline.training_cache_payload() != changed.training_cache_payload()
    assert IndexAwareWZcfgQuant._V2_CACHE_FIELDS <= baseline.training_cache_payload().keys()
    assert baseline.training_cache_payload()["raw_index_bits"] == 24


def test_coordinates_are_appended_and_restart_for_extra_pairs() -> None:
    target, side_information = _problem()
    quantizer = IndexAwareDedupedWZQuantizer(_config(), si_size=1)
    prepared, metadata = quantizer.preprocess_x(target)
    side_info = quantizer.normalize_side_info_tensor(metadata, side_information)

    expected = torch.arange(target.numel(), dtype=torch.float32, device=prepared.device)
    assert prepared.shape == (target.numel(), 2)
    assert side_info.shape == (target.numel(), 2)
    torch.testing.assert_close(prepared[:, -1], expected)
    torch.testing.assert_close(side_info[:, -1], expected)

    stacked_x, stacked_si = quantizer._stack_training_pairs(
        prepared, side_info, ((target.flip(0), [side_information[0].flip(0)]),)
    )
    assert stacked_x.shape == (2 * target.numel(), 2)
    assert stacked_si.shape == (2 * target.numel(), 2)
    torch.testing.assert_close(stacked_x[:, -1], expected.repeat(2))
    torch.testing.assert_close(stacked_si[:, -1], expected.repeat(2))

    retry_model = quantizer._new_coding_model(quantizer.side_info_size)
    assert retry_model.index_conditioner.vector_size == target.numel()


def test_extra_pairs_reserve_most_samples_for_the_current_problem() -> None:
    target, side_information = _problem()
    quantizer = IndexAwareDedupedWZQuantizer(_config(
        index_cache_version="index-side-information/v2",
        current_problem_sampling_fraction=0.75,
    ), si_size=1)
    prepared, metadata = quantizer.preprocess_x(target)
    side_info = quantizer.normalize_side_info_tensor(metadata, side_information)

    quantizer._stack_training_pairs(
        prepared,
        side_info,
        (
            (target.flip(0), [side_information[0].flip(0)]),
            (target.roll(1), [side_information[0].roll(1)]),
            (target.roll(2), [side_information[0].roll(2)]),
        ),
    )

    assert quantizer.training_block_probabilities == (0.75, 1 / 12, 1 / 12, 1 / 12)


def test_explicit_problem_weights_override_indexed_fallback_and_enter_cache() -> None:
    weights = (0.50, 0.33, 0.17)
    target, side_information = _problem()
    config = _config(
        index_cache_version="index-side-information/v2",
        current_problem_sampling_fraction=0.95,
        training_problem_weights=weights,
    )
    quantizer = IndexAwareDedupedWZQuantizer(config, si_size=1)
    prepared, metadata = quantizer.preprocess_x(target)
    side_info = quantizer.normalize_side_info_tensor(metadata, side_information)
    quantizer._stack_training_pairs(
        prepared,
        side_info,
        (
            (target.roll(1), [side_information[0].roll(1)]),
            (target.roll(2), [side_information[0].roll(2)]),
        ),
    )

    assert quantizer.training_block_probabilities == weights
    assert config.training_cache_payload()["training_problem_weights"] == weights


def test_explicit_problem_weights_apply_to_ordinary_quantizer() -> None:
    weights = (0.50, 0.33, 0.17)
    config = wz_quantizer.WZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        norm_slices=[slice(None)],
        hidden_dim=4,
        layers=1,
        fused_optimizer=False,
        training_problem_weights=weights,
    )
    quantizer = wz_quantizer.DedupedWZQuantizerCancer(config, si_size=1)

    quantizer._set_training_block_probabilities(extra_pair_count=2)

    assert quantizer.training_block_probabilities == weights
    assert config.training_cache_payload()["training_problem_weights"] == weights


def test_indexed_candidate_rejects_constant_refinement_planes() -> None:
    quantizer = IndexAwareDedupedWZQuantizer(_config(), si_size=1)
    reconstruction = torch.zeros(8)

    assert quantizer.candidate_failure_tolerance == 8
    assert not quantizer._candidate_is_usable(torch.zeros((2, 8)), reconstruction)
    assert quantizer._candidate_is_usable(
        torch.tensor([[0, 1] * 4, [1, 0] * 4]), reconstruction
    )


def test_training_seed_is_reproducible_and_rng_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    draws: list[torch.Tensor] = []
    initial_parameters: list[torch.Tensor] = []

    def draw_only(
        self: wz_quantizer.DedupedWZQuantizerCancer,
        x_raw: torch.Tensor,
        si_raw_list: list[torch.Tensor],
        extra_pairs: object,
    ) -> None:
        del x_raw, si_raw_list, extra_pairs
        draws.append(torch.rand(4))
        initial_parameters.append(next(self.coding_model.parameters()).detach().clone())

    monkeypatch.setattr(
        wz_quantizer.DedupedWZQuantizerCancer,
        "_train_model_uncached",
        draw_only,
    )
    target, side_information = _problem()
    quantizer = IndexAwareDedupedWZQuantizer(_config(training_seed=91), si_size=1)
    torch.manual_seed(7)
    expected_next = torch.rand(1)
    torch.manual_seed(7)
    quantizer._train_model_uncached(target, side_information)
    actual_next = torch.rand(1)
    quantizer._train_model_uncached(target, side_information)

    torch.testing.assert_close(draws[0], draws[1])
    torch.testing.assert_close(initial_parameters[0], initial_parameters[1])
    torch.testing.assert_close(actual_next, expected_next)


def test_coordinates_prevent_deduplication_across_positions() -> None:
    prepared = torch.stack((torch.ones(7), torch.arange(7)), dim=1)
    quantizer = IndexAwareDedupedWZQuantizer(_config(), si_size=1)
    representatives, inverse, collisions = unique_row_groups((prepared,), quantizer.match_bits)

    assert representatives.numel() == prepared.shape[0]
    assert inverse.unique().numel() == prepared.shape[0]
    assert not collisions.numel()


def test_v2_inference_never_deduplicates_distinct_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[int] = []

    def fake_encode(
        self: wz_quantizer.WZQuantizerCancer,
        x_prep: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self, batch_size
        observed_rows.append(x_prep.shape[0])
        return (
            torch.zeros((2, x_prep.shape[0]), dtype=torch.uint8),
            torch.zeros((2, x_prep.shape[0], 2)),
        )

    monkeypatch.setattr(wz_quantizer.WZQuantizerCancer, "encode_prepared", fake_encode)
    quantizer = IndexAwareDedupedWZQuantizer(_config(
        index_cache_version="index-side-information/v2"
    ), si_size=1)
    prepared = torch.stack((torch.ones(7), torch.arange(7)), dim=1)

    quantizer.encode_prepared(prepared, batch_size=7)

    assert observed_rows == [7]
    assert quantizer.prior_calculator is wz_quantizer.PriorCalculator


def test_indexed_loss_targets_only_values_and_trains_conditioner() -> None:
    config = _config(train_epochs=2, source_perturbation="none")
    quantizer = IndexAwareDedupedWZQuantizer(config, si_size=1)
    target, side_information = _problem()
    indexed_x, metadata = quantizer.preprocess_x(target)
    indexed_si = quantizer.normalize_side_info_tensor(metadata, side_information)
    quantizer.coding_model.to(indexed_x.device).train()

    loss = quantizer.compute_loss(
        quantizer.coding_model,
        (indexed_x,),
        indexed_si,
        epoch=0,
        c_cfg=config,
        sample_weights=None,
        wmspe_denom=1.0,
        perturb_generator=torch.Generator().manual_seed(0),
    )
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    for name, parameter in quantizer.coding_model.index_conditioner.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_v2_router_regularization_is_added_to_the_rd_loss() -> None:
    config = _config(
        index_cache_version="index-side-information/v2",
        train_epochs=2,
        index_dropout=0.0,
        routing_balance_loss_weight=0.0,
        routing_z_loss_weight=0.0,
        embedding_diversity_loss_weight=0.0,
    )
    regularized = config.model_copy(update={
        "routing_balance_loss_weight": 0.1,
        "routing_z_loss_weight": 0.1,
        "embedding_diversity_loss_weight": 0.1,
    })
    quantizer = IndexAwareDedupedWZQuantizer(config, si_size=1)
    target, side_information = _problem()
    indexed_x, metadata = quantizer.preprocess_x(target)
    indexed_si = quantizer.normalize_side_info_tensor(metadata, side_information)
    quantizer.coding_model.to(indexed_x.device).train()

    def loss(c_cfg: IndexAwareWZcfgQuant) -> torch.Tensor:
        torch.manual_seed(19)
        return quantizer.compute_loss(
            quantizer.coding_model,
            (indexed_x,),
            indexed_si,
            epoch=0,
            c_cfg=c_cfg,
            sample_weights=torch.arange(
                1, indexed_x.shape[0] + 1, device=indexed_x.device, dtype=torch.float32),
            wmspe_denom=1.0,
            perturb_generator=torch.Generator().manual_seed(0),
        )

    rd_loss = loss(config)
    regularized_loss = loss(regularized)
    assert torch.isfinite(regularized_loss) and regularized_loss > rd_loss


def test_cached_training_uses_indexed_class_and_restores_runtime_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target, side_information = _problem()
    observed_class: type[wz_quantizer.WZQuantizerCancer] | None = None

    def fake_cache_call(
        function: object,
        cache_dir: Path,
        hash_func: object,
        overwrite: bool,
        **kwargs: object,
    ) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
        del function, cache_dir, hash_func, overwrite
        nonlocal observed_class
        observed_class = kwargs["quantizer_class"]  # type: ignore[assignment]
        cached = observed_class(c_cfg=kwargs["c_cfg"], si_size=1)
        return cached.coding_model.state_dict(), (0.25, 0.5)

    monkeypatch.setattr(wz_quantizer, "cache_call", fake_cache_call)
    quantizer = IndexAwareDedupedWZQuantizer(_config(cache_dir=tmp_path), si_size=1)
    quantizer.train_model(target, side_information)

    assert observed_class is IndexAwareDedupedWZQuantizer
    assert quantizer.vector_size == target.numel()
    assert quantizer.coding_model.index_conditioner.vector_size == target.numel()
    assert quantizer.training_prior_stage_rates == (0.25, 0.5)
    assert quantizer.training_prior_rate == 0.75


def test_compact_artifact_contains_only_conditioner_state() -> None:
    target, _ = _problem()
    quantizer = IndexAwareDedupedWZQuantizer(_config(), si_size=1)
    quantizer.preprocess_x(target)
    artifact = quantizer.index_artifact()

    assert artifact["schema_version"] == "index-side-information/v1"
    assert artifact["vector_size"] == target.numel()
    assert artifact["conditioner_state"]
    assert all(key.startswith("index_conditioner.") for key in artifact["conditioner_state"])
    assert all(tensor.shape[0] != target.numel() for tensor in artifact["conditioner_state"].values())


def test_v2_artifact_captures_complete_config_and_final_routing_diagnostics() -> None:
    target, _ = _problem()
    config = _config(index_cache_version="index-side-information/v2")
    quantizer = IndexAwareDedupedWZQuantizer(config, si_size=1)
    quantizer.preprocess_x(target)
    quantizer.coding_model.index_conditioner.train()

    artifact = quantizer.index_artifact()
    diagnostics = artifact["routing_diagnostics"]

    assert artifact["schema_version"] == "index-side-information/v2"
    assert IndexAwareWZcfgQuant._V2_CACHE_FIELDS <= artifact["config"].keys()
    assert diagnostics["scope"] == "index_only_final_schedule"
    assert diagnostics["sample_size"] == target.numel()
    assert diagnostics["routing_temperature"] == config.routing_temperature_end
    assert 1 <= diagnostics["routing_top1_active_routes"] <= config.route_count
    assert quantizer.coding_model.index_conditioner.training
    assert all(
        np.isfinite(value)
        for key, value in diagnostics.items()
        if key != "scope"
    )
    assert "index_conditioner.index_router.weight" in artifact["conditioner_state"]
    assert "index_conditioner.content_router.weight" in artifact["conditioner_state"]
    assert "index_conditioner.routed_scale" in artifact["conditioner_state"]
