from __future__ import annotations

from typing import Any

import torch

from FL_code.FL_core.cache import model_training_hash

from .index_quantizer import IndexAwareDedupedWZQuantizer, IndexAwareWZcfgQuant


def _config(**updates: Any) -> IndexAwareWZcfgQuant:
    defaults: dict[str, Any] = {
        "index_cache_version": "index-side-information/v1",
    }
    return IndexAwareWZcfgQuant(
        bins_per_plane=2,
        num_planes=2,
        norm_slices=[slice(None)],
        hidden_dim=4,
        layers=1,
        hash_table_count=1,
        hash_dim=2,
        route_count=3,
        embedding_dim=2,
        fused_optimizer=False,
        **(defaults | updates),
    )


def _training_problem(
    size: int = 8,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor],
    tuple[tuple[torch.Tensor, tuple[torch.Tensor]], ...],
]:
    current = torch.linspace(-2, 3, size)
    current_si = (torch.linspace(4, -1, size),)
    extras = tuple(
        (
            torch.linspace(-2 + offset, 3 + offset, size),
            (torch.linspace(4 + 2 * offset, -1 + 2 * offset, size),),
        )
        for offset in (1, 2, 3)
    )
    return current, current_si, extras


def _cache_kwargs(
    config: IndexAwareWZcfgQuant,
) -> dict[str, Any]:
    current, current_si, extras = _training_problem()
    return {
        "quantizer_class": IndexAwareDedupedWZQuantizer,
        "c_cfg": config,
        "x_raw": current,
        "side_information": current_si,
        "extra_training_pairs": extras,
    }


def _hash(kwargs: dict[str, Any]) -> str:
    return model_training_hash((), kwargs)


def test_multi_round_training_hash_covers_every_training_input_and_seed() -> None:
    baseline = _cache_kwargs(_config(training_seed=43))
    exact_repeat = _cache_kwargs(_config(training_seed=43))
    assert _hash(baseline) == _hash(exact_repeat)

    changed_target = _cache_kwargs(_config(training_seed=43))
    changed_target["extra_training_pairs"][1][0][3] += 0.25

    changed_si = _cache_kwargs(_config(training_seed=43))
    changed_si["extra_training_pairs"][2][1][0][4] -= 0.5

    changed_order = _cache_kwargs(_config(training_seed=43))
    changed_order["extra_training_pairs"] = tuple(
        reversed(changed_order["extra_training_pairs"])
    )

    changed_seed = _cache_kwargs(_config(training_seed=44))
    no_extras = _cache_kwargs(_config(training_seed=43))
    no_extras.pop("extra_training_pairs")

    baseline_hash = _hash(baseline)
    assert {
        _hash(changed_target),
        _hash(changed_si),
        _hash(changed_order),
        _hash(changed_seed),
        _hash(no_extras),
    }.isdisjoint({baseline_hash})


def test_three_extra_pairs_restart_encoder_and_si_coordinates_per_block() -> None:
    current, current_si, extras = _training_problem()
    quantizer = IndexAwareDedupedWZQuantizer(_config(), si_size=1)
    current_prepared, metadata = quantizer.preprocess_x(current)
    current_prepared_si = quantizer.normalize_side_info_tensor(metadata, current_si)
    stacked_x, stacked_si = quantizer._stack_training_pairs(
        current_prepared,
        current_prepared_si,
        extras,
    )

    block_count = 1 + len(extras)
    expected = torch.arange(
        current.numel(), device=stacked_x.device, dtype=torch.float32
    ).expand(block_count, -1)
    x_coordinates = stacked_x.reshape(block_count, current.numel(), -1)[:, :, -1]
    si_coordinates = stacked_si.reshape(block_count, current.numel(), -1)[:, :, -1]

    torch.testing.assert_close(x_coordinates, expected)
    torch.testing.assert_close(si_coordinates, expected)
    torch.testing.assert_close(x_coordinates, si_coordinates)
    assert x_coordinates[:, 0].eq(0).all()
    assert x_coordinates[:, -1].eq(current.numel() - 1).all()
