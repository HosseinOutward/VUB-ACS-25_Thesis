from __future__ import annotations

import torch
import pytest
from torch import nn

from FL_code.FL_core.cache import model_training_hash
from FL_code.cancer_protocol.history import _CHUNK_ELEMENTS, CompressedHistoryVector, HistoryVector
from FL_code.cancer_protocol.wz_quantizer import (
    SideInfoMatrix,
    WZQuantizerCancer,
    WZcfgQuant,
    unique_row_groups,
    wz_model_training_loop,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training loop requires CUDA")
def test_training_decodes_sampled_side_information_once_per_epoch() -> None:
    class CountingSideInfo:
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values
            self.shape = values.shape
            self.calls = 0

        def to(self, device: torch.device | str) -> CountingSideInfo:
            self.values = self.values.to(device)
            return self

        def __getitem__(self, rows: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.values[rows]

    model = nn.Linear(1, 1)
    model.num_planes = 1  # type: ignore[attr-defined]
    side_info = CountingSideInfo(torch.arange(12, dtype=torch.float32).unsqueeze(1))
    config = WZcfgQuant(
        bins_per_plane=2, num_planes=1, train_epochs=2, train_sample_size=8,
        train_batch_size=2, deduplicate_training=False,
    )

    def loss(
        candidate: nn.Module,
        training_inputs: tuple[torch.Tensor],
        si_batch: torch.Tensor,
        _epoch: int,
        _config: WZcfgQuant,
        *,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        assert sample_weights is None
        return (candidate(training_inputs[0]) - si_batch).square().mean()

    wz_model_training_loop(
        loss, model, model.parameters(), (torch.ones(12, 1),), side_info, config,
        batch_size=config.train_batch_size,
    )

    assert side_info.calls == config.train_epochs


def test_side_information_matrix_decodes_compressed_rows_and_normalizes() -> None:
    values = torch.arange(20, dtype=torch.float32) / 10
    compressed = CompressedHistoryVector(
        values, normalization_factors=torch.tensor([[1.0, 0.0]])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = SideInfoMatrix(
        [compressed, HistoryVector(values + 1)], 20, torch.tensor([[2.0, 1.0]]),
        (slice(0, 20),), device,
    )
    expected = torch.stack([compressed.materialize(), values + 1], dim=1).to(device)
    expected = (expected - 1) / 2
    rows = torch.tensor([19, 0, 7, 3])
    assert torch.equal(matrix[rows], expected[rows])
    assert torch.equal(matrix[4:12], expected[4:12])
    chunks = matrix._columns[0].chunks
    assert chunks is not None and chunks[0].layout == "raw"
    assert chunks[0].values.device == matrix.device
    assert matrix.device.type == device.type


def test_side_information_matrix_gathers_uint16_codes_on_accelerator() -> None:
    distinct = 65_536
    patterns = (
        torch.arange(distinct, dtype=torch.int64) * (1 << (32 - 18)) + 0x10000000
    ).to(torch.int32)
    values = patterns.repeat(4).view(torch.float32)
    compressed = CompressedHistoryVector(
        values, normalization_factors=torch.tensor([[1.0, 0.0]])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = SideInfoMatrix(
        [compressed], values.numel(), torch.tensor([[1.0, 0.0]]),
        (slice(0, values.numel()),), device,
    )
    rows = torch.tensor([65_535, 0, 128], device=device)

    chunks = matrix._columns[0].chunks
    assert chunks is not None and chunks[0].codes is not None
    assert chunks[0].layout == "direct" and chunks[0].codes.dtype is torch.uint16
    assert torch.equal(matrix[rows].cpu().squeeze(1), compressed.materialize()[rows.cpu()])
    assert matrix.code_rows(rows).shape == (3, 1)


def test_side_information_matrix_gathers_packed_codes_on_accelerator() -> None:
    distinct = 16_384
    patterns = (
        torch.arange(distinct, dtype=torch.int64) * (1 << (32 - 18)) + 0x10000000
    ).to(torch.int32)
    values = patterns.repeat(16).view(torch.float32)
    compressed = CompressedHistoryVector(
        values, normalization_factors=torch.tensor([[1.0, 0.0]])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = SideInfoMatrix(
        [compressed], values.numel(), torch.tensor([[1.0, 0.0]]),
        (slice(0, values.numel()),), device,
    )
    rows = torch.tensor([0, 1, 255, 256, 16_383, values.numel() - 1], device=device)

    chunks = matrix._columns[0].chunks
    assert chunks is not None and chunks[0].layout == "packed"
    assert torch.equal(matrix[rows].cpu().squeeze(1), compressed.materialize()[rows.cpu()])
    assert torch.unique(matrix.code_rows(rows)).numel() == rows.numel() - 1


def test_side_information_matrix_decodes_across_adaptive_chunk_layouts() -> None:
    patterns = (
        torch.arange(16_384, dtype=torch.int64) * (1 << (32 - 18)) + 0x10000000
    ).to(torch.int32)
    packed = patterns.repeat(_CHUNK_ELEMENTS // patterns.numel())
    raw = patterns[:100]
    values = torch.cat((packed, raw)).view(torch.float32)
    compressed = CompressedHistoryVector(
        values, normalization_factors=torch.tensor([[1.0, 0.0]])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = SideInfoMatrix(
        [compressed], values.numel(), torch.tensor([[1.0, 0.0]]),
        (slice(0, values.numel()),), device,
    )
    rows = torch.tensor(
        [0, _CHUNK_ELEMENTS - 1, _CHUNK_ELEMENTS, values.numel() - 1], device=device
    )

    chunks = matrix._columns[0].chunks
    assert chunks is not None and [chunk.layout for chunk in chunks] == ["packed", "raw"]
    assert torch.equal(matrix[rows].cpu().squeeze(1), compressed.materialize()[rows.cpu()])
    assert torch.unique(matrix.code_rows(rows)).numel() == rows.numel()


def test_unique_row_groups_uses_compact_codes_and_normalization_slice() -> None:
    values = torch.zeros(4)
    slices = (slice(0, 2), slice(2, 4))
    matrix = SideInfoMatrix(
        [CompressedHistoryVector(values, slices)], 4, torch.tensor([[2.0, 1.0], [3.0, 1.0]]),
        slices, torch.device("cpu"),
    )
    _, inverse, collisions = unique_row_groups((matrix,), 1)
    assert torch.equal(inverse, torch.tensor([0, 0, 1, 1]))
    assert collisions.numel() == 0


def test_unique_row_groups_accepts_empty_side_information_matrix() -> None:
    matrix = SideInfoMatrix([], 4, torch.tensor([[1.0, 0.0]]), (slice(0, 4),), torch.device("cpu"))
    _, inverse, collisions = unique_row_groups((matrix,), 1)
    assert torch.equal(inverse, torch.zeros(4, dtype=torch.long))
    assert collisions.numel() == 0


def test_unique_row_groups_flattens_multi_dimensional_training_inputs() -> None:
    block = torch.tensor([
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ])

    actual = unique_row_groups((block,), 32)
    expected = unique_row_groups((block.flatten(start_dim=1),), 32)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


def test_normalize_side_info_matches_preprocessed_history() -> None:
    size = 100
    slices = (slice(0, 30), slice(30, size))
    quantizer = WZQuantizerCancer(
        WZcfgQuant(bins_per_plane=2, num_planes=1, norm_slices=slices), si_size=1,
    )
    target = torch.linspace(-1, 1, size)
    _, metadata = quantizer.preprocess_x(target)
    side_info = torch.randn(size)
    matrix = quantizer.normalize_side_info_tensor(metadata, [HistoryVector(side_info)])
    expected = quantizer.preprocess_x(
        side_info, skip_outliers=True, normalization_factors=metadata.norm_factors,
    )[0].squeeze(1)
    assert torch.equal(matrix[:], expected[:, None])


def test_model_training_hash_ignores_side_information_contents_but_keeps_size() -> None:
    class Config:
        def training_cache_payload(self) -> dict[str, int]:
            return {"model": 1}

    kwargs = {
        "c_cfg": Config(),
        "x_raw": torch.arange(5, dtype=torch.float32),
        "side_information": (HistoryVector(torch.zeros(5)),),
    }
    changed = kwargs | {"side_information": (HistoryVector(torch.ones(5)),)}
    wider = kwargs | {"side_information": (HistoryVector(torch.zeros(5)), HistoryVector(torch.zeros(5)))}
    assert model_training_hash((), kwargs) == model_training_hash((), changed)
    assert model_training_hash((), kwargs) != model_training_hash((), wider)


def test_model_training_hash_keeps_extra_targets_and_schedule_discriminator() -> None:
    target = torch.arange(5, dtype=torch.float32)
    history = HistoryVector(torch.zeros(5))
    base = {
        "c_cfg": WZcfgQuant(
            bins_per_plane=2,
            num_planes=1,
            training_cache_discriminator=("wz_cancer", "T"),
        ),
        "x_raw": target,
        "side_information": (history,),
        "extra_training_pairs": ((target + 1, (history,)),),
    }
    changed_si = base | {
        "side_information": (HistoryVector(torch.ones(5)),),
        "extra_training_pairs": ((target + 1, (HistoryVector(torch.ones(5)),)),),
    }
    changed_target = base | {"extra_training_pairs": ((target + 2, (history,)),)}
    changed_schedule = base | {
        "c_cfg": base["c_cfg"].model_copy(
            update={"training_cache_discriminator": ("wz_cancer", "O")}
        )
    }

    expected = model_training_hash((), base)
    assert model_training_hash((), changed_si) == expected
    assert model_training_hash((), changed_target) != expected
    assert model_training_hash((), changed_schedule) != expected


def test_prior_training_hash_ignores_side_info_matrix_contents() -> None:
    config = WZcfgQuant(bins_per_plane=2, num_planes=1)
    shared = {
        "c_cfg": config,
        "bins_vec": torch.zeros((1, 5), dtype=torch.long),
        "soft_codes": torch.zeros((1, 5, 2)),
    }
    first = shared | {"side_info": torch.zeros((5, 2))}
    changed = shared | {"side_info": torch.ones((5, 2))}
    wider = shared | {"side_info": torch.zeros((5, 3))}

    assert model_training_hash((), first) == model_training_hash((), changed)
    assert model_training_hash((), first) != model_training_hash((), wider)
