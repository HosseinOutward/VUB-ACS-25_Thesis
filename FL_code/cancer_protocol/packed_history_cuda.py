"""Fused CUDA row access for split-packed reconstruction-history codes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 256


# Triton pointer parameters must remain unannotated; only compile-time scalars use tl.constexpr.
@triton.jit
def _packed_ranks_kernel(
    low_codes,
    packed_high,
    indices,
    output,
    count,
    high_bits: tl.constexpr,
    block_size: tl.constexpr,
):
    coordinates = tl.program_id(0) * block_size + tl.arange(0, block_size)
    selected = coordinates < count
    rows = tl.load(indices + coordinates, mask=selected).to(tl.int64)
    bit_offsets = rows * high_bits
    byte_indices, shifts = bit_offsets >> 3, bit_offsets & 7
    words = tl.load(packed_high + byte_indices, mask=selected).to(tl.int32)
    words |= tl.load(packed_high + byte_indices + 1, mask=selected).to(tl.int32) << 8
    if high_bits > 9:
        words |= tl.load(packed_high + byte_indices + 2, mask=selected).to(tl.int32) << 16
    high = (words >> shifts) & ((1 << high_bits) - 1)
    ranks = tl.load(low_codes + rows, mask=selected).to(tl.int32) | (high << 8)
    tl.store(output + coordinates, ranks, mask=selected)


@triton.jit
def _packed_values_kernel(
    low_codes,
    packed_high,
    dictionary,
    indices,
    output,
    count,
    high_bits: tl.constexpr,
    block_size: tl.constexpr,
):
    coordinates = tl.program_id(0) * block_size + tl.arange(0, block_size)
    selected = coordinates < count
    rows = tl.load(indices + coordinates, mask=selected).to(tl.int64)
    bit_offsets = rows * high_bits
    byte_indices, shifts = bit_offsets >> 3, bit_offsets & 7
    words = tl.load(packed_high + byte_indices, mask=selected).to(tl.int32)
    words |= tl.load(packed_high + byte_indices + 1, mask=selected).to(tl.int32) << 8
    if high_bits > 9:
        words |= tl.load(packed_high + byte_indices + 2, mask=selected).to(tl.int32) << 16
    high = (words >> shifts) & ((1 << high_bits) - 1)
    ranks = tl.load(low_codes + rows, mask=selected).to(tl.int32) | (high << 8)
    tl.store(output + coordinates, tl.load(dictionary + ranks, mask=selected), mask=selected)


def packed_ranks(
    low_codes: torch.Tensor,
    packed_high: torch.Tensor,
    indices: torch.Tensor,
    high_bits: int,
) -> torch.Tensor:
    """Return dictionary ranks for arbitrary CUDA row indices."""
    assert low_codes.is_cuda and packed_high.is_cuda and indices.is_cuda
    output = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    if indices.numel() == 0:
        return output
    _packed_ranks_kernel[(triton.cdiv(indices.numel(), _BLOCK_SIZE),)](
        low_codes,
        packed_high,
        indices,
        output,
        count=indices.numel(),
        high_bits=high_bits,
        block_size=_BLOCK_SIZE,
    )
    return output


def packed_values(
    low_codes: torch.Tensor,
    packed_high: torch.Tensor,
    dictionary: torch.Tensor,
    indices: torch.Tensor,
    high_bits: int,
) -> torch.Tensor:
    """Decode dictionary values for arbitrary CUDA row indices."""
    assert low_codes.is_cuda and packed_high.is_cuda and dictionary.is_cuda and indices.is_cuda
    output = torch.empty(indices.shape, dtype=torch.float32, device=indices.device)
    if indices.numel() == 0:
        return output
    _packed_values_kernel[(triton.cdiv(indices.numel(), _BLOCK_SIZE),)](
        low_codes,
        packed_high,
        dictionary,
        indices,
        output,
        count=indices.numel(),
        high_bits=high_bits,
        block_size=_BLOCK_SIZE,
    )
    return output
