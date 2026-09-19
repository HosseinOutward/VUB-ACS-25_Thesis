"""Temporary n-split helper used only by the current sparse initial-SI experiment."""

from __future__ import annotations

import torch


def nsplit_quantize(values: torch.Tensor, splits: int) -> torch.Tensor:
    """Replace values in each empirical quantile interval with that interval's mean."""
    assert values.dtype is torch.float32 and values.device.type == "cpu" and values.ndim == 1
    assert isinstance(splits, int) and 2 <= splits <= 256
    thresholds = torch.quantile(values, torch.arange(1, splits, dtype=values.dtype) / splits)
    bins = torch.bucketize(values, thresholds)
    counts = torch.bincount(bins, minlength=splits)
    sums = torch.zeros(splits, dtype=torch.float64).scatter_add_(0, bins, values.double())
    means = torch.zeros(splits, dtype=torch.float32)
    occupied = counts > 0
    means[occupied] = (sums[occupied] / counts[occupied]).float()
    return means[bins]


__all__ = ["nsplit_quantize"]
