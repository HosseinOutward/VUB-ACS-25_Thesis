"""Top-k transformations used by the report's sparsification experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from FL_code.FL_core.codec import record_reconstruction_metrics


def sparsify_with_feedback(
    delta: torch.Tensor,
    residual: torch.Tensor,
    keep_fraction: float,
    support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the transmitted coordinates and the residual carried into the next round."""
    effective = delta + residual
    sent = torch.zeros_like(effective)
    if support is None:
        keep = round(effective.numel() * keep_fraction)
        assert keep >= 1, (
            f"sparsify-keep={keep_fraction} keeps no coordinates of {effective.numel()}.")
        kept = effective.abs().topk(keep).indices
        sent[kept] = effective[kept]
    else:
        assert support.shape == effective.shape and support.dtype == torch.bool
        sent[support] = effective[support]
    return sent, effective - sent


def shared_support_mask(
    effectives: dict[int, torch.Tensor], keep_fraction: float
) -> torch.Tensor:
    """Return the union of every worker's independently selected top-k support."""
    assert effectives, "A shared support needs at least one worker's vector."
    size = next(iter(effectives.values())).numel()
    keep = round(size * keep_fraction)
    assert keep >= 1, f"sparsify-keep={keep_fraction} keeps no coordinates of {size}."
    mask = torch.zeros(size, dtype=torch.bool)
    for effective in effectives.values():
        assert effective.numel() == size
        mask[effective.abs().topk(keep).indices] = True
    return mask


def append_sparsification_record(
    save_dir: Path,
    round_id: int,
    client_id: int,
    keep_fraction: float,
    sent: torch.Tensor,
    delta: torch.Tensor,
    reconstructed: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    """Record sparse reconstruction error against both the sent and original vectors."""
    row = {
        "round_id": round_id,
        "client_id": client_id,
        "keep_fraction": keep_fraction,
        "kept_count": int(torch.count_nonzero(sent).item()),
        "sent_wmape": record_reconstruction_metrics(sent, reconstructed)["wmape"],
        "raw_wmape": record_reconstruction_metrics(delta, reconstructed)["wmape"],
        "residual_norm_ratio": float(residual.norm().item() / delta.norm().item()),
    }
    csv_path = save_dir / "sparsification_records.csv"
    with csv_path.open("a", newline="") as sparse_file:
        writer = csv.DictWriter(sparse_file, fieldnames=list(row))
        if sparse_file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
