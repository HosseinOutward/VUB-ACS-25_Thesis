from __future__ import annotations

from pathlib import Path

import pytest
import torch

from FL_code.run_fl import FLConfig

from . import trace as trace_module
from .explicit_cache import CacheEntry, TorchCodec
from .trace import Trace

DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not (DATA_FOLDER / "SVHN").is_dir(), reason="needs the local SVHN dataset")
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_resumed_stream_matches_uninterrupted_stream(
    device: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream resumed from a saved state skips earlier rounds and reproduces their successors exactly."""
    # CUDA runs the production precision path, whose GradScaler state is part of the checkpoint.
    production = device != "cpu"
    cfg = FLConfig(
        protocol="identity", model_name="resnet56", dataset_name="SVHN", num_classes=10,
        data_folder=DATA_FOLDER, num_clients=2, num_loader_workers=0, local_epochs=1, batch_size=64,
        lr=0.1, dataset_fraction=0.01, channels_last=production, fused_optimizer=production,
        mixed_precision=production,
    )
    trace = Trace(cfg, dataset_revision="test", device=torch.device(device))
    checkpoint = CacheEntry(tmp_path, trace.identity(), TorchCodec())
    uninterrupted = dict(trace.stream(0, 3))
    for _ in trace.stream(0, 1, checkpoint, save_round=2):
        pass
    assert checkpoint.load()["round"] == 2

    calls = 0
    train_client_update = trace_module.train_client_update

    def counted(*args: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return train_client_update(*args)

    monkeypatch.setattr(trace_module, "train_client_update", counted)
    resumed = dict(trace.stream(2, 3, checkpoint))

    assert calls == 2 * cfg.num_clients
    assert resumed.keys() == {2, 3}
    for round_id, deltas in resumed.items():
        assert all(torch.equal(left, right) for left, right in zip(deltas, uninterrupted[round_id], strict=True))
