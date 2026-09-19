from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from FL_code.FL_core import trace as trace_module
from FL_code.FL_core.models import FLModelTemplate
from FL_code.FL_core.server import aggregate_and_update
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager
from FL_code.run_fl import FLConfig


class _TinyModel(FLModelTemplate):
    def __init__(self, cfg: FLConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        self.linear = nn.Linear(2, 2).to(device)
        self.register_buffer("visits", torch.zeros(1, device=device))

    def configure_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.6)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, labels = (value.to(self.device) for value in batch)
        if self.cfg.augment:
            features = features + torch.rand_like(features) * 0.1
        self.visits.add_(1)
        return nn.functional.cross_entropy(self.linear(features), labels)


def _cfg(**updates: object) -> FLConfig:
    values: dict[str, object] = dict(
        protocol="identity", model_name="resnet18_cifar", dataset_name="SYNTHETIC",
        num_classes=2, image_size=4, num_clients=2, num_loader_workers=0, rounds=4,
        local_epochs=1, batch_size=2, lr=0.07, seed=19, server_lr=0.8,
        server_momentum=0.5, augment=True, channels_last=False, fused_optimizer=False,
        mixed_precision=False, recalibrate_bn=False,
    )
    return FLConfig.model_validate(values | updates)


@pytest.fixture(autouse=True)
def _tiny_training(monkeypatch: pytest.MonkeyPatch) -> None:
    features = torch.tensor([
        [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0],
        [-0.7, 0.3], [0.8, -0.2], [-0.4, -0.8], [0.2, 0.9],
    ]).share_memory_()
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).share_memory_()
    monkeypatch.setattr(
        "FL_code.FL_core.trace.precompute_dataset_to_shared",
        lambda *_args, **_kwargs: (features, labels, None),
    )
    monkeypatch.setattr(
        "FL_code.FL_core.trace.initialize_model",
        lambda cfg, device: _TinyModel(cfg, device),
    )


def _spy_on_training(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Wrap the real shared training call and record every (round, client) it actually ran."""
    calls: list[tuple[int, int]] = []
    original = trace_module.train_client_update

    def spy(model, loader, optimizer, scaler, manager, cfg, round_id, client_id):  # type: ignore[no-untyped-def]
        calls.append((round_id, client_id))
        return original(model, loader, optimizer, scaler, manager, cfg, round_id, client_id)

    monkeypatch.setattr(trace_module, "train_client_update", spy)
    return calls


def _deltas(directory: Path) -> list[torch.Tensor]:
    return [torch.load(path, weights_only=True) for path in sorted(directory.glob("round_*_client_*.pt"))]


def test_trace_is_identical_when_uninterrupted_resumed_and_rng_is_disturbed(tmp_path: Path) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    uninterrupted = Trace(tmp_path / "uninterrupted", cfg, dataset_revision="rev1")
    uninterrupted.ensure(range(1, 4), device)

    resumed = Trace(tmp_path / "resumed", cfg, dataset_revision="rev1")
    resumed.ensure((1,), device)
    torch.manual_seed(999)
    _ = torch.rand(100)
    resumed.ensure((2, 3), device)

    assert len(_deltas(uninterrupted.directory)) == 6
    assert all(torch.equal(left, right) for left, right in zip(
        _deltas(uninterrupted.directory), _deltas(resumed.directory), strict=True
    ))
    assert sorted(path.name for path in uninterrupted.checkpoint_dir.iterdir()) == [
        "round_0.pt", "round_3.pt"
    ]
    assert sorted(path.name for path in resumed.checkpoint_dir.iterdir()) == [
        "round_0.pt", "round_1.pt", "round_3.pt"
    ]
    assert FLConfig.model_validate_json(resumed.config_path.read_text()).num_classes == 2


def test_trace_rejects_unmanaged_files_and_semantic_config_mismatch(tmp_path: Path) -> None:
    unmanaged = tmp_path / "unmanaged"
    unmanaged.mkdir()
    torch.save(torch.zeros(1), unmanaged / "round_0_client_0.pt")
    with pytest.raises(AssertionError, match="no managed trace.json marker"):
        Trace(unmanaged, _cfg(), dataset_revision="rev1")

    managed = Trace(tmp_path / "managed", _cfg(), dataset_revision="rev1")
    with pytest.raises(AssertionError, match="configuration differs"):
        Trace(managed.directory, _cfg(lr=0.08), dataset_revision="rev1")


@pytest.mark.parametrize("field, value", [
    ("seed", 999), ("model_name", "resnet18"), ("dataset_name", "MNIST"),
])
def test_semantic_field_mismatch_fails_loudly(tmp_path: Path, field: str, value: object) -> None:
    managed = Trace(tmp_path / "managed", _cfg(), dataset_revision="rev1")
    with pytest.raises(AssertionError, match="configuration differs"):
        Trace(managed.directory, _cfg(**{field: value}), dataset_revision="rev1")


def test_generator_version_mismatch_fails_loudly(tmp_path: Path) -> None:
    managed = Trace(tmp_path / "managed", _cfg(), dataset_revision="rev1")
    marker = json.loads(managed.marker_path.read_text())
    marker["version"] = trace_module._TRACE_VERSION + 1
    managed.marker_path.write_text(json.dumps(marker))
    with pytest.raises(AssertionError, match="generator version differs"):
        Trace(managed.directory, _cfg(), dataset_revision="rev1")


def test_dataset_revision_mismatch_fails_loudly(tmp_path: Path) -> None:
    managed = Trace(tmp_path / "managed", _cfg(), dataset_revision="rev-a")
    with pytest.raises(AssertionError, match="dataset_revision differs"):
        Trace(managed.directory, _cfg(), dataset_revision="rev-b")


def test_num_classes_none_resolution_does_not_invalidate_immediate_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, directory = torch.device("cpu"), tmp_path / "trace"
    Trace(directory, _cfg(num_classes=None), dataset_revision="rev1").ensure((1,), device)

    calls = _spy_on_training(monkeypatch)
    rerun = Trace(directory, _cfg(num_classes=None), dataset_revision="rev1")
    rerun.ensure((1,), device)

    assert calls == []
    assert FLConfig.model_validate_json(rerun.config_path.read_text()).num_classes == 2


def test_no_training_when_every_requested_round_is_already_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    tr.ensure((1, 2), device)

    calls = _spy_on_training(monkeypatch)
    tr.ensure((1, 2), device)
    assert calls == []


def test_extending_to_a_later_window_does_not_retrain_earlier_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    tr.ensure((1, 2), device)

    calls = _spy_on_training(monkeypatch)
    tr.ensure((1, 2, 3), device)
    assert calls == [(3, client_id) for client_id in range(cfg.num_clients)]


def test_requested_case_starts_create_preceding_server_checkpoints(tmp_path: Path) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    tr.ensure(range(5), device, checkpoint_rounds=(2, 4))

    assert {path.name for path in tr.checkpoint_dir.iterdir()} >= {
        "round_1.pt", "round_3.pt", "round_4.pt"}


def test_regenerating_a_missing_round_resumes_from_the_checkpoint_strictly_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    tr.ensure((1, 3), device)
    assert sorted(p.name for p in tr.checkpoint_dir.iterdir()) == [
        "round_0.pt", "round_1.pt", "round_2.pt", "round_3.pt"
    ]

    # Lose round 3's deltas while every checkpoint, including the one strictly before it
    # (round_2.pt), remains on disk.
    for client_id in range(cfg.num_clients):
        tr._delta_path(3, client_id).unlink()

    calls = _spy_on_training(monkeypatch)
    tr.ensure((3,), device)
    assert calls == [(3, client_id) for client_id in range(cfg.num_clients)]


def test_callers_rng_state_is_unchanged_after_ensure(tmp_path: Path) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")

    random.seed(4242)
    np.random.seed(4242)
    torch.manual_seed(4242)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    tr.ensure((1, 2), device)

    assert random.getstate() == python_state
    algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
    algorithm2, keys2, position2, has_gauss2, cached_gaussian2 = np.random.get_state()
    assert (algorithm, position, has_gauss, cached_gaussian) == (algorithm2, position2, has_gauss2, cached_gaussian2)
    assert np.array_equal(keys, keys2)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_identity_is_available_only_after_ensure_and_has_expected_keys(tmp_path: Path) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev-x")
    with pytest.raises(AssertionError, match="num_classes"):
        tr.identity()

    tr.ensure((1,), device)
    identity = tr.identity()
    assert set(identity) == {"trace_version", "fl_config", "dataset_revision", "device"}
    assert identity["trace_version"] == trace_module._TRACE_VERSION
    assert identity["dataset_revision"] == "rev-x"
    assert identity["fl_config"]["num_classes"] == 2
    assert identity["fl_config"]["seed"] == cfg.seed
    assert identity["device"]["type"] == "cpu"
    assert json.dumps(identity)  # must be JSON-serializable as-is


def test_load_delta_returns_flat_cpu_float32_tensor(tmp_path: Path) -> None:
    device, cfg = torch.device("cpu"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    tr.ensure((1,), device)

    delta = tr.load_delta(1, 0)
    assert delta.ndim == 1 and delta.dtype == torch.float32 and delta.device.type == "cpu"
    assert torch.equal(delta, torch.load(tr._delta_path(1, 0), weights_only=True))
    with pytest.raises(AssertionError, match="Missing trace delta"):
        tr.load_delta(99, 0)


def test_trace_rejects_nonzero_loader_workers(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="num_loader_workers"):
        Trace(tmp_path / "trace", _cfg(num_loader_workers=2), dataset_revision="rev1")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="host needs two CUDA devices for this check")
def test_trace_seeds_and_restores_the_requested_cuda_device_not_the_ambient_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ambient current device is cuda:0; the trace is asked to run on cuda:1.

    Proves the requested device is authoritative (training actually runs on index 1) and that
    the caller's ambient current device (0) is restored, not silently left on 1.
    """
    torch.cuda.set_device(0)
    requested = torch.device("cuda:1")
    cfg = _cfg(num_classes=None, mixed_precision=False)

    seen_indices: set[int | None] = set()
    original = trace_module.train_client_update

    def spy(model, loader, optimizer, scaler, manager, cfg_, round_id, client_id):  # type: ignore[no-untyped-def]
        seen_indices.add(next(model.parameters()).device.index)
        return original(model, loader, optimizer, scaler, manager, cfg_, round_id, client_id)

    monkeypatch.setattr(trace_module, "train_client_update", spy)

    first = Trace(tmp_path / "first", cfg, dataset_revision="gpu-rev")
    first.ensure((1, 2), requested)
    assert torch.cuda.current_device() == 0
    assert seen_indices == {1}

    # Disturb cuda:0's RNG between the two runs; an independent trace generated on cuda:1 must
    # be unaffected, proving isolation is keyed on the requested device, not the ambient one.
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(777)
    _ = torch.rand(1000, device="cuda:0")

    second = Trace(tmp_path / "second", cfg, dataset_revision="gpu-rev")
    second.ensure((1, 2), requested)
    assert torch.cuda.current_device() == 0

    for round_id in (1, 2):
        for client_id in range(cfg.num_clients):
            assert torch.equal(first.load_delta(round_id, client_id), second.load_delta(round_id, client_id))


def test_unavailable_cuda_device_fails_clearly_instead_of_falling_back_to_cpu(tmp_path: Path) -> None:
    device, cfg = torch.device(f"cuda:{torch.cuda.device_count() + 5}"), _cfg(num_classes=None)
    tr = Trace(tmp_path / "trace", cfg, dataset_revision="rev1")
    with pytest.raises(AssertionError, match="CUDA device"):
        tr.ensure((1,), device)


def test_aggregation_preserves_input_and_uses_server_momentum() -> None:
    model = nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    manager = StateDictManager(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.5)
    first = {"weight": torch.tensor([[1.0]])}
    second = {"weight": torch.tensor([[3.0]])}

    aggregate_and_update(model, [first, second], [1, 1], manager, optimizer)
    aggregate_and_update(model, [first, second], [1, 1], manager, optimizer)

    assert torch.equal(first["weight"], torch.tensor([[1.0]]))
    assert torch.equal(second["weight"], torch.tensor([[3.0]]))
    assert torch.equal(model.weight, initial + 5.0)
