"""Checkpointed deterministic raw-client-update traces for sparse FL experiments."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
import json
import platform
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from FL_code.run_fl import FLConfig
from .client import train_client_update
from .dataset import GROUP_FIELD, create_dataloader, precompute_dataset_to_shared
from .models import FLModelTemplate, initialize_model
from .server import aggregate_and_update
from .utils import StateDictManager, write_fl_config_snapshot


# Bump whenever `_generate`'s algorithm changes in a way that could produce different deltas for
# the same configuration, so an existing trace on disk cannot be silently reused as if it still
# matched. Version 2 fixed cross-device RNG leakage (`_seeded` seeded the ambient current CUDA
# device rather than the requested one) and added the fixed cudnn determinism policy below.
_TRACE_VERSION = 2
# `data_folder` is excluded: a path is not data identity, `dataset_revision` is.
_SEMANTIC_FIELDS = {
    "model_name", "dataset_name", "image_size", "num_clients",
    "local_epochs", "batch_size", "lr", "seed", "server_lr", "server_momentum",
    "server_nesterov", "augment", "label_smoothing", "grad_clip_norm", "non_iid_partition",
    "dirichlet_alpha", "channels_last", "fused_optimizer", "mixed_precision",
    "dataset_fraction", "num_loader_workers",
}
# Forced rather than merely observed, and recorded in the device signature: cudnn's autotuned
# ("benchmark") algorithm choice is itself a source of run-to-run nondeterminism, independent of
# any RNG seed.
_CUDNN_DETERMINISTIC = True
_CUDNN_BENCHMARK = False


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer, saved: dict[str, Any], device: torch.device
) -> None:
    """Load one optimizer's saved state, moving only floating buffers to `device`.

    SGD's momentum buffer is the only tensor this checkpoint format carries. A non-floating
    tensor would be a scalar step counter, which a fused/capturable optimizer requires in a
    specific placement (often CPU) that blindly moving to `device` could silently violate.
    """
    optimizer.load_state_dict(saved)
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                assert value.dtype.is_floating_point, (
                    f"Unexpected non-floating optimizer state {key!r}={value!r}; refusing to "
                    f"blindly move it to {device}."
                )
                optimizer_state[key] = value.to(device)


@contextmanager
def _seeded(seed: int, device: torch.device) -> Iterator[None]:
    """Fork RNG and cudnn-determinism state for `device`, seed it, then restore the caller's.

    `device` must carry an explicit CUDA index rather than relying on the ambient current
    device: `torch.cuda.manual_seed` and kernel dispatch both key off "current device", so a
    caller-selected device is only authoritative if this context also makes it current for its
    duration - which `torch.cuda.device` does, restoring the prior current device on exit.
    """
    assert device.type != "cuda" or device.index is not None, (
        f"Trace generation needs an explicit CUDA device index; got {device}."
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cudnn_state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    cuda_devices = [device.index] if device.type == "cuda" else []
    device_ctx = torch.cuda.device(device.index) if device.type == "cuda" else contextlib.nullcontext()
    with torch.random.fork_rng(devices=cuda_devices), device_ctx:
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = _CUDNN_DETERMINISTIC
                torch.backends.cudnn.benchmark = _CUDNN_BENCHMARK
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_state


class Trace:
    """A runner-created FL trace consumed by every sparse replay case.

    The directory owns an identity ``fl_config.json``, a managed marker (generator version,
    dataset revision, and resolved device signature), requested client deltas, and sparse
    boundary checkpoints containing all state needed to continue the same trajectory. `ensure`
    is the only entry point that trains; `load_delta` and `identity` only read what it produced.
    """

    def __init__(self, directory: Path, cfg: FLConfig, *, dataset_revision: str) -> None:
        """Open or create a trace directory, asserting it matches `cfg` and `dataset_revision`.

        `dataset_revision` is an explicit, caller-supplied fingerprint for the local dataset
        `cfg.data_folder` points at - a folder path alone is not data identity, and this is
        recorded once here rather than re-hashed by every caller on every client-round.
        """
        self.directory = directory
        self.config_path = directory / "fl_config.json"
        self.marker_path = directory / "trace.json"
        self.checkpoint_dir = directory / "checkpoints"
        assert cfg.protocol == "identity", f"Trace config protocol must be identity, got {cfg.protocol!r}."
        assert cfg.num_loader_workers == 0, (
            f"Trace supports only num_loader_workers=0 (no process-local worker state to "
            f"checkpoint); got {cfg.num_loader_workers}."
        )

        directory.mkdir(parents=True, exist_ok=True)
        if not self.marker_path.exists():
            unrelated = tuple(directory.iterdir())
            assert not unrelated, (
                f"Trace directory {directory} has files but no managed trace.json marker; use a new directory."
            )
            write_fl_config_snapshot(cfg, directory)
            self._write_marker({"version": _TRACE_VERSION, "device": None, "dataset_revision": dataset_revision})
        marker = self._read_marker()
        assert set(marker) == {"version", "device", "dataset_revision"}, (
            f"Invalid managed trace marker: {self.marker_path}."
        )
        assert marker["version"] == _TRACE_VERSION, (
            f"Trace generator version differs at {self.marker_path}: expected {_TRACE_VERSION}, "
            f"found {marker['version']}."
        )
        assert marker["dataset_revision"] == dataset_revision, (
            f"Trace dataset_revision differs at {self.marker_path}: expected {dataset_revision!r}, "
            f"found {marker['dataset_revision']!r}."
        )
        assert self.config_path.is_file(), f"Managed trace is missing {self.config_path}."
        stored = FLConfig.model_validate_json(self.config_path.read_text())
        expected = cfg.model_dump(mode="json", include=_SEMANTIC_FIELDS)
        actual = stored.model_dump(mode="json", include=_SEMANTIC_FIELDS)
        assert actual == expected, (
            f"Trace FL configuration differs at {self.config_path}: expected {expected}, found {actual}."
        )
        assert cfg.num_classes in (None, stored.num_classes), (
            f"Trace num_classes differs at {self.config_path}: expected {cfg.num_classes}, "
            f"found {stored.num_classes}."
        )
        self.cfg = stored

    def ensure(self, rounds: Iterable[int], device: torch.device) -> None:
        """Generate every missing full-client requested round, resuming from a boundary checkpoint."""
        requested = tuple(sorted(set(rounds)))
        assert requested and requested[0] >= 0, "Trace rounds must contain non-negative integers."
        if device.type == "cuda":
            assert torch.cuda.is_available(), f"Requested {device} but CUDA is not available."
            assert device.index is not None, f"Trace needs an explicit CUDA device index; got {device}."
            assert device.index < torch.cuda.device_count(), (
                f"Requested {device} but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        signature = self._device_signature(device)
        marker = self._read_marker()
        assert marker["device"] in (None, signature), (
            f"Trace device differs at {self.marker_path}: expected {marker['device']}, got {signature}."
        )
        if marker["device"] is None:
            self._write_marker(marker | {"device": signature})

        missing = [round_id for round_id in requested if any(
            not self._delta_path(round_id, client_id).is_file()
            for client_id in range(self.cfg.num_clients)
        )]
        if not missing:
            return
        self.checkpoint_dir.mkdir(exist_ok=True)
        checkpoints = {
            int(match.group(1)): path
            for path in self.checkpoint_dir.glob("round_*.pt")
            if (match := re.fullmatch(r"round_(\d+)\.pt", path.name))
        }
        preceding = [round_id for round_id in checkpoints if round_id < missing[0]]
        checkpoint_round = max(preceding, default=-1)
        self._generate(
            requested=set(requested), device=device, start_round=checkpoint_round + 1,
            final_round=max(missing), checkpoint=checkpoints.get(checkpoint_round),
        )

    def identity(self) -> dict[str, Any]:
        """Return the finalized, JSON-serializable trace identity; D records it without rebuilding it.

        Only meaningful once `ensure` has resolved `num_classes` and recorded a generation
        device: raises before that, rather than returning identity that a later `ensure` call
        could still legitimately change.
        """
        marker = self._read_marker()
        assert self.cfg.num_classes is not None, (
            "Trace identity requested before ensure() resolved num_classes."
        )
        assert marker["device"] is not None, (
            "Trace identity requested before ensure() recorded a generation device."
        )
        return {
            "trace_version": _TRACE_VERSION,
            "fl_config": self.cfg.model_dump(mode="json", include=_SEMANTIC_FIELDS | {"num_classes"}),
            "dataset_revision": marker["dataset_revision"],
            "device": marker["device"],
        }

    def load_delta(self, round_id: int, client_id: int) -> torch.Tensor:
        """Load one previously generated raw client delta: flat, float32, on CPU.

        The round/client filename convention lives only here, so callers (e.g. the sparse
        replay runner) must not keep a second implementation of it.
        """
        path = self._delta_path(round_id, client_id)
        assert path.is_file(), f"Missing trace delta {path}; call ensure() for round {round_id} first."
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.dtype == torch.float32, (
            f"Trace delta at {path} must be a flat float32 tensor; got {type(tensor).__name__} "
            f"ndim={getattr(tensor, 'ndim', None)} dtype={getattr(tensor, 'dtype', None)}."
        )
        return tensor

    def _generate(
        self, *, requested: set[int], device: torch.device, start_round: int,
        final_round: int, checkpoint: Path | None,
    ) -> None:
        with _seeded(self.cfg.seed, device):
            features, labels, groups = precompute_dataset_to_shared(
                self.cfg.dataset_name, self.cfg.data_folder, "train", torch.float32,
                self.cfg.dataset_fraction, self.cfg.seed, self.cfg.image_size,
                self.cfg.num_clients if self.cfg.dataset_name.upper() in GROUP_FIELD else None,
            )
            num_classes = int(labels.max()) + 1
            assert self.cfg.num_classes in (None, num_classes), (
                f"Trace dataset has {num_classes} classes, config declares {self.cfg.num_classes}."
            )
            if self.cfg.num_classes is None:
                self.cfg.num_classes = num_classes
                write_fl_config_snapshot(self.cfg, self.directory)
            model = initialize_model(self.cfg, device)
            manager = StateDictManager(model)
            loaders = [create_dataloader(
                features, labels, self.cfg, device, True, client_id, self.cfg.num_clients, groups
            ) for client_id in range(self.cfg.num_clients)]
            optimizers = [model.configure_optimizer() for _ in loaders]
            scalers = [torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision) for _ in loaders]
            buffers = [{name: value.detach().cpu().clone() for name, value in model.named_buffers()}
                       for _ in loaders]
            server_opt = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=self.cfg.server_lr, momentum=self.cfg.server_momentum,
                nesterov=self.cfg.server_nesterov,
            )
            if checkpoint is not None:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert state["round"] + 1 == start_round, f"Invalid checkpoint round in {checkpoint}."
                manager.load_trainable_state(model, manager.unflatten(state["server"]))
                buffers = state["buffers"]
                for optimizer, saved in zip(optimizers, state["client_optimizers"], strict=True):
                    _restore_optimizer_state(optimizer, saved, device)
                for scaler, saved in zip(scalers, state["scalers"], strict=True):
                    scaler.load_state_dict(saved)
                for loader, saved in zip(loaders, state["loaders"], strict=True):
                    assert loader.generator is not None
                    loader.generator.set_state(saved)
                _restore_optimizer_state(server_opt, state["server_optimizer"], device)

            starts = {round_id for round_id in requested if round_id - 1 not in requested}
            ends = {round_id for round_id in requested if round_id + 1 not in requested}
            for round_id in range(start_round, final_round + 1):
                if round_id in starts and round_id > start_round:
                    self._save_checkpoint(round_id - 1, model, manager, buffers, optimizers,
                                          scalers, loaders, server_opt)
                server = manager.clone_trainable(model.state_dict())
                deltas: list[torch.Tensor] = []
                for client_id, (loader, optimizer, scaler) in enumerate(
                    zip(loaders, optimizers, scalers, strict=True)
                ):
                    manager.load_trainable_state(model, server)
                    with torch.no_grad():
                        for name, value in model.named_buffers():
                            value.copy_(buffers[client_id][name].to(device=value.device))
                    with _seeded(self.cfg.seed + round_id * self.cfg.num_clients + client_id + 1, device):
                        delta = train_client_update(
                            model, loader, optimizer, scaler, manager, self.cfg, round_id, client_id
                        )
                    buffers[client_id] = {
                        name: value.detach().cpu().clone() for name, value in model.named_buffers()
                    }
                    deltas.append(delta)
                    if round_id in requested:
                        path = self._delta_path(round_id, client_id)
                        if not path.exists():
                            _atomic_save(delta, path)
                manager.load_trainable_state(model, server)
                aggregate_and_update(
                    model, [manager.unflatten(delta) for delta in deltas],
                    [len(loader.dataset) for loader in loaders], manager, server_opt,
                )
                if round_id in ends or round_id == final_round:
                    self._save_checkpoint(round_id, model, manager, buffers, optimizers,
                                          scalers, loaders, server_opt)

    def _save_checkpoint(
        self, round_id: int, model: FLModelTemplate, manager: StateDictManager,
        buffers: list[dict[str, torch.Tensor]], optimizers: list[torch.optim.Optimizer],
        scalers: list[torch.amp.GradScaler], loaders: list[DataLoader],
        server_optimizer: torch.optim.Optimizer,
    ) -> None:
        assert all(loader.generator is not None for loader in loaders)
        _atomic_save({
            "round": round_id,
            "server": manager.flatten(model.state_dict()),
            "buffers": buffers,
            "client_optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "scalers": [scaler.state_dict() for scaler in scalers],
            "loaders": [cast(torch.Generator, loader.generator).get_state() for loader in loaders],
            "server_optimizer": server_optimizer.state_dict(),
        }, self.checkpoint_dir / f"round_{round_id}.pt")

    def _delta_path(self, round_id: int, client_id: int) -> Path:
        return self.directory / f"round_{round_id}_client_{client_id}.pt"

    def _read_marker(self) -> dict[str, Any]:
        return json.loads(self.marker_path.read_text())

    def _write_marker(self, marker: dict[str, Any]) -> None:
        self.marker_path.write_text(json.dumps(marker, indent=2))

    @staticmethod
    def _device_signature(device: torch.device) -> dict[str, Any]:
        if device.type == "cuda":
            return {
                "type": "cuda", "name": torch.cuda.get_device_name(device.index),
                "capability": list(torch.cuda.get_device_capability(device.index)),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn_deterministic": _CUDNN_DETERMINISTIC, "cudnn_benchmark": _CUDNN_BENCHMARK,
            }
        return {"type": device.type, "machine": platform.machine(), "torch": torch.__version__}
