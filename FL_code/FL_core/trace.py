"""Deterministic identity-FL client-update streams with an optional continuation checkpoint."""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
import platform
import random
from typing import Any, Iterator, cast

import numpy as np
import torch

from FL_code.run_fl import FLConfig
from .client import train_client_update
from .dataset import GROUP_FIELD, create_dataloader, precompute_dataset_to_shared
from .explicit_cache import CacheEntry
from .models import initialize_model
from .server import aggregate_and_update
from .utils import StateDictManager


# Bump whenever `_run`'s algorithm changes in a way that could produce different deltas for the
# same configuration, so completed experiments and saved continuation states cannot be mistaken
# for the same trajectory.
# Version 2 fixed cross-device RNG leakage (`_seeded` seeded the ambient
# current CUDA device rather than the requested one) and added the fixed cudnn determinism policy.
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
    """The one uncompressed FL trajectory every sparse replay case reads its raw updates from.

    Client updates are never stored: a stream regenerates the sequential training state and yields
    only the requested rounds. The caller may supply one checkpoint entry holding the complete
    continuation state at the start of some round, which lets a later stream skip the rounds before it.
    """

    def __init__(
        self, cfg: FLConfig, *, dataset_revision: str, device: torch.device,
    ) -> None:
        """Bind one FL configuration, dataset fingerprint, and generation device.

        `dataset_revision` is an explicit, caller-supplied fingerprint for the local dataset
        `cfg.data_folder` points at - a folder path alone is not data identity, and this is
        recorded once here rather than re-hashed by every caller on every client-round.
        """
        assert cfg.protocol == "identity", f"Trace config protocol must be identity, got {cfg.protocol!r}."
        assert cfg.num_loader_workers == 0, (
            f"Trace supports only num_loader_workers=0; got {cfg.num_loader_workers}."
        )
        assert cfg.num_classes is not None, (
            "Trace needs an explicit num_classes in its FL config: it is part of trace identity, "
            "which must be known before any round is generated."
        )
        if device.type == "cuda":
            assert torch.cuda.is_available(), f"Requested {device} but CUDA is not available."
            assert device.index is not None, f"Trace needs an explicit CUDA device index; got {device}."
            assert device.index < torch.cuda.device_count(), (
                f"Requested {device} but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        self.cfg = cfg
        self.dataset_revision = dataset_revision
        self.device = device

    def identity(self) -> dict[str, Any]:
        """Return the JSON-serializable identity of the trajectory this trace generates."""
        return {
            "trace_version": _TRACE_VERSION,
            "fl_config": self.cfg.model_dump(mode="json", include=_SEMANTIC_FIELDS | {"num_classes"}),
            "dataset_revision": self.dataset_revision,
            "device": self._device_signature(self.device),
        }

    def stream(
        self, first_round: int, last_round: int,
        checkpoint: CacheEntry[dict[str, Any]] | None = None, save_round: int | None = None,
    ) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
        """Yield `(round_id, per-client deltas)` for `first_round..last_round` inclusive.

        Earlier rounds are retrained but not yielded because aggregation makes the trajectory
        sequential. Training resumes from `checkpoint` when it holds a state at or before
        `first_round`, and `checkpoint` is overwritten with the state at the start of `save_round`.
        The caller keys `checkpoint` on `identity()`, the only thing a stored state must match.
        """
        assert 0 <= first_round <= last_round, f"Invalid trace round range {first_round}..{last_round}."
        assert save_round is None or (checkpoint is not None and first_round <= save_round <= last_round + 1), (
            f"save_round={save_round} must lie within first_round={first_round} and "
            f"last_round + 1={last_round + 1}, with a checkpoint to save into.")
        yield from self._run(first_round, last_round, checkpoint, save_round)

    def _run(
        self, first_round: int, final_round: int,
        checkpoint: CacheEntry[dict[str, Any]] | None, save_round: int | None,
    ) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
        # Setup has to finish before the first yield: a `_seeded` scope may not span one, or the
        # consumer's own seeding would run inside this trace's forked RNG state and leak into the
        # rounds after it. Each client-round below is seeded on its own, so nothing else needs to.
        with _seeded(self.cfg.seed, self.device):
            features, labels, groups = precompute_dataset_to_shared(
                self.cfg.dataset_name, self.cfg.data_folder, "train", torch.float32,
                self.cfg.dataset_fraction, self.cfg.seed, self.cfg.image_size,
                self.cfg.num_clients if self.cfg.dataset_name.upper() in GROUP_FIELD else None,
            )
            assert self.cfg.num_classes == int(labels.max()) + 1, (
                f"Trace dataset has {int(labels.max()) + 1} classes, config declares {self.cfg.num_classes}.")
            model = initialize_model(self.cfg, self.device)
            manager = StateDictManager(model)
            loaders = [create_dataloader(
                features, labels, self.cfg, self.device, True, client_id, self.cfg.num_clients, groups
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
            assert all(loader.generator is not None for loader in loaders)
            start_round = 0
            state = checkpoint.load() if checkpoint is not None and checkpoint.path.exists() else None
            if state is not None and state["round"] <= first_round:
                start_round = state["round"]
                manager.load_trainable_state(model, manager.unflatten(state["server"]))
                buffers = state["buffers"]
                # Optimizers move their loaded state onto their parameters' device themselves.
                for optimizer, saved in zip(optimizers, state["client_optimizers"], strict=True):
                    optimizer.load_state_dict(saved)
                for scaler, saved in zip(scalers, state["scalers"], strict=True):
                    scaler.load_state_dict(saved)
                for loader, saved in zip(loaders, state["loaders"], strict=True):
                    loader.generator.set_state(saved)
                server_opt.load_state_dict(state["server_optimizer"])
        for round_id in range(start_round, final_round + 1):
            server = manager.clone_trainable(model.state_dict())
            deltas: list[torch.Tensor] = []
            for client_id, (loader, optimizer, scaler) in enumerate(
                zip(loaders, optimizers, scalers, strict=True)
            ):
                manager.load_trainable_state(model, server)
                with torch.no_grad():
                    for name, value in model.named_buffers():
                        value.copy_(buffers[client_id][name].to(device=value.device))
                with _seeded(self.cfg.seed + round_id * self.cfg.num_clients + client_id + 1, self.device):
                    delta = train_client_update(
                        model, loader, optimizer, scaler, manager, self.cfg, round_id, client_id
                    )
                buffers[client_id] = {
                    name: value.detach().cpu().clone() for name, value in model.named_buffers()
                }
                deltas.append(delta)
            manager.load_trainable_state(model, server)
            # Aggregation clones what it reads, so the yielded deltas stay the raw updates.
            aggregate_and_update(
                model, [manager.unflatten(delta) for delta in deltas],
                [len(loader.dataset) for loader in loaders], manager, server_opt,
            )
            if round_id + 1 == save_round:
                cast(CacheEntry, checkpoint).save({
                    "round": save_round,
                    "server": manager.flatten(model.state_dict()),
                    "buffers": buffers,
                    "client_optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "scalers": [scaler.state_dict() for scaler in scalers],
                    "loaders": [loader.generator.get_state() for loader in loaders],
                    "server_optimizer": server_opt.state_dict(),
                })
            if round_id >= first_round:
                yield round_id, tuple(deltas)

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
