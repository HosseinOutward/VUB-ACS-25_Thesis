from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from hashlib import sha256
from pathlib import Path
import struct
from typing import Any, Callable, TypeVar, cast

from cachier import cachier
from pydantic import BaseModel
import torch


T = TypeVar("T")

# This explicit contract prevents new logging or execution fields from silently
# invalidating every learned-model entry. Add a field only when changing it must
# select a separately trained model.
MODEL_TRAINING_CONFIG_FIELDS: frozenset[str] = frozenset({
    "bins_per_plane",
    "num_planes",
    "norm_slices",
    "outlier_threshold",
    "marginal_loss",
    "train_epochs",
    "reconst_ld",
    "train_sample_size",
    "train_batch_size",
    "lr",
    "lr_step",
    "tau",
    "prior_train_repeats",
    "fused_optimizer",
    "mixed_precision",
    "source_perturbation",
    "shared_heads",
    "prior_context",
    "prior_probabilities",
    "hidden_dim",
    "layers",
})


def cache_call(
    function: Callable[..., T],
    cache_dir: Path | None,
    hash_func: Callable[[tuple[Any, ...], dict[str, Any]], str],
    overwrite: bool,
    **kwargs: Any,
) -> T:
    """Call a function directly or through a separate-file persistent Cachier cache."""
    if cache_dir is None:
        return function(**kwargs)
    # Cachier fixes its storage directory when decorating, so configurable cache
    # directories require creating the wrapper at the computation boundary.
    cached_function = cachier(
        cache_dir=cache_dir,
        separate_files=True,
        hash_func=hash_func,
    )(function)
    return cast(T, cached_function(**kwargs, cachier__overwrite_cache=overwrite))


def content_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Hash nested configuration values and tensor contents for persistent computation caches."""
    digest = sha256()
    _update_hash(digest, (args, kwargs))
    return digest.hexdigest()


def client_update_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Hash the FL inputs that define a reusable client-update trace entry."""
    assert not args
    cfg = kwargs["cfg"]
    assert isinstance(cfg, BaseModel)
    # Client updates form a reusable uncompressed FL trace. Protocol and runtime
    # output settings therefore must not prevent replaying that trace.
    ignored_fields = {
        "protocol",
        "run_name",
        "rounds",
        "num_loader_workers",
        "recalibrate_bn",
        "bn_recalib_batches",
        "client_eval_every_n_rounds",
        "training_progress_bar",
        "records_dir",
        "master_addr",
        "master_port",
        "protocol_cache_dir",
        "protocol_cache_overwrite",
        "debug_save_train_data",
        "debug_data_folder",
        "debug_save_deltas",
        "debug_save_recons",
        "debug_save_recons_rounds",
        "debug_load_from_saved_data",
        "debug_continue_from_saved_data",
        "debug_continue_then_save",
    }
    assert not ignored_fields - type(cfg).model_fields.keys()
    return content_hash(
        (),
        {
            "cache_version": 1,
            "fl_config": cfg.model_dump(exclude=ignored_fields),
            "round_id": kwargs["round_id"],
            "client_id": kwargs["client_id"],
        },
    )


def model_training_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Hash inputs that determine the learned model, excluding execution-only settings."""
    assert not args
    c_cfg = kwargs["c_cfg"]
    assert isinstance(c_cfg, BaseModel)
    assert MODEL_TRAINING_CONFIG_FIELDS <= type(c_cfg).model_fields.keys()
    # Target, side-information, bins, and soft codes remain content-hashed.
    # Configuration is opt-in so unrelated future fields reuse existing work.
    hashed_kwargs = kwargs | {
        "c_cfg": c_cfg.model_dump(include=MODEL_TRAINING_CONFIG_FIELDS) | {
            # Preserve keys written before the allowlist without letting these
            # non-semantic historical fields vary the cache identity.
            "max_side_info_count": 5,
            "pretrain_pth_dir": Path("FL_code/data/pre_trained_pth"),
            "training_progress_bar": False,
            "training_log_every_epochs": None,
            "tf32": False,
        }
    }
    return content_hash((), {"cache_version": 1, **hashed_kwargs})


def _update_hash(digest: Any, value: Any) -> None:
    digest.update(f"{type(value).__module__}.{type(value).__qualname__}:".encode())
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        # Hash semantic tensor identity before its raw bytes so equal byte streams
        # with different shapes or dtypes cannot share an entry.
        digest.update(str(tensor.dtype).encode())
        digest.update(struct.pack(f"!{tensor.ndim}q", *tensor.shape))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
        return
    if isinstance(value, BaseModel):
        _update_hash(digest, value.model_dump())
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            _update_hash(digest, key)
            _update_hash(digest, value[key])
        return
    if isinstance(value, tuple | list):
        for item in value:
            _update_hash(digest, item)
        return
    if isinstance(value, slice):
        _update_hash(digest, (value.start, value.stop, value.step))
        return
    if isinstance(value, Path):
        digest.update(str(value).encode())
        return
    if isinstance(value, Enum):
        _update_hash(digest, value.value)
        return
    if isinstance(value, type):
        digest.update(f"{value.__module__}.{value.__qualname__}".encode())
        return
    if isinstance(value, str | bytes):
        digest.update(value.encode() if isinstance(value, str) else value)
        return
    if isinstance(value, bool | int):
        digest.update(str(value).encode())
        return
    if isinstance(value, float):
        digest.update(struct.pack("!d", value))
        return
    raise TypeError(f"Cannot build a stable cache hash for {type(value).__name__}.")
