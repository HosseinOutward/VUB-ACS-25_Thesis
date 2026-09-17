from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from hashlib import sha256
from pathlib import Path
import struct
from typing import Any, Callable, Protocol, TypeVar, cast, runtime_checkable

from cachier import cachier
from pydantic import BaseModel
import torch


T = TypeVar("T")
HASH_CHUNK_ELEMENTS = 1 << 24


@runtime_checkable
class ModelTrainingCacheConfig(Protocol):
    """Configuration contract consumed by the generic learned-model cache."""

    def training_cache_payload(self) -> Mapping[str, Any]:
        """Return only configuration values that determine trained model parameters."""


@runtime_checkable
class ContentHashable(Protocol):
    """Contract for values whose logical contents can be streamed into a cache digest."""

    logical_size: int

    def update_content_hash(self, digest: Any) -> None:
        """Append ordered logical float32 bytes to ``digest`` without materializing the value."""


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
    assert isinstance(c_cfg, ModelTrainingCacheConfig), (
        f"{type(c_cfg).__name__} must implement training_cache_payload() before it can use "
        "model_training_hash.")
    # Tensor inputs and model-specific options remain content-hashed; the config owns the decision
    # about which of its fields are semantic to training.
    hashed_kwargs = kwargs | {"c_cfg": c_cfg.training_cache_payload()}
    return content_hash((), {"cache_version": 1, **hashed_kwargs})


def _update_hash(digest: Any, value: Any) -> None:
    # Tensor-like logical values share one framing so a retained history handle hashes exactly
    # like the flat float32 tensor it represents, regardless of its physical backend.
    if isinstance(value, torch.Tensor):
        flat = value.detach().contiguous().reshape(-1)
        _update_typed_hash(digest, value.dtype, tuple(value.shape), (
            flat[start:start + HASH_CHUNK_ELEMENTS].cpu().view(torch.uint8).numpy()
            for start in range(0, flat.numel(), HASH_CHUNK_ELEMENTS)
        ))
        return
    if isinstance(value, ContentHashable):
        # Physical storage (chunking, on-disk dtype, deduplication) must not leak into cache
        # identity, so a stored history vector presents its header as a flat CPU float32 tensor
        # and streams its own logical bytes afterward.
        _update_typed_hash(digest, torch.float32, (value.logical_size,), ())
        value.update_content_hash(digest)
        return
    digest.update(f"{type(value).__module__}.{type(value).__qualname__}:".encode())
    if value is None:
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


def _update_typed_hash(
    digest: Any, dtype: torch.dtype, shape: tuple[int, ...], byte_chunks: Iterable[Any],
) -> None:
    """Hash a tensor-shaped value's dtype, shape, and ordered raw byte chunks.

    Hashing semantic identity (dtype, shape) before raw bytes keeps equal byte streams of
    different shapes or dtypes from sharing an entry; existing model-training cache identities
    depend on this exact framing.
    """
    digest.update(f"{torch.Tensor.__module__}.{torch.Tensor.__qualname__}:".encode())
    digest.update(str(dtype).encode())
    digest.update(struct.pack(f"!{len(shape)}q", *shape))
    for chunk in byte_chunks:
        digest.update(chunk)
