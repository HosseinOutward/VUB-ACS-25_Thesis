"""Content hashing, atomic publication, and caches explicitly declared at their call site.

No operation is cached today. A future call site supplies the directory and the complete
semantic match value itself, e.g. ``CacheEntry(directory, match).resolve(compute)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import is_dataclass, asdict
from enum import Enum
from hashlib import sha256
import fcntl
import os
from pathlib import Path
import pickle
import struct
import tempfile
from typing import Any, Iterator, Protocol, cast

import torch
from pydantic import BaseModel


HASH_CHUNK_ELEMENTS = 1 << 24


class CacheCodec[T](Protocol):
    """Serialization contract for one explicit cache entry type."""

    suffix: str

    def load(self, path: Path) -> T:
        """Load one value from ``path``."""

    def save(self, value: T, path: Path) -> None:
        """Save one value to ``path``."""


class PickleCodec[T]:
    """Trusted-local pickle serialization for arbitrary Python values."""

    suffix = ".pkl"

    def load(self, path: Path) -> T:
        with path.open("rb") as stream:
            return cast(T, pickle.load(stream))  # noqa: S301 - explicit local artifact format

    def save(self, value: T, path: Path) -> None:
        with path.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


class TorchCodec[T]:
    """``torch.save`` serialization for tensor containers, loaded onto the CPU as weights only."""

    suffix = ".pt"

    def load(self, path: Path) -> T:
        return cast(T, torch.load(path, map_location="cpu", weights_only=True))

    def save(self, value: T, path: Path) -> None:
        torch.save(value, path)


def content_hash(value: object) -> str:
    """Return a deterministic SHA-256 identity for supported match values."""
    digest = sha256()
    _update_hash(digest, value)
    return digest.hexdigest()


def _part(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("!Q", len(value)))
    digest.update(value)


def _update_hash(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().reshape(-1).cpu().view(torch.uint8)
        _part(digest, b"torch.Tensor")
        _part(digest, str(value.dtype).encode())
        _part(digest, struct.pack(f"!{value.ndim}q", *value.shape))
        digest.update(struct.pack("!Q", tensor.numel()))
        for start in range(0, tensor.numel(), HASH_CHUNK_ELEMENTS):
            digest.update(tensor[start:start + HASH_CHUNK_ELEMENTS].numpy())
        return

    _part(digest, f"{type(value).__module__}.{type(value).__qualname__}".encode())
    if value is None:
        return
    if isinstance(value, BaseModel):
        _update_hash(digest, value.model_dump())
        return
    if is_dataclass(value) and not isinstance(value, type):
        _update_hash(digest, asdict(value))
        return
    if isinstance(value, Mapping):
        pairs = sorted(((content_hash(key), key, item) for key, item in value.items()), key=lambda pair: pair[0])
        _part(digest, struct.pack("!Q", len(pairs)))
        for _, key, item in pairs:
            _update_hash(digest, key)
            _update_hash(digest, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        _part(digest, struct.pack("!Q", len(value)))
        for item in value:
            _update_hash(digest, item)
        return
    if isinstance(value, set | frozenset):
        for item_hash in sorted(content_hash(item) for item in value):
            _part(digest, bytes.fromhex(item_hash))
        return
    if isinstance(value, Path):
        _part(digest, str(value).encode())
        return
    if isinstance(value, Enum):
        _update_hash(digest, value.value)
        return
    if isinstance(value, type):
        _part(digest, f"{value.__module__}.{value.__qualname__}".encode())
        return
    if isinstance(value, str):
        _part(digest, value.encode())
        return
    if isinstance(value, bytes | bytearray):
        _part(digest, bytes(value))
        return
    if isinstance(value, bool):
        _part(digest, b"1" if value else b"0")
        return
    if isinstance(value, int):
        _part(digest, str(value).encode())
        return
    if isinstance(value, float):
        _part(digest, struct.pack("!d", value))
        return
    raise TypeError(f"Cannot build a stable content hash for {type(value).__name__}.")


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling path and atomically publish it on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        yield temporary_path
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class CacheEntry[T]:
    """One explicitly identified, lazily computed local artifact."""

    def __init__(self, directory: Path, match: object, codec: CacheCodec[T] | None = None) -> None:
        self.directory = directory
        self.codec = codec or PickleCodec()
        assert self.codec.suffix.startswith("."), "cache codec suffix must start with '.'"
        self.identity = content_hash(match)
        self.path = self.directory / f"{self.identity}{self.codec.suffix}"

    def resolve(self, compute: Callable[[], T], *, refresh: bool = False) -> T:
        """Load this entry or compute and publish it atomically when absent."""
        if self.path.exists() and not refresh:
            return self.load()
        self.directory.mkdir(parents=True, exist_ok=True)
        # Held until the file closes, so parallel cases wait for one computation instead of repeating it.
        with self.path.with_name(f".{self.path.name}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists() and not refresh:
                return self.load()
            value = compute()
            self.save(value)
            return value

    def save(self, value: T) -> None:
        """Atomically publish ``value``, replacing whatever this identity held before."""
        with atomic_output(self.path) as temporary_path:
            self.codec.save(value, temporary_path)

    def load(self) -> T:
        """Load the published value; a missing or unreadable entry fails naming its path."""
        try:
            return self.codec.load(self.path)
        except Exception as error:
            error.add_note(f"while loading explicit cache entry {self.path}")
            raise
