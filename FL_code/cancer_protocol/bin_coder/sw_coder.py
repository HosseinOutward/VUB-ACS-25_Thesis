"""Fixed-rate Slepian-Wolf coding for learned quantizer symbols.

This module implements the compression-relevant part of Zhang and Tao's multilevel
syndrome design: quantizer indices are split least-significant bit first, each binary plane is
encoded as an LDPC syndrome, and the decoder conditions each plane on its decoded lower planes
and its own side information.  The paper's Gaussian-CEO estimator is replaced by a generic
decoder-side conditional prior supplied by the quantizer.

Two code families live here.  ``balanced_4cycle_v1`` builds one purpose-made parity matrix per
rate, which is the better code at any single rate but forces the rate to be decided before a bit
is sent.  ``ldpca_split_v1`` is Varodayan et al.'s rate-adaptive accumulator: one rate-one mother
code per plane whose accumulated syndrome any nested subset of levels can be read from, so a
decoder can climb from level to level, one bit per level, until the block decodes.

Both are used the same way at a fixed rate: :meth:`SlepianWolfCodec.encode_syndromes` receives
only symbols and an immutable rate plan, and the prior and side information exist only at the
decoder.  :meth:`SlepianWolfCodec.decode_ladder` is the LDPCA-only path where the rate is instead
discovered per block over a feedback channel; it needs a per-block verification, because a
satisfied syndrome is not proof of a correct decode.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from functools import cache
from heapq import heappop, heappush
from math import ceil, log2
from multiprocessing import get_context
import os
import struct
from time import perf_counter
from typing import Literal, Protocol, Self

from ldpc.mod2 import rank as gf2_rank
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import sparse
import torch


# Zhang and Tao do not publish their degree profile. This operational default is the
# node-perspective conversion of Liveris et al.'s rate-1/2 irregular ensemble in Eqs. (5)-(6),
# independently evaluated here at the paper's shorter block length and variable rates.
LIVERIS_2002_RATE_HALF_NODE_DISTRIBUTION: tuple[tuple[int, float], ...] = (
    (2, 0.4831563221664648),
    (3, 0.29236969702105664),
    (6, 0.10109102919383119),
    (7, 0.0606613419357665),
    (20, 0.0627216096828809),
)
DEFAULT_DEGREE_DISTRIBUTION = LIVERIS_2002_RATE_HALF_NODE_DISTRIBUTION
MATRIX_BACKEND: Literal["balanced_4cycle_v1"] = "balanced_4cycle_v1"
LDPCA_BACKEND: Literal["ldpca_split_v1"] = "ldpca_split_v1"
MatrixBackend = Literal["balanced_4cycle_v1", "ldpca_split_v1"]
PAYLOAD_HEADER = struct.Struct(">Q8s")
# Sentinel load for check rows this variable already uses; larger than any reachable degree sum.
_MASKED_LOAD = np.iinfo(np.int64).max
# Parity matrices are ~160 KB each and a rate search touches a new one per probe, so the
# process-local cache is bounded.
MATRIX_MEMORY_CACHE_SIZE = 256
# Below this block length a matrix builds in milliseconds and forking costs more than it saves.
PARALLEL_BUILD_MIN_BLOCK_LENGTH = 512

# Identifies one parity matrix: everything the construction reads, in plan order.
MatrixKey = tuple[str, int, int, tuple[tuple[int, float], ...], int, int, int, int, int]
# One LDPCA mother code serves every level of its plane, so neither the row count nor the
# full-rank retry budget that only the per-rate constructor reads belong in its key.
MotherKey = tuple[str, int, int, tuple[tuple[int, float], ...], int, int, int]


class ConditionalPrior(Protocol):
    """Decoder-owned provider of a stage prior conditioned on decoded stages and side info."""

    def __call__(self, decoded_stages: torch.Tensor, side_info: torch.Tensor, stage: int) -> torch.Tensor:
        """Return ``q(U_stage | U_<stage, side_info)`` with shape ``(positions, alphabet)``."""


class BlockRecovery(Protocol):
    """Encoder-side supplier of the raw bits of blocks whose syndrome failed to decode.

    Calling one is a request over the feedback channel and costs ``block_length`` uplink bits per
    block; the decoder invokes it before the next plane reads the recovered prefix, so a failure
    never propagates.  Passing no recovery leaves belief-propagation failures in the output, which
    is the paper's one-shot behaviour.
    """

    def __call__(self, stage: int, bit: int, blocks: torch.Tensor) -> torch.Tensor:
        """Return the ``(blocks.numel(), block_length)`` true plane bits of the named blocks."""


class PlaneVerification(Protocol):
    """Decoder-side check that a decoded block really is the source block.

    A satisfied syndrome only says the decode is *a* solution.  Near and below the conditional
    entropy belief propagation regularly settles on a different word with the same syndrome:
    measured here, one in sixty blocks still decoded wrongly while every syndrome checked out.
    Varodayan et al. send a per-block CRC for exactly this reason; the transport supplies whatever
    it chose to send, and without one no amount of repair makes a decode exact.
    """

    def __call__(self, stage: int, bit: int, plane: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """Return a mask over ``blocks`` that is False where the block has to be resent."""


class LadderChannel(Protocol):
    """Encoder-side LDPCA accumulator, serving one rung of the ladder per decoder request.

    ``request`` is the uplink.  Level ``k + 1``'s transmitted positions are level ``k``'s plus one,
    so asking for a higher rung costs exactly the bits it adds and the decoder keeps everything it
    already holds.  ``raw`` is the ladder's terminal rung -- the block itself -- which exists only
    because the merged code at level ``block_length`` is not quite full rank and belief propagation
    is not guaranteed; a run in which it is ever called says so in its result.
    """

    def request(self, plane: int, blocks: torch.Tensor, level: int) -> torch.Tensor:
        """Return the ``(blocks.numel(), level)`` accumulated bits transmitted at ``level``."""

    def raw(self, plane: int, blocks: torch.Tensor) -> torch.Tensor:
        """Return the ``(blocks.numel(), block_length)`` true plane bits of the named blocks."""


class SWCoderConfig(BaseModel):
    """Runtime settings used by the fixed-rate Slepian-Wolf encoder and decoder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int = Field(default=2400, ge=32)
    max_bp_iterations: int = Field(default=100, ge=1)
    damping: float = Field(default=0.9, gt=0.0, le=1.0)
    llr_clip: float = Field(default=20.0, gt=0.0)
    code_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    degree_distribution: tuple[tuple[int, float], ...] = DEFAULT_DEGREE_DISTRIBUTION
    girth_candidates: int = Field(default=8, ge=1)
    matrix_attempts: int = Field(default=8, ge=1)
    message_budget: int = Field(default=8_000_000, ge=1)
    # Rows added per LDPCA rung.  Purely a decoder-side choice -- the encoder serves whatever
    # level it is asked for -- trading a mean overshoot of half a step against one belief
    # propagation pass per rung.  At the default block length a step is 1% of the rate.
    ladder_step: int = Field(default=24, ge=1)
    matrix_backend: MatrixBackend = MATRIX_BACKEND

    @model_validator(mode="after")
    def validate_degree_distribution(self) -> Self:
        """Validate the structural invariants of the irregular variable-node profile."""
        assert self.degree_distribution, "degree_distribution must not be empty."
        assert all(degree >= 2 and fraction > 0 for degree, fraction in self.degree_distribution)
        assert abs(sum(fraction for _, fraction in self.degree_distribution) - 1.0) < 1e-9, (
            "degree_distribution fractions must sum to one."
        )
        return self


class SWRatePlan(BaseModel):
    """Shared one-shot syndrome dimensions produced by the decoder and consumed at both ends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 2
    block_length: int = Field(ge=32)
    alphabet: int = Field(ge=2)
    syndrome_rows: tuple[tuple[int, ...], ...]
    code_seed: int = Field(ge=0, le=2**32 - 1)
    degree_distribution: tuple[tuple[int, float], ...]
    girth_candidates: int = Field(ge=1)
    matrix_attempts: int = Field(ge=1)
    matrix_backend: MatrixBackend

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Validate the stage/bit layout and every parity-check row count."""
        assert self.version == 2, f"unsupported Slepian-Wolf plan version {self.version}."
        assert self.alphabet & (self.alphabet - 1) == 0, f"alphabet {self.alphabet} must be a power of two."
        bit_width = int(log2(self.alphabet))
        assert self.syndrome_rows, "syndrome_rows must contain at least one stage."
        assert all(len(stage) == bit_width for stage in self.syndrome_rows), (
            f"every stage must contain {bit_width} bit-plane rates."
        )
        assert all(
            0 <= rows <= self.block_length for stage in self.syndrome_rows for rows in stage
        ), f"syndrome rows must lie in [0, {self.block_length}]."
        assert self.degree_distribution, "degree_distribution must not be empty."
        assert all(degree >= 2 and fraction > 0 for degree, fraction in self.degree_distribution)
        assert abs(sum(fraction for _, fraction in self.degree_distribution) - 1.0) < 1e-9
        return self

    @classmethod
    def initial(cls, num_stages: int, alphabet: int, cfg: SWCoderConfig) -> Self:
        """Create the rate-one bootstrap plan used before decoder observations exist."""
        assert num_stages >= 1
        assert alphabet >= 2 and alphabet & (alphabet - 1) == 0
        full_rate = (cfg.block_length,) * int(log2(alphabet))
        return cls(
            block_length=cfg.block_length,
            alphabet=alphabet,
            syndrome_rows=(full_rate,) * num_stages,
            code_seed=cfg.code_seed,
            degree_distribution=cfg.degree_distribution,
            girth_candidates=cfg.girth_candidates,
            matrix_attempts=cfg.matrix_attempts,
            matrix_backend=cfg.matrix_backend,
        )

    @property
    def identifier(self) -> str:
        """Return the stable 64-bit identifier carried by each payload."""
        return sha256(self.model_dump_json().encode()).hexdigest()[:16]

    @property
    def description_bits(self) -> int:
        """Return the canonical plan size if the complete JSON description is transmitted."""
        return len(self.model_dump_json().encode()) * 8

    @property
    def num_stages(self) -> int:
        return len(self.syndrome_rows)

    @property
    def bit_width(self) -> int:
        return int(log2(self.alphabet))


class SWPayload(BaseModel):
    """Wire payload produced by a worker encoder and consumed by the server decoder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions: int = Field(gt=0, le=2**64 - 1)
    plan_id: str = Field(min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")
    syndromes: bytes

    def to_bytes(self) -> bytes:
        """Serialize the fixed-size header and packed syndrome stream."""
        return PAYLOAD_HEADER.pack(self.positions, bytes.fromhex(self.plan_id)) + self.syndromes

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        """Parse a serialized Slepian-Wolf payload."""
        assert len(payload) >= PAYLOAD_HEADER.size, (
            f"payload has {len(payload)} bytes; expected at least {PAYLOAD_HEADER.size}."
        )
        positions, plan_id = PAYLOAD_HEADER.unpack_from(payload)
        return cls(positions=positions, plan_id=plan_id.hex(), syndromes=payload[PAYLOAD_HEADER.size :])

    @property
    def wire_bits(self) -> int:
        return len(self.to_bytes()) * 8



class PlaneHashes:
    """Random GF(2) block hashes, sent with the syndromes and checked after decoding.

    The role Varodayan et al.'s CRC plays: it is what turns "the syndrome is satisfied" into
    "this is the source block".  ``hash_bits`` is a rate, ``hash_bits / block_length`` per plane,
    against a false-accept probability of ``2 ** -hash_bits`` per decode attempt -- and a ladder
    makes many attempts per block, so it needs more bits than a one-shot transport does.
    """

    def __init__(self, block_length: int, hash_bits: int, seed: int) -> None:
        assert hash_bits >= 1, "block verification needs at least one hash bit."
        self.block_length = block_length
        self.hash_bits = hash_bits
        self.projection = torch.as_tensor(
            np.random.default_rng([seed, block_length, hash_bits]).integers(
                0, 2, (block_length, hash_bits)
            ),
            dtype=torch.float32,
        )

    def digest(self, plane: torch.Tensor) -> torch.Tensor:
        """Return each block's hash; the float product is exact at these block lengths."""
        return (plane @ self.projection.to(plane)) % 2

    def digest_planes(self, planes: torch.Tensor) -> torch.Tensor:
        """Return the block hashes of every ``(planes, blocks, block_length)`` plane."""
        return torch.stack([self.digest(plane) for plane in planes])

    def pack(self, digests: torch.Tensor) -> bytes:
        """Pack every plane's hashes into the bytes that travel with the syndromes."""
        return np.packbits(digests.reshape(-1).to(torch.uint8).numpy(), bitorder="little").tobytes()

    def unpack(self, packed: bytes, planes: int) -> torch.Tensor:
        """Recover transmitted hashes as ``(planes, blocks, hash_bits)``."""
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        return torch.as_tensor(unpacked.astype(np.float32)).view(planes, -1, self.hash_bits)


class HashVerification:
    """Compare decoded blocks against the hashes their encoder sent."""

    def __init__(self, hashes: PlaneHashes, digests: torch.Tensor, bit_width: int) -> None:
        self.hashes = hashes
        self.digests = digests
        self.bit_width = bit_width

    def __call__(
        self, stage: int, bit: int, plane: torch.Tensor, blocks: torch.Tensor
    ) -> torch.Tensor:
        expected = self.digests[stage * self.bit_width + bit].to(plane.device)[blocks]
        return (self.hashes.digest(plane) == expected).all(dim=1)


class SWLadderPayload(BaseModel):
    """Every accumulated LDPCA bit one worker sent, and the rung each block stopped at.

    Produced by :class:`SWLadder` once the decode that drove the feedback has finished, and
    replayed at the receiver by :class:`SWLadderReplay`.  ``levels[plane][block]`` is both the rung
    the decoder settled on and that block's cost in bits, which is what makes the accounting of a
    rate-adaptive transport a sum rather than a guess.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions: int = Field(gt=0)
    plan_id: str = Field(min_length=16, max_length=16, pattern=r"^[0-9a-f]{16}$")
    block_length: int = Field(ge=32)
    levels: tuple[tuple[int, ...], ...]
    accumulated: bytes
    requests: int = Field(ge=0)
    raw: tuple[tuple[int, int], ...] = ()
    raw_planes: bytes = b""

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Require one packed accumulator prefix per block and one packed block per raw rung."""
        assert self.levels and all(len(plane) == len(self.levels[0]) for plane in self.levels)
        assert all(1 <= level <= self.block_length for plane in self.levels for level in plane), (
            f"every ladder level must lie in 1..{self.block_length}."
        )
        assert len(self.accumulated) == ceil(self.uplink_syndrome_bits / 8), (
            f"{self.uplink_syndrome_bits} accumulated bits do not fit {len(self.accumulated)} bytes."
        )
        assert len(self.raw_planes) == ceil(len(self.raw) * self.block_length / 8), (
            f"{len(self.raw)} raw blocks do not fit {len(self.raw_planes)} bytes."
        )
        return self

    @property
    def uplink_syndrome_bits(self) -> int:
        return sum(map(sum, self.levels))

    @property
    def uplink_bits(self) -> int:
        """Return what the ladder cost the uplink: one bit per rung climbed, plus any raw block."""
        return self.uplink_syndrome_bits + len(self.raw) * self.block_length

    @property
    def blocks(self) -> int:
        return len(self.levels[0])


@cache
def _boundary_index(block_length: int, level: int) -> torch.Tensor:
    """Return the accumulator positions transmitted at one ladder level."""
    return torch.as_tensor(split_boundaries(block_length, level), dtype=torch.long) - 1


def _ladder_syndrome(accumulated: torch.Tensor) -> torch.Tensor:
    """Difference an accumulated prefix back into the merged code's syndrome.

    The accumulator is why a rung costs one bit: consecutive transmitted values differ by the sum
    of the mother syndromes in between, which is exactly the merged check the decoder needs.
    """
    syndrome = accumulated.clone()
    syndrome[:, 1:] = (accumulated[:, 1:] - accumulated[:, :-1]) % 2
    return syndrome


class SWLadder:
    """Encoder side of the LDPCA feedback channel: the accumulated syndrome, served in rungs.

    The encoder computes the whole rate-one accumulator without knowing any rate; which prefix of
    it leaves the machine is the decoder's decision, arriving later over the feedback channel.
    """

    def __init__(self, accumulated: torch.Tensor, planes: torch.Tensor, block_length: int) -> None:
        assert accumulated.shape == planes.shape
        self.accumulated = accumulated
        self.planes = planes
        self.block_length = block_length
        self.levels = torch.zeros(accumulated.shape[:2], dtype=torch.long)
        self.requests = 0
        self.raw_requests: list[tuple[int, int]] = []

    def request(self, plane: int, blocks: torch.Tensor, level: int) -> torch.Tensor:
        """Serve one rung and charge the bits it added to each named block."""
        wanted = blocks.cpu()
        assert 1 <= level <= self.block_length, f"rung {level} outside 1..{self.block_length}."
        assert bool((self.levels[plane, wanted] <= level).all()), "a ladder never climbs down."
        self.levels[plane, wanted] = level
        self.requests += int(wanted.numel())
        return self.accumulated[plane][wanted][:, _boundary_index(self.block_length, level)].float()

    def raw(self, plane: int, blocks: torch.Tensor) -> torch.Tensor:
        """Serve the terminal rung: the plane bits themselves."""
        self.raw_requests.extend((plane, int(block)) for block in blocks.cpu())
        return self.planes[plane][blocks.cpu()]

    def payload(self, positions: int, plan_id: str) -> SWLadderPayload:
        """Return the bits that actually crossed the uplink, packed in plane and block order."""
        levels = [[int(level) for level in plane] for plane in self.levels]
        prefixes = [
            self.accumulated[plane, block][_boundary_index(self.block_length, level)]
            for plane, plane_levels in enumerate(levels)
            for block, level in enumerate(plane_levels)
        ]
        raw = [self.planes[plane][block] for plane, block in self.raw_requests]
        return SWLadderPayload(
            positions=positions,
            plan_id=plan_id,
            block_length=self.block_length,
            levels=tuple(tuple(plane) for plane in levels),
            accumulated=_pack(torch.cat(prefixes)),
            requests=self.requests,
            raw=tuple(self.raw_requests),
            raw_planes=_pack(torch.cat(raw)) if raw else b"",
        )


class SWLadderReplay:
    """Serve a recorded ladder to a rerun of the decode and refuse any rung nobody sent.

    The worker ran the decoder's half of the protocol to discover the levels; the receiver runs it
    again for real.  Both must ask for the same rungs, so asking for one the payload does not carry
    is a determinism failure and is raised rather than papered over.
    """

    def __init__(self, payload: SWLadderPayload) -> None:
        self.payload = payload
        self.levels = torch.as_tensor(payload.levels, dtype=torch.long)
        self.requests = 0
        block_length = payload.block_length
        self.accumulated = torch.zeros((*self.levels.shape, block_length), dtype=torch.uint8)
        sent = _unpack(payload.accumulated, payload.uplink_syndrome_bits)
        offset = 0
        for plane, plane_levels in enumerate(payload.levels):
            for block, level in enumerate(plane_levels):
                self.accumulated[plane, block, _boundary_index(block_length, level)] = sent[
                    offset : offset + level
                ].to(torch.uint8)
                offset += level
        self.raw_rows = {pair: row for row, pair in enumerate(payload.raw)}
        self.raw_bits = _unpack(payload.raw_planes, len(payload.raw) * block_length).view(-1, block_length)

    def request(self, plane: int, blocks: torch.Tensor, level: int) -> torch.Tensor:
        """Serve a rung the worker already paid for, or fail loudly."""
        wanted = blocks.cpu()
        assert bool((self.levels[plane, wanted] >= level).all()), (
            f"the rerun asked plane {plane} for rung {level}; the worker stopped at "
            f"{int(self.levels[plane, wanted].min())}, so belief propagation did not reproduce."
        )
        self.requests += int(wanted.numel())
        return self.accumulated[plane][wanted][
            :, _boundary_index(self.payload.block_length, level)
        ].float()

    def raw(self, plane: int, blocks: torch.Tensor) -> torch.Tensor:
        """Serve a terminal rung the worker already sent, or fail loudly."""
        missing = [int(block) for block in blocks.cpu() if (plane, int(block)) not in self.raw_rows]
        assert not missing, (
            f"the rerun asked for raw plane {plane} blocks {missing}, which the worker never sent."
        )
        return self.raw_bits[[self.raw_rows[(plane, int(block))] for block in blocks.cpu()]]


def _pack(bits: torch.Tensor) -> bytes:
    """Pack a flat 0/1 tensor into little-endian bytes."""
    return np.packbits(bits.reshape(-1).to(torch.uint8).numpy(), bitorder="little").tobytes()


def _unpack(packed: bytes, count: int) -> torch.Tensor:
    """Unpack ``count`` little-endian bits and refuse a payload whose byte padding is dirty."""
    unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
    assert unpacked.size >= count and not unpacked[count:].any(), "dirty byte padding in payload."
    return torch.as_tensor(unpacked[:count].astype(np.float32))


@dataclass(frozen=True, slots=True)
class SWDecodeResult:
    """Decoder output, diagnostics, and the rate plan inferred for the following round."""

    symbols: torch.Tensor
    plane_model_entropies: tuple[tuple[float, ...], ...]
    plane_decoded_cross_entropies: tuple[tuple[float, ...], ...]
    decoded_cross_entropy_stage_rates: tuple[float, ...]
    recovered_blocks: int
    recovery_bits: int
    verification_failures: int
    nonconverged_blocks: int
    bp_attempted_blocks: int
    syndrome_consistent_blocks: int
    plane_blocks: int
    decode_passes: int
    bp_iterations: int
    next_plan: SWRatePlan
    seconds: float


@dataclass(frozen=True, slots=True)
class SWLadderResult:
    """What one rate-adaptive decode recovered, what each block cost, and how it was found."""

    symbols: torch.Tensor
    plane_levels: torch.Tensor
    block_length: int
    plane_model_entropies: tuple[tuple[float, ...], ...]
    plane_decoded_cross_entropies: tuple[tuple[float, ...], ...]
    raw_blocks: int
    verification_failures: int
    bp_attempted_blocks: int
    syndrome_consistent_blocks: int
    bp_iterations: int
    seconds: float

    @property
    def plane_rates(self) -> tuple[float, ...]:
        """Return each plane's mean settled level as a rate in bits per coded position."""
        return tuple(float(level) / self.block_length for level in self.plane_levels.float().mean(dim=1))

    @property
    def syndrome_rate(self) -> float:
        """Return the rate every plane's ladder settled on, in bits per position."""
        return sum(self.plane_rates)


@dataclass(frozen=True, slots=True)
class SWResult:
    """Round-trip compression rate and residual-error measurements for one quantized vector."""

    syndrome_stage_rates: tuple[float, ...]
    model_entropy_stage_rates: tuple[float, ...]
    cross_entropy_stage_rates: tuple[float, ...]
    plane_bit_errors: tuple[tuple[int, ...], ...]
    plane_block_errors: tuple[tuple[int, ...], ...]
    bit_errors: int
    symbol_errors: int
    block_errors: int
    undetected_block_errors: int
    nonconverged_blocks: int
    bp_attempted_blocks: int
    syndrome_consistent_blocks: int
    bp_iterations: int
    plane_blocks: int
    positions: int
    source_bits: int
    source_symbols: int
    syndrome_bits: int
    wire_bits: int
    plan_description_bits: int
    next_plan: SWRatePlan
    seconds: float

    @property
    def syndrome_rate(self) -> float:
        return sum(self.syndrome_stage_rates)

    @property
    def model_entropy_rate(self) -> float:
        return sum(self.model_entropy_stage_rates)

    @property
    def cross_entropy_rate(self) -> float:
        return sum(self.cross_entropy_stage_rates)

    @property
    def wire_rate(self) -> float:
        return self.wire_bits / self.positions

    @property
    def plan_description_rate(self) -> float:
        return self.plan_description_bits / self.positions

    @property
    def bit_error_rate(self) -> float:
        return self.bit_errors / self.source_bits

    @property
    def symbol_error_rate(self) -> float:
        return self.symbol_errors / self.source_symbols

    @property
    def block_error_rate(self) -> float:
        return self.block_errors / self.plane_blocks

    @property
    def undetected_block_error_rate(self) -> float:
        return self.undetected_block_errors / max(1, self.bp_attempted_blocks)

    @property
    def mean_bp_iterations(self) -> float:
        return self.bp_iterations / max(1, self.bp_attempted_blocks)

    def report(self) -> str:
        """Render the per-stage rate and reliability table used by the command-line benchmark."""
        lines = [f"{'stage':>5} {'model H':>9} {'cross H':>9} {'syndrome':>9} {'excess':>9}"]
        for stage, (syndrome, model, cross) in enumerate(
            zip(
                self.syndrome_stage_rates,
                self.model_entropy_stage_rates,
                self.cross_entropy_stage_rates,
                strict=True,
            ),
            1,
        ):
            excess = f"{syndrome / model - 1:>8.1%}" if model else "      n/a"
            lines.append(f"{stage:>5} {model:>9.4f} {cross:>9.4f} {syndrome:>9.4f} {excess}")
        excess = (
            f"{self.syndrome_rate / self.model_entropy_rate - 1:>8.1%}"
            if self.model_entropy_rate
            else "      n/a"
        )
        lines.append(
            f"{'all':>5} {self.model_entropy_rate:>9.4f} "
            f"{self.cross_entropy_rate:>9.4f} {self.syndrome_rate:>9.4f} {excess}"
        )
        lines.append(
            f"BER {self.bit_error_rate:.3%} ({self.bit_errors}) | "
            f"SER {self.symbol_error_rate:.3%} ({self.symbol_errors}) | "
            f"BLER {self.block_error_rate:.3%} ({self.block_errors}) | "
            f"undetected BLER {self.undetected_block_error_rate:.3%} "
            f"({self.undetected_block_errors})"
        )
        lines.append(
            f"BP consistent {self.syndrome_consistent_blocks}/{self.bp_attempted_blocks} | "
            f"non-converged {self.nonconverged_blocks} | mean iterations "
            f"{self.mean_bp_iterations:.1f} | {self.seconds:.1f}s"
        )
        lines.append(
            f"syndrome {self.syndrome_bits} bits | wire {self.wire_bits} bits | "
            f"plan description {self.plan_description_bits} bits | wire/plan-description rates "
            f"{self.wire_rate:.4f}/{self.plan_description_rate:.4f}"
        )
        return "\n".join(lines)


def _degree_counts(block_length: int, distribution: tuple[tuple[int, float], ...]) -> np.ndarray:
    counts = np.asarray([round(fraction * block_length) for _, fraction in distribution])
    counts[int(np.argmax(counts))] += block_length - int(counts.sum())
    return np.repeat([degree for degree, _ in distribution], counts)


def _parity_structure(
    block_length: int,
    check_count: int,
    rng: np.random.Generator,
    distribution: tuple[tuple[int, float], ...],
    girth_candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a balanced irregular Tanner graph while suppressing repeated four-cycles."""
    degrees = _degree_counts(block_length, distribution)
    rng.shuffle(degrees)
    load = np.zeros(check_count, dtype=np.int64)
    check_variables: list[set[int]] = [set() for _ in range(check_count)]
    edge_variable: list[int] = []
    edge_check: list[int] = []

    for variable in rng.permutation(block_length).tolist():
        degree = min(int(degrees[variable]), check_count)
        # Rows already used by this variable are masked to an unreachable load rather than
        # rescanned, which keeps the candidate array (and therefore rng consumption) identical.
        masked_load = load.copy()
        neighbours: set[int] = set()
        for _ in range(degree):
            available = np.flatnonzero(masked_load == masked_load.min())
            if available.size > girth_candidates:
                available = rng.choice(available, girth_candidates, replace=False)
            overlap = [len(neighbours.intersection(check_variables[int(row)])) for row in available]
            row = int(available[overlap.index(min(overlap))])
            edge_variable.append(variable)
            edge_check.append(row)
            neighbours.update(check_variables[row])
            check_variables[row].add(variable)
            load[row] += 1
            masked_load[row] = _MASKED_LOAD

    return np.asarray(edge_variable, dtype=np.int64), np.asarray(edge_check, dtype=np.int64)


def _full_row_rank(edge_variable: np.ndarray, edge_check: np.ndarray, check_count: int, block_length: int) -> bool:
    matrix = sparse.csr_matrix(
        (np.ones(edge_variable.size, dtype=np.uint8), (edge_check, edge_variable)),
        shape=(check_count, block_length),
    )
    return gf2_rank(matrix, method="dense") == check_count


def _parity_matrix(
    block_length: int,
    rows: int,
    code_seed: int,
    stage: int,
    bit: int,
    distribution: tuple[tuple[int, float], ...],
    girth_candidates: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one plane's full-row-rank Tanner graph; free-standing so worker processes can run it."""
    for attempt in range(attempts):
        rng = np.random.default_rng(np.random.SeedSequence([code_seed, stage, bit, rows, attempt]))
        structure = _parity_structure(block_length, rows, rng, distribution, girth_candidates)
        if _full_row_rank(*structure, rows, block_length):
            return structure
    raise AssertionError(
        f"could not construct a full-row-rank {rows}x{block_length} "
        f"parity matrix for stage {stage + 1}, bit {bit + 1}."
    )


@cache
def split_boundaries(block_length: int, level: int) -> tuple[int, ...]:
    """Return the end position of each interval of the mother code at one ladder level.

    Varodayan et al.'s accumulator lets a decoder use any nested subset of the accumulated
    syndrome, and which subsets exist is the ladder's whole design.  Splitting the widest interval
    in half, leftmost of equals first, makes level ``k + 1`` a refinement of level ``k`` for every
    ``k``, so asking for more redundancy costs one bit per level instead of a jump to the next
    rung of a coarse ladder.  Yu and Sharma instead choose each split to shape the merged degree
    distribution; this does not, which is the obvious thing to try if the merged code disappoints.
    """
    assert 1 <= level <= block_length, f"level {level} outside 1..{block_length}."
    intervals = [(-block_length, 0, block_length)]
    for _ in range(level - 1):
        width, start, end = heappop(intervals)
        assert width < -1, "cannot split an interval of one row; level exceeds the mother code."
        middle = (start + end) // 2
        heappush(intervals, (start - middle, start, middle))
        heappush(intervals, (middle - end, middle, end))
    return tuple(sorted(end for _, _, end in intervals))


def _merged_structure(
    mother: tuple[np.ndarray, np.ndarray], order: np.ndarray, boundaries: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Merge the mother code's checks into one check per interval, cancelling repeated edges.

    A variable that reaches the same merged check an even number of times drops out of it, which
    is what costs the merged code its degree at low levels.
    """
    edge_variable, edge_check = mother
    group = np.searchsorted(boundaries, order[edge_check], side="right")
    pairs, counts = np.unique(np.stack((group, edge_variable)), axis=1, return_counts=True)
    kept = pairs[:, counts % 2 == 1]
    return np.ascontiguousarray(kept[1]), np.ascontiguousarray(kept[0])


def _mother_matrix(
    block_length: int,
    code_seed: int,
    stage: int,
    bit: int,
    distribution: tuple[tuple[int, float], ...],
    girth_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one plane's rate-one LDPCA mother code and the order its checks are transmitted in.

    Free-standing so worker processes can run it.  No rank rejection: a square sparse GF(2) matrix
    is rarely full rank, the levels a plan actually codes at are far below rate one, and a level
    that cannot decode is repaired rather than assumed away.
    """
    rng = np.random.default_rng(np.random.SeedSequence([code_seed, stage, bit, block_length]))
    edge_variable, edge_check = _parity_structure(
        block_length, block_length, rng, distribution, girth_candidates
    )
    return edge_variable, edge_check, rng.permutation(block_length)


def _bp_decode(
    llr: torch.Tensor, syndrome: torch.Tensor, edge_variable: torch.Tensor, edge_check: torch.Tensor, cfg: SWCoderConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a batch of source syndromes with sum-product belief propagation."""
    blocks, checks = syndrome.shape
    edge_syndrome = syndrome[:, edge_check]
    variable_to_check = llr[:, edge_variable]
    check_to_variable = torch.zeros_like(variable_to_check)
    decoded = (llr < 0).to(llr.dtype)
    succeeded = torch.zeros(blocks, dtype=torch.bool, device=llr.device)
    iterations_used = torch.zeros(blocks, dtype=torch.long, device=llr.device)

    for iteration in range(1, cfg.max_bp_iterations + 1):
        magnitude = variable_to_check.abs().clamp(1e-6, cfg.llr_clip)
        log_tanh = torch.log(torch.tanh(magnitude / 2))
        negative = (variable_to_check < 0).to(llr.dtype)
        totals = torch.zeros(blocks, checks, device=llr.device).index_add_(1, edge_check, log_tanh)
        parities = torch.zeros(blocks, checks, device=llr.device).index_add_(1, edge_check, negative)
        sign = 1 - 2 * ((parities[:, edge_check] - negative + edge_syndrome) % 2)
        check_to_variable = check_to_variable.lerp(
            sign
            * 2
            * torch.atanh(
                torch.exp((totals[:, edge_check] - log_tanh).clamp(max=-1e-7)).clamp(max=1 - 1e-7)
            ),
            cfg.damping,
        )

        posterior = llr + torch.zeros_like(llr).index_add_(1, edge_variable, check_to_variable)
        bits = (posterior < 0).to(llr.dtype)
        consistent = (
            torch.zeros(blocks, checks, device=llr.device).index_add_(1, edge_check, bits[:, edge_variable]) % 2
            == syndrome
        ).all(dim=1)
        newly_succeeded = consistent & ~succeeded
        decoded = torch.where((~succeeded)[:, None], bits, decoded)
        iterations_used[newly_succeeded] = iteration
        succeeded |= consistent
        if bool(succeeded.all()):
            return decoded, succeeded, iterations_used
        variable_to_check = (posterior[:, edge_variable] - check_to_variable).clamp(-cfg.llr_clip, cfg.llr_clip)
    iterations_used[~succeeded] = cfg.max_bp_iterations
    return decoded, succeeded, iterations_used


def _lsb_plane_probabilities(
    prior: torch.Tensor, prefix: torch.Tensor, bit: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero/one probabilities conditioned on the less-significant bits, and the prefix weight.

    The weight is the prefix's own probability.  A decoder always passes a prefix it has decoded
    and therefore observed, but the rate controller averages over prefixes nobody has observed,
    where a zero weight is an ordinary outcome rather than a fault.
    """
    alphabet = prior.shape[1]
    groups = alphabet >> (bit + 1)
    bases = prefix[:, None] + torch.arange(groups, device=prior.device)[None, :] * (2 << bit)
    zero = prior.gather(1, bases).sum(dim=1)
    one = prior.gather(1, bases + (1 << bit)).sum(dim=1)
    weight = zero + one
    return zero / weight.clamp(min=1e-30), one / weight.clamp(min=1e-30), weight


def _binary_entropy(zero: torch.Tensor, one: torch.Tensor) -> torch.Tensor:
    """Return the entropy in bits of each position's binary distribution."""
    return -(torch.xlogy(zero, zero) + torch.xlogy(one, one)) / float(np.log(2))


def _plane_cross_entropy(zero: torch.Tensor, one: torch.Tensor, bits: torch.Tensor) -> float:
    """Return the mean code length in bits that this plane's realized bits cost under the prior."""
    return float(-torch.log2(torch.where(bits.bool(), one, zero).clamp(min=1e-12)).mean())


def _plane_llr(prior: torch.Tensor, prefix: torch.Tensor, bit: int, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior source LLRs and binary entropies for one LSB-first plane."""
    zero, one, weight = _lsb_plane_probabilities(prior, prefix, bit)
    assert bool((weight > 0).all()), f"decoded lower-bit prefix has zero probability at bit {bit + 1}."
    return (
        (zero.clamp(min=1e-12).log() - one.clamp(min=1e-12).log()).clamp(-clip, clip),
        _binary_entropy(zero, one),
    )


def blocked_planes(bins: torch.Tensor, plan: SWRatePlan) -> torch.Tensor:
    """Return every bit plane of every stage as ``(planes, blocks, block_length)``, zero padded.

    The layout both the syndrome and the block hash are computed over, in the plan's plane order:
    stage-major, least-significant bit first.
    """
    positions = bins.shape[1]
    blocks = ceil(positions / plan.block_length)
    padding = blocks * plan.block_length - positions
    return torch.stack([
        torch.nn.functional.pad(
            ((bins[stage].long().cpu() >> bit) & 1).to(torch.float32), (0, padding)
        ).view(blocks, plan.block_length)
        for stage in range(plan.num_stages)
        for bit in range(plan.bit_width)
    ])


def plane_rows(
    plane_rates: Sequence[Sequence[float]], margin: float, block_length: int
) -> tuple[tuple[int, ...], ...]:
    """Convert per-plane rates in bits per position into syndrome row counts.

    The single place a rate becomes a row count.  The margin belongs to whoever plans the rate:
    a fixed-rate LDPC plane needs slack above its conditional entropy, and where that slack comes
    from -- a constant, or a measurement on data the decoder already holds -- is the whole
    difference between the one-shot controller and the calibrated one.
    """
    assert margin >= 0.0
    counts = (
        tuple(min(block_length, max(0, ceil(block_length * (rate + margin)))) for rate in stage)
        for stage in plane_rates
    )
    # Two checks can never be independent: every variable has degree >= 2 by the distribution's own
    # invariant, so both rows touch every variable and the matrix has rank 1.  It is the single
    # unconstructible width, and a plane whose conditional entropy has collapsed lands on it.
    return tuple(tuple(3 if rows == 2 else rows for rows in stage) for stage in counts)


def expected_plane_rates(
    prior_provider: ConditionalPrior,
    side_info: torch.Tensor,
    num_stages: int,
    alphabet: int,
    samples: int = 2,
    seed: int = 0,
) -> tuple[tuple[float, ...], ...]:
    """Return every plane's model conditional entropy without observing one source symbol.

    The learned-prior form of Zhang and Tao's Eq. (45) rate controller, and the only rate a
    decoder can name before anything is sent.  Within a stage the already split lower bits are
    enumerated exactly and weighted by their own model probability; the earlier stages that a
    later stage conditions on are drawn from the model rather than enumerated, which keeps the
    cost linear instead of exponential in the stage index at the price of a sampling error that
    averages away over positions.  These are model entropies: the gap to the cross-entropy of a
    real symbol stream is exactly what a pilot exists to measure.
    """
    assert samples >= 1 and alphabet >= 2 and alphabet & (alphabet - 1) == 0
    positions, bit_width = side_info.shape[0], int(log2(alphabet))
    generator = torch.Generator(device=side_info.device).manual_seed(seed)
    totals = np.zeros((num_stages, bit_width))

    for _ in range(samples):
        drawn = torch.zeros((num_stages, positions), dtype=torch.long)
        for stage in range(num_stages):
            prior = prior_provider(drawn, side_info, stage).to(side_info.device)
            assert prior.shape == (positions, alphabet), (
                f"stage {stage + 1} prior has shape {tuple(prior.shape)}; expected {(positions, alphabet)}."
            )
            for bit in range(bit_width):
                for prefix_value in range(1 << bit):
                    prefix = torch.full((positions,), prefix_value, dtype=torch.long, device=prior.device)
                    zero, one, weight = _lsb_plane_probabilities(prior, prefix, bit)
                    totals[stage, bit] += float((weight * _binary_entropy(zero, one)).mean())
            uniform = torch.rand(positions, 1, device=prior.device, generator=generator)
            drawn[stage] = (prior.cumsum(dim=1) < uniform).sum(dim=1).clamp(max=alphabet - 1).cpu()
    return tuple(tuple(stage) for stage in (totals / samples).tolist())


class SlepianWolfCodec:
    """Fixed-rate encoder and decoder owning the parity matrices its plans need.

    Matrix construction, not belief propagation, dominates coding time, so one codec should be
    shared by everything that touches the same plans; resident matrices make a rate search and
    encode/decode round trip cheap.
    """

    def __init__(self, cfg: SWCoderConfig | None = None) -> None:
        self.cfg = cfg or SWCoderConfig()
        self._matrices: OrderedDict[MatrixKey, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._edge_tensors: dict[MatrixKey, tuple[torch.Tensor, torch.Tensor]] = {}
        # One resident mother per plane, not per rate: merging it down to a level costs milliseconds.
        self._mothers: dict[MotherKey, tuple[np.ndarray, ...]] = {}

    def initial_plan(self, num_stages: int, alphabet: int) -> SWRatePlan:
        """Return the rate-one plan used to bootstrap previous-round adaptation."""
        return SWRatePlan.initial(num_stages, alphabet, self.cfg)

    @staticmethod
    def _key(plan: SWRatePlan, stage: int, bit: int, rows: int) -> MatrixKey:
        return (
            plan.matrix_backend,
            plan.block_length,
            plan.code_seed,
            plan.degree_distribution,
            plan.girth_candidates,
            plan.matrix_attempts,
            stage,
            bit,
            rows,
        )

    @classmethod
    def _coded_planes(cls, plan: SWRatePlan) -> dict[MatrixKey, tuple[int, int, int]]:
        """Return the plan's syndrome-coded planes, keyed by the matrix each one needs."""
        return {
            cls._key(plan, stage, bit, rows): (stage, bit, rows)
            for stage, stage_rows in enumerate(plan.syndrome_rows)
            for bit, rows in enumerate(stage_rows)
            if 0 < rows < plan.block_length
        }

    def _prepare(self, plan: SWRatePlan, planes: dict[MatrixKey, tuple[int, int, int]] | None = None) -> None:
        """Make every matrix the plan needs resident, building the missing ones in parallel."""
        if planes is None:
            planes = self._coded_planes(plan)
            self._edge_tensors = {key: value for key, value in self._edge_tensors.items() if key in planes}
        missing = {key: spec for key, spec in planes.items() if key not in self._matrices}
        if not missing:
            return
        if plan.matrix_backend == LDPCA_BACKEND:
            for key, (stage, bit, rows) in missing.items():
                edge_variable, edge_check, order = self._mother(plan, stage, bit)
                self._remember(key, _merged_structure(
                    (edge_variable, edge_check), order, split_boundaries(plan.block_length, rows)))
            return
        assert plan.matrix_backend == MATRIX_BACKEND, f"unsupported matrix backend {plan.matrix_backend}."
        arguments = [
            (plan.block_length, rows, plan.code_seed, stage, bit,
             plan.degree_distribution, plan.girth_candidates, plan.matrix_attempts)
            for stage, bit, rows in missing.values()
        ]
        if len(arguments) == 1 or plan.block_length < PARALLEL_BUILD_MIN_BLOCK_LENGTH:
            built = [_parity_matrix(*argument) for argument in arguments]
        else:
            # Construction is single-threaded Python; the builds are independent and deterministic.
            # Fork is safe here because the workers only touch numpy.
            with ProcessPoolExecutor(
                min(len(arguments), os.cpu_count() or 1), mp_context=get_context("fork")
            ) as pool:
                built = list(pool.map(_parity_matrix, *zip(*arguments, strict=True)))
        for key, structure in zip(missing, built, strict=True):
            self._remember(key, structure)

    def _remember(self, key: MatrixKey, structure: tuple[np.ndarray, np.ndarray]) -> None:
        self._matrices[key] = structure
        self._matrices.move_to_end(key)
        while len(self._matrices) > MATRIX_MEMORY_CACHE_SIZE:
            self._matrices.popitem(last=False)

    def _mother(self, plan: SWRatePlan, stage: int, bit: int) -> tuple[np.ndarray, ...]:
        """Return the plane's rate-one LDPCA mother code and its check transmission order."""
        key: MotherKey = (
            plan.matrix_backend, plan.block_length, plan.code_seed,
            plan.degree_distribution, plan.girth_candidates, stage, bit,
        )
        cached = self._mothers.get(key)
        if cached is None:
            cached = _mother_matrix(
                plan.block_length, plan.code_seed, stage, bit,
                plan.degree_distribution, plan.girth_candidates,
            )
        self._mothers[key] = cached
        return cached

    def _matrix(self, plan: SWRatePlan, stage: int, bit: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
        key = self._key(plan, stage, bit, rows)
        self._prepare(plan, {key: (stage, bit, rows)})
        self._matrices.move_to_end(key)
        return self._matrices[key]

    def _edges(
        self, plan: SWRatePlan, stage: int, bit: int, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the plane's edge indices on ``device``, uploaded once per plan rather than per call."""
        key = self._key(plan, stage, bit, rows)
        cached = self._edge_tensors.get(key)
        if cached is None or cached[0].device != device:
            edge_variable, edge_check = self._matrix(plan, stage, bit, rows)
            cached = (torch.as_tensor(edge_variable, device=device), torch.as_tensor(edge_check, device=device))
            self._edge_tensors[key] = cached
        return cached

    @staticmethod
    def _expected_syndrome_bits(plan: SWRatePlan, positions: int) -> int:
        blocks = ceil(positions / plan.block_length)
        return blocks * sum(map(sum, plan.syndrome_rows))

    def encode_syndromes(self, bins: torch.Tensor, plan: SWRatePlan) -> SWPayload:
        """Encode quantizer symbols using only the shared plan and return packed syndromes."""
        assert bins.ndim == 2, f"bins must have shape (stages, positions), got {bins.shape}."
        assert bins.shape[0] == plan.num_stages
        assert bins.shape[1] > 0
        assert bool(((bins >= 0) & (bins < plan.alphabet)).all()), f"bins must lie in [0, {plan.alphabet})."
        positions = bins.shape[1]
        blocks = ceil(positions / plan.block_length)
        padding = blocks * plan.block_length - positions
        self._prepare(plan)
        coded_planes: list[torch.Tensor] = []

        for stage, stage_rows in enumerate(plan.syndrome_rows):
            symbols = bins[stage].long()
            for bit, rows in enumerate(stage_rows):
                plane = torch.nn.functional.pad(
                    ((symbols >> bit) & 1).to(torch.float32), (0, padding)
                ).view(blocks, plan.block_length)
                if rows == plan.block_length:
                    syndrome = plane
                elif rows == 0:
                    continue
                else:
                    syndrome = self._plane_syndrome(plane, plan, stage, bit, rows)
                coded_planes.append(syndrome.to(torch.uint8).reshape(-1))

        # One host transfer for the whole payload; per-plane copies would sync the device each time.
        syndrome_bits = (
            torch.cat(coded_planes).cpu().numpy() if coded_planes else np.empty(0, dtype=np.uint8)
        )
        syndromes = np.packbits(syndrome_bits, bitorder="little").tobytes()
        assert syndrome_bits.size == self._expected_syndrome_bits(plan, positions)
        return SWPayload(positions=positions, plan_id=plan.identifier, syndromes=syndromes)

    def _bp_plane(
        self, llr: torch.Tensor, syndrome: torch.Tensor, plan: SWRatePlan, stage: int, bit: int, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Decode one plane's blocks in message-budget sized passes."""
        variable_index, check_index = self._edges(plan, stage, bit, rows, llr.device)
        per_pass = max(1, self.cfg.message_budget // variable_index.numel())
        plane = torch.empty_like(llr)
        succeeded = torch.empty(llr.shape[0], dtype=torch.bool, device=llr.device)
        iterations = passes = 0
        for chunk in torch.arange(llr.shape[0], device=llr.device).split(per_pass):
            plane[chunk], succeeded[chunk], used = _bp_decode(
                llr[chunk], syndrome[chunk], variable_index, check_index, self.cfg
            )
            iterations += int(used.sum())
            passes += 1
        return plane, succeeded, iterations, passes

    def _plane_syndrome(
        self, plane: torch.Tensor, plan: SWRatePlan, stage: int, bit: int, rows: int
    ) -> torch.Tensor:
        """Multiply one plane's blocks by their parity-check matrix over GF(2)."""
        variable_index, check_index = self._edges(plan, stage, bit, rows, plane.device)
        return torch.zeros(
            plane.shape[0], rows, dtype=plane.dtype, device=plane.device
        ).index_add_(1, check_index, plane[:, variable_index]) % 2

    def calibrate_plane_rows(
        self,
        bins: torch.Tensor,
        plan: SWRatePlan,
        prior_provider: ConditionalPrior,
        side_info: torch.Tensor,
        row_step: int,
        grid_points: int,
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]]:
        """Return the smallest per-plane rows that decode known symbols exactly, and their cost.

        Nothing here is oracle as long as the caller owns the symbols it passes: on a pilot the
        decoder has already received, the rows this returns are then applied to positions that
        took no part in the search.  Each plane's candidate grid starts at the rows of ``plan``
        and always ends at the full rate, so a plane that cannot be calibrated ends uncompressed
        and visibly so instead of silently under-rated.  The prefix a later plane conditions on
        is the true one, because the protocol that uses these rows repairs failed blocks before
        the next plane reads them.
        """
        assert row_step >= 1 and grid_points >= 1
        assert bins.shape == (plan.num_stages, side_info.shape[0])
        positions = bins.shape[1]
        blocks = ceil(positions / plan.block_length)
        padding = blocks * plan.block_length - positions
        device = side_info.device
        known = torch.zeros((plan.num_stages, positions), dtype=torch.long)
        selected: list[tuple[int, ...]] = []
        measured: list[tuple[float, ...]] = []

        for stage, stage_rows in enumerate(plan.syndrome_rows):
            prior = prior_provider(known, side_info, stage).to(device)
            symbols = bins[stage].long().to(device)
            prefix = torch.zeros(positions, dtype=torch.long, device=device)
            stage_selected: list[int] = []
            stage_measured: list[float] = []

            for bit, start_rows in enumerate(stage_rows):
                zero, one, _ = _lsb_plane_probabilities(prior, prefix, bit)
                truth = (symbols >> bit) & 1
                stage_measured.append(_plane_cross_entropy(zero, one, truth))
                llr, _ = _plane_llr(prior, prefix, bit, self.cfg.llr_clip)
                llr = torch.nn.functional.pad(
                    llr, (0, padding), value=self.cfg.llr_clip
                ).view(blocks, plan.block_length)
                plane = torch.nn.functional.pad(
                    truth.to(llr.dtype), (0, padding)
                ).view(blocks, plan.block_length)
                candidates = sorted(
                    {min(plan.block_length, start_rows + step * row_step) for step in range(grid_points)}
                    | {plan.block_length}
                )
                # Built in one batch: a matrix costs seconds and belief propagation milliseconds,
                # so the whole grid is worth constructing in parallel even where the scan stops early.
                self._prepare(
                    plan,
                    {
                        self._key(plan, stage, bit, rows): (stage, bit, rows)
                        for rows in candidates
                        if 0 < rows < plan.block_length
                    },
                )
                stage_selected.append(
                    next(
                        rows
                        for rows in candidates
                        if self._decodes_exactly(plane, llr, plan, stage, bit, rows)
                    )
                )
                prefix |= truth << bit

            known[stage] = symbols.cpu()
            selected.append(tuple(stage_selected))
            measured.append(tuple(stage_measured))
        return tuple(selected), tuple(measured)

    def _decodes_exactly(
        self,
        plane: torch.Tensor,
        llr: torch.Tensor,
        plan: SWRatePlan,
        stage: int,
        bit: int,
        rows: int,
    ) -> bool:
        """Report whether one candidate row count recovers every bit of a known plane."""
        if rows == plan.block_length:
            return True
        if rows == 0:
            return bool(((llr < 0).to(plane.dtype) == plane).all())
        syndrome = self._plane_syndrome(plane, plan, stage, bit, rows)
        return bool((self._bp_plane(llr, syndrome, plan, stage, bit, rows)[0] == plane).all())

    def decode_syndromes(
        self,
        payload: SWPayload,
        plan: SWRatePlan,
        prior_provider: ConditionalPrior,
        side_info: torch.Tensor,
        recovery: BlockRecovery | None = None,
        verify: PlaneVerification | None = None,
    ) -> SWDecodeResult:
        """Decode a packed syndrome payload using only decoder-available conditional information."""
        start = perf_counter()
        assert payload.plan_id == plan.identifier, (
            f"payload plan {payload.plan_id} does not match decoder plan {plan.identifier}."
        )
        assert side_info.shape[0] == payload.positions, (
            f"side information has {side_info.shape[0]} positions; payload has {payload.positions}."
        )
        expected_bits = self._expected_syndrome_bits(plan, payload.positions)
        expected_bytes = ceil(expected_bits / 8)
        assert len(payload.syndromes) == expected_bytes, (
            f"payload has {len(payload.syndromes)} syndrome bytes; expected {expected_bytes}."
        )
        unpacked_with_padding = np.unpackbits(np.frombuffer(payload.syndromes, dtype=np.uint8), bitorder="little")
        unpacked = unpacked_with_padding[:expected_bits]
        assert not unpacked_with_padding[expected_bits:].any(), "non-zero byte-padding bits in syndrome payload."

        positions = payload.positions
        blocks = ceil(positions / plan.block_length)
        padding = blocks * plan.block_length - positions
        device = side_info.device
        self._prepare(plan)
        decoded = torch.zeros((plan.num_stages, positions), dtype=torch.long)
        entropies: list[tuple[float, ...]] = []
        plane_cross_entropies: list[tuple[float, ...]] = []
        decoded_cross_entropies: list[float] = []
        offset = passes = iterations = bp_attempted = syndrome_consistent = 0
        recovered = unverified = 0

        for stage, stage_rows in enumerate(plan.syndrome_rows):
            prior = self._stage_prior(prior_provider, decoded, side_info, stage, plan.alphabet)
            prefix = torch.zeros(positions, dtype=torch.long, device=device)
            stage_entropies: list[float] = []
            stage_cross_entropies: list[float] = []

            for bit, rows in enumerate(stage_rows):
                zero, one, _ = _lsb_plane_probabilities(prior, prefix, bit)
                llr, entropy = _plane_llr(prior, prefix, bit, self.cfg.llr_clip)
                stage_entropies.append(float(entropy.mean()))
                llr = torch.nn.functional.pad(
                    llr, (0, padding), value=self.cfg.llr_clip
                ).view(blocks, plan.block_length)
                count = blocks * rows
                syndrome = torch.as_tensor(
                    unpacked[offset : offset + count].copy(), dtype=torch.float32, device=device
                ).view(blocks, rows)
                offset += count

                if rows == plan.block_length:
                    plane = syndrome
                    succeeded = torch.ones(blocks, dtype=torch.bool, device=device)
                elif rows == 0:
                    # A rate-zero plane is the prior's guess: nothing constrains it but the check.
                    plane = (llr < 0).to(llr.dtype)
                    succeeded = torch.ones(blocks, dtype=torch.bool, device=device)
                else:
                    plane, succeeded, plane_iterations, plane_passes = self._bp_plane(
                        llr, syndrome, plan, stage, bit, rows
                    )
                    iterations += plane_iterations
                    passes += plane_passes
                    bp_attempted += blocks
                    syndrome_consistent += int(succeeded.sum())
                if verify is not None and rows != plan.block_length:
                    verified = verify(stage, bit, plane, torch.arange(blocks, device=device))
                    unverified += int((succeeded & ~verified).sum())
                    succeeded &= verified
                if recovery is not None and not bool(succeeded.all()):
                    failed = torch.nonzero(~succeeded, as_tuple=False).squeeze(1)
                    plane[failed] = recovery(stage, bit, failed).to(llr.dtype).to(device)
                    recovered += int(failed.numel())
                bits = plane.flatten()[:positions]
                stage_cross_entropies.append(_plane_cross_entropy(zero, one, bits))
                prefix |= bits.long() << bit

            decoded[stage] = prefix.cpu()
            decoded_cross_entropies.append(
                float(-torch.log2(prior.gather(1, prefix[:, None]).squeeze(1).clamp(min=1e-12)).mean())
            )
            entropies.append(tuple(stage_entropies))
            plane_cross_entropies.append(tuple(stage_cross_entropies))

        assert offset == expected_bits
        return SWDecodeResult(
            symbols=decoded,
            plane_model_entropies=tuple(entropies),
            plane_decoded_cross_entropies=tuple(plane_cross_entropies),
            decoded_cross_entropy_stage_rates=tuple(decoded_cross_entropies),
            recovered_blocks=recovered,
            recovery_bits=recovered * plan.block_length,
            verification_failures=unverified,
            nonconverged_blocks=bp_attempted - syndrome_consistent,
            bp_attempted_blocks=bp_attempted,
            syndrome_consistent_blocks=syndrome_consistent,
            plane_blocks=blocks * plan.num_stages * plan.bit_width,
            decode_passes=passes,
            bp_iterations=iterations,
            next_plan=plan.model_copy(
                update={"syndrome_rows": plane_rows(entropies, 0.0, plan.block_length)}
            ),
            seconds=perf_counter() - start,
        )

    def _stage_prior(
        self,
        prior_provider: ConditionalPrior,
        decoded: torch.Tensor,
        side_info: torch.Tensor,
        stage: int,
        alphabet: int,
    ) -> torch.Tensor:
        """Return one stage's decoder-side prior after checking that it is a distribution."""
        positions = side_info.shape[0]
        prior = prior_provider(decoded, side_info, stage).to(side_info.device)
        assert prior.shape == (positions, alphabet), (
            f"stage {stage + 1} prior has shape {tuple(prior.shape)}; expected {(positions, alphabet)}."
        )
        assert bool(torch.isfinite(prior).all()) and bool((prior >= 0).all())
        assert torch.allclose(
            prior.sum(dim=1), torch.ones(positions, device=side_info.device), atol=1e-4, rtol=1e-4
        ), f"stage {stage + 1} prior rows must sum to one."
        return prior

    def accumulate_ladder(self, bins: torch.Tensor, plan: SWRatePlan) -> SWLadder:
        """Return the LDPCA encoder's entire output: every plane's rate-one accumulated syndrome.

        This is the whole of what the encoder knows how to do.  It never sees a rate: the rate is
        how far up the accumulator the decoder later asks it to go, which is what makes the
        transport adaptive without the encoder having any decoder-side information.
        """
        assert plan.matrix_backend == LDPCA_BACKEND, (
            f"the ladder needs the {LDPCA_BACKEND} backend; the plan names {plan.matrix_backend}."
        )
        assert bins.ndim == 2 and bins.shape[0] == plan.num_stages, (
            f"bins must have shape ({plan.num_stages}, positions), got {tuple(bins.shape)}."
        )
        assert bool(((bins >= 0) & (bins < plan.alphabet)).all()), f"bins must lie in [0, {plan.alphabet})."
        planes = blocked_planes(bins, plan)
        accumulated: list[torch.Tensor] = []

        for index, plane in enumerate(planes):
            edge_variable, edge_check, order = self._mother(
                plan, index // plan.bit_width, index % plan.bit_width)
            syndrome = torch.zeros(plane.shape[0], plan.block_length).index_add_(
                1, torch.as_tensor(edge_check), plane[:, torch.as_tensor(edge_variable)]
            ) % 2
            # The accumulator is taken in transmission order, which is what makes every level's
            # transmitted positions a subset of every higher level's.
            transmitted = syndrome[:, torch.as_tensor(np.argsort(order))]
            accumulated.append((transmitted.cumsum(dim=1) % 2).to(torch.uint8))
        return SWLadder(torch.stack(accumulated), planes, plan.block_length)

    def decode_ladder(
        self,
        ladder: LadderChannel,
        plan: SWRatePlan,
        positions: int,
        prior_provider: ConditionalPrior,
        side_info: torch.Tensor,
        verify: PlaneVerification,
    ) -> SWLadderResult:
        """Climb each plane's LDPCA ladder until every block verifies, and report what it cost.

        This is where the rate is chosen, per block, by trying: a block stops costing bits the
        moment its decode verifies, so no margin has to be guessed and a hard block cannot make an
        easy one pay for it.  ``plan.syndrome_rows`` only says where the climb starts -- starting
        too low costs belief-propagation passes and no bits at all, starting too high overpays
        every block, so the model's own conditional entropy is the right guess.

        ``verify`` is not optional here.  Every rung asks "is this the source block?", and a
        satisfied syndrome cannot answer it, so without a hash the ladder would stop at the first
        confident wrong word.
        """
        start_time = perf_counter()
        assert plan.matrix_backend == LDPCA_BACKEND, (
            f"the ladder needs the {LDPCA_BACKEND} backend; the plan names {plan.matrix_backend}."
        )
        assert side_info.shape[0] == positions, (
            f"side information has {side_info.shape[0]} positions; the ladder codes {positions}."
        )
        blocks = ceil(positions / plan.block_length)
        padding = blocks * plan.block_length - positions
        device = side_info.device
        decoded = torch.zeros((plan.num_stages, positions), dtype=torch.long)
        levels = torch.zeros((plan.num_stages * plan.bit_width, blocks), dtype=torch.long)
        entropies: list[tuple[float, ...]] = []
        cross_entropies: list[tuple[float, ...]] = []
        iterations = attempted = consistent = unverified = raw_blocks = 0

        for stage, stage_rows in enumerate(plan.syndrome_rows):
            prior = self._stage_prior(prior_provider, decoded, side_info, stage, plan.alphabet)
            prefix = torch.zeros(positions, dtype=torch.long, device=device)
            stage_entropies: list[float] = []
            stage_cross_entropies: list[float] = []

            for bit, start_level in enumerate(stage_rows):
                index = stage * plan.bit_width + bit
                zero, one, _ = _lsb_plane_probabilities(prior, prefix, bit)
                llr, entropy = _plane_llr(prior, prefix, bit, self.cfg.llr_clip)
                stage_entropies.append(float(entropy.mean()))
                llr = torch.nn.functional.pad(
                    llr, (0, padding), value=self.cfg.llr_clip
                ).view(blocks, plan.block_length)
                plane = torch.zeros_like(llr)
                active = torch.arange(blocks, device=device)
                level = max(1, start_level)

                while active.numel():
                    syndrome = _ladder_syndrome(ladder.request(index, active, level).to(device))
                    recovered, succeeded, plane_iterations, _ = self._bp_plane(
                        llr[active], syndrome, plan, stage, bit, level
                    )
                    iterations += plane_iterations
                    attempted += int(active.numel())
                    consistent += int(succeeded.sum())
                    verified = verify(stage, bit, recovered, active)
                    unverified += int((succeeded & ~verified).sum())
                    succeeded &= verified
                    plane[active[succeeded]] = recovered[succeeded]
                    levels[index, active[succeeded].cpu()] = level
                    active = active[~succeeded]
                    if level == plan.block_length:
                        break
                    level = min(level + self.cfg.ladder_step, plan.block_length)

                if active.numel():
                    # The merged code at level block_length is one row short of full rank and
                    # belief propagation is not exact anyway, so the ladder keeps a last rung.
                    plane[active] = ladder.raw(index, active).to(plane)
                    levels[index, active.cpu()] = plan.block_length
                    raw_blocks += int(active.numel())

                bits = plane.flatten()[:positions]
                stage_cross_entropies.append(_plane_cross_entropy(zero, one, bits))
                prefix |= bits.long() << bit

            decoded[stage] = prefix.cpu()
            entropies.append(tuple(stage_entropies))
            cross_entropies.append(tuple(stage_cross_entropies))

        return SWLadderResult(
            symbols=decoded,
            plane_levels=levels,
            block_length=plan.block_length,
            plane_model_entropies=tuple(entropies),
            plane_decoded_cross_entropies=tuple(cross_entropies),
            raw_blocks=raw_blocks,
            verification_failures=unverified,
            bp_attempted_blocks=attempted,
            syndrome_consistent_blocks=consistent,
            bp_iterations=iterations,
            seconds=perf_counter() - start_time,
        )


def _true_prior_stage_statistics(
    bins: torch.Tensor, prior_provider: ConditionalPrior, side_info: torch.Tensor, alphabet: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return model entropy and true-symbol cross-entropy under true prior-stage context."""
    decoded = torch.zeros_like(bins, dtype=torch.long, device="cpu")
    entropies: list[float] = []
    cross_entropies: list[float] = []
    for stage, symbols in enumerate(bins.long()):
        prior = prior_provider(decoded, side_info, stage).to(side_info.device)
        assert prior.shape == (bins.shape[1], alphabet)
        symbols = symbols.to(side_info.device)
        entropies.append(float((-(torch.xlogy(prior, prior).sum(dim=1)) / float(np.log(2))).mean()))
        cross_entropies.append(
            float(-torch.log2(prior.gather(1, symbols[:, None]).squeeze(1).clamp(min=1e-12)).mean())
        )
        decoded[stage] = symbols.cpu()
    return tuple(entropies), tuple(cross_entropies)


def _undetected_block_errors(
    differences: torch.Tensor, plan: SWRatePlan, codec: SlepianWolfCodec, device: torch.device
) -> int:
    """Count erroneous BP-decoded plane blocks that still satisfy their syndrome."""
    blocks = ceil(differences.shape[1] / plan.block_length)
    padding = blocks * plan.block_length - differences.shape[1]
    undetected = 0
    for stage, stage_rows in enumerate(plan.syndrome_rows):
        for bit, rows in enumerate(stage_rows):
            if rows in (0, plan.block_length):
                continue
            errors = torch.nn.functional.pad(
                ((differences[stage].to(device) >> bit) & 1).float(), (0, padding)
            ).view(blocks, plan.block_length)
            variable_index, check_index = codec._edges(plan, stage, bit, rows, device)
            parity = torch.zeros(blocks, rows, device=device).index_add_(
                1, check_index, errors[:, variable_index]
            ) % 2
            undetected += int((errors.bool().any(dim=1) & ~parity.bool().any(dim=1)).sum())
    return undetected


def evaluate_slepian_wolf(
    bins: torch.Tensor,
    plan: SWRatePlan,
    prior_provider: ConditionalPrior,
    side_info: torch.Tensor,
    codec: SlepianWolfCodec | None = None,
) -> SWResult:
    """Round-trip a quantized vector and measure rate beside residual decoding errors.

    Pass the caller's own codec whenever there is one: a fresh codec rebuilds every parity
    matrix the plan names, which costs far more than the round trip being measured.
    """
    start = perf_counter()
    codec = codec or SlepianWolfCodec()
    payload = codec.encode_syndromes(bins, plan)
    decoded = codec.decode_syndromes(payload, plan, prior_provider, side_info)
    source = bins.long().cpu()
    differences = decoded.symbols ^ source
    model_entropies, cross_entropies = _true_prior_stage_statistics(bins, prior_provider, side_info, plan.alphabet)
    blocks = ceil(bins.shape[1] / plan.block_length)
    padding = blocks * plan.block_length - bins.shape[1]
    plane_bit_errors: list[tuple[int, ...]] = []
    plane_block_errors: list[tuple[int, ...]] = []
    for stage in range(plan.num_stages):
        stage_bit_errors: list[int] = []
        stage_block_errors: list[int] = []
        blocked = torch.nn.functional.pad(differences[stage], (0, padding)).view(blocks, plan.block_length)
        for bit in range(plan.bit_width):
            errors = (blocked >> bit) & 1
            stage_bit_errors.append(int(errors.sum()))
            stage_block_errors.append(int(errors.any(dim=1).sum()))
        plane_bit_errors.append(tuple(stage_bit_errors))
        plane_block_errors.append(tuple(stage_block_errors))
    syndrome_stage_rates = tuple(blocks * sum(rows) / bins.shape[1] for rows in plan.syndrome_rows)
    return SWResult(
        syndrome_stage_rates=syndrome_stage_rates,
        model_entropy_stage_rates=model_entropies,
        cross_entropy_stage_rates=cross_entropies,
        plane_bit_errors=tuple(plane_bit_errors),
        plane_block_errors=tuple(plane_block_errors),
        bit_errors=sum(map(sum, plane_bit_errors)),
        symbol_errors=int((differences != 0).sum()),
        block_errors=sum(map(sum, plane_block_errors)),
        undetected_block_errors=_undetected_block_errors(differences, plan, codec, side_info.device),
        nonconverged_blocks=decoded.nonconverged_blocks,
        bp_attempted_blocks=decoded.bp_attempted_blocks,
        syndrome_consistent_blocks=decoded.syndrome_consistent_blocks,
        bp_iterations=decoded.bp_iterations,
        plane_blocks=decoded.plane_blocks,
        positions=bins.shape[1],
        source_bits=bins.numel() * plan.bit_width,
        source_symbols=bins.numel(),
        syndrome_bits=codec._expected_syndrome_bits(plan, bins.shape[1]),
        wire_bits=payload.wire_bits,
        plan_description_bits=plan.description_bits,
        next_plan=decoded.next_plan,
        seconds=perf_counter() - start,
    )


def _ladder_report(
    codec: SlepianWolfCodec,
    bins: torch.Tensor,
    plan: SWRatePlan,
    prior_provider: ConditionalPrior,
    side_info: torch.Tensor,
    hash_bits: int,
) -> str:
    """Run the rate-adaptive ladder over the same symbols and render what each block cost."""
    ladder = codec.accumulate_ladder(bins, plan)
    hashes = PlaneHashes(plan.block_length, hash_bits, plan.code_seed)
    digests = hashes.digest_planes(ladder.planes)
    result = codec.decode_ladder(
        ladder, plan, bins.shape[1], prior_provider, side_info,
        HashVerification(hashes, digests, plan.bit_width),
    )
    exact = bool((result.symbols == bins.long().cpu()).all())
    hash_rate = digests.shape[0] * digests.shape[1] * hash_bits / bins.shape[1]
    levels = result.plane_levels
    return (
        f"LDPCA ladder  rungs {result.syndrome_rate:.4f} + hashes {hash_rate:.4f} + raw "
        f"{result.raw_blocks * plan.block_length / bins.shape[1]:.4f} = "
        f"{result.syndrome_rate + hash_rate + result.raw_blocks * plan.block_length / bins.shape[1]:.4f} "
        f"bit/position | exact {exact} | levels {int(levels.min())}-{int(levels.max())} "
        f"of {plan.block_length} | {result.verification_failures} hash rejections | "
        f"{result.raw_blocks} raw blocks | {result.seconds:.0f}s"
    )


def main() -> None:
    """Benchmark a previous-round rate plan on the quantizer's synthetic Gaussian source."""
    from FL_code.FL_core.utils import set_global_seed

    from ..prior_code import PriorCalculator
    from ..history import HistoryVector
    from ..wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1_000_000, help="coded vector length")
    parser.add_argument("--noise-power", type=float, default=0.01)
    parser.add_argument("--bins-per-plane", type=int, default=2)
    parser.add_argument("--num-planes", type=int, default=3)
    parser.add_argument("--reconst-ld", type=float, default=120.0)
    parser.add_argument("--block-length", type=int, default=SWCoderConfig().block_length)
    parser.add_argument("--margin", type=float, default=0.0, help="bits per plane added to the entropy plan")
    parser.add_argument("--ladder-step", type=int, default=SWCoderConfig().ladder_step)
    parser.add_argument("--ladder-hash-bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    noise_std = args.noise_power**0.5

    def source(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        si = torch.randn(count, generator=generator)
        return si + torch.randn(count, generator=generator) * noise_std, si

    quantizer_config = WZcfgQuant(
        bins_per_plane=args.bins_per_plane,
        num_planes=args.num_planes,
        norm_slices=(slice(None),),
        reconst_ld=args.reconst_ld,
        train_epochs=180,
        train_sample_size=200_000,
        train_batch_size=1_000,
        lr=3e-3,
        lr_step=40,
        tau=1.0,
        training_progress_bar=False,
        training_log_every_epochs=20,
        quantizer_train_repeats=1,
        shared_heads=True,
        prior_context="hard",
        prior_probabilities="categorical",
        fused_optimizer=True,
        hidden_dim=100,
        layers=3,
    )
    train_target, train_si = source(args.size)
    quantizer = DedupedWZQuantizerCancer(quantizer_config, si_size=1)
    quantizer.train_model(train_target, [HistoryVector(train_si)])

    target, si = source(args.size)
    quantizer.side_info_list_used = [HistoryVector(si)]
    bins, soft_codes, metadata = quantizer.encoding_process(target)
    assert soft_codes is not None
    side_info = quantizer.normalize_side_info_tensor(metadata, [HistoryVector(si)])
    prior_config = quantizer_config.model_copy(update={"marginal_loss": False})
    prior_model = PriorCalculator.make_trained_prior_model(
        source_model=quantizer.coding_model,
        bins_vec=bins.long(),
        soft_codes=soft_codes,
        side_info=side_info,
        c_cfg=prior_config,
    )

    def prior_provider(decoded_stages: torch.Tensor, decoder_side_info: torch.Tensor, stage: int) -> torch.Tensor:
        return PriorCalculator.compute_prior_from_network(
            prior_model, decoded_stages, decoder_side_info
        )[stage].to(decoder_side_info.device)

    cfg = SWCoderConfig(block_length=args.block_length, code_seed=args.seed)
    codec = SlepianWolfCodec(cfg)
    bootstrap = codec.initial_plan(args.num_planes, args.bins_per_plane)
    bootstrap_payload = codec.encode_syndromes(bins, bootstrap)
    bootstrap_decode = codec.decode_syndromes(bootstrap_payload, bootstrap, prior_provider, side_info)
    plan = bootstrap.model_copy(update={"syndrome_rows": plane_rows(
        bootstrap_decode.plane_model_entropies, args.margin, cfg.block_length)})
    result = evaluate_slepian_wolf(bins, plan, prior_provider, side_info, codec)
    marginal = sum(PriorCalculator.compute_plane_rates(
        PriorCalculator.compute_marginal_prior(bins, args.bins_per_plane, args.num_planes),
        bins, args.num_planes,
    ))
    print(
        f"\n{args.bins_per_plane} bins x {args.num_planes} stages, {args.size} floats, "
        f"block {args.block_length}; rate plan calibrated on the preceding decode\n"
    )
    print(result.report())
    print(
        f"encoder-side marginal rate {marginal:.4f} "
        f"({1 - result.syndrome_rate / marginal:.1%} saved by Slepian-Wolf coding)"
    )
    # The same plan, read as a starting rung instead of a fixed rate: the ladder climbs from the
    # model entropy per block, so this line and the table above differ only in the code family.
    ladder_codec = SlepianWolfCodec(cfg.model_copy(update={
        "matrix_backend": LDPCA_BACKEND, "ladder_step": args.ladder_step}))
    ladder_plan = ladder_codec.initial_plan(args.num_planes, args.bins_per_plane).model_copy(
        update={"syndrome_rows": plan.syndrome_rows})
    print(_ladder_report(
        ladder_codec, bins, ladder_plan, prior_provider, side_info, args.ladder_hash_bits))


if __name__ == "__main__":
    main()
