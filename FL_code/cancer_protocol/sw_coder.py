"""Fixed-rate Slepian-Wolf coding for learned quantizer symbols.

This module implements the compression-relevant part of Zhang and Tao's multilevel
syndrome design: quantizer indices are split least-significant bit first, each binary plane is
encoded as an LDPC syndrome, and the decoder conditions each plane on its decoded lower planes
and its own side information.  The paper's Gaussian-CEO estimator is replaced by a generic
decoder-side conditional prior supplied by the quantizer.

Encoding and decoding are deliberately separate.  The encoder receives only symbols and an
immutable rate plan; the prior and side information exist only at the decoder.  There is no
feedback, checksum, or lossless fallback.  As in the paper's practical experiments, residual
belief-propagation errors are therefore measured rather than hidden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, log2
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
PAYLOAD_HEADER = struct.Struct(">Q8s")


class ConditionalPrior(Protocol):
    """Decoder-owned provider of a stage prior conditioned on decoded stages and side info."""

    def __call__(self, decoded_stages: torch.Tensor, side_info: torch.Tensor, stage: int) -> torch.Tensor:
        """Return ``q(U_stage | U_<stage, side_info)`` with shape ``(positions, alphabet)``."""


class SWCoderConfig(BaseModel):
    """Runtime settings used by the fixed-rate Slepian-Wolf encoder and decoder."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int = Field(default=2400, ge=32)
    max_bp_iterations: int = Field(default=100, ge=1)
    damping: float = Field(default=0.9, gt=0.0, le=1.0)
    llr_clip: float = Field(default=20.0, gt=0.0)
    code_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    rate_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    degree_distribution: tuple[tuple[int, float], ...] = DEFAULT_DEGREE_DISTRIBUTION
    girth_candidates: int = Field(default=8, ge=1)
    matrix_attempts: int = Field(default=8, ge=1)
    message_budget: int = Field(default=8_000_000, ge=1)
    matrix_backend: Literal["balanced_4cycle_v1"] = MATRIX_BACKEND

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
    matrix_backend: Literal["balanced_4cycle_v1"]

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


@dataclass(frozen=True, slots=True)
class SWDecodeResult:
    """Decoder output, diagnostics, and the rate plan inferred for the following round."""

    symbols: torch.Tensor
    plane_model_entropies: tuple[tuple[float, ...], ...]
    decoded_cross_entropy_stage_rates: tuple[float, ...]
    nonconverged_blocks: int
    bp_attempted_blocks: int
    syndrome_consistent_blocks: int
    plane_blocks: int
    decode_passes: int
    bp_iterations: int
    next_plan: SWRatePlan
    seconds: float


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
    def bp_convergence_rate(self) -> float:
        return self.syndrome_consistent_blocks / max(1, self.bp_attempted_blocks)

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
        chosen: set[int] = set()
        neighbours: set[int] = set()
        for _ in range(degree):
            available = np.asarray([row for row in range(check_count) if row not in chosen])
            available = available[load[available] == load[available].min()]
            if available.size > girth_candidates:
                available = rng.choice(available, girth_candidates, replace=False)
            overlap = np.asarray([len(neighbours.intersection(check_variables[int(row)])) for row in available])
            row = int(available[np.flatnonzero(overlap == overlap.min())[0]])
            edge_variable.append(variable)
            edge_check.append(row)
            chosen.add(row)
            neighbours.update(check_variables[row])
            check_variables[row].add(variable)
            load[row] += 1

    return np.asarray(edge_variable, dtype=np.int64), np.asarray(edge_check, dtype=np.int64)


def _full_row_rank(edge_variable: np.ndarray, edge_check: np.ndarray, check_count: int, block_length: int) -> bool:
    matrix = sparse.csr_matrix(
        (np.ones(edge_variable.size, dtype=np.uint8), (edge_check, edge_variable)),
        shape=(check_count, block_length),
    )
    return gf2_rank(matrix, method="dense") == check_count


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


def _lsb_plane_probabilities(prior: torch.Tensor, prefix: torch.Tensor, bit: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero/one probabilities conditioned on already decoded less-significant bits."""
    alphabet = prior.shape[1]
    groups = alphabet >> (bit + 1)
    bases = prefix[:, None] + torch.arange(groups, device=prior.device)[None, :] * (2 << bit)
    zero = prior.gather(1, bases).sum(dim=1)
    one = prior.gather(1, bases + (1 << bit)).sum(dim=1)
    normalization = zero + one
    assert bool((normalization > 0).all()), f"decoded lower-bit prefix has zero probability at bit {bit + 1}."
    return zero / normalization, one / normalization


def _plane_llr(prior: torch.Tensor, prefix: torch.Tensor, bit: int, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return posterior source LLRs and binary entropies for one LSB-first plane."""
    zero, one = _lsb_plane_probabilities(prior, prefix, bit)
    entropy = -(torch.xlogy(zero, zero) + torch.xlogy(one, one)) / float(np.log(2))
    return (zero.clamp(min=1e-12).log() - one.clamp(min=1e-12).log()).clamp(-clip, clip), entropy


class SlepianWolfCodec:
    """Fixed-rate encoder and decoder with an instance-local parity-matrix cache."""

    def __init__(self, cfg: SWCoderConfig | None = None) -> None:
        self.cfg = cfg or SWCoderConfig()
        self._matrices: dict[
            tuple[str, int, int, tuple[tuple[int, float], ...], int, int, int, int, int], tuple[np.ndarray, np.ndarray]
        ] = {}

    def initial_plan(self, num_stages: int, alphabet: int) -> SWRatePlan:
        """Return the rate-one plan used to bootstrap previous-round adaptation."""
        return SWRatePlan.initial(num_stages, alphabet, self.cfg)

    def _matrix(self, plan: SWRatePlan, stage: int, bit: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
        key = (
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
        if key not in self._matrices:
            assert plan.matrix_backend == MATRIX_BACKEND
            for attempt in range(plan.matrix_attempts):
                rng = np.random.default_rng(np.random.SeedSequence([plan.code_seed, stage, bit, rows, attempt]))
                structure = _parity_structure(
                    plan.block_length, rows, rng, plan.degree_distribution, plan.girth_candidates
                )
                if _full_row_rank(*structure, rows, plan.block_length):
                    self._matrices[key] = structure
                    break
            else:
                raise AssertionError(
                    f"could not construct a full-row-rank {rows}x{plan.block_length} "
                    f"parity matrix for stage {stage + 1}, bit {bit + 1}."
                )
        return self._matrices[key]

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
        packed_planes: list[np.ndarray] = []

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
                    edge_variable, edge_check = self._matrix(plan, stage, bit, rows)
                    variable_index = torch.as_tensor(edge_variable, device=plane.device)
                    check_index = torch.as_tensor(edge_check, device=plane.device)
                    syndrome = torch.zeros(
                        blocks, rows, dtype=plane.dtype, device=plane.device
                    ).index_add_(1, check_index, plane[:, variable_index]) % 2
                packed_planes.append(syndrome.to(torch.uint8).cpu().numpy().reshape(-1))

        syndrome_bits = np.concatenate(packed_planes) if packed_planes else np.empty(0, dtype=np.uint8)
        syndromes = np.packbits(syndrome_bits, bitorder="little").tobytes()
        assert syndrome_bits.size == self._expected_syndrome_bits(plan, positions)
        return SWPayload(positions=positions, plan_id=plan.identifier, syndromes=syndromes)

    def decode_syndromes(
        self, payload: SWPayload, plan: SWRatePlan, prior_provider: ConditionalPrior, side_info: torch.Tensor
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
        decoded = torch.zeros((plan.num_stages, positions), dtype=torch.long)
        entropies: list[tuple[float, ...]] = []
        decoded_cross_entropies: list[float] = []
        next_rows: list[tuple[int, ...]] = []
        offset = passes = iterations = bp_attempted = syndrome_consistent = 0

        for stage, stage_rows in enumerate(plan.syndrome_rows):
            prior = prior_provider(decoded, side_info, stage).to(device)
            assert prior.shape == (positions, plan.alphabet), (
                f"stage {stage + 1} prior has shape {prior.shape}; "
                f"expected {(positions, plan.alphabet)}."
            )
            assert bool(torch.isfinite(prior).all()) and bool((prior >= 0).all())
            assert torch.allclose(
                prior.sum(dim=1), torch.ones(positions, device=device), atol=1e-4, rtol=1e-4
            ), f"stage {stage + 1} prior rows must sum to one."
            prefix = torch.zeros(positions, dtype=torch.long, device=device)
            stage_entropies: list[float] = []

            for bit, rows in enumerate(stage_rows):
                llr, entropy = _plane_llr(prior, prefix, bit, self.cfg.llr_clip)
                mean_entropy = float(entropy.mean())
                stage_entropies.append(mean_entropy)
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
                    plane = (llr < 0).to(llr.dtype)
                    succeeded = torch.ones(blocks, dtype=torch.bool, device=device)
                else:
                    edge_variable, edge_check = self._matrix(plan, stage, bit, rows)
                    variable_index = torch.as_tensor(edge_variable, device=device)
                    check_index = torch.as_tensor(edge_check, device=device)
                    per_pass = max(1, self.cfg.message_budget // edge_variable.size)
                    plane = torch.empty_like(llr)
                    succeeded = torch.empty(blocks, dtype=torch.bool, device=device)
                    for chunk in torch.arange(blocks, device=device).split(per_pass):
                        plane[chunk], succeeded[chunk], used = _bp_decode(
                            llr[chunk], syndrome[chunk], variable_index, check_index, self.cfg
                        )
                        iterations += int(used.sum())
                        bp_attempted += chunk.numel()
                        passes += 1
                    syndrome_consistent += int(succeeded.sum())
                prefix |= plane.flatten()[:positions].long() << bit

            decoded[stage] = prefix.cpu()
            decoded_cross_entropies.append(
                float(-torch.log2(prior.gather(1, prefix[:, None]).squeeze(1).clamp(min=1e-12)).mean())
            )
            entropies.append(tuple(stage_entropies))
            next_rows.append(
                tuple(
                    min(plan.block_length, ceil(plan.block_length * (entropy + self.cfg.rate_margin)))
                    for entropy in stage_entropies
                )
            )

        assert offset == expected_bits
        return SWDecodeResult(
            symbols=decoded,
            plane_model_entropies=tuple(entropies),
            decoded_cross_entropy_stage_rates=tuple(decoded_cross_entropies),
            nonconverged_blocks=bp_attempted - syndrome_consistent,
            bp_attempted_blocks=bp_attempted,
            syndrome_consistent_blocks=syndrome_consistent,
            plane_blocks=blocks * plan.num_stages * plan.bit_width,
            decode_passes=passes,
            bp_iterations=iterations,
            next_plan=plan.model_copy(update={"syndrome_rows": tuple(next_rows)}),
            seconds=perf_counter() - start,
        )


def encode_syndromes(bins: torch.Tensor, plan: SWRatePlan, cfg: SWCoderConfig | None = None) -> SWPayload:
    """Encode quantizer symbols without exposing decoder-only state to the encoder."""
    return SlepianWolfCodec(cfg).encode_syndromes(bins, plan)


def decode_syndromes(
    payload: SWPayload,
    plan: SWRatePlan,
    prior_provider: ConditionalPrior,
    side_info: torch.Tensor,
    cfg: SWCoderConfig | None = None,
) -> SWDecodeResult:
    """Decode a syndrome payload and produce the following round's rate plan."""
    return SlepianWolfCodec(cfg).decode_syndromes(payload, plan, prior_provider, side_info)


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


def _undetected_block_errors(differences: torch.Tensor, plan: SWRatePlan, codec: SlepianWolfCodec) -> int:
    """Count erroneous BP-decoded plane blocks that still satisfy their syndrome."""
    blocks = ceil(differences.shape[1] / plan.block_length)
    padding = blocks * plan.block_length - differences.shape[1]
    undetected = 0
    for stage, stage_rows in enumerate(plan.syndrome_rows):
        for bit, rows in enumerate(stage_rows):
            if rows in (0, plan.block_length):
                continue
            errors = torch.nn.functional.pad(
                ((differences[stage] >> bit) & 1).float(), (0, padding)
            ).view(blocks, plan.block_length)
            edge_variable, edge_check = codec._matrix(plan, stage, bit, rows)
            parity = torch.zeros(blocks, rows).index_add_(
                1, torch.as_tensor(edge_check), errors[:, torch.as_tensor(edge_variable)]
            ) % 2
            undetected += int((errors.bool().any(dim=1) & ~parity.bool().any(dim=1)).sum())
    return undetected


def evaluate_slepian_wolf(
    bins: torch.Tensor,
    plan: SWRatePlan,
    prior_provider: ConditionalPrior,
    side_info: torch.Tensor,
    cfg: SWCoderConfig | None = None,
) -> SWResult:
    """Round-trip a quantized vector and measure rate beside residual decoding errors."""
    start = perf_counter()
    codec = SlepianWolfCodec(cfg)
    payload = codec.encode_syndromes(bins, plan)
    decoded = codec.decode_syndromes(payload, plan, prior_provider, side_info)
    source = bins.long().cpu()
    differences = decoded.symbols ^ source
    model_entropies, cross_entropies = _true_prior_stage_statistics(bins, prior_provider, side_info, plan.alphabet)
    blocks = ceil(bins.shape[1] / plan.block_length)
    plane_bit_errors: list[tuple[int, ...]] = []
    plane_block_errors: list[tuple[int, ...]] = []
    for stage in range(plan.num_stages):
        stage_bit_errors: list[int] = []
        stage_block_errors: list[int] = []
        for bit in range(plan.bit_width):
            errors = (differences[stage] >> bit) & 1
            stage_bit_errors.append(int(errors.sum()))
            stage_block_errors.append(sum(bool(chunk.any()) for chunk in errors.split(plan.block_length)))
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
        undetected_block_errors=_undetected_block_errors(differences, plan, codec),
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


def main() -> None:
    """Benchmark a previous-round rate plan on the quantizer's synthetic Gaussian source."""
    from FL_code.FL_core.utils import set_global_seed

    from .prior_code import PriorCalculator
    from .wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1_000_000, help="coded vector length")
    parser.add_argument("--noise-power", type=float, default=0.01)
    parser.add_argument("--bins-per-plane", type=int, default=2)
    parser.add_argument("--num-planes", type=int, default=3)
    parser.add_argument("--reconst-ld", type=float, default=120.0)
    parser.add_argument("--block-length", type=int, default=SWCoderConfig().block_length)
    parser.add_argument("--rate-margin", type=float, default=SWCoderConfig().rate_margin)
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
        tf32=False,
        fused_optimizer=True,
        hidden_dim=100,
        layers=3,
    )
    train_target, train_si = source(args.size)
    quantizer = DedupedWZQuantizerCancer(quantizer_config, si_size=1)
    quantizer.train_model(train_target, [train_si])

    target, si = source(args.size)
    quantizer.side_info_list_used = [si]
    bins, soft_codes, metadata = quantizer.encoding_process(target)
    assert soft_codes is not None
    side_info = quantizer.normalize_side_info_tensor(metadata, [si])
    prior_config = quantizer_config.model_copy(update={"marginal_loss": False})
    prior_model = PriorCalculator.make_trained_prior_model(bins.long(), soft_codes, side_info, prior_config)

    def prior_provider(decoded_stages: torch.Tensor, decoder_side_info: torch.Tensor, stage: int) -> torch.Tensor:
        return PriorCalculator.compute_prior_from_network(
            prior_model, decoded_stages, decoder_side_info
        )[stage].to(decoder_side_info.device)

    cfg = SWCoderConfig(block_length=args.block_length, rate_margin=args.rate_margin, code_seed=args.seed)
    codec = SlepianWolfCodec(cfg)
    bootstrap = codec.initial_plan(args.num_planes, args.bins_per_plane)
    bootstrap_payload = codec.encode_syndromes(bins, bootstrap)
    plan = codec.decode_syndromes(bootstrap_payload, bootstrap, prior_provider, side_info).next_plan
    result = evaluate_slepian_wolf(bins, plan, prior_provider, side_info, cfg)
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


if __name__ == "__main__":
    main()
