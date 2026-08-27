"""Feedback-driven Slepian--Wolf transports for codecs that produce staged indices.

The encoder here owns nothing but its symbols.  Side information, the prior network and the
decoder live at the server, so every rate decision is made there and travels back down, which is
what separates this from :mod:`oracle_sw_mixin`: no plan is ever chosen by looking at the vector
it will be spent on.

One round, in the order things cross the link:

1. **up** the quantizer's preprocessing metadata, which the decoder needs before it can normalize
   its own side information the same way.  It already rides in the ordinary payload.
2. **down** a pilot plan.  The decoder gets it from :func:`expected_plane_rates`, the learned-prior
   form of Zhang and Tao's Eq. (45) controller, plus a fixed bootstrap margin for the LDPC
   finite-length gap.  No symbol has been sent at this point and none is needed.
3. **up** syndromes for a seeded pilot subset, each block carrying a short hash.  Blocks whose
   syndrome or hash fails are named on the way down and resent raw on the way up, so the symbols
   the next step calibrates on are the source symbols.
4. **down** the bulk plan: for every plane, the smallest row count that decoded the pilot without
   a single bit error, plus a safety margin.  The pilot took no part in the bulk, so this is a
   held-out rate, not a fitted one.
5. **up** the bulk syndromes, hashed and repaired the same way.

``pilot_blocks=0`` drops steps 3 and 4 and ships the step-2 plan directly, which is the paper's
one-shot controller and the A/B that says what the pilot is worth.

``LDPCASWRoundMixin`` is the same idea with the rate decided per block instead of per plane.  Its
encoder sends nothing but a prefix of Varodayan et al.'s rate-one accumulated syndrome, and the
decoder climbs one rung at a time until each block's hash verifies, so there is no pilot, no
margin and no calibration: steps 2-4 above collapse into "start at the model entropy and ask for
more where it is needed".  A block that is easy stops early and a block that is hard cannot make
it pay, which is where the rate goes.

Both ends run in this process, so the encoder simply calls the decoder's half; the point of the
exercise is that every bit in either direction is counted, not that the two halves sit on
separate machines.
"""

from __future__ import annotations

from math import ceil, log2
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from FL_code.FL_core.utils import decompress_data_list

from ..round_codecs import CancerRecord
from ..wz_quantizer import PreprocessMetadata
from .sw_coder import (
    BlockRecovery,
    ConditionalPrior,
    HashVerification,
    LDPCA_BACKEND,
    PlaneHashes,
    PlaneVerification,
    SlepianWolfCodec,
    SWCoderConfig,
    SWLadderPayload,
    SWLadderReplay,
    SWLadderResult,
    SWPayload,
    SWRatePlan,
    blocked_planes,
    expected_plane_rates,
    plane_rows,
)


class PracticalSWConfig(BaseModel):
    """Settings of the feedback rate controller wrapped around the fixed-rate SW core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pilot_blocks: int = Field(default=20, ge=0)
    permutation_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    rate_samples: int = Field(default=2, ge=1)
    # Bits per plane added to the a-priori entropy for the pilot only. It has to cover both the
    # LDPC finite-length gap and however over-confident the prior turns out to be, which is the
    # very thing the pilot exists to measure -- so it is deliberately generous. Swept on synthetic
    # Gaussian data, 0.12 left a sixth of the pilot needing repair and 0.20 left none; at or above
    # 1.0 the pilot goes uncoded, which is the simplest thing that certainly works.
    bootstrap_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    # Rows added to what the pilot certified. The decoding threshold is a ragged band rather than
    # a waterfall, so the first error-free point on a few blocks is optimistic for the twenty times
    # more blocks in the bulk. Swept over 24-240 rows, the total rate bottomed out near 72.
    safety_rows: int = Field(default=72, ge=0)
    search_row_step: int = Field(default=24, ge=1)
    search_grid_points: int = Field(default=24, ge=1)
    # Hash bits per coded block. Zero degenerates to no verification, because an empty hash
    # compares equal for every block -- and then a repaired decode is no longer exact.
    verification_bits: int = Field(default=16, ge=0)
    recover_failed_blocks: bool = True

    @model_validator(mode="after")
    def validate_pilot(self) -> Self:
        """Require a pilot the decoder can trust: verified and repaired, or uncoded."""
        exact = (self.recover_failed_blocks and self.verification_bits > 0) or self.bootstrap_margin >= 1.0
        assert not self.pilot_blocks or exact, (
            "a pilot calibrates on symbols the decoder holds exactly, so it needs both "
            "recover_failed_blocks and verification_bits, or bootstrap_margin >= 1.0 to send the "
            "pilot uncoded."
        )
        return self


class SWRepair(BaseModel):
    """What one coded chunk spent on being exact: a hash per block, and the blocks resent raw."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_length: int = Field(ge=32)
    hash_bits: int = Field(ge=0)
    hashed_blocks: int = Field(ge=0)
    hashes: bytes = b""
    requests: tuple[tuple[int, int, tuple[int, ...]], ...] = ()
    bits: bytes = b""

    @model_validator(mode="after")
    def validate_bits(self) -> Self:
        """Require one packed hash per coded block and one packed block per repair request."""
        assert len(self.hashes) == ceil(self.hashed_blocks * self.hash_bits / 8), (
            f"{self.hashed_blocks} hashes of {self.hash_bits} bits do not fit {len(self.hashes)} bytes."
        )
        assert len(self.bits) == ceil(self.block_count * self.block_length / 8), (
            f"{self.block_count} repaired blocks of {self.block_length} bits do not fit "
            f"{len(self.bits)} bytes."
        )
        return self

    @property
    def block_count(self) -> int:
        return sum(len(blocks) for _, _, blocks in self.requests)

    @property
    def uplink_bits(self) -> int:
        """Return what being exact cost the uplink: every hash, and every block resent raw."""
        return self.hashed_blocks * self.hash_bits + self.block_count * self.block_length

    def planes(self) -> torch.Tensor:
        """Unpack every repaired block, in request order."""
        unpacked = np.unpackbits(np.frombuffer(self.bits, dtype=np.uint8), bitorder="little")
        return torch.as_tensor(
            unpacked[: self.block_count * self.block_length].astype(np.float32)
        ).view(self.block_count, self.block_length)


class PracticalSWPayload(BaseModel):
    """Everything one worker put on the uplink, in the order the protocol sent it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    positions: int = Field(gt=0)
    permutation_seed: int = Field(ge=0, le=2**32 - 1)
    pilot_plan: SWRatePlan | None
    pilot: SWPayload | None
    pilot_repair: SWRepair | None
    bulk_plan: SWRatePlan
    bulk: SWPayload
    bulk_repair: SWRepair

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        """Require the pilot to be present or absent as a whole and every part to match its plan."""
        assert len({part is None for part in (self.pilot_plan, self.pilot, self.pilot_repair)}) == 1, (
            "a pilot plan, its syndromes and its repairs are sent together or not at all."
        )
        assert self.bulk.plan_id == self.bulk_plan.identifier
        if self.pilot is None:
            assert self.bulk.positions == self.positions
        else:
            assert self.pilot_plan is not None and self.pilot.plan_id == self.pilot_plan.identifier
            assert self.pilot.positions + self.bulk.positions == self.positions
        return self

    @property
    def parts(self) -> tuple[tuple[SWRatePlan, SWPayload, SWRepair], ...]:
        """Return the coded chunks that were actually sent, pilot first."""
        pilot = (
            ()
            if self.pilot is None
            else ((self.pilot_plan, self.pilot, self.pilot_repair),)
        )
        return (*pilot, (self.bulk_plan, self.bulk, self.bulk_repair))

    @property
    def uplink_bits(self) -> int:
        """Return every bit the worker sent: syndromes, block hashes and raw repairs."""
        return sum(part.wire_bits + repair.uplink_bits for _, part, repair in self.parts)

    @property
    def downlink_bits(self) -> int:
        """Return what the server sent back: the rate plans and the indices of failed blocks."""
        index_bits = max(1, ceil(log2(max(2, ceil(self.positions / self.bulk_plan.block_length)))))
        return sum(
            plan_row_bits(plan) + repair.block_count * index_bits for plan, _, repair in self.parts
        )

    @property
    def repaired_blocks(self) -> int:
        return sum(repair.block_count for _, _, repair in self.parts)


def plan_row_bits(plan: SWRatePlan) -> int:
    """Return a plan's size on the wire as two bytes per plane.

    Only the row counts change from round to round; block length, seed and degree profile are
    agreed once and are not charged here.
    """
    return 16 * plan.num_stages * plan.bit_width


class _RepairRecorder:
    """Answer a decoder's repair requests from the true symbols and keep what it asked for."""

    def __init__(self, planes: torch.Tensor, bit_width: int, block_length: int) -> None:
        self.source_planes = planes
        self.bit_width = bit_width
        self.block_length = block_length
        self.requests: list[tuple[int, int, tuple[int, ...]]] = []
        self.planes: list[torch.Tensor] = []

    def __call__(self, stage: int, bit: int, blocks: torch.Tensor) -> torch.Tensor:
        wanted = blocks.cpu()
        plane = self.source_planes[stage * self.bit_width + bit][wanted]
        self.requests.append((stage, bit, tuple(wanted.tolist())))
        self.planes.append(plane)
        return plane

    def payload(self, hashes: PlaneHashes, digests: torch.Tensor) -> SWRepair:
        """Return the hashes and the recorded repairs as the bits they cost on the uplink."""
        packed = (
            np.packbits(
                torch.cat(self.planes).reshape(-1).to(torch.uint8).numpy(), bitorder="little"
            ).tobytes()
            if self.planes
            else b""
        )
        return SWRepair(
            block_length=self.block_length,
            hash_bits=hashes.hash_bits,
            hashed_blocks=digests.shape[0] * digests.shape[1],
            hashes=hashes.pack(digests),
            requests=tuple(self.requests),
            bits=packed,
        )


class _RepairReplay:
    """Serve recorded repairs to a rerun of the decode and refuse anything else."""

    def __init__(self, repair: SWRepair) -> None:
        self.repair = repair
        self.planes = repair.planes()
        self.request_index = 0
        self.block_index = 0

    def __call__(self, stage: int, bit: int, blocks: torch.Tensor) -> torch.Tensor:
        requested = (stage, bit, tuple(blocks.cpu().tolist()))
        assert self.request_index < len(self.repair.requests), (
            f"decoding asked to repair stage {stage + 1} bit {bit + 1} beyond the "
            f"{len(self.repair.requests)} repairs recorded at encode time; belief propagation did "
            "not reproduce."
        )
        expected = self.repair.requests[self.request_index]
        assert expected == requested, (
            f"decoding asked to repair stage {stage + 1} bit {bit + 1} blocks {requested[2]}, but "
            f"the payload carries {expected[2]} for stage {expected[0] + 1} bit {expected[1] + 1}; "
            "belief propagation did not reproduce, which CUDA reductions can cause near the "
            "decoding threshold."
        )
        served = self.planes[self.block_index : self.block_index + len(expected[2])]
        self.request_index += 1
        self.block_index += len(expected[2])
        return served


class PracticalSWRecord(CancerRecord):
    """Compression record carrying what the feedback rate controller decided and what it cost."""

    sw_positions: int | None = None
    sw_pilot_positions: int | None = None
    sw_uplink_rate: float | None = None
    sw_downlink_rate: float | None = None
    sw_pilot_rate: float | None = None
    sw_bulk_rate: float | None = None
    sw_repair_rate: float | None = None
    sw_repaired_blocks: int | None = None
    sw_pilot_repaired_blocks: int | None = None
    sw_apriori_rate: float | None = None
    sw_pilot_cross_entropy_rate: float | None = None
    sw_apriori_plane_rates: list[list[float]] | None = None
    sw_pilot_cross_entropy_plane_rates: list[list[float]] | None = None
    sw_bulk_plane_rows: list[list[int]] | None = None


class _NonOracleSWMixin:
    """What both non-oracle transports share: one codec, one hash family, and the decoder context.

    Everything here is deliberately rate-agnostic.  The two transports differ only in how they
    turn a decoder-side prior into bits on the wire.
    """

    payload_class: type[BaseModel]
    sw_config: SWCoderConfig
    sw_codec: SlepianWolfCodec
    sw_hashes: PlaneHashes

    def __init__(self, *args: Any, sw_config: SWCoderConfig | None, verification_bits: int, **kwargs: Any) -> None:
        self.sw_config = SWCoderConfig() if sw_config is None else sw_config
        # One codec for both ends: its parity matrices are the expensive part and both halves of
        # the protocol want the same ones.
        self.sw_codec = SlepianWolfCodec(self.sw_config)
        self.sw_hashes = PlaneHashes(
            self.sw_config.block_length, verification_bits, self.sw_config.code_seed
        )
        super().__init__(*args, **kwargs)

    def _staged_alphabet(self, bins: torch.Tensor) -> int:
        """Return the stage alphabet after checking the shape and power-of-two contract."""
        alphabet = self.quantizer.bins_per_plane
        assert bins.ndim == 2 and bins.shape[0] == self.quantizer.num_planes, (
            f"expected staged indices with shape ({self.quantizer.num_planes}, positions), "
            f"got {tuple(bins.shape)}."
        )
        assert alphabet >= 2 and alphabet & (alphabet - 1) == 0, (
            f"Slepian-Wolf coding requires a power-of-two stage alphabet; got {alphabet}."
        )
        return alphabet

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Replace the ordinary staged indices with what the feedback protocol put on the wire."""
        outer_payload = super().get_encod_payload(delta_vec, record)
        bins, metadata = outer_payload["payload_content"]
        outer_payload["payload_content"] = (
            self.encode_staged_indices(bins, metadata, record), metadata)
        return outer_payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Recover staged indices and return the quantizer's normal final reconstruction."""
        transmitted, metadata = decompress_data_list(payload)["payload_content"]
        assert isinstance(transmitted, self.payload_class), (
            f"{type(self).__name__} decodes {self.payload_class.__name__}, not "
            f"{type(transmitted).__name__}."
        )
        bins = self.decode_staged_indices(transmitted, metadata)
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]

    def _sw_context(self, metadata: PreprocessMetadata) -> tuple[torch.Tensor, ConditionalPrior]:
        """Return decoder side information and the quantizer's causal stage prior.

        The side information is moved to the host because the encoder-side run of the protocol and
        the decoder's own rerun have to agree bitwise about which blocks failed, and CUDA
        reductions do not agree near the decoding threshold.  The prior network itself still runs
        wherever the quantizer lives.
        """
        side_info = self.quantizer.side_info_tensor(metadata).cpu()

        def prior_provider(
            decoded_stages: torch.Tensor, decoder_side_info: torch.Tensor, stage: int
        ) -> torch.Tensor:
            return self.quantizer.prior_calculator.compute_prior_from_network(
                self.quantizer.coding_model, decoded_stages, decoder_side_info
            )[stage].to(decoder_side_info.device)

        return side_info, prior_provider


class PracticalSWRoundMixin(_NonOracleSWMixin):
    """Fixed-rate staged-index SW transport whose rate is calibrated on a decoded pilot."""

    record_class = PracticalSWRecord
    payload_class = PracticalSWPayload
    practical_config: PracticalSWConfig

    def __init__(
        self,
        *args: Any,
        sw_config: SWCoderConfig | None = None,
        practical_config: PracticalSWConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.practical_config = PracticalSWConfig() if practical_config is None else practical_config
        super().__init__(
            *args,
            sw_config=sw_config,
            verification_bits=self.practical_config.verification_bits,
            **kwargs,
        )

    def _split(self, positions: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the seeded pilot and bulk position indices.

        One permutation does two jobs: its head is a uniform random pilot, so what the pilot
        certifies transfers to the bulk, and its tail interleaves the bulk, so a block is a sample
        of the whole vector instead of one contiguous slab of one layer.
        """
        pilot_positions = self.practical_config.pilot_blocks * self.sw_config.block_length
        assert pilot_positions + self.sw_config.block_length <= positions, (
            f"{self.practical_config.pilot_blocks} pilot blocks of {self.sw_config.block_length} "
            f"leave no full bulk block in {positions} positions."
        )
        permutation = torch.randperm(
            positions,
            generator=torch.Generator().manual_seed(self.practical_config.permutation_seed),
        )
        return permutation[:pilot_positions], permutation[pilot_positions:]

    def _code_chunk(
        self,
        bins: torch.Tensor,
        plan: SWRatePlan,
        prior_provider: ConditionalPrior,
        side_info: torch.Tensor,
    ) -> tuple[SWPayload, SWRepair, torch.Tensor]:
        """Encode one chunk, run the decoder over it, and repair whatever it could not verify."""
        payload = self.sw_codec.encode_syndromes(bins, plan)
        planes = blocked_planes(bins, plan)
        digests = self.sw_hashes.digest_planes(planes)
        recorder = _RepairRecorder(planes, plan.bit_width, plan.block_length)
        recovery: BlockRecovery | None = (
            recorder if self.practical_config.recover_failed_blocks else None
        )
        verify: PlaneVerification = HashVerification(self.sw_hashes, digests, plan.bit_width)
        decoded = self.sw_codec.decode_syndromes(
            payload, plan, prior_provider, side_info, recovery, verify
        )
        return payload, recorder.payload(self.sw_hashes, digests), decoded.symbols

    def encode_staged_indices(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, record: CancerRecord | None = None
    ) -> PracticalSWPayload:
        """Run the feedback protocol and return everything that crossed the uplink."""
        alphabet = self._staged_alphabet(bins)
        config = self.practical_config
        side_info, prior_provider = self._sw_context(metadata)
        pilot_index, bulk_index = self._split(bins.shape[1])
        base = self.sw_codec.initial_plan(bins.shape[0], alphabet)

        apriori = expected_plane_rates(
            prior_provider,
            side_info[pilot_index if config.pilot_blocks else bulk_index],
            bins.shape[0],
            alphabet,
            config.rate_samples,
            config.permutation_seed,
        )
        measured: tuple[tuple[float, ...], ...] | None = None

        if config.pilot_blocks:
            pilot_plan = base.model_copy(update={
                "syndrome_rows": plane_rows(apriori, config.bootstrap_margin, base.block_length)})
            pilot_payload, pilot_repair, pilot_symbols = self._code_chunk(
                bins[:, pilot_index], pilot_plan, prior_provider, side_info[pilot_index])
            assert bool((pilot_symbols == bins[:, pilot_index].long()).all()), (
                "the pilot the calibration is about to trust does not match the source; its "
                "hashes and repairs did not restore every block."
            )
            # The floor the search starts from carries no margin: the calibration is free to land
            # under the bootstrap plan when the code turns out better than that constant assumed.
            floor = base.model_copy(update={
                "syndrome_rows": plane_rows(apriori, 0.0, base.block_length)})
            calibrated, measured = self.sw_codec.calibrate_plane_rows(
                pilot_symbols,
                floor,
                prior_provider,
                side_info[pilot_index],
                config.search_row_step,
                config.search_grid_points,
            )
            bulk_rows = tuple(
                tuple(min(base.block_length, rows + config.safety_rows) for rows in stage)
                for stage in calibrated
            )
        else:
            pilot_plan = pilot_payload = pilot_repair = None
            bulk_rows = plane_rows(apriori, config.bootstrap_margin, base.block_length)

        bulk_plan = base.model_copy(update={"syndrome_rows": bulk_rows})
        bulk_payload, bulk_repair, _ = self._code_chunk(
            bins[:, bulk_index], bulk_plan, prior_provider, side_info[bulk_index])

        payload = PracticalSWPayload(
            positions=bins.shape[1],
            permutation_seed=config.permutation_seed,
            pilot_plan=pilot_plan,
            pilot=pilot_payload,
            pilot_repair=pilot_repair,
            bulk_plan=bulk_plan,
            bulk=bulk_payload,
            bulk_repair=bulk_repair,
        )
        if record is not None:
            self._record_rates(payload, apriori, measured, record)
        return payload

    def _record_rates(
        self,
        payload: PracticalSWPayload,
        apriori: tuple[tuple[float, ...], ...],
        measured: tuple[tuple[float, ...], ...] | None,
        record: CancerRecord,
    ) -> None:
        """Attach the controller's decisions and their cost, all in bits per original position."""
        assert isinstance(record, PracticalSWRecord), (
            f"{type(record).__name__} cannot hold the practical Slepian-Wolf accounting.")
        positions = payload.positions
        record.update_record_fields(
            sw_positions=positions,
            sw_pilot_positions=0 if payload.pilot is None else payload.pilot.positions,
            sw_uplink_rate=payload.uplink_bits / positions,
            sw_downlink_rate=payload.downlink_bits / positions,
            sw_bulk_rate=self.sw_codec._expected_syndrome_bits(
                payload.bulk_plan, payload.bulk.positions) / positions,
            sw_repair_rate=sum(
                repair.uplink_bits for _, _, repair in payload.parts) / positions,
            sw_repaired_blocks=payload.repaired_blocks,
            sw_pilot_repaired_blocks=(
                0 if payload.pilot_repair is None else payload.pilot_repair.block_count),
            sw_apriori_rate=sum(map(sum, apriori)),
            sw_apriori_plane_rates=[list(stage) for stage in apriori],
            sw_bulk_plane_rows=[list(stage) for stage in payload.bulk_plan.syndrome_rows],
        )
        if payload.pilot is not None and payload.pilot_plan is not None:
            record.update_record_fields(
                sw_pilot_rate=self.sw_codec._expected_syndrome_bits(
                    payload.pilot_plan, payload.pilot.positions) / positions,
            )
        if measured is not None:
            record.update_record_fields(
                sw_pilot_cross_entropy_rate=sum(map(sum, measured)),
                sw_pilot_cross_entropy_plane_rates=[list(stage) for stage in measured],
            )

    def decode_staged_indices(
        self, payload: PracticalSWPayload, metadata: PreprocessMetadata
    ) -> torch.Tensor:
        """Recover staged indices from the transmitted plans, syndromes, hashes and repairs."""
        assert payload.bulk_plan.block_length == self.sw_config.block_length, (
            f"payload block length {payload.bulk_plan.block_length} does not match "
            f"decoder block length {self.sw_config.block_length}."
        )
        assert payload.bulk_plan.num_stages == self.quantizer.num_planes
        assert payload.bulk_plan.alphabet == self.quantizer.bins_per_plane
        side_info, prior_provider = self._sw_context(metadata)
        pilot_index, bulk_index = self._split(payload.positions)
        symbols = torch.zeros((payload.bulk_plan.num_stages, payload.positions), dtype=torch.long)

        for index, (plan, part, repair) in zip(
            (bulk_index,) if payload.pilot is None else (pilot_index, bulk_index),
            payload.parts,
            strict=True,
        ):
            symbols[:, index] = self.sw_codec.decode_syndromes(
                part,
                plan,
                prior_provider,
                side_info[index],
                _RepairReplay(repair),
                HashVerification(
                    self.sw_hashes,
                    self.sw_hashes.unpack(repair.hashes, plan.num_stages * plan.bit_width),
                    plan.bit_width,
                ),
            ).symbols
        return symbols


class LDPCAConfig(BaseModel):
    """Settings of the rate-adaptive LDPCA ladder wrapped around the same fixed-rate SW core."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate_samples: int = Field(default=2, ge=1)
    rate_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    # Bits per plane added to the model entropy before the climb starts. It buys nothing in rate:
    # the ladder charges each block for the rung it stops at, so a start below every block's
    # requirement costs belief-propagation passes only, while a start above one costs that block
    # the difference forever. Zero is therefore the rate-optimal choice and the default; raise it
    # only to trade a little rate for a shorter climb.
    start_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    # Hash bits per coded block. A ladder tests one candidate word per rung, so its false-accept
    # exposure is the number of rungs climbed times 2**-verification_bits -- far more attempts
    # than the one-shot transport makes, which is why this default is twice that one's. At the
    # default block length it costs 0.013 bit/position per plane.
    verification_bits: int = Field(default=32, ge=1)


class LDPCASWPayload(BaseModel):
    """Everything one worker put on the uplink for a rate-adaptive Slepian-Wolf round.

    ``plan`` is only the rung the climb started from, agreed before any symbol was sent; what each
    block actually cost is in ``ladder.levels``, which is the point of the transport.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: SWRatePlan
    ladder: SWLadderPayload
    hash_bits: int = Field(ge=1)
    hashes: bytes

    @model_validator(mode="after")
    def validate_parts(self) -> Self:
        """Require the ladder to name this plan and one packed hash per coded block."""
        assert self.plan.matrix_backend == LDPCA_BACKEND, (
            f"a ladder payload needs the {LDPCA_BACKEND} backend; the plan names "
            f"{self.plan.matrix_backend}."
        )
        assert self.ladder.plan_id == self.plan.identifier, (
            f"ladder plan {self.ladder.plan_id} does not match the transmitted plan "
            f"{self.plan.identifier}."
        )
        assert len(self.ladder.levels) == self.plan.num_stages * self.plan.bit_width, (
            f"the ladder covers {len(self.ladder.levels)} planes; the plan has "
            f"{self.plan.num_stages * self.plan.bit_width}."
        )
        assert len(self.hashes) == ceil(self.hashed_blocks * self.hash_bits / 8), (
            f"{self.hashed_blocks} hashes of {self.hash_bits} bits do not fit "
            f"{len(self.hashes)} bytes."
        )
        return self

    @property
    def hashed_blocks(self) -> int:
        return len(self.ladder.levels) * self.ladder.blocks

    @property
    def uplink_bits(self) -> int:
        """Return every bit the worker sent: accumulated rungs, block hashes and raw blocks."""
        return self.ladder.uplink_bits + self.hashed_blocks * self.hash_bits

    @property
    def downlink_bits(self) -> int:
        """Return what the server sent back: the starting plan, and one bit per rung requested."""
        return plan_row_bits(self.plan) + self.ladder.requests

    @property
    def stage_uplink_bits(self) -> tuple[int, ...]:
        """Return the uplink bits each refinement stage is responsible for."""
        stage_bits = [0] * self.plan.num_stages
        for plane, levels in enumerate(self.ladder.levels):
            stage_bits[plane // self.plan.bit_width] += (
                sum(levels) + self.ladder.blocks * self.hash_bits
            )
        for plane, _ in self.ladder.raw:
            stage_bits[plane // self.plan.bit_width] += self.ladder.block_length
        return tuple(stage_bits)


class LDPCASWRecord(CancerRecord):
    """Compression record carrying the rungs the ladder settled on and what they cost."""

    sw_positions: int | None = None
    sw_uplink_rate: float | None = None
    sw_downlink_rate: float | None = None
    sw_syndrome_rate: float | None = None
    sw_hash_rate: float | None = None
    sw_raw_rate: float | None = None
    sw_raw_blocks: int | None = None
    sw_apriori_rate: float | None = None
    sw_apriori_plane_rates: list[list[float]] | None = None
    sw_decoded_cross_entropy_rate: float | None = None
    sw_plane_mean_levels: list[float] | None = None
    sw_plane_max_levels: list[int] | None = None
    sw_plane_min_levels: list[int] | None = None
    sw_verification_failures: int | None = None
    sw_ladder_requests: int | None = None


class LDPCASWRoundMixin(_NonOracleSWMixin):
    """Rate-adaptive staged-index SW transport: one accumulated syndrome, climbed per block."""

    record_class = LDPCASWRecord
    payload_class = LDPCASWPayload
    ldpca_config: LDPCAConfig

    def __init__(
        self,
        *args: Any,
        sw_config: SWCoderConfig | None = None,
        ldpca_config: LDPCAConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.ldpca_config = LDPCAConfig() if ldpca_config is None else ldpca_config
        super().__init__(
            *args,
            sw_config=sw_config,
            verification_bits=self.ldpca_config.verification_bits,
            **kwargs,
        )
        assert self.sw_config.matrix_backend == LDPCA_BACKEND, (
            f"the ladder needs the {LDPCA_BACKEND} matrix backend; the coder config names "
            f"{self.sw_config.matrix_backend}."
        )

    def _start_plan(
        self, stages: int, alphabet: int, prior_provider: ConditionalPrior, side_info: torch.Tensor
    ) -> tuple[SWRatePlan, tuple[tuple[float, ...], ...]]:
        """Return the rung the climb starts at, and the model entropy it came from."""
        apriori = expected_plane_rates(
            prior_provider, side_info, stages, alphabet,
            self.ldpca_config.rate_samples, self.ldpca_config.rate_seed,
        )
        plan = self.sw_codec.initial_plan(stages, alphabet).model_copy(update={
            "syndrome_rows": plane_rows(
                apriori, self.ldpca_config.start_margin, self.sw_config.block_length)})
        return plan, apriori

    def encode_staged_indices(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, record: CancerRecord | None = None
    ) -> LDPCASWPayload:
        """Run the decoder's climb against the true symbols and return what it asked for."""
        alphabet = self._staged_alphabet(bins)
        side_info, prior_provider = self._sw_context(metadata)
        plan, apriori = self._start_plan(bins.shape[0], alphabet, prior_provider, side_info)
        ladder = self.sw_codec.accumulate_ladder(bins, plan)
        digests = self.sw_hashes.digest_planes(ladder.planes)
        result = self.sw_codec.decode_ladder(
            ladder, plan, bins.shape[1], prior_provider, side_info,
            HashVerification(self.sw_hashes, digests, plan.bit_width),
        )
        assert bool((result.symbols == bins.long().cpu()).all()), (
            "every block's hash verified yet the decode differs from the source: with "
            f"{self.sw_hashes.hash_bits} hash bits and {result.bp_attempted_blocks} rungs climbed, a "
            "wrong word collided with its hash. Raise verification_bits."
        )
        payload = LDPCASWPayload(
            plan=plan,
            ladder=ladder.payload(bins.shape[1], plan.identifier),
            hash_bits=self.sw_hashes.hash_bits,
            hashes=self.sw_hashes.pack(digests),
        )
        if record is not None:
            self._record_ladder(payload, apriori, result, record)
        return payload

    def _record_ladder(
        self,
        payload: LDPCASWPayload,
        apriori: tuple[tuple[float, ...], ...],
        result: SWLadderResult,
        record: CancerRecord,
    ) -> None:
        """Attach what the climb settled on and what it cost, all in bits per original position."""
        assert isinstance(record, LDPCASWRecord), (
            f"{type(record).__name__} cannot hold the LDPCA ladder accounting.")
        positions = payload.ladder.positions
        record.update_record_fields(
            sw_positions=positions,
            sw_uplink_rate=payload.uplink_bits / positions,
            sw_downlink_rate=payload.downlink_bits / positions,
            sw_syndrome_rate=payload.ladder.uplink_syndrome_bits / positions,
            sw_hash_rate=payload.hashed_blocks * payload.hash_bits / positions,
            sw_raw_rate=len(payload.ladder.raw) * payload.ladder.block_length / positions,
            sw_raw_blocks=len(payload.ladder.raw),
            sw_apriori_rate=sum(map(sum, apriori)),
            sw_apriori_plane_rates=[list(stage) for stage in apriori],
            sw_decoded_cross_entropy_rate=sum(map(sum, result.plane_decoded_cross_entropies)),
            sw_plane_mean_levels=result.plane_levels.float().mean(dim=1).tolist(),
            sw_plane_max_levels=result.plane_levels.max(dim=1).values.tolist(),
            sw_plane_min_levels=result.plane_levels.min(dim=1).values.tolist(),
            sw_verification_failures=result.verification_failures,
            sw_ladder_requests=payload.ladder.requests,
        )

    def decode_staged_indices(
        self, payload: LDPCASWPayload, metadata: PreprocessMetadata
    ) -> torch.Tensor:
        """Recover staged indices by climbing again over exactly the rungs the worker sent."""
        assert payload.plan.block_length == self.sw_config.block_length, (
            f"payload block length {payload.plan.block_length} does not match "
            f"decoder block length {self.sw_config.block_length}."
        )
        assert payload.plan.num_stages == self.quantizer.num_planes
        assert payload.plan.alphabet == self.quantizer.bins_per_plane
        side_info, prior_provider = self._sw_context(metadata)
        replay = SWLadderReplay(payload.ladder)
        result = self.sw_codec.decode_ladder(
            replay, payload.plan, payload.ladder.positions, prior_provider, side_info,
            HashVerification(
                self.sw_hashes,
                self.sw_hashes.unpack(payload.hashes, len(payload.ladder.levels)),
                payload.plan.bit_width,
            ),
        )
        assert replay.requests == payload.ladder.requests, (
            f"the rerun climbed {replay.requests} rungs where the worker recorded "
            f"{payload.ladder.requests}; belief propagation did not reproduce."
        )
        return result.symbols


__all__ = [
    "plan_row_bits",
    "LDPCAConfig",
    "LDPCASWPayload",
    "LDPCASWRecord",
    "LDPCASWRoundMixin",
    "PracticalSWConfig",
    "PracticalSWPayload",
    "PracticalSWRecord",
    "PracticalSWRoundMixin",
    "SWRepair",
]
