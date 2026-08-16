"""Prior-conditioned entropy coding of staged quantizer symbols with interleaved rANS.

Unlike the Slepian-Wolf coder in ``sw_coder.py``, which withholds the prior from the encoder and
buys it back with a channel code, this is the ordinary conditional-entropy regime: both ends
evaluate the *same* conditional prior, coding is lossless, and the realized rate lands within a
fraction of a percent of ``mean(-log2 p)``.  It is a usable wire format for the learned quantizer
and the reference point every distributed-coding result is measured against.

``lanes`` rANS states advance as one vector, so a stage costs ``ceil(positions / lanes)`` tensor
steps instead of one Python iteration per symbol -- the reason this repo's earlier rANS was
unusable.  The coding loops run on whatever device the conditional prior returns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import blake2b
from math import ceil
import struct
from time import perf_counter
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

# rANS invariants: the state lives in [2**RENORM_BITS, 2**32) and renormalization moves exactly
# one RENORM_BITS-wide word.  Capping precision at RENORM_BITS is what makes that "exactly one"
# true, which is in turn what lets renormalization be a masked step instead of a while loop.
RENORM_BITS = 16
STATE_LOWER = 1 << RENORM_BITS
WORD_MASK = STATE_LOWER - 1
HEADER = struct.Struct(">QIIIB8s")


class ConditionalPrior(Protocol):
    """Provider of one stage's conditional prior, evaluated identically at both ends.

    Side information is the caller's business and belongs in the closure; what the coder controls
    is ``decoded``, which it masks so the encoder sees exactly the state the decoder will see.
    """

    def __call__(self, decoded: torch.Tensor, stage: int) -> torch.Tensor:
        """Return ``q(U_stage | U_<stage)`` with shape ``(positions, alphabet)``."""


class RansConfig(BaseModel):
    """Encoder settings; the decoder reads them back from the payload header instead."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Lanes trade rate against wall time and nothing else: the flush costs 4 bytes per lane,
    # while the step count -- which is what the coding loops actually cost -- is positions/lanes.
    # The default is sized for the millions of positions a model delta has; short vectors want
    # fewer lanes, and `RansResult.report` prints the flush so the choice stays visible.
    lanes: int = Field(default=2048, ge=1, le=2**20)
    precision: int = Field(default=RENORM_BITS, ge=2, le=RENORM_BITS)


def quantize_pmf(pmf: torch.Tensor, precision: int, rows: int) -> torch.Tensor:
    """Return an exact integer CDF of shape ``(rows, alphabet + 1)`` summing to ``2 ** precision``.

    Rows past the PMF are the degenerate table that makes lane padding free: symbol zero takes the
    whole mass, so coding it leaves the state untouched and never renormalizes.  Clamping every
    frequency to at least one is what makes vanishing probabilities safe, and unlike the earlier
    implementation's fix-up it reads only the prior, so nothing data-dependent enters the table.
    """
    positions, alphabet = pmf.shape
    assert rows >= positions
    total = 1 << precision
    freq = (pmf * total).floor().long().clamp(min=1)
    freq.scatter_add_(1, freq.argmax(1, keepdim=True), total - freq.sum(1, keepdim=True))
    assert bool((freq >= 1).all()), (
        f"an alphabet of {alphabet} cannot be represented at precision {precision}.")
    cdf = torch.zeros((rows, alphabet + 1), dtype=torch.long, device=pmf.device)
    cdf[:positions, 1:] = freq.cumsum(1)
    cdf[positions:, 1:] = total
    return cdf


def stage_prior(prior: ConditionalPrior, symbols: torch.Tensor, stage: int) -> torch.Tensor:
    """Evaluate one stage's prior on the masked context that both ends share.

    Stages at or after ``stage`` are zeroed even at the encoder, which already knows them: a
    deduplicating prior groups rows by their symbols, so an encoder passing the true later stages
    would form different groups than the decoder and could return a different final bit.
    """
    context = symbols.clone()
    context[stage:] = 0
    pmf = prior(context, stage)
    assert pmf.ndim == 2 and pmf.shape[0] == symbols.shape[1], (
        f"stage {stage} prior has shape {tuple(pmf.shape)}; "
        f"expected ({symbols.shape[1]}, alphabet).")
    assert bool(torch.isfinite(pmf).all()) and bool((pmf >= 0).all()), (
        f"stage {stage} prior is not a probability tensor.")
    assert torch.allclose(pmf.sum(1), torch.ones(pmf.shape[0], device=pmf.device), atol=1e-4), (
        f"stage {stage} prior rows must sum to one.")
    return pmf


def _digest(symbols: torch.Tensor) -> bytes:
    return blake2b(symbols.long().cpu().numpy().tobytes(), digest_size=8).digest()


def encode(bins: torch.Tensor, prior: ConditionalPrior, cfg: RansConfig | None = None) -> bytes:
    """Code staged symbols against their conditional prior and return the complete payload.

    Stages are pushed last to first so that the stack pops them in the order the causal prior
    needs, which keeps the whole vector on one stream and one lane flush.
    """
    cfg = cfg or RansConfig()
    assert bins.ndim == 2 and bins.shape[1] > 0, (
        f"bins must have shape (stages, positions), got {tuple(bins.shape)}.")
    stages, positions = bins.shape
    bins = bins.long()
    steps = ceil(positions / cfg.lanes)
    rows = steps * cfg.lanes
    scale = 1 << (32 - cfg.precision)

    state: torch.Tensor | None = None
    alphabet = 0
    words: list[torch.Tensor] = []

    for stage in reversed(range(stages)):
        pmf = stage_prior(prior, bins, stage)
        device, alphabet = pmf.device, pmf.shape[1]
        if state is None:
            state = torch.full((cfg.lanes,), STATE_LOWER, dtype=torch.long, device=device)
        assert state.device == device, "the conditional prior changed device between stages."
        symbols = torch.zeros((rows, 1), dtype=torch.long, device=device)
        symbols[:positions, 0] = bins[stage].to(device)
        assert bool((symbols < alphabet).all()), f"stage {stage} symbols exceed the prior alphabet."
        # Unlike the decoder, the encoder knows every symbol before it starts, so the interval
        # lookups leave the loop entirely; the loop is bound by how many tensor calls it issues.
        cdf = quantize_pmf(pmf, cfg.precision, rows)
        del pmf
        starts = cdf.gather(1, symbols).squeeze(1)
        freqs = cdf.gather(1, symbols + 1).squeeze(1) - starts
        limits = freqs * scale
        del cdf, symbols
        # The trailing slot absorbs the writes of non-emitting lanes, so the scatter stays
        # unconditional and the step loop never synchronizes with the host.
        buffer = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        written = torch.zeros((), dtype=torch.long, device=device)

        for step in reversed(range(steps)):
            window = slice(step * cfg.lanes, (step + 1) * cfg.lanes)
            freq, start = freqs[window], starts[window]
            emit = state >= limits[window]
            buffer.scatter_(
                0,
                torch.where(emit, written + emit.long().cumsum(0) - 1, rows),
                (state & WORD_MASK).int(),
            )
            written += emit.sum()
            state = torch.where(emit, state >> RENORM_BITS, state)
            state = (
                (torch.div(state, freq, rounding_mode="floor") << cfg.precision)
                + state % freq + start
            )

        words.append(buffer[: int(written)].clone())
        del starts, freqs, limits, buffer

    assert state is not None, "bins must contain at least one stage."
    header = HEADER.pack(positions, stages, alphabet, cfg.lanes, cfg.precision, _digest(bins))
    flush = state.cpu().numpy().astype(">u4").tobytes()
    return header + flush + torch.cat(words).cpu().numpy().astype(">u2").tobytes()


def decode(payload: bytes, prior: ConditionalPrior) -> torch.Tensor:
    """Recover staged symbols, re-deriving each stage's prior from the stages already decoded."""
    assert len(payload) >= HEADER.size, (
        f"payload is {len(payload)} bytes; the header alone needs {HEADER.size}.")
    positions, stages, alphabet, lanes, precision, digest = HEADER.unpack_from(payload)
    flush_end = HEADER.size + 4 * lanes
    assert len(payload) >= flush_end, f"payload is truncated before its {lanes} lane states."
    assert (len(payload) - flush_end) % 2 == 0, "the coded stream is not a whole number of words."

    steps = ceil(positions / lanes)
    rows = steps * lanes
    decoded = torch.zeros((stages, positions), dtype=torch.long)
    state: torch.Tensor | None = None
    stream: torch.Tensor = torch.empty(0, dtype=torch.long)
    pointer: torch.Tensor = torch.zeros((), dtype=torch.long)

    for stage in range(stages):
        pmf = stage_prior(prior, decoded, stage)
        device = pmf.device
        assert pmf.shape[1] == alphabet, (
            f"stage {stage} prior has alphabet {pmf.shape[1]}; "
            f"the payload was coded with {alphabet}.")
        if state is None:
            state = torch.from_numpy(
                np.frombuffer(payload[HEADER.size : flush_end], dtype=">u4").astype(np.int64)
            ).to(device)
            stream = torch.from_numpy(
                np.frombuffer(payload[flush_end:], dtype=">u2").astype(np.int64)
            ).to(device)
            pointer = torch.tensor(stream.numel(), dtype=torch.long, device=device)
        assert state.device == device, "the conditional prior changed device between stages."
        cdf = quantize_pmf(pmf, precision, rows)
        del pmf
        symbols = torch.zeros(rows, dtype=torch.long, device=device)

        for step in range(steps):
            window = slice(step * lanes, (step + 1) * lanes)
            table = cdf[window]
            slot = state & ((1 << precision) - 1)
            symbol = torch.searchsorted(table, slot[:, None], right=True) - 1
            start = table.gather(1, symbol).squeeze(1)
            freq = table.gather(1, symbol + 1).squeeze(1) - start
            symbols[window] = symbol.squeeze(1)
            state = freq * (state >> precision) + slot - start
            refill = state < STATE_LOWER
            pointer -= refill.sum()
            # Clamped so a desynchronized prior reaches the per-stage check below with a
            # diagnosis instead of dying on a raw out-of-range index mid-loop.
            word = stream.index_select(
                0, torch.where(refill, pointer + refill.long().cumsum(0) - 1, 0).clamp(min=0))
            state = torch.where(refill, (state << RENORM_BITS) | word, state)

        assert int(pointer) >= 0, (
            f"stage {stage} read past the coded stream; the conditional prior is not "
            "producing the same table at both ends.")
        decoded[stage] = symbols[:positions].cpu()
        del cdf, symbols

    assert int(pointer) == 0, f"{int(pointer)} coded words were left unread after the final stage."
    assert _digest(decoded) == digest, (
        "decoded symbols do not match the encoder's digest; the conditional prior is not "
        "producing the same table at both ends.")
    return decoded


@dataclass(frozen=True, slots=True)
class RansResult:
    """Round-trip rate accounting for one staged vector, in bits per original value."""

    stage_prior_rates: tuple[float, ...]
    stage_table_rates: tuple[float, ...]
    positions: int
    stages: int
    alphabet: int
    lanes: int
    precision: int
    steps: int
    header_bytes: int
    flush_bytes: int
    stream_bytes: int
    encode_seconds: float
    decode_seconds: float

    @property
    def prior_rate(self) -> float:
        return sum(self.stage_prior_rates)

    @property
    def table_rate(self) -> float:
        return sum(self.stage_table_rates)

    @property
    def stream_rate(self) -> float:
        return 8 * self.stream_bytes / self.positions

    @property
    def wire_rate(self) -> float:
        return 8 * (self.header_bytes + self.flush_bytes + self.stream_bytes) / self.positions

    def report(self) -> str:
        """Render per-stage rates and the two sources of excess over the analytic prior."""
        lines = [f"{'stage':>5} {'prior H':>9} {'table H':>9} {'excess':>9}"]
        for stage, (analytic, table) in enumerate(
            zip(self.stage_prior_rates, self.stage_table_rates, strict=True), 1
        ):
            excess = f"{table / analytic - 1:>8.2%}" if analytic else "      n/a"
            lines.append(f"{stage:>5} {analytic:>9.4f} {table:>9.4f} {excess}")
        excess = f"{self.table_rate / self.prior_rate - 1:>8.2%}" if self.prior_rate else "      n/a"
        lines.append(f"{'all':>5} {self.prior_rate:>9.4f} {self.table_rate:>9.4f} {excess}")
        over = (
            f"{self.wire_rate / self.prior_rate - 1:+.2%} over the analytic prior"
            if self.prior_rate else "no analytic prior to compare against")
        lines.append(
            f"stream {self.stream_rate:.4f} | wire {self.wire_rate:.4f} bit/value ({over}): "
            f"{self.stream_bytes} B coded + {self.flush_bytes} B lane flush "
            f"+ {self.header_bytes} B header")
        lines.append(
            f"{self.positions} values x {self.stages} stages, alphabet {self.alphabet}, "
            f"{self.lanes} lanes, precision {self.precision}, {self.steps} steps/stage | "
            f"encode {self.encode_seconds:.2f}s | decode {self.decode_seconds:.2f}s")
        return "\n".join(lines)


def stage_code_lengths(
    bins: torch.Tensor, prior: ConditionalPrior, cfg: RansConfig
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return each stage's analytic and quantized-table cross-entropy, in bits per value."""
    bins = bins.long()
    total = float(1 << cfg.precision)
    analytic: list[float] = []
    table: list[float] = []
    for stage in range(bins.shape[0]):
        pmf = stage_prior(prior, bins, stage)
        index = bins[stage].to(pmf.device)[:, None]
        analytic.append(float(-torch.log2(pmf.gather(1, index).clamp(min=1e-12)).mean()))
        cdf = quantize_pmf(pmf, cfg.precision, pmf.shape[0])
        freq = cdf.gather(1, index + 1) - cdf.gather(1, index)
        table.append(float(-torch.log2(freq.double() / total).mean()))
    return tuple(analytic), tuple(table)


def evaluate_rans(
    bins: torch.Tensor, prior: ConditionalPrior, cfg: RansConfig | None = None
) -> RansResult:
    """Round-trip a staged vector and measure what the coder spends against the analytic prior."""
    cfg = cfg or RansConfig()
    start = perf_counter()
    payload = encode(bins, prior, cfg)
    encode_seconds = perf_counter() - start

    start = perf_counter()
    recovered = decode(payload, prior)
    decode_seconds = perf_counter() - start
    assert bool((recovered == bins.long().cpu()).all()), "the rANS round trip is not lossless."

    analytic, table = stage_code_lengths(bins, prior, cfg)
    flush_bytes = 4 * cfg.lanes
    return RansResult(
        stage_prior_rates=analytic,
        stage_table_rates=table,
        positions=bins.shape[1],
        stages=bins.shape[0],
        alphabet=HEADER.unpack_from(payload)[2],
        lanes=cfg.lanes,
        precision=cfg.precision,
        steps=ceil(bins.shape[1] / cfg.lanes),
        header_bytes=HEADER.size,
        flush_bytes=flush_bytes,
        stream_bytes=len(payload) - HEADER.size - flush_bytes,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
    )


def main() -> None:
    """Benchmark the coder on the learned quantizer's synthetic Gaussian Wyner-Ziv source."""
    from FL_code.FL_core.utils import set_global_seed

    from .prior_code import PriorCalculator
    from .wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1_000_000, help="coded vector length")
    parser.add_argument("--noise-power", type=float, default=0.01)
    parser.add_argument("--bins-per-plane", type=int, default=3)
    parser.add_argument("--num-planes", type=int, default=3)
    parser.add_argument("--reconst-ld", type=float, default=120.0)
    parser.add_argument("--lanes", type=int, default=RansConfig().lanes)
    parser.add_argument("--precision", type=int, default=RansConfig().precision)
    parser.add_argument("--device", default=None, help="coder device; default is the prior's")
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
    prior_model = PriorCalculator.make_trained_prior_model(
        bins.long(), soft_codes, side_info,
        quantizer_config.model_copy(update={"marginal_loss": False}))
    coder_device = torch.device(args.device) if args.device else side_info.device

    def prior(decoded: torch.Tensor, stage: int) -> torch.Tensor:
        return PriorCalculator.compute_prior_from_network(
            prior_model, decoded, side_info)[stage].to(coder_device)

    result = evaluate_rans(bins, prior, RansConfig(lanes=args.lanes, precision=args.precision))
    marginal = sum(PriorCalculator.compute_plane_rates(
        PriorCalculator.compute_marginal_prior(bins, args.bins_per_plane, args.num_planes),
        bins, args.num_planes))
    print(f"\n{args.bins_per_plane} bins x {args.num_planes} stages, "
          f"{args.size} floats on {coder_device}\n")
    print(result.report())
    print(f"encoder-side marginal rate {marginal:.4f} "
          f"({1 - result.wire_rate / marginal:.1%} saved by conditional entropy coding)")


if __name__ == "__main__":
    main()
