"""Paper-conformance benchmarks for the analytical path in ``sw_coder``.

Run the fast rate and structural checks with::

    python -m pytest -q -s Experiments/slepian_wolf/paper_replication_benchmark.py

Set ``SW_PAPER_RUN_LDPC=1`` to additionally run the slower, finite-length LDPC
reliability checks. These treat zero backoff as an asymptotic-bound diagnostic,
not as a paper BER claim. ``SW_PAPER_POSITIONS`` can reduce the default one-million
source positions for smoke runs.

The reference below implements the paper's IID Gaussian-CEO sign model directly.
It does not import another repository implementation or any neural prior code.
Paper rates count syndrome bits only; wire headers and rate-plan serialization are
reported separately by ``sw_coder`` and are not part of those comparisons.
The multi-bit points of Figure 10 and Table 1 are deliberately excluded because
the paper does not publish the uniform quantizer range, thresholds, saturation
rule, or index mapping needed to reproduce them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, comb, log2, pi, sqrt
import os
from statistics import NormalDist
from typing import Final

import numpy as np
import pytest
from scipy.special import ndtr, roots_hermitenorm, xlogy
import torch

from FL_code.cancer_protocol.sw_coder import (
    LIVERIS_2002_RATE_HALF_NODE_DISTRIBUTION,
    SWCoderConfig,
    SWRatePlan,
    SWResult,
    SlepianWolfCodec,
    _full_row_rank,
    _bp_decode,
    _lsb_plane_probabilities,
    _plane_llr,
    evaluate_slepian_wolf,
)


PAPER_BLOCK_LENGTH: Final = 2_400
PAPER_BP_ITERATIONS: Final = 100
RUN_LDPC: Final = os.environ.get("SW_PAPER_RUN_LDPC") == "1"
LDPC_POSITIONS: Final = ceil(
    int(os.environ.get("SW_PAPER_POSITIONS", "1000000")) / PAPER_BLOCK_LENGTH
) * PAPER_BLOCK_LENGTH


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Return binary entropy in bits while preserving exact zero-probability limits."""
    return -(xlogy(probability, probability) + xlogy(1 - probability, 1 - probability)) / np.log(2)


@dataclass(frozen=True, slots=True)
class GaussianCEOSignReference:
    """Exact sign-quantized Gaussian-CEO reference used by the paper's server."""

    signal_variance: float
    noise_variance: float
    quadrature_order: int = 1_024

    def __post_init__(self) -> None:
        assert self.signal_variance > 0
        assert self.noise_variance > 0
        assert self.quadrature_order >= 16

    def context_statistics(self, previous_workers: int) -> tuple[np.ndarray, np.ndarray]:
        """Return each sign-count context probability and the next positive-sign posterior."""
        assert previous_workers >= 0
        nodes, weights = roots_hermitenorm(self.quadrature_order)
        positive_given_x = ndtr(
            sqrt(self.signal_variance / self.noise_variance) * nodes
        )
        context_probability = np.empty(previous_workers + 1)
        positive_probability = np.empty(previous_workers + 1)
        for positive_count in range(previous_workers + 1):
            context_integrand = positive_given_x**positive_count * (
                1 - positive_given_x
            ) ** (previous_workers - positive_count)
            context_probability[positive_count] = (
                weights @ context_integrand / sqrt(2 * pi)
            )
            positive_probability[positive_count] = (
                weights @ (context_integrand * positive_given_x)
                / sqrt(2 * pi)
                / context_probability[positive_count]
            )
        return context_probability, positive_probability

    def worker_rate(self, worker: int) -> float:
        """Return ``H(U_k | U_1, ..., U_{k-1})`` for one-indexed worker ``k``."""
        assert worker >= 1
        previous_workers = worker - 1
        context_probability, positive_probability = self.context_statistics(previous_workers)
        multiplicity = np.asarray(
            [comb(previous_workers, count) for count in range(previous_workers + 1)]
        )
        return float(
            multiplicity @ (context_probability * _binary_entropy(positive_probability))
        )

    def semi_analytical_rate(self, worker: int, samples: int, seed: int) -> float:
        """Estimate Eq. (45) by the paper's semi-analytical Monte Carlo construction."""
        assert worker >= 1
        assert samples >= 1
        previous_workers = worker - 1
        rng = np.random.default_rng(seed)
        global_gradient = rng.normal(0, sqrt(self.signal_variance), samples)
        positive_given_x = ndtr(global_gradient / sqrt(self.noise_variance))
        context_probability = np.empty(previous_workers + 1)
        positive_probability = np.empty(previous_workers + 1)
        for positive_count in range(previous_workers + 1):
            context_weight = positive_given_x**positive_count * (
                1 - positive_given_x
            ) ** (previous_workers - positive_count)
            context_probability[positive_count] = context_weight.mean()
            positive_probability[positive_count] = (
                (context_weight * positive_given_x).mean()
                / context_probability[positive_count]
            )
        multiplicity = np.asarray(
            [comb(previous_workers, count) for count in range(previous_workers + 1)]
        )
        return float(
            multiplicity @ (context_probability * _binary_entropy(positive_probability))
        )

    def traditional_monte_carlo_rate(self, worker: int, samples: int, seed: int) -> float:
        """Estimate the rate from sampled discrete contexts as in the paper's Eq. (42)."""
        assert worker >= 1
        assert samples >= 1
        rng = np.random.default_rng(seed)
        global_gradient = rng.normal(0, sqrt(self.signal_variance), samples)
        observations = global_gradient + rng.normal(
            0, sqrt(self.noise_variance), (worker, samples)
        )
        signs = observations >= 0
        if worker == 1:
            return float(_binary_entropy(np.asarray([signs[0].mean()]))[0])

        context = np.packbits(
            signs[:-1].T,
            axis=1,
            bitorder="little",
        )
        # K=10 in Fig. 7 needs two packed bytes, so rows are viewed as one opaque key.
        _, inverse = np.unique(context, axis=0, return_inverse=True)
        rate = 0.0
        for label in range(int(inverse.max()) + 1):
            selected = inverse == label
            probability = np.asarray([signs[-1, selected].mean()])
            rate += selected.mean() * float(_binary_entropy(probability)[0])
        return rate

    def prior(self, previous_signs: torch.Tensor) -> torch.Tensor:
        """Return the exact decoder prior using only already decoded worker signs."""
        assert previous_signs.ndim == 2
        _, positive_probability = self.context_statistics(previous_signs.shape[1])
        lookup = torch.as_tensor(
            positive_probability,
            dtype=torch.float32,
            device=previous_signs.device,
        )
        positive = lookup[previous_signs.sum(dim=1).long()]
        return torch.stack((1 - positive, positive), dim=1)


@dataclass(frozen=True, slots=True)
class LDPCObservation:
    """One finite-length result produced by the benchmark and compared with the paper."""

    source_seed: int
    matrix_seed: int
    matrix_backend: str
    degree_profile: str
    worker: int
    margin: float
    entropy_rate: float
    syndrome_rate: float
    wire_rate: float
    prior_only_ber: float
    bit_error_rate: float
    bit_error_upper_95: float
    block_error_rate: float
    nonconverged_blocks: int
    undetected_block_errors: int
    mean_bp_iterations: float
    seconds: float


def _wilson_upper(errors: int, trials: int, confidence: float = 0.95) -> float:
    """Return the one-sided Wilson upper confidence bound for a Bernoulli rate."""
    assert 0 <= errors <= trials
    assert trials > 0
    z = NormalDist().inv_cdf(confidence)
    estimate = errors / trials
    denominator = 1 + z**2 / trials
    center = estimate + z**2 / (2 * trials)
    radius = z * sqrt(estimate * (1 - estimate) / trials + z**2 / (4 * trials**2))
    return (center + radius) / denominator


def _sample_signs(
    reference: GaussianCEOSignReference,
    workers: int,
    positions: int,
    seed: int,
) -> torch.Tensor:
    """Draw sign-quantized worker observations from the paper's Gaussian-CEO model."""
    assert workers >= 1
    assert positions >= 1
    generator = torch.Generator().manual_seed(seed)
    global_gradient = torch.randn(positions, generator=generator) * sqrt(
        reference.signal_variance
    )
    observations = global_gradient + torch.randn(
        (workers, positions), generator=generator
    ) * sqrt(reference.noise_variance)
    return (observations >= 0).long()


def _binary_plan(rate: float, cfg: SWCoderConfig, code_seed: int) -> SWRatePlan:
    """Create the one-plane syndrome plan prescribed by Eq. (36)."""
    rows = min(cfg.block_length, ceil(cfg.block_length * rate))
    return SWRatePlan(
        block_length=cfg.block_length,
        alphabet=2,
        syndrome_rows=((rows,),),
        code_seed=code_seed,
        degree_distribution=cfg.degree_distribution,
        girth_candidates=cfg.girth_candidates,
        matrix_attempts=cfg.matrix_attempts,
        matrix_backend=cfg.matrix_backend,
    )


def _prior_provider(
    reference: GaussianCEOSignReference,
) -> Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]:
    """Bind a paper prior to the codec's decoder-owned provider interface."""

    def provide(
        decoded_stages: torch.Tensor,
        side_info: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        assert stage == 0
        assert decoded_stages.shape[0] == 1
        return reference.prior(side_info)

    return provide


def _evaluate_oracle_worker(
    reference: GaussianCEOSignReference,
    signs: torch.Tensor,
    worker: int,
    margin: float,
    seed: int,
) -> SWResult:
    """Decode one worker while supplying the true preceding worker signs as side information."""
    assert 2 <= worker <= signs.shape[0]
    entropy_rate = reference.worker_rate(worker)
    cfg = SWCoderConfig(
        block_length=PAPER_BLOCK_LENGTH,
        max_bp_iterations=PAPER_BP_ITERATIONS,
        rate_margin=margin,
        code_seed=seed,
    )
    plan = _binary_plan(entropy_rate + margin, cfg, seed)
    return evaluate_slepian_wolf(
        signs[worker - 1 : worker],
        plan,
        _prior_provider(reference),
        signs[: worker - 1].T,
        cfg,
    )


def test_multilevel_lsb_chain_rule() -> None:
    """Match the LSB ordering and entropy chain in the paper's Eq. (33)--(35)."""
    priors = torch.tensor(
        [
            [0.02, 0.08, 0.15, 0.05, 0.25, 0.10, 0.20, 0.15],
            [0.20, 0.05, 0.05, 0.20, 0.10, 0.25, 0.10, 0.05],
        ],
        dtype=torch.float64,
    )
    categorical_entropy = -torch.xlogy(priors, priors).sum(dim=1) / np.log(2)
    plane_entropy = torch.zeros(priors.shape[0], dtype=priors.dtype)

    for bit in range(3):
        for prefix_value in range(1 << bit):
            prefix = torch.full((priors.shape[0],), prefix_value, dtype=torch.long)
            zero, one = _lsb_plane_probabilities(priors, prefix, bit)
            symbols = torch.arange(priors.shape[1])
            prefix_probability = priors[:, (symbols % (1 << bit)) == prefix_value].sum(
                dim=1
            ) if bit else torch.ones(priors.shape[0], dtype=priors.dtype)
            plane_entropy += prefix_probability * (
                -(torch.xlogy(zero, zero) + torch.xlogy(one, one)) / np.log(2)
            )

    torch.testing.assert_close(plane_entropy, categorical_entropy, atol=1e-12, rtol=1e-12)


def test_syndrome_math_rank_packing_and_padding() -> None:
    """Verify that the packed payload is exactly the claimed full-rank syndrome stream."""
    cfg = SWCoderConfig(block_length=64, code_seed=17)
    plan = _binary_plan(23 / cfg.block_length, cfg, cfg.code_seed)
    codec = SlepianWolfCodec(cfg)
    bins = (torch.arange(77) % 3 == 0).long()[None, :]
    payload = codec.encode_syndromes(bins, plan)
    edge_variable, edge_check = codec._matrix(plan, 0, 0, 23)
    assert _full_row_rank(edge_variable, edge_check, 23, cfg.block_length)

    blocks = 2
    plane = torch.nn.functional.pad(bins.float(), (0, 51)).view(blocks, cfg.block_length)
    expected = (
        torch.zeros(blocks, 23)
        .index_add_(
            1,
            torch.as_tensor(edge_check),
            plane[:, torch.as_tensor(edge_variable)],
        )
        .remainder(2)
        .to(torch.uint8)
        .numpy()
        .reshape(-1)
    )
    unpacked = np.unpackbits(
        np.frombuffer(payload.syndromes, dtype=np.uint8), bitorder="little"
    )
    np.testing.assert_array_equal(unpacked[: expected.size], expected)
    assert not unpacked[expected.size :].any()
    assert len(payload.syndromes) == ceil(expected.size / 8)

    fresh_edges = SlepianWolfCodec(cfg)._matrix(plan, 0, 0, 23)
    np.testing.assert_array_equal(fresh_edges[0], edge_variable)
    np.testing.assert_array_equal(fresh_edges[1], edge_check)
    assert plan.matrix_backend == "balanced_4cycle_v1"


def test_liveris_edge_to_node_degree_profile() -> None:
    """Verify the operational profile is the node view of Liveris et al. Eq. (5)."""
    expected_edge = np.asarray((0.234029, 0.212425, 0.146898, 0.102840, 0.303808))
    degrees = np.asarray([degree for degree, _ in LIVERIS_2002_RATE_HALF_NODE_DISTRIBUTION])
    node_fractions = np.asarray(
        [fraction for _, fraction in LIVERIS_2002_RATE_HALF_NODE_DISTRIBUTION]
    )
    edge_fractions = degrees * node_fractions / (degrees @ node_fractions)
    np.testing.assert_allclose(edge_fractions, expected_edge, atol=1e-7)

    check_edge = np.asarray((0.71875, 0.28125))
    check_degrees = np.asarray((8, 9))
    average_check_degree = 1 / np.sum(check_edge / check_degrees)
    assert 1 - (degrees @ node_fractions) / average_check_degree == pytest.approx(0.5)


def test_bp_matches_exact_marginals_on_a_tree() -> None:
    """Validate the syndrome sign and LLR convention where sum-product is exact."""
    edge_variable = torch.tensor([0, 1, 1, 2, 2, 3])
    edge_check = torch.tensor([0, 0, 1, 1, 2, 2])
    source = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    syndrome = torch.zeros(1, 3).index_add_(
        1, edge_check, source[:, edge_variable]
    ) % 2
    probability_one = torch.tensor([0.8, 0.3, 0.7, 0.9])
    llr = ((1 - probability_one).log() - probability_one.log())[None, :]
    decoded, succeeded, _ = _bp_decode(
        llr,
        syndrome,
        edge_variable,
        edge_check,
        SWCoderConfig(block_length=32, max_bp_iterations=20, damping=1.0),
    )

    candidates = torch.tensor(
        [[(value >> bit) & 1 for bit in range(4)] for value in range(16)],
        dtype=torch.float32,
    )
    candidate_syndromes = torch.zeros(16, 3).index_add_(
        1, edge_check, candidates[:, edge_variable]
    ) % 2
    valid = (candidate_syndromes == syndrome).all(dim=1)
    weights = torch.where(
        candidates.bool(), probability_one, 1 - probability_one
    ).prod(dim=1)[valid]
    exact = ((weights[:, None] * candidates[valid]).sum(dim=0) / weights.sum() > 0.5).float()
    torch.testing.assert_close(decoded[0], exact)
    assert bool(succeeded[0])


def test_full_rate_and_deterministic_zero_rate_are_exact() -> None:
    """Exercise the two boundary codes without allowing a hidden recovery mechanism."""
    cfg = SWCoderConfig(block_length=64, code_seed=23)
    bins = (torch.arange(79) % 5 < 2).long()[None, :]
    uniform_side_info = torch.zeros((bins.shape[1], 0), dtype=torch.long)

    def uniform_prior(
        decoded_stages: torch.Tensor,
        side_info: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        del decoded_stages, stage
        return torch.full((side_info.shape[0], 2), 0.5)

    full_plan = _binary_plan(1.0, cfg, cfg.code_seed)
    full = evaluate_slepian_wolf(bins, full_plan, uniform_prior, uniform_side_info, cfg)
    assert full.bit_errors == 0
    assert full.syndrome_bits == 2 * cfg.block_length
    assert full.model_entropy_rate == pytest.approx(1.0)
    assert full.cross_entropy_rate == pytest.approx(1.0)
    assert full.syndrome_rate == 2 * cfg.block_length / bins.shape[1]

    deterministic_side_info = bins[0, :, None]

    def deterministic_prior(
        decoded_stages: torch.Tensor,
        side_info: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        del decoded_stages, stage
        return torch.nn.functional.one_hot(side_info[:, 0], 2).float()

    zero_plan = _binary_plan(0.0, cfg, cfg.code_seed)
    zero = evaluate_slepian_wolf(
        bins,
        zero_plan,
        deterministic_prior,
        deterministic_side_info,
        cfg,
    )
    assert zero.bit_errors == 0
    assert zero.syndrome_bits == 0
    assert zero.wire_bits > zero.syndrome_bits
    assert zero.bp_attempted_blocks == 0


def test_paper_llr_matches_symmetric_sign_posterior() -> None:
    """Match Eq. (47) after converting the paper's bit-one LLR to BP's bit-zero sign."""
    reference = GaussianCEOSignReference(10.0, 1.0)
    previous_signs = torch.tensor([[0], [1]], dtype=torch.long)
    prior = reference.prior(previous_signs)
    llr, _ = _plane_llr(prior, torch.zeros(2, dtype=torch.long), 0, clip=100.0)
    expected = prior[:, 0].log() - prior[:, 1].log()
    torch.testing.assert_close(llr, expected)


def test_nonuniform_prior_is_an_intentional_generalization() -> None:
    """Record that the generic codec uses posterior source LLRs, not paper channel-only LLRs."""
    prior = torch.tensor([[0.8, 0.2]])
    llr, _ = _plane_llr(prior, torch.zeros(1, dtype=torch.long), 0, clip=100.0)
    assert float(llr[0]) == pytest.approx(log2(4) * np.log(2))
    assert float(llr[0]) != pytest.approx(0.0)  # Eq. (47) is zero when there is no channel output.


def test_figure_7_semi_analytical_rate_anchor() -> None:
    """Reproduce Fig. 7's approximately 0.363-bit worker-10 rate and low-sample bias."""
    reference = GaussianCEOSignReference(10.0, 1.0)
    exact = reference.worker_rate(10)
    semi_analytical = np.mean(
        [reference.semi_analytical_rate(10, 100, seed) for seed in range(100)]
    )
    traditional = np.mean(
        [reference.traditional_monte_carlo_rate(10, 10, seed) for seed in range(100)]
    )
    print(
        f"Fig. 7 worker 10: exact={exact:.4f}, "
        f"semi-analytical(L=100)={semi_analytical:.4f}, "
        f"traditional(L=10)={traditional:.4f}"
    )
    assert exact == pytest.approx(0.3630, abs=5e-4)
    assert semi_analytical == pytest.approx(exact, abs=0.02)
    assert traditional < exact - 0.15


def test_figure_9_rates_decrease_with_correlation_and_worker_index() -> None:
    """Reproduce the ordering and correlation trends of the five curves in Fig. 9."""
    ratios = (0.01, 0.1, 1.0, 10.0, 100.0)
    rates = np.asarray(
        [
            [GaussianCEOSignReference(ratio, 1.0).worker_rate(worker) for worker in range(1, 6)]
            for ratio in ratios
        ]
    )
    print("Fig. 9 rows (SNR -> workers 1..5):")
    for ratio, row in zip(ratios, rates, strict=True):
        print(f"{ratio:>6g}: " + " ".join(f"{rate:.4f}" for rate in row))
    assert np.all(np.diff(rates, axis=1) <= 1e-12)
    assert np.all(np.diff(rates[:, 1:], axis=0) <= 1e-12)
    np.testing.assert_allclose(
        rates[3],
        np.asarray([1.0, 0.5757, 0.4816, 0.4381, 0.4127]),
        atol=5e-4,
    )


def test_figure_10_one_bit_saving_anchor() -> None:
    """Reproduce the paper's approximately 72% one-bit saving at variance ratio 100."""
    sw_rate = GaussianCEOSignReference(100.0, 1.0).worker_rate(2)
    relative_saving = 1 - sw_rate
    print(
        f"Fig. 10 one-bit point: SW={sw_rate:.4f} bit/symbol, "
        f"saving={relative_saving:.1%} (paper: 72%)"
    )
    assert sw_rate == pytest.approx(0.2640, abs=5e-4)
    assert relative_saving == pytest.approx(0.72, abs=0.02)


def test_next_plan_matches_independent_paper_rate() -> None:
    """Calibrate on one round and verify the plan against an independent same-law round."""
    reference = GaussianCEOSignReference(10.0, 1.0)
    cfg = SWCoderConfig(
        block_length=PAPER_BLOCK_LENGTH,
        max_bp_iterations=PAPER_BP_ITERATIONS,
        code_seed=29,
    )
    calibration = _sample_signs(reference, workers=5, positions=48_000, seed=29)
    codec = SlepianWolfCodec(cfg)
    bootstrap = codec.initial_plan(num_stages=1, alphabet=2)
    decoded = codec.decode_syndromes(
        codec.encode_syndromes(calibration[4:5], bootstrap),
        bootstrap,
        _prior_provider(reference),
        calibration[:4].T,
    )
    planned_rate = decoded.next_plan.syndrome_rows[0][0] / cfg.block_length
    exact_rate = reference.worker_rate(5)
    assert planned_rate == pytest.approx(exact_rate, abs=0.01 + 1 / cfg.block_length)

    evaluation = _sample_signs(reference, workers=5, positions=48_000, seed=30)
    payload = codec.encode_syndromes(evaluation[4:5], decoded.next_plan)
    paper_rate = codec._expected_syndrome_bits(decoded.next_plan, 48_000) / 48_000
    assert paper_rate == planned_rate
    assert payload.wire_bits / 48_000 > paper_rate


@pytest.fixture(scope="module")
def ldpc_observations() -> tuple[LDPCObservation, ...]:
    """Measure the independent local LDPC at and above the paper's entropy rate."""
    reference = GaussianCEOSignReference(10.0, 1.0)
    observations: list[LDPCObservation] = []
    for source_seed, matrix_seed in ((5, 101), (11, 211)):
        signs = _sample_signs(
            reference, workers=5, positions=LDPC_POSITIONS, seed=source_seed
        )
        for margin in (0.0, 0.10, 0.15, 0.25):
            for worker in range(2, 6):
                result = _evaluate_oracle_worker(
                    reference, signs, worker, margin, matrix_seed + worker
                )
                observations.append(
                    LDPCObservation(
                        source_seed=source_seed,
                        matrix_seed=matrix_seed + worker,
                        matrix_backend="balanced_4cycle_v1",
                        degree_profile="liveris_2002_rate_half_node",
                        worker=worker,
                        margin=margin,
                        entropy_rate=reference.worker_rate(worker),
                        syndrome_rate=result.syndrome_rate,
                        wire_rate=result.wire_rate,
                        prior_only_ber=float(
                            (
                                reference.prior(signs[: worker - 1].T).argmax(dim=1)
                                != signs[worker - 1]
                            ).float().mean()
                        ),
                        bit_error_rate=result.bit_error_rate,
                        bit_error_upper_95=_wilson_upper(
                            result.bit_errors, result.source_bits
                        ),
                        block_error_rate=result.block_error_rate,
                        nonconverged_blocks=result.nonconverged_blocks,
                        undetected_block_errors=result.undetected_block_errors,
                        mean_bp_iterations=result.mean_bp_iterations,
                        seconds=result.seconds,
                    )
                )
    print(
        "paper-omitted/operational fields: source seed, matrix seed/backend/profile, and backoff\n"
        "source matrix backend profile worker margin entropy syndrome wire priorBER BER upper95 "
        "BLER nonconv undetected mean_iter seconds"
    )
    for result in observations:
        print(
            f"{result.source_seed:>6} {result.matrix_seed:>6} "
            f"{result.matrix_backend:>20} {result.degree_profile:>31} {result.worker:>6} "
            f"{result.margin:>6.2f} "
            f"{result.entropy_rate:>7.4f} {result.syndrome_rate:>8.4f} "
            f"{result.wire_rate:>7.4f} {result.prior_only_ber:>8.3%} "
            f"{result.bit_error_rate:>7.3%} {result.bit_error_upper_95:>7.3%} "
            f"{result.block_error_rate:>7.1%} {result.nonconverged_blocks:>12} "
            f"{result.undetected_block_errors:>10} "
            f"{result.mean_bp_iterations:>9.1f} {result.seconds:>7.2f}"
        )
    return tuple(observations)


@pytest.mark.skipif(not RUN_LDPC, reason="set SW_PAPER_RUN_LDPC=1 to run LDPC benchmarks")
def test_rate_backoff_can_reach_the_paper_ber_target(
    ldpc_observations: tuple[LDPCObservation, ...],
) -> None:
    """Report each worker's smallest held-out backoff with a sub-1% upper BER bound."""
    reliable: dict[int, float] = {}
    for worker in range(2, 6):
        worker_results = [
            result for result in ldpc_observations if result.worker == worker
        ]
        reliable[worker] = min(
            margin
            for margin in sorted({result.margin for result in worker_results})
            if max(
                result.bit_error_upper_95
                for result in worker_results
                if result.margin == margin
            )
            < 0.01
        )
    print(
        "smallest held-out reliable backoff: "
        + ", ".join(
            f"worker {worker}=+{margin:.2f}" for worker, margin in reliable.items()
        )
    )
    assert max(reliable.values()) <= 0.25


@pytest.mark.skipif(not RUN_LDPC, reason="set SW_PAPER_RUN_LDPC=1 to run LDPC benchmarks")
def test_zero_backoff_is_an_asymptotic_bound_diagnostic(
    ldpc_observations: tuple[LDPCObservation, ...],
) -> None:
    """Record zero-backoff behavior without attributing this policy to the paper."""
    entropy_rate = [result for result in ldpc_observations if result.margin == 0.0]
    assert entropy_rate
    assert all(
        result.syndrome_rate >= result.entropy_rate for result in entropy_rate
    )
    backed_off = [result for result in ldpc_observations if result.margin == 0.25]
    assert np.mean([result.bit_error_rate for result in entropy_rate]) > np.mean(
        [result.bit_error_rate for result in backed_off]
    )


@pytest.mark.skipif(not RUN_LDPC, reason="set SW_PAPER_RUN_LDPC=1 to run LDPC benchmarks")
def test_successive_worker_decoding_exposes_error_propagation() -> None:
    """Decode workers in paper order using decoded, never oracle, preceding workers as side info."""
    reference = GaussianCEOSignReference(10.0, 1.0)
    signs = _sample_signs(reference, workers=5, positions=LDPC_POSITIONS, seed=41)
    aggregate_rates: list[float] = []
    aggregate_errors: list[float] = []
    for margin in (0.0, 0.25):
        decoded_workers = [signs[0]]
        communication_bits = signs.shape[1]  # Paper worker 1 is sent at one bit/symbol.
        bit_errors = 0
        for worker in range(2, 6):
            cfg = SWCoderConfig(
                block_length=PAPER_BLOCK_LENGTH,
                max_bp_iterations=PAPER_BP_ITERATIONS,
                rate_margin=margin,
                code_seed=410 + worker,
            )
            plan = _binary_plan(reference.worker_rate(worker) + margin, cfg, cfg.code_seed)
            codec = SlepianWolfCodec(cfg)
            side_info = torch.stack(decoded_workers, dim=1)
            payload = codec.encode_syndromes(signs[worker - 1 : worker], plan)
            decoded = codec.decode_syndromes(
                payload,
                plan,
                _prior_provider(reference),
                side_info,
            ).symbols[0]
            decoded_workers.append(decoded)
            communication_bits += codec._expected_syndrome_bits(plan, signs.shape[1])
            bit_errors += int((decoded != signs[worker - 1]).sum())
        aggregate_rates.append(communication_bits / signs.shape[1])
        aggregate_errors.append(bit_errors / ((signs.shape[0] - 1) * signs.shape[1]))

    print(
        "successive workers: "
        f"margin=0 rate={aggregate_rates[0]:.4f} BER={aggregate_errors[0]:.3%}; "
        f"margin=.25 rate={aggregate_rates[1]:.4f} BER={aggregate_errors[1]:.3%}"
    )
    assert aggregate_errors[1] < 0.01
    assert aggregate_errors[0] > aggregate_errors[1]
