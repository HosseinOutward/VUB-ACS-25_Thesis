"""Compare the O and T rounds coordinate by coordinate, using the per-element dumps.

O sees everything T sees and more: T conditions on the five reconstructions of its own client,
which are a subset of the server ledger O conditions on. O should therefore never lose. This
script measures where it does, and whether the losses look like training noise or like structure.

Inputs are the ``elements/`` directories two replays wrote (see ``dump_element_metrics``); nothing
is recomputed, so a later inspection only pays the file reads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.models import initialize_model  # noqa: E402
from FL_code.FL_core.utils import StateDictManager  # noqa: E402
from FL_code.cancer_protocol.wz_quantizer import normalization_params  # noqa: E402
from FL_code.run_fl import FLConfig  # noqa: E402


DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_FL_CONFIG = EXPERIMENT_DIR / "replay_fl_config.json"
PLOTS_DIR = REPORT_DIR / "results" / "analysis"
DECILES = 10


def wmape_to_db(wmape_percent: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert WMAPE percentages to dB relative to 100 percent error."""
    assert (wmape_percent > 0).all(), "WMAPE must be positive before converting to dB."
    return 20.0 * np.log10(wmape_percent / 100.0)


@dataclass(frozen=True)
class RoundElements:
    """One client-round's per-element rate and distortion as a replay dumped them.

    ``cumulative_bits`` is the conditional code length paid up to and including each stage, and
    ``abs_error`` is that stage's reconstruction error in units of the vector's mean magnitude,
    which is the denominator WMAPE uses. Both are (stages, coordinates).
    """

    round_id: int
    client_id: int
    cumulative_bits: NDArray[np.float32]
    abs_error: NDArray[np.float32]

    @property
    def stages(self) -> int:
        return self.cumulative_bits.shape[0]


def load_round(elements_dir: Path, round_id: int, client_id: int) -> RoundElements:
    """Read one client-round's dumped arrays, in float32 for accumulation."""
    path = elements_dir / f"round_{round_id}_client_{client_id}.npz"
    assert path.is_file(), (
        f"Missing element dump: {path}. Only rounds the replay dumped arrays for can be compared.")
    with np.load(path) as dump:
        assert {"bits", "errors"} <= set(dump.files), (
            f"{path} holds {dump.files}; it was written without the per-element arrays.")
        return RoundElements(
            round_id,
            client_id,
            np.cumsum(dump["bits"].astype(np.float32), axis=0),
            np.abs(dump["errors"].astype(np.float32)),
        )


def model_layout(fl_config: Path) -> tuple[NDArray[np.int32], list[str], list[slice]]:
    """Label every coordinate with its model parameter, and return the codec's slice boundaries."""
    cfg = FLConfig.model_validate(
        json.loads(fl_config.read_text())
        | {"protocol": "wz_cancer", "run_name": "elements", "records_dir": "/tmp", "seed": 43}
    )
    manager = StateDictManager(initialize_model(cfg, torch.device("cpu")))
    return (
        np.repeat(np.arange(len(manager.keys), dtype=np.int32), manager.numels),
        manager.keys,
        list(manager.get_slices()),
    )


def slice_normalized(
    delta: torch.Tensor, previous: torch.Tensor, slices: list[slice]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Normalize both vectors per parameter slice exactly as the quantizer's preprocessing does.

    Raw magnitudes are not comparable across parameters — the codec sees each slice centred and
    scaled — so deciles of the raw vector would only be re-sorting the layers. The previous round
    is scaled by this round's factors, which is how side information is normalized too.
    """
    normalized_delta, normalized_previous = delta.clone(), previous.clone()
    for parameter_slice in slices:
        scale, center = normalization_params(normalized_delta[parameter_slice])
        normalized_delta[parameter_slice] = (normalized_delta[parameter_slice] - center) / scale
        normalized_previous[parameter_slice] = (
            normalized_previous[parameter_slice] - center) / scale
    return normalized_delta.numpy(), normalized_previous.numpy()


def decile_labels(values: NDArray[np.float32]) -> NDArray[np.int32]:
    """Bucket coordinates into equal-count deciles of the given magnitude."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(values.size, dtype=np.int32)
    labels[order] = np.arange(values.size, dtype=np.int64) * DECILES // values.size
    return labels


def stratifications(
    delta: NDArray[np.float32],
    previous_delta: NDArray[np.float32],
    parameter_of: NDArray[np.int32],
    parameter_names: list[str],
) -> dict[str, tuple[NDArray[np.int32], list[str]]]:
    """Build every coordinate grouping the comparison is broken down by.

    Both deltas arrive already slice-normalized. Magnitude splits the vector by how much signal a
    coordinate carries; novelty splits it by how far the coordinate moved since the client's
    previous round, which is the axis along which the client's own history — all the side
    information T gets — is informative.
    """
    return {
        "parameter": (parameter_of, parameter_names),
        "magnitude": (
            decile_labels(np.abs(delta)),
            [f"|x| d{index}" for index in range(DECILES)],
        ),
        "novelty": (
            decile_labels(np.abs(delta - previous_delta)),
            [f"|x - prev| d{index}" for index in range(DECILES)],
        ),
    }


def stratum_means(
    per_stage: NDArray[np.float32], labels: NDArray[np.int32], counts: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Average a per-coordinate, per-stage quantity within each stratum."""
    sums = np.stack([
        np.bincount(labels, weights=stage, minlength=counts.size) for stage in per_stage
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (sums / counts).T


def average_slope(rates: NDArray[np.float64], errors: NDArray[np.float64]) -> float:
    """Mean stage-to-stage fall in absolute error per extra bit, over both sides' curves."""
    steps = np.diff(rates)
    assert (steps > 0).all(), "Cumulative rate must increase from stage to stage."
    slope = float(np.mean(-np.diff(errors) / steps))
    assert slope > 0, "Later stages must reduce error; the curve is not a rate-distortion curve."
    return slope


def element_gap_bits(
    oracle: RoundElements, temporal: RoundElements, slope: float
) -> NDArray[np.float32]:
    """Per-coordinate rate-equivalent gap of T against O, in bits, one row per stage.

    A coordinate's two sides sit at different points in the rate-distortion plane, so their gap is
    only meaningful once distortion is converted to the rate that buys it. The conversion uses the
    average stage-to-stage slope of the aggregate curves, which is the exchange rate the codec
    itself operates at: it optimizes one Lagrangian for the whole vector, so every coordinate is
    priced at the same margin. Positive means T spends more than O for the same reconstruction
    quality.

    Distortion enters linearly rather than in dB, which makes the metric additive: the mean of
    these coordinate gaps is exactly the gap read off the two whole-vector curves, so the
    histogram and the aggregate number can never tell different stories.
    """
    return (temporal.cumulative_bits - oracle.cumulative_bits) + (
        temporal.abs_error - oracle.abs_error) / slope


GAP_EDGES = np.linspace(-2.0, 2.0, 161)


def gap_histogram(gaps: NDArray[np.float32], round_id: int, client_id: int) -> pd.DataFrame:
    """Bin the per-coordinate gaps so their shape survives without keeping every coordinate.

    ``coordinates`` is the full vector length, not the binned total: the edges clip the tails, so
    normalizing by the counts alone would inflate every share.
    """
    return pd.DataFrame({
        "round_id": round_id,
        "client_id": client_id,
        "stage": np.repeat(np.arange(1, gaps.shape[0] + 1), GAP_EDGES.size - 1),
        "gap_bits_left_edge": np.tile(GAP_EDGES[:-1], gaps.shape[0]),
        "count": np.concatenate([np.histogram(stage, bins=GAP_EDGES)[0] for stage in gaps]),
        "coordinates": gaps.shape[1],
    })


# Margins the win shares are reported at, in bits per coordinate. The whole-vector gap is around
# 0.02 bits, so a coordinate decided by more than 0.05 already moved more than the entire average
# difference between the two codecs, and one past 0.5 bits is over half a stage's worth of rate.
MARGINS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def margin_shares(gaps: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
    """Share of coordinates each side wins by more than each margin, per stage."""
    return {
        f"{side}_better_by_{margin}": (
            (gaps < -margin) if side == "t" else (gaps > margin)
        ).mean(axis=1)
        for margin in MARGINS
        for side in ("t", "o")
    }


def dominance_shares(oracle: RoundElements, temporal: RoundElements) -> pd.DataFrame:
    """Per stage, the share of coordinates each side wins on rate, on error, and on both."""
    rows = []
    for stage in range(oracle.stages):
        o_cheaper = oracle.cumulative_bits[stage] < temporal.cumulative_bits[stage]
        o_closer = oracle.abs_error[stage] < temporal.abs_error[stage]
        t_cheaper = temporal.cumulative_bits[stage] < oracle.cumulative_bits[stage]
        t_closer = temporal.abs_error[stage] < oracle.abs_error[stage]
        rows.append({
            "stage": stage + 1,
            "o_dominates": float(np.mean(o_cheaper & o_closer)),
            "t_dominates": float(np.mean(t_cheaper & t_closer)),
            "o_cheaper": float(np.mean(o_cheaper)),
            "o_closer": float(np.mean(o_closer)),
        })
    return pd.DataFrame(rows)


def compare_client_round(
    oracle_dir: Path,
    temporal_dir: Path,
    round_id: int,
    client_id: int,
    deltas: Path,
    parameter_of: NDArray[np.int32],
    parameter_names: list[str],
    slices: list[slice],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare one client-round: whole-vector curves, per-coordinate gaps, and per-stratum gaps."""
    oracle = load_round(oracle_dir, round_id, client_id)
    temporal = load_round(temporal_dir, round_id, client_id)
    assert oracle.cumulative_bits.shape == temporal.cumulative_bits.shape, (
        f"Round {round_id} client {client_id} was coded with different stage counts or vector "
        f"lengths on the two sides: {oracle.cumulative_bits.shape} vs {temporal.cumulative_bits.shape}.")

    whole = pd.DataFrame({
        "round_id": round_id,
        "client_id": client_id,
        "stage": np.arange(1, oracle.stages + 1),
        "o_rate": oracle.cumulative_bits.mean(axis=1),
        "t_rate": temporal.cumulative_bits.mean(axis=1),
        "o_wmape": oracle.abs_error.mean(axis=1) * 100.0,
        "t_wmape": temporal.abs_error.mean(axis=1) * 100.0,
    })
    slope = np.mean([
        average_slope(whole[f"{side}_rate"].to_numpy(), whole[f"{side}_wmape"].to_numpy() / 100.0)
        for side in ("o", "t")
    ])

    gaps = element_gap_bits(oracle, temporal, float(slope))
    per_element = dominance_shares(oracle, temporal).assign(
        round_id=round_id,
        client_id=client_id,
        mean_gap_bits=gaps.mean(axis=1),
        **margin_shares(gaps),
    )
    histogram = gap_histogram(gaps, round_id, client_id)

    delta, previous = slice_normalized(
        torch.load(deltas / f"round_{round_id}_client_{client_id}.pt", weights_only=True),
        torch.load(deltas / f"round_{round_id - 1}_client_{client_id}.pt", weights_only=True),
        slices,
    )
    strata_frames = []
    for name, (labels, group_names) in stratifications(
        delta, previous, parameter_of, parameter_names
    ).items():
        counts = np.bincount(labels, minlength=len(group_names))
        # Averaging the per-coordinate gap inside a stratum is the same slope-converted quantity
        # the histogram shows, so the two views cannot disagree. The interpolated curve gap is not
        # used here: a stratum whose two curves do not overlap in distortion — which is exactly the
        # case where one side dominates outright — leaves it undefined.
        columns = {
            "gap_bits": stratum_means(gaps, labels, counts),
            "o_rate": stratum_means(oracle.cumulative_bits, labels, counts),
            "t_rate": stratum_means(temporal.cumulative_bits, labels, counts),
            "o_wmape": stratum_means(oracle.abs_error, labels, counts) * 100.0,
            "t_wmape": stratum_means(temporal.abs_error, labels, counts) * 100.0,
        }
        strata_frames.append(pd.DataFrame({
            "round_id": round_id,
            "client_id": client_id,
            "stratification": name,
            "group": np.repeat(group_names, oracle.stages),
            "stage": np.tile(np.arange(1, oracle.stages + 1), len(group_names)),
            "count": np.repeat(counts, oracle.stages),
            **{key: value.ravel() for key, value in columns.items()},
        }))
    del gaps
    strata = pd.concat(strata_frames, ignore_index=True)
    return whole, per_element, histogram, strata[strata["count"] > 0]


def sign_agreement(strata: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-client stratum gaps into a mean and the share of clients that agree on its sign.

    A stratum whose gap keeps the same sign across every worker and round is structure; one whose
    sign flips is the training noise two independently fitted models differ by.
    """
    grouped = strata.groupby(["stratification", "group", "stage"], sort=False)
    summary = grouped.agg(
        count=("count", "mean"),
        mean_gap_bits=("gap_bits", "mean"),
        std_gap_bits=("gap_bits", "std"),
        o_rate=("o_rate", "mean"),
        o_wmape=("o_wmape", "mean"),
        t_wmape=("t_wmape", "mean"),
        samples=("gap_bits", "size"),
    )
    summary["positive_share"] = grouped["gap_bits"].apply(lambda gaps: float((gaps > 0).mean()))
    return summary.reset_index()


def plot_report(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Draw the four views the O-versus-T question is read from."""
    curves, dominance = frames["whole_vector_curves"], frames["per_element_dominance"]
    histogram = frames["gap_histogram"]
    # A stratum's gap over its whole curve is the mean of its stage gaps, matching how the rate
    # gap is read off an RD curve rather than at one operating point.
    strata = frames["strata_summary"].groupby(
        ["stratification", "group"], sort=False, as_index=False).mean(numeric_only=True)
    rounds = sorted(int(round_id) for round_id in curves["round_id"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    rd_ax, hist_ax, parameter_ax, decile_ax = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, round_id in enumerate(rounds):
        of_round = curves[curves["round_id"].eq(round_id)].groupby("stage").mean(numeric_only=True)
        for side, style in (("o", "-"), ("t", "--")):
            rd_ax.plot(
                of_round[f"{side}_rate"],
                wmape_to_db(of_round[f"{side}_wmape"].to_numpy()),
                style,
                marker="o" if side == "o" else "s",
                color=colors[index % len(colors)],
                label=f"round {round_id} {side.upper()}",
            )
    rd_ax.set_title("Whole-vector curves, worker-averaged\nsolid = O, dashed = T")
    rd_ax.set_xlabel("Cumulative prior rate by stage")
    rd_ax.set_ylabel("Stage WMAPE, dB re 100%")
    rd_ax.legend(fontsize="xx-small", ncol=2)

    for stage, of_stage in histogram.groupby("stage"):
        totals = of_stage.groupby("gap_bits_left_edge")["count"].sum()
        coordinates = of_stage.groupby("gap_bits_left_edge")["coordinates"].sum()
        hist_ax.step(totals.index, totals / coordinates, where="post", label=f"stage {stage}")
    hist_ax.axvline(0.0, color="0.4", linewidth=1.0)
    hist_ax.set_yscale("log")
    hist_ax.set_title(
        "Per-coordinate rate-equivalent gap\npositive = T pays more than O for the same quality")
    hist_ax.set_xlabel("T minus O (bits)")
    hist_ax.set_ylabel("Share of coordinates")
    hist_ax.legend(fontsize="x-small")

    by_parameter = strata[strata["stratification"].eq("parameter")]
    scatter = parameter_ax.scatter(
        by_parameter["count"],
        by_parameter["mean_gap_bits"],
        c=by_parameter["positive_share"],
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        s=22,
    )
    parameter_ax.axhline(0.0, color="0.4", linewidth=1.0)
    parameter_ax.set_xscale("log")
    parameter_ax.set_title(
        "Rate gap per model parameter\ncolour = share of client-rounds where T lost (1.0 = O always won)")
    parameter_ax.set_xlabel("Coordinates in the parameter")
    parameter_ax.set_ylabel("T minus O rate gap (bits)")
    fig.colorbar(scatter, ax=parameter_ax)

    for name, marker in (("magnitude", "o"), ("novelty", "s")):
        of_name = strata[strata["stratification"].eq(name)]
        decile_ax.errorbar(
            np.arange(len(of_name)),
            of_name["mean_gap_bits"],
            yerr=of_name["std_gap_bits"],
            marker=marker,
            capsize=3,
            label=name,
        )
    decile_ax.axhline(0.0, color="0.4", linewidth=1.0)
    decile_ax.set_title("Rate gap by decile, error bars over client-rounds")
    decile_ax.set_xlabel("Decile (0 = smallest)")
    decile_ax.set_ylabel("T minus O rate gap (bits)")
    decile_ax.legend(fontsize="x-small")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "O against T coordinate by coordinate — O conditions on the whole server ledger, "
        f"T only on its own client's\nrounds {rounds}, "
        f"O wins both rate and error on {dominance['o_dominates'].mean():.1%} of coordinates")
    fig.savefig(out / "elements_o_vs_t.png", dpi=180)
    plt.close(fig)
    logging.info("Wrote %s.", out / "elements_o_vs_t.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-run", type=Path, help="Records dir of the O run.")
    parser.add_argument("--temporal-run", type=Path, help="Records dir of the T run.")
    parser.add_argument("--rounds", type=int, nargs="+", default=(12, 14, 16, 18, 20))
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--out", type=Path, default=PLOTS_DIR / "elements_o_vs_t")
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Re-plot from a previous pass instead of re-reading the element dumps.",
    )
    args = parser.parse_args()
    assert args.from_csv or (args.oracle_run and args.temporal_run), (
        "Both run directories are needed unless --from-csv replots an earlier pass.")
    return args


CSV_NAMES = (
    "whole_vector_curves", "per_element_dominance", "gap_histogram", "strata_gaps", "strata_summary")


def measure(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Read every dumped client-round once and reduce it to the comparison tables."""
    parameter_of, parameter_names, slices = model_layout(args.fl_config)
    whole, per_element, histograms, strata = [], [], [], []
    for round_id in args.rounds:
        for client_id in range(args.clients):
            logging.info("Comparing round %d, client %d.", round_id, client_id)
            results = compare_client_round(
                args.oracle_run / "elements",
                args.temporal_run / "elements",
                round_id,
                client_id,
                args.deltas,
                parameter_of,
                parameter_names,
                slices,
            )
            for collected, result in zip(
                (whole, per_element, histograms, strata), results, strict=True
            ):
                collected.append(result)

    frames = {
        "whole_vector_curves": pd.concat(whole, ignore_index=True),
        "per_element_dominance": pd.concat(per_element, ignore_index=True),
        "gap_histogram": pd.concat(histograms, ignore_index=True),
        "strata_gaps": pd.concat(strata, ignore_index=True),
    }
    frames["strata_summary"] = sign_agreement(frames["strata_gaps"])
    return frames


def main() -> None:
    """Write the whole-vector, per-coordinate, and per-stratum O-versus-T comparisons."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    if args.from_csv:
        frames = {name: pd.read_csv(args.out / f"{name}.csv") for name in CSV_NAMES}
    else:
        frames = measure(args)
        args.out.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(args.out / f"{name}.csv", index=False)
            logging.info("Wrote %s (%d rows).", args.out / f"{name}.csv", len(frame))

    plot_report(frames, args.out)
    dominance = frames["per_element_dominance"].groupby("stage").mean(numeric_only=True)
    logging.info("Per-coordinate outcome by stage:\n%s", dominance[
        ["o_dominates", "t_dominates", "mean_gap_bits"]
        + [f"{side}_better_by_{margin}" for margin in MARGINS for side in ("t", "o")]
    ].to_string())


if __name__ == "__main__":
    main()
