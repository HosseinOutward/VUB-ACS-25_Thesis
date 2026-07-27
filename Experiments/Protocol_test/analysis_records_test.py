from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from numpy.typing import NDArray


PROJECT_DIR = Path(__file__).resolve().parent
RECORDS_DIR = PROJECT_DIR / "exp_records"
PLOTS_DIR = PROJECT_DIR / "analysis_plots"
RECORD_FILE_NAME = "compression_records.csv"
REQUIRED_COLUMNS = {
    "round_id",
    "client_id",
    "round_name_full",
    "prior_stage_rates",
    "stage_wmapes",
    "protocol_name_full",
}
LINE_STYLES = ("-", "--", "-.", ":")
CLIENT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
VARIANT_TOKEN = re.compile(r"replace_O_with='(\w+)'")
GAP_METRICS = {
    "rate_gap_bits": "T minus O rate at matched distortion (bits)",
    "rate_gap_percent": "T minus O rate at matched distortion (%)",
}


@dataclass(frozen=True)
class Curve:
    """A stage-wise rate-distortion curve produced from one experiment records subset."""

    subset: str
    label: str
    cumulative_prior_rates: NDArray[np.float64]
    distortion_db: NDArray[np.float64]
    client_id: int | None = None


def parse_float_list(value: str, csv_path: Path, field_name: str) -> list[float]:
    """Parse one CSV cell containing a Python-style numeric list."""
    try:
        values = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Could not parse {field_name} in {csv_path}: {value!r}") from error

    if not isinstance(values, list) or not all(isinstance(item, int | float) for item in values):
        raise TypeError(f"{field_name} must be a numeric list in {csv_path}: {value!r}")
    return [float(item) for item in values]


def wmape_to_db(wmape_percent: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert WMAPE percentages to dB relative to 100 percent error."""
    assert (wmape_percent > 0).all(), "WMAPE must be positive before converting to dB."
    return 20.0 * np.log10(wmape_percent / 100.0)


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    """Fail when a records file lacks the fields needed for stage-wise RD curves."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path} is missing required column(s): {sorted(missing)}")


def load_records(csv_paths: list[Path]) -> dict[Path, pd.DataFrame]:
    """Load and validate each experiment records file once for all plot subsets."""
    records_by_path: dict[Path, pd.DataFrame] = {}
    for csv_path in csv_paths:
        records = pd.read_csv(csv_path)
        validate_columns(records, csv_path)
        assert not records.empty, f"Records file is empty: {csv_path}"
        protocols = records["protocol_name_full"].drop_duplicates().tolist()
        assert len(protocols) == 1, f"Expected one protocol_name_full in {csv_path}, found {protocols}."
        records_by_path[csv_path] = records
        logging.info(
            "Loaded %s: %d rows, rounds %s, training round %d, protocol %r",
            csv_path,
            len(records),
            sorted(records["round_id"].unique().tolist()),
            training_round_id(records, csv_path),
            protocols[0],
        )
    return records_by_path


def averaged_curve(df: pd.DataFrame, csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average rows into cumulative prior rates paired with stage-wise WMAPE dB."""
    validate_columns(df, csv_path)
    stage_rates = df["prior_stage_rates"].map(
        lambda value: parse_float_list(str(value), csv_path, "prior_stage_rates")
    )
    stage_wmapes = df["stage_wmapes"].map(lambda value: parse_float_list(str(value), csv_path, "stage_wmapes"))
    stage_count = len(stage_rates.iloc[0])
    assert stage_count > 0, f"prior_stage_rates must contain at least one stage in {csv_path}."
    assert all(len(rates) == stage_count for rates in stage_rates), (
        f"All prior_stage_rates rows must have the same stage count in {csv_path}."
    )
    assert all(len(wmapes) == stage_count for wmapes in stage_wmapes), (
        f"stage_wmapes must match prior_stage_rates length in {csv_path}."
    )
    return (
        np.cumsum(np.array(stage_rates.tolist(), dtype=np.float64).mean(axis=0)),
        wmape_to_db(np.array(stage_wmapes.tolist(), dtype=np.float64).mean(axis=0)),
    )


def experiment_label(csv_path: Path, df: pd.DataFrame) -> str:
    """Build a concise label from the experiment folder and protocol name."""
    protocols = df["protocol_name_full"].drop_duplicates().tolist()
    assert len(protocols) == 1, f"Expected one protocol_name_full in {csv_path}, found {protocols}."
    return f"{csv_path.parent.name}\n{protocols[0]}"


def build_curves(
    records_by_path: dict[Path, pd.DataFrame],
    subset: str,
    selector: Callable[[pd.DataFrame, Path], pd.DataFrame],
    per_client: bool = False,
) -> list[Curve]:
    """Build averaged curves for every experiment file that has rows in a subset, optionally split per worker."""
    curves: list[Curve] = []
    for csv_path, records in records_by_path.items():
        df = selector(records, csv_path)
        if df.empty:
            logging.warning("Skipping %s for subset %r: selected 0/%d rows.", csv_path, subset, len(records))
            continue

        logging.info(
            "Subset %r selects %d/%d rows from %s%s.",
            subset,
            len(df),
            len(records),
            csv_path,
            f" at source indices {df.index.tolist()}" if len(df) <= 5 else "",
        )
        label = experiment_label(csv_path, df)
        groups = df.groupby("client_id") if per_client else [(None, df)]
        for client_id, group in groups:
            cumulative_prior_rates, distortion_db = averaged_curve(group, csv_path)
            curves.append(
                Curve(
                    subset,
                    label,
                    cumulative_prior_rates,
                    distortion_db,
                    None if client_id is None else int(client_id),
                )
            )
    assert curves, f"Subset {subset!r} matched no rows in any records file."
    return curves


def rate_at_distortion(curve: Curve, distortion_db: NDArray[np.float64]) -> NDArray[np.float64]:
    """Read a curve's cumulative rate at the given distortions by interpolating between its stage points."""
    order = np.argsort(curve.distortion_db)
    return np.interp(distortion_db, curve.distortion_db[order], curve.cumulative_prior_rates[order])


def curve_rate_gap(curve: Curve, reference: Curve) -> tuple[float, float]:
    """Mean extra rate a curve spends over a reference at matched distortion, in bits and in percent.

    Each curve is its own piecewise-linear rate(distortion) function through its actual stage points,
    and the horizontal gap between them is averaged over the distortion span both curves cover. No
    fitted or stage-averaged slope is involved, and the trapezoid integral is exact rather than
    approximate because every stage point of both curves is a breakpoint of the integration grid.
    A positive result means the curve needs more rate than the reference for the same distortion.
    """
    low = max(float(curve.distortion_db.min()), float(reference.distortion_db.min()))
    high = min(float(curve.distortion_db.max()), float(reference.distortion_db.max()))
    if high <= low:
        logging.warning(
            "%r and %r (%s) share no distortion range, so their rate gap is undefined.",
            curve.subset,
            reference.subset,
            curve.label,
        )
        return float("nan"), float("nan")

    grid = np.unique(np.concatenate([curve.distortion_db, reference.distortion_db, [low, high]]))
    grid = grid[(grid >= low) & (grid <= high)]
    curve_rates, reference_rates = rate_at_distortion(curve, grid), rate_at_distortion(reference, grid)
    gap = float(np.trapezoid(curve_rates - reference_rates, grid)) / (high - low)
    return gap, 100.0 * gap / (float(np.trapezoid(reference_rates, grid)) / (high - low))


def training_round_id(records: pd.DataFrame, csv_path: Path) -> int:
    """Find the one round in which the model was trained; every later round reuses that frozen model.

    "F" is the only frozen-round marker, so any other round_name_full prefix (O, T, OM, TM, ...) names
    a trained variant. A run trains once and then replays, so the trained rows must all sit in a single
    round, one row per participating worker; more than one such round means the file mixes runs.
    """
    prefixes = records["round_name_full"].str.split("|").str[0]
    rounds = sorted(records.loc[prefixes.ne("F"), "round_id"].unique().tolist())
    assert len(rounds) == 1, f"Expected exactly one training round in {csv_path}, found {rounds}."
    return int(rounds[0])


def tagged_records(records_by_path: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """Stack every records file, tagged with the O/T side it ran plus the mode and base config it shares.

    A protocol either replaces O with a variant name or is plain O. The variant's first letter is the
    side and its suffix is the mode, so "TM" is the T side of mode "M" and pairs with "OM" on its own,
    without any variant being named here.
    """
    frames = []
    for csv_path, records in records_by_path.items():
        tokens = records["protocol_name_full"].iloc[0].split("|")
        variants = [match[1] for match in map(VARIANT_TOKEN.fullmatch, tokens) if match]
        assert len(variants) <= 1, f"{csv_path} names more than one variant: {variants}."
        variant = variants[0] if variants else "O"
        assert variant[0] in "OT", f"{csv_path} runs variant {variant!r}, which is neither an O nor a T side."
        frames.append(
            records.assign(
                side=variant[0],
                mode=variant[1:],
                base_config="|".join(token for token in tokens if not VARIANT_TOKEN.fullmatch(token)),
                source=csv_path,
            )
        )
    return pd.concat(frames, ignore_index=True)


def variant_gap_table(records_by_path: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """Rate gap of the T side against the O side of every mode that was run both ways, per round and per worker.

    Rows with a null round_id hold the all-rounds gap, which pools every round into one curve per
    side; rows with a null client_id are worker-averaged.
    """

    def gap_row(config: str, round_id: int | None, client_id: int | None, pair: pd.DataFrame) -> dict[str, object]:
        curves = {
            side: Curve(side, config, *averaged_curve(rows, rows["source"].iloc[0]))
            for side, rows in pair.groupby("side")
        }
        gap_bits, gap_percent = curve_rate_gap(curves["T"], curves["O"])
        return {
            "config": config,
            "round_id": round_id,
            "client_id": client_id,
            "rate_gap_bits": gap_bits,
            "rate_gap_percent": gap_percent,
        }

    rows: list[dict[str, object]] = []
    for (base, mode), pair in tagged_records(records_by_path).groupby(["base_config", "mode"]):
        config = f"{base} · T{mode} vs O{mode}"
        if set(pair["side"]) != {"O", "T"}:
            logging.warning("%s: only the %s side was run, so it has nothing to compare against.", config, set(pair["side"]))
            continue
        assert pair.groupby("side")["source"].nunique().eq(1).all(), (
            f"{config}: a side has more than one run: {sorted({str(path) for path in pair['source']})}."
        )
        # Only a round and worker that both sides recorded can be compared; the log below counts the drops.
        shared = pair.groupby(["round_id", "client_id"]).filter(lambda group: group["side"].nunique() == 2)
        assert not shared.empty, f"{config}: the two sides share no round and worker."
        logging.info(
            "%s: comparing rounds %s over %d workers (%d of %d rows paired).",
            config,
            sorted(set(shared["round_id"])),
            shared["client_id"].nunique(),
            len(shared),
            len(pair),
        )

        rows.append(gap_row(config, None, None, shared))
        for round_id, in_round in shared.groupby("round_id"):
            rows.append(gap_row(config, int(round_id), None, in_round))
            rows += [
                gap_row(config, int(round_id), int(client_id), of_client)
                for client_id, of_client in in_round.groupby("client_id")
            ]

    assert rows, "No mode was run both as O and as T, so no T-vs-O gap can be measured."
    return pd.DataFrame(rows).astype({"round_id": "Int64", "client_id": "Int64"})


def shade(color: str | tuple[float, float, float], amount: float) -> tuple[float, float, float]:
    """Blend a color toward white (amount > 0) or black (amount < 0) to separate same-protocol client curves."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + ((1.0 if amount > 0 else 0.0) - rgb) * abs(amount))


def protocol_styles(labels: list[str]) -> dict[str, tuple[str, str]]:
    """Map every experiment label to a distinct color and line-style pair."""
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(labels) > len(palette) * len(LINE_STYLES):
        logging.warning(
            "Only %d color/style pairs available for %d labels.", len(palette) * len(LINE_STYLES), len(labels)
        )
    return {
        label: (palette[index % len(palette)], LINE_STYLES[index % len(LINE_STYLES)])
        for index, label in enumerate(labels)
    }


def client_styles(client_ids: list[int]) -> dict[int, tuple[str, float]]:
    """Map every client id to a marker shape and a color-shading amount."""
    span = max(len(client_ids) - 1, 1)
    return {
        client_id: (CLIENT_MARKERS[index % len(CLIENT_MARKERS)], 0.7 * (index / span - 0.5))
        for index, client_id in enumerate(client_ids)
    }


def plot_curve(
    ax: Axes,
    curve: Curve,
    color: str | tuple[float, float, float] | None = None,
    alpha: float = 0.9,
    annotate: bool = True,
    include_in_legend: bool = True,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    """Draw one stage-wise curve on an axis."""
    ax.plot(
        curve.cumulative_prior_rates,
        curve.distortion_db,
        marker=marker,
        markersize=6,
        linestyle=linestyle,
        linewidth=2.0,
        color=color,
        alpha=alpha,
        label=curve.label if include_in_legend else None,
    )
    if annotate:
        for stage_id, (rate, distortion_db) in enumerate(
            zip(curve.cumulative_prior_rates, curve.distortion_db), start=1
        ):
            ax.annotate(f"s{stage_id}", (rate, distortion_db), textcoords="offset points", xytext=(4, 5), alpha=alpha)


def style_axis(ax: Axes, title: str) -> None:
    """Apply shared RD plot labels and grid styling."""
    ax.set_title(title)
    ax.set_xlabel("Cumulative prior rate by stage")
    ax.set_ylabel("Stage WMAPE distortion, dB re 100%")
    ax.grid(True, alpha=0.3)


def save_plot(fig: Figure, output_path: Path, dpi: int) -> None:
    """Save a plot while reporting whether an existing artifact is replaced."""
    if output_path.exists():
        logging.warning("Overwriting existing plot: %s", output_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    logging.info("Saved plot: %s", output_path)


def plot_subset(
    curves: list[Curve],
    title: str,
    output_path: Path,
) -> None:
    """Plot averaged cumulative prior-rate curves for one row subset."""
    assert curves, f"Refusing to create an empty plot at {output_path}."
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for curve in curves:
        plot_curve(ax, curve)
    style_axis(ax, title)
    ax.legend(fontsize="small")

    save_plot(fig, output_path, dpi=200)


def axis_limits(curves: list[Curve]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded x and y limits covering every curve."""
    assert curves, "Cannot calculate axis limits without curves."
    all_x = np.concatenate([curve.cumulative_prior_rates for curve in curves])
    all_y = np.concatenate([curve.distortion_db for curve in curves])
    x_pad = max((float(all_x.max()) - float(all_x.min())) * 0.06, 0.01)
    y_pad = max((float(all_y.max()) - float(all_y.min())) * 0.08, 0.1)
    return (float(all_x.min()) - x_pad, float(all_x.max()) + x_pad), (
        float(all_y.min()) - y_pad,
        float(all_y.max()) + y_pad,
    )


def plot_comparison(
    curves_by_subset: dict[str, list[Curve]],
    client_curves_by_subset: dict[str, list[Curve]],
    titles_by_subset: dict[str, str],
    output_path: Path,
) -> None:
    """Plot every subset as a row: worker-averaged curves with faint context, beside per-worker curves."""
    all_curves = [curve for curves in curves_by_subset.values() for curve in curves]
    all_client_curves = [curve for curves in client_curves_by_subset.values() for curve in curves]
    assert all_curves and all_client_curves, "No curves available for comparison plot."
    xlim, ylim = axis_limits(all_curves + all_client_curves)
    styles = protocol_styles(sorted({curve.label for curve in all_curves}))
    clients = client_styles(sorted({curve.client_id for curve in all_client_curves}))
    fig, axes = plt.subplots(
        len(curves_by_subset),
        2,
        figsize=(15, 6 * len(curves_by_subset)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for (mean_ax, client_ax), subset in zip(np.atleast_2d(axes), curves_by_subset, strict=True):
        context_curves = [curve for curve in all_curves if curve.subset != subset]
        logging.info(
            "Comparison row %r draws %d worker-averaged (plus %d faint context) and %d per-worker curves.",
            subset,
            len(curves_by_subset[subset]),
            len(context_curves),
            len(client_curves_by_subset[subset]),
        )
        for curve in context_curves:
            color, linestyle = styles[curve.label]
            plot_curve(mean_ax, curve, color, alpha=0.08, annotate=False, include_in_legend=False, linestyle=linestyle)
        for curve in curves_by_subset[subset]:
            color, linestyle = styles[curve.label]
            plot_curve(mean_ax, curve, color, linestyle=linestyle)
        style_axis(mean_ax, f"{titles_by_subset[subset]}\nworker-averaged")
        mean_ax.legend(fontsize="x-small")

        for curve in client_curves_by_subset[subset]:
            color, linestyle = styles[curve.label]
            marker, shading = clients[curve.client_id]
            plot_curve(
                client_ax,
                curve,
                shade(color, shading),
                annotate=False,
                include_in_legend=False,
                linestyle=linestyle,
                marker=marker,
            )
        style_axis(client_ax, f"{titles_by_subset[subset]}\nper worker")
        client_ax.legend(
            handles=[
                Line2D([], [], color=shade("0.35", shading), marker=marker, linestyle="", label=f"client {client_id}")
                for client_id, (marker, shading) in clients.items()
            ],
            fontsize="x-small",
            title="worker (color/style = protocol)",
            title_fontsize="x-small",
            ncol=2,
        )

    for ax in np.atleast_2d(axes).ravel():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    save_plot(fig, output_path, dpi=220)


GAP_PLOT_TITLE = (
    "The T side against the O side of every mode, over rounds\n"
    "positive = T pays more rate than O for the same distortion; dotted = that pair's all-rounds gap"
)


def plot_variant_gaps(gaps: pd.DataFrame, output_path: Path) -> None:
    """Plot every variant pair's gap over rounds, worker-averaged beside per worker, one row per metric."""
    per_round = gaps.dropna(subset=["round_id"])
    worker_averaged, per_client = per_round[per_round["client_id"].isna()], per_round.dropna(subset=["client_id"])
    all_rounds = gaps[gaps["round_id"].isna() & gaps["client_id"].isna()].set_index("config")
    assert not worker_averaged.empty, "No worker-averaged gaps to plot."
    styles = protocol_styles(sorted(gaps["config"].unique().tolist()))
    clients = client_styles(sorted(per_client["client_id"].astype(int).unique().tolist()))
    fig, axes = plt.subplots(
        len(GAP_METRICS),
        2,
        figsize=(15, 5 * len(GAP_METRICS)),
        sharex=True,
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )

    for (mean_ax, client_ax), (metric, ylabel) in zip(axes, GAP_METRICS.items(), strict=True):
        for config, group in worker_averaged.groupby("config"):
            color, linestyle = styles[config]
            mean_ax.plot(
                group["round_id"],
                group[metric],
                marker="o",
                linewidth=2.0,
                color=color,
                linestyle=linestyle,
                label=config,
            )
            mean_ax.axhline(all_rounds.loc[config, metric], color=color, linestyle=":", linewidth=1.5, alpha=0.8)
        for (config, client_id), group in per_client.groupby(["config", "client_id"]):
            color, linestyle = styles[config]
            marker, shading = clients[int(client_id)]
            client_ax.plot(
                group["round_id"],
                group[metric],
                marker=marker,
                markersize=5,
                linewidth=1.2,
                color=shade(color, shading),
                linestyle=linestyle,
            )
        for ax in (mean_ax, client_ax):
            ax.axhline(0.0, color="0.4", linewidth=1.0)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    axes[0][0].set_title("worker-averaged")
    axes[0][1].set_title("per worker")

    axes[0][0].legend(fontsize="x-small")
    axes[0][1].legend(
        handles=[
            Line2D([], [], color=shade("0.35", shading), marker=marker, linestyle="", label=f"client {client_id}")
            for client_id, (marker, shading) in clients.items()
        ],
        fontsize="x-small",
        title="worker (color/style = pair)",
        title_fontsize="x-small",
        ncol=2,
    )
    for ax in axes[-1]:
        ax.set_xlabel("Round")
    fig.suptitle(GAP_PLOT_TITLE)

    save_plot(fig, output_path, dpi=200)


def main() -> None:
    """Generate read-only stage-wise rate-distortion summaries from protocol-test records."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    csv_paths = sorted(RECORDS_DIR.glob(f"*/{RECORD_FILE_NAME}"))
    assert csv_paths, f"No {RECORD_FILE_NAME} files found under {RECORDS_DIR}."
    logging.info("Discovered %d records files under %s.", len(csv_paths), RECORDS_DIR)
    records_by_path = load_records(csv_paths)
    if not PLOTS_DIR.exists():
        logging.info("Creating plot directory: %s", PLOTS_DIR)
    PLOTS_DIR.mkdir(exist_ok=True)

    selectors: dict[str, tuple[str, Path, Callable[[pd.DataFrame, Path], pd.DataFrame]]] = {
        "all": ("Average over all rows", PLOTS_DIR / "rate_distortion_all_rows.png", lambda df, _path: df),
        "training_round": (
            "Average over the training round",
            PLOTS_DIR / "rate_distortion_training_round.png",
            lambda df, path: df[df["round_id"].eq(training_round_id(df, path))],
        ),
        "frozen_rounds": (
            "Average over the frozen rounds",
            PLOTS_DIR / "rate_distortion_frozen_rounds.png",
            lambda df, path: df[df["round_id"].ne(training_round_id(df, path))],
        ),
    }
    curves_by_subset = {
        subset: build_curves(records_by_path, subset, selector)
        for subset, (_title, _output_path, selector) in selectors.items()
    }
    client_curves_by_subset = {
        subset: build_curves(records_by_path, subset, selector, per_client=True)
        for subset, (_title, _output_path, selector) in selectors.items()
    }
    for subset, (title, output_path, _selector) in selectors.items():
        plot_subset(curves_by_subset[subset], title, output_path)
    plot_comparison(
        curves_by_subset,
        client_curves_by_subset,
        {subset: title for subset, (title, _output_path, _selector) in selectors.items()},
        PLOTS_DIR / "rate_distortion_comparison_triple.png",
    )

    gaps = variant_gap_table(records_by_path)
    gaps.to_csv(PLOTS_DIR / "variant_gap_over_rounds.csv", index=False)
    plot_variant_gaps(gaps, PLOTS_DIR / "variant_gap_over_rounds.png")

    logging.info("Completed %d plots in %s.", len(selectors) + 2, PLOTS_DIR)


if __name__ == "__main__":
    main()
