from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from numpy.typing import NDArray


PROJECT_DIR = Path(__file__).resolve().parent
RECORDS_DIR = PROJECT_DIR / "exp_records"
PLOTS_DIR = PROJECT_DIR / "analysis_plots"
RECORD_FILE_NAME = "compression_records.csv"
REQUIRED_COLUMNS = {"round_id", "prior_stage_rates", "stage_wmapes", "protocol_name_full"}


@dataclass(frozen=True)
class Curve:
    """A stage-wise rate-distortion curve produced from one experiment records subset."""

    subset: str
    label: str
    cumulative_prior_rates: NDArray[np.float64]
    distortion_db: NDArray[np.float64]


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
            "Loaded %s: %d rows, rounds %s, protocol %r",
            csv_path,
            len(records),
            sorted(records["round_id"].unique().tolist()),
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


def round_14_first_5_indices(df: pd.DataFrame) -> pd.Index:
    """Return the selected row indices for the first five round-14 records."""
    return df.index[df["round_id"].eq(14)][:5]


def experiment_label(csv_path: Path, df: pd.DataFrame) -> str:
    """Build a concise label from the experiment folder and protocol name."""
    protocols = df["protocol_name_full"].drop_duplicates().tolist()
    assert len(protocols) == 1, f"Expected one protocol_name_full in {csv_path}, found {protocols}."
    return f"{csv_path.parent.name}\n{protocols[0]}"


def build_curves(
    records_by_path: dict[Path, pd.DataFrame],
    subset: str,
    selector: Callable[[pd.DataFrame], pd.DataFrame],
) -> list[Curve]:
    """Build averaged curves for every experiment file that has rows in a subset."""
    curves: list[Curve] = []
    for csv_path, records in records_by_path.items():
        df = selector(records)
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
        cumulative_prior_rates, distortion_db = averaged_curve(df, csv_path)
        curves.append(Curve(subset, experiment_label(csv_path, df), cumulative_prior_rates, distortion_db))
    assert curves, f"Subset {subset!r} matched no rows in any records file."
    return curves


def plot_curve(
    ax: Axes,
    curve: Curve,
    color: str | None = None,
    alpha: float = 0.9,
    annotate: bool = True,
    include_in_legend: bool = True,
) -> None:
    """Draw one stage-wise curve on an axis."""
    ax.plot(
        curve.cumulative_prior_rates,
        curve.distortion_db,
        marker="o",
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
    titles_by_subset: dict[str, str],
    output_path: Path,
) -> None:
    """Plot all subsets side-by-side with shared axes and faint non-active context curves."""
    all_curves = [curve for curves in curves_by_subset.values() for curve in curves]
    assert all_curves, "No curves available for comparison plot."
    xlim, ylim = axis_limits(all_curves)
    labels = sorted({curve.label for curve in all_curves})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(labels) > len(palette):
        logging.warning("Reusing %d plot colors across %d experiment labels.", len(palette), len(labels))
    colors = dict(zip(labels, cycle(palette), strict=False))
    fig, axes = plt.subplots(
        1,
        len(curves_by_subset),
        figsize=(18, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (subset, active_curves) in zip(np.atleast_1d(axes), curves_by_subset.items(), strict=True):
        context_curves = [curve for curve in all_curves if curve.subset != subset]
        logging.info(
            "Comparison panel %r draws %d active and %d faint context curves; "
            "context curves are excluded from the legend.",
            subset,
            len(active_curves),
            len(context_curves),
        )
        for curve in context_curves:
            plot_curve(
                ax,
                curve,
                color=colors[curve.label],
                alpha=0.08,
                annotate=False,
                include_in_legend=False,
            )
        for curve in active_curves:
            plot_curve(ax, curve, color=colors[curve.label], alpha=0.9, annotate=True)
        style_axis(ax, titles_by_subset[subset])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(fontsize="x-small")

    save_plot(fig, output_path, dpi=220)


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

    selectors: dict[str, tuple[str, Path, Callable[[pd.DataFrame], pd.DataFrame]]] = {
        "all": ("Average over all rows", PLOTS_DIR / "rate_distortion_all_rows.png", lambda df: df),
        "round14_first5": (
            "Average over first five round-14 rows",
            PLOTS_DIR / "rate_distortion_round14_first5.png",
            lambda df: df.loc[round_14_first_5_indices(df)],
        ),
        "except_round14_first5": (
            "Average over rows excluding first five round-14 rows",
            PLOTS_DIR / "rate_distortion_except_round14_first5.png",
            lambda df: df.drop(index=round_14_first_5_indices(df)),
        ),
    }
    curves_by_subset = {
        subset: build_curves(records_by_path, subset, selector)
        for subset, (_title, _output_path, selector) in selectors.items()
    }
    for subset, (title, output_path, _selector) in selectors.items():
        plot_subset(curves_by_subset[subset], title, output_path)
    plot_comparison(
        curves_by_subset,
        {subset: title for subset, (title, _output_path, _selector) in selectors.items()},
        PLOTS_DIR / "rate_distortion_comparison_triple.png",
    )

    logging.info("Completed %d plots in %s.", len(selectors) + 1, PLOTS_DIR)


if __name__ == "__main__":
    main()
