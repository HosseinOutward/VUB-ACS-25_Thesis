"""Plot the per-round rate-distortion curves from one protocol replay."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
import pandas as pd  # noqa: E402

from Experiments.Protocol_test.rd_metrics import rd_curve_gap  # noqa: E402

REQUIRED_COLUMNS = {"round_id", "prior_stage_rates", "stage_wmapes", "protocol_name_full"}


def parse_args() -> argparse.Namespace:
    """Parse the records input and optional image path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="A replay's compression_records.csv.")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def numeric_list(value: str, field: str, records_path: Path) -> NDArray[np.float64]:
    """Parse one list-valued compression-record field."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Could not parse {field} in {records_path}: {value!r}.") from error
    assert isinstance(parsed, list) and parsed and all(isinstance(item, int | float) for item in parsed), (
        f"{field} must be a non-empty numeric list in {records_path}; got {value!r}."
    )
    return np.asarray(parsed, dtype=np.float64)


def round_curve(records: pd.DataFrame, records_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average clients into cumulative prior rate and WMAPE-dB stage coordinates."""
    rates = [numeric_list(str(value), "prior_stage_rates", records_path)
             for value in records["prior_stage_rates"]]
    wmapes = [numeric_list(str(value), "stage_wmapes", records_path)
              for value in records["stage_wmapes"]]
    stage_counts = {len(values) for values in (*rates, *wmapes)}
    assert len(stage_counts) == 1, f"Stage counts differ within {records_path}: {sorted(stage_counts)}."
    mean_wmape = np.stack(wmapes).mean(axis=0)
    assert (mean_wmape > 0).all(), f"WMAPE must be positive in {records_path}."
    return np.cumsum(np.stack(rates).mean(axis=0)), 20 * np.log10(mean_wmape / 100)


def plot_records(records_path: Path, output_path: Path) -> None:
    """Render one averaged RD curve for each replayed round."""
    assert records_path.is_file(), f"Missing compression records: {records_path}."
    records = pd.read_csv(records_path)
    missing = REQUIRED_COLUMNS - set(records.columns)
    assert not missing, f"{records_path} is missing columns: {sorted(missing)}."
    protocols = records["protocol_name_full"].drop_duplicates().tolist()
    assert len(protocols) == 1, f"Expected one protocol in {records_path}; found {protocols}."

    rounds = sorted(records["round_id"].unique().tolist())
    normalization = plt.Normalize(rounds[0] - 0.5, rounds[-1] + 0.5)
    colormap = matplotlib.colormaps["viridis"]
    figure, axis = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
    for round_id, group in records.groupby("round_id", sort=True):
        rate, distortion = round_curve(group, records_path)
        axis.plot(rate, distortion, color=colormap(normalization(round_id)), marker="o", linewidth=1.8)

    colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(normalization, colormap), ax=axis)
    colorbar.set_label("Round")
    axis.set_xlabel("Cumulative prior rate (bits/element)")
    axis.set_ylabel("WMAPE (dB relative to 100%)")
    axis.set_title(protocols[0])
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote {output_path}")


def plot_sparse_results(output_root: Path, comparisons: tuple[tuple[str, str], ...]) -> None:
    """Plot completed sparse codecs and write declared matched-case RD comparisons."""
    records_by_codec = {
        path.parent.name: pd.read_csv(path)
        for path in sorted((output_root / "records").glob("*/compression_records.csv"))
    }
    assert records_by_codec, f"No aggregated sparse records under {output_root}."
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
    for codec_id, records in records_by_codec.items():
        rate, distortion = round_curve(records, output_root / "records" / codec_id / "compression_records.csv")
        axis.plot(rate, distortion, marker="o", linewidth=1.8, label=codec_id)
    axis.set_xlabel("Cumulative prior rate (bits/element)")
    axis.set_ylabel("WMAPE (dB relative to 100%)")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(analysis_dir / "rate_distortion_comparison.png", dpi=180)
    plt.close(figure)

    gaps: list[dict[str, int | float | str | None]] = []
    keys = ["case_id", "relative_round", "client_id"]
    for candidate_id, reference_id in comparisons:
        if candidate_id not in records_by_codec or reference_id not in records_by_codec:
            continue
        candidate, reference = records_by_codec[candidate_id], records_by_codec[reference_id]
        candidate_keys = pd.MultiIndex.from_frame(candidate[keys])
        reference_keys = pd.MultiIndex.from_frame(reference[keys])
        shared = candidate_keys.intersection(reference_keys)
        candidate, reference = candidate[candidate_keys.isin(shared)], reference[reference_keys.isin(shared)]
        for relative_round in (None, *sorted(candidate["relative_round"].unique().tolist())):
            candidate_rows = candidate if relative_round is None else candidate[candidate["relative_round"].eq(relative_round)]
            reference_rows = reference if relative_round is None else reference[reference["relative_round"].eq(relative_round)]
            if candidate_rows.empty:
                continue
            candidate_curve = round_curve(candidate_rows, output_root / "records" / candidate_id / "compression_records.csv")
            reference_curve = round_curve(reference_rows, output_root / "records" / reference_id / "compression_records.csv")
            gaps.append({
                "candidate": candidate_id,
                "reference": reference_id,
                "relative_round": relative_round,
                **rd_curve_gap(*candidate_curve, *reference_curve, name=f"{candidate_id} vs {reference_id}"),
            })
    if gaps:
        pd.DataFrame(gaps).to_csv(analysis_dir / "comparison_gaps.csv", index=False)


def main() -> None:
    """Plot the selected replay records."""
    args = parse_args()
    output = args.out or Path(__file__).parent / "analysis_plots" / f"{args.records.parent.name}_rd.png"
    plot_records(args.records, output)


if __name__ == "__main__":
    main()
