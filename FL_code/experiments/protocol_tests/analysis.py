from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_RECORDS_ROOT = Path(__file__).parent / "exp_records"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize protocol replay records and plot RD points.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--start-round", type=int, default=14)
    parser.add_argument("--end-round", type=int, default=18)
    parser.add_argument("--clients", type=int)
    return parser.parse_args()


def _db_wmape(values: pd.Series) -> float:
    mean_wmape = values.mean()
    assert mean_wmape > 0, "WMAPE must be positive to convert to dB."
    return 10 * math.log10(mean_wmape)


def _records_paths(records_root: Path) -> list[Path]:
    assert records_root.is_dir(), f"Missing records directory: {records_root}."
    paths = sorted(path / "compression_records.csv" for path in records_root.iterdir() if path.is_dir())
    existing = [path for path in paths if path.is_file()]
    assert existing, f"No compression_records.csv files found under {records_root}."
    return existing


def summarize_records(
    records_path: Path,
    start_round: int,
    end_round: int,
    clients: int | None,
) -> dict[str, object]:
    """Return one aggregate row for a replay records file."""
    records = pd.read_csv(records_path)
    first = records[records.round_id == start_round]
    later = records[(records.round_id > start_round) & (records.round_id <= end_round)]
    client_count = clients if clients is not None else len(first)
    assert client_count > 0, "clients must be positive."
    assert len(first) == client_count and len(later) == client_count * (end_round - start_round), (
        f"Expected {client_count} first-round and {client_count * (end_round - start_round)} later rows "
        f"in {records_path}; got {len(first)} and {len(later)}."
    )

    first_db_wmape = _db_wmape(first.wmape)
    later_db_wmape = _db_wmape(later.wmape)
    later_weight = end_round - start_round
    averages = (
        records.mean(numeric_only=True)
        .drop(labels=["round_id", "client_id"], errors="ignore")
        .add_prefix("avg_")
        .to_dict()
    )
    return {
        "name": records_path.parent.name,
        "first_db_wmape": first_db_wmape,
        "first_prior_rate": first.prior_rate.mean(),
        "later_db_wmape": later_db_wmape,
        "later_prior_rate": later.prior_rate.mean(),
        "weighted_db_wmape": (first_db_wmape + later_weight * later_db_wmape) / (1 + later_weight),
        "weighted_prior_rate": (
            first.prior_rate.mean() + later_weight * later.prior_rate.mean()
        ) / (1 + later_weight),
        "protocol": records.protocol_name_full.iloc[-1],
        "start_round": start_round,
        "end_round": end_round,
        "clients": client_count,
        **averages,
        "records_path": str(records_path),
    }


def build_stats(records_root: Path, start_round: int, end_round: int, clients: int | None) -> Path:
    """Write protocol_stats.csv by summarizing every replay records directory."""
    assert start_round < end_round, "start-round must be less than end-round for later-round analysis."
    stats = pd.DataFrame(
        summarize_records(path, start_round, end_round, clients)
        for path in _records_paths(records_root)
    )
    stats_path = records_root / "protocol_stats.csv"
    stats.to_csv(stats_path, index=False)
    return stats_path


def plot_stats(stats_path: Path) -> Path:
    """Plot RD summaries from protocol_stats.csv."""
    import matplotlib.pyplot as plt

    stats = pd.read_csv(stats_path)
    plot_path = stats_path.with_suffix(".png")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    panels = (
        ("first_prior_rate", "first_db_wmape", "First replay round"),
        ("later_prior_rate", "later_db_wmape", "Later replay rounds"),
        ("weighted_prior_rate", "weighted_db_wmape", "Weighted replay"),
    )
    for axis, (x_field, y_field, title) in zip(axes, panels):
        axis.scatter(stats[x_field], stats[y_field])
        for name, x, y in stats[["name", x_field, y_field]].itertuples(index=False, name=None):
            axis.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
        axis.set(title=title, xlabel="Prior rate")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("WMAPE (dB)")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def main() -> None:
    """Build aggregate protocol replay stats and the RD comparison plot."""
    args = _parse_args()
    stats_path = build_stats(args.records_root, args.start_round, args.end_round, args.clients)
    plot_path = plot_stats(stats_path)
    print(f"Stats written to {stats_path}\nPlot written to {plot_path}")


if __name__ == "__main__":
    main()
