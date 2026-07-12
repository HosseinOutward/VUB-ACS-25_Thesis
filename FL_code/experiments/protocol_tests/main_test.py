from __future__ import annotations

import argparse
from enum import StrEnum
import math
from pathlib import Path
import sys
from typing import Any, Hashable
import warnings

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import (
    Access,
    BaseProtocol,
    CompressionRecord,
    create_protocol,
    simulate_compression,
)
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.run_fl import FLConfig


DEFAULT_PROTOCOL = "wz_cancer"
DEFAULT_RECONSTRUCTIONS = PROJECT_ROOT / "_/data/cancer_protocol/T_round_checkpoints"
DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/client_deltas"
DEFAULT_RECORDS_ROOT = Path(__file__).parent / "exp_records"


class HistorySource(StrEnum):
    """Selects whether the protocol starts from decoder reconstructions or lossless deltas."""

    RECONSTRUCTIONS = "reconstructions"
    RAW_DELTAS = "raw_deltas"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved client deltas through a protocol without running federated learning."
    )
    parser.add_argument("--name", required=True, help="Unique short label stored in the stats CSV.")
    parser.add_argument("--description", default="", help="Free-text note stored as the final field.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--start-round", type=int, default=14)
    parser.add_argument("--end-round", type=int, default=18)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument(
        "--history-source",
        type=HistorySource,
        choices=tuple(HistorySource),
        default=HistorySource.RECONSTRUCTIONS,
    )
    parser.add_argument("--reconstructions", type=Path, default=DEFAULT_RECONSTRUCTIONS)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def _load_tensor(directory: Path, round_id: int, client_id: int) -> torch.Tensor:
    path = directory / f"round_{round_id}_client_{client_id}.pt"
    assert path.is_file(), f"Missing replay tensor: {path}."
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(tensor, torch.Tensor), f"Replay file does not contain a tensor: {path}."
    assert tensor.dtype == torch.float32 and tensor.device == torch.device("cpu"), (
        f"Expected a CPU float32 tensor in {path}; got {tensor.dtype} on {tensor.device}."
    )
    return tensor


def _build_state_manager(config_path: Path) -> StateDictManager:
    assert config_path.is_file(), f"Missing replay FL configuration: {config_path}."
    cfg = FLConfig.model_validate_json(config_path.read_text())
    model = initialize_model(cfg, torch.device("cpu"))
    return StateDictManager(model)


def seed_history(
    protocol: BaseProtocol,
    source: HistorySource,
    start_round: int,
    client_ids: range,
    reconstructions_dir: Path,
    deltas_dir: Path,
) -> None:
    """Recreate the bounded protocol history immediately before the first replayed round."""
    history_depth = protocol._recons_history.max_per_client
    assert history_depth > 0, (
        f"Protocol {protocol.protocol_name_full!r} does not retain reconstruction history."
    )
    assert start_round >= history_depth, (
        f"Round {start_round} has fewer than {history_depth} preceding rounds to seed."
    )

    for round_id in range(start_round - history_depth, start_round):
        for client_id in client_ids:
            raw_delta = _load_tensor(deltas_dir, round_id, client_id)
            history_tensor = (
                _load_tensor(reconstructions_dir, round_id, client_id)
                if source is HistorySource.RECONSTRUCTIONS
                else raw_delta
            )
            record = CompressionRecord(round_id, client_id, f"history_seed_{source.value}")
            protocol.commit_reconstruction(
                raw_delta, history_tensor, record, Access.TEMPORAL_TOO
            )


def save_stats(
    records_path: Path,
    stats_path: Path,
    name: str,
    description: str,
    protocol: str,
    history_source: HistorySource,
    start_round: int,
    end_round: int,
    clients: int,
    seed: int,
) -> None:
    """Append one named aggregate row and reject duplicate experiment names."""
    assert records_path.is_file(), f"Missing compression records: {records_path}."
    records = pd.read_csv(records_path)
    first = records[records.round_id == start_round]
    rest = records[records.round_id > start_round]
    assert len(first) == clients and len(rest) == clients * (end_round - start_round), (
        f"Expected {clients} first-round and {clients * (end_round - start_round)} later rows; "
        f"got {len(first)} and {len(rest)}."
    )

    first_wmape, rest_wmape = first.wmape.mean(), rest.wmape.mean()
    first_prior, rest_prior = first.prior_rate.mean(), rest.prior_rate.mean()
    averages = (
        records.mean(numeric_only=True)
        .drop(labels=["round_id", "client_id"], errors="ignore")
        .add_prefix("avg_")
        .to_dict()
    )
    row: dict[str | Hashable, str | float | int | Any] = {
        "name": name,
        "round14_db_wmape": 10 * math.log10(first_wmape),
        "round14_prior_rate": first_prior,
        "later_db_wmape": 10 * math.log10(rest_wmape),
        "later_prior_rate": rest_prior,
        "weighted_db_wmape": (10 * math.log10(first_wmape) + 4 * 10 * math.log10(rest_wmape)) / 5,
        "weighted_prior_rate": (first_prior + 4 * rest_prior) / 5,
        "protocol": protocol,
        "history_source": history_source.value,
        "start_round": start_round,
        "end_round": end_round,
        "clients": clients,
        "seed": seed,
        **averages,
        "records_path": str(records_path),
        "description": description,
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(stats_path, keep_default_na=False) if stats_path.exists() else pd.DataFrame()
    assert existing.empty or name not in existing.name.values, (
        f"Experiment name {name!r} already exists in {stats_path}."
    )
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = updated[[column for column in updated if column != "description"] + ["description"]]
    updated.to_csv(stats_path, index=False)


def plot_stats(stats_path: Path, plot_path: Path) -> None:
    """Plot three rate-distortion views and circle the newest experiment."""
    import matplotlib.pyplot as plt

    stats = pd.read_csv(stats_path)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    panels = (
        ("round14_prior_rate", "round14_db_wmape", "Round 14"),
        ("later_prior_rate", "later_db_wmape", "Rounds 15–18"),
        ("weighted_prior_rate", "weighted_db_wmape", "1:4 weighted"),
    )
    latest = stats.iloc[-1]
    for axis, (x_field, y_field, title) in zip(axes, panels):
        axis.scatter(stats[x_field], stats[y_field])
        axis.scatter(
            latest[x_field], latest[y_field], s=180, facecolors="none",
            edgecolors="black", linewidths=1.5,
        )
        for name, x, y in stats[["name", x_field, y_field]].itertuples(index=False, name=None):
            axis.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points")
        axis.set(title=title, xlabel="Prior rate")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("WMAPE (dB)")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Replay client-rounds and append one named aggregate row to the experiment stats."""
    args = _parse_args()
    assert args.start_round <= args.end_round, "start-round must not exceed end-round."
    assert args.clients > 0, "clients must be positive."
    assert Path(args.name).name == args.name and args.name not in {".", ".."}, (
        "name must be a single directory-safe path component."
    )

    if args.history_source is HistorySource.RAW_DELTAS:
        message = (
            "LOSSLESS-HISTORY EXPERIMENT: history entries are raw client deltas, not "
            "decoder reconstructions. Results do not represent the normal decoder state."
        )
        warnings.warn(message, stacklevel=1)
        print(f"\n{'!' * 88}\n{message}\n{'!' * 88}\n")

    set_global_seed(args.seed)
    state_manager = _build_state_manager(args.reconstructions / "fl_config.json")
    protocol = create_protocol(args.protocol, state_manager.get_slices())
    client_ids = range(args.clients)
    seed_history(
        protocol,
        args.history_source,
        args.start_round,
        client_ids,
        args.reconstructions,
        args.deltas,
    )

    output_dir = args.records_root / args.name
    assert not (output_dir / "compression_records.csv").exists(), (
        f"Output records already exist: {output_dir / 'compression_records.csv'}."
    )

    for round_id in range(args.start_round, args.end_round + 1):
        for client_id in client_ids:
            delta = _load_tensor(args.deltas, round_id, client_id)
            assert delta.numel() == state_manager.param_count, (
                f"round {round_id}, client {client_id} has {delta.numel()} values; "
                f"the configured model has {state_manager.param_count}."
            )
            print(f"Simulating round {round_id}, client {client_id} with {args.protocol}")
            simulate_compression(
                protocol,
                delta,
                client_id,
                round_id,
                sd_manager=state_manager,
                save_dir=output_dir,
                server_eval_metrics={},
                worker_eval_metrics=(),
                metric_keys=(),
            )

    generated_path = output_dir / "compression_records.csv"
    print(f"\nRecords written to {generated_path}")
    stats_path = args.records_root / "protocol_stats.csv"
    save_stats(
        generated_path,
        stats_path,
        args.name,
        args.description,
        args.protocol,
        args.history_source,
        args.start_round,
        args.end_round,
        args.clients,
        args.seed,
    )
    plot_path = stats_path.with_suffix(".png")
    plot_stats(stats_path, plot_path)
    print(f"Stats appended to {stats_path}\nPlot written to {plot_path}")


if __name__ == "__main__":
    main()
