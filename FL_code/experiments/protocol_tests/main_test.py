from __future__ import annotations

import argparse
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
import sys
import warnings

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
from FL_code.FL_core.utils import StateDictManager, _prepare_records_dir, set_global_seed
from FL_code.run_fl import FLConfig


DEFAULT_PROTOCOL = "wz_cancer"
DEFAULT_RECONSTRUCTIONS = PROJECT_ROOT / "_/data/federated_learning/T_RD_reconst"
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
    parser.add_argument("--name", required=True, help="Short label used in the replay records directory.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--start-round", type=int, default=14)
    parser.add_argument("--end-round", type=int, default=18)
    parser.add_argument("--clients", type=int)
    parser.add_argument(
        "--history-source",
        type=HistorySource,
        choices=tuple(HistorySource),
        default=HistorySource.RECONSTRUCTIONS,
    )
    parser.add_argument("--reconstructions", type=Path, default=DEFAULT_RECONSTRUCTIONS)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument(
        "--replace-commits-with-raw",
        action="store_true",
        help="After each simulated decode, replace the committed reconstruction history with the raw delta.",
    )
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


def _load_replay_config(config_path: Path, protocol: str, name: str, records_root: Path) -> FLConfig:
    """Load the cached FL setup and set replay-only protocol and records fields."""
    assert config_path.is_file(), f"Missing replay FL configuration: {config_path}."
    cached_cfg = FLConfig.model_validate_json(config_path.read_text())
    return FLConfig(
        **(cached_cfg.model_dump() | {
            "protocol": protocol,
            "run_name": name,
            "records_dir": records_root,
        })
    )


def _build_state_manager(cfg: FLConfig) -> StateDictManager:
    """Build the trainable-state flattener used by compression replay."""
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
            history_tensor = (
                _load_tensor(reconstructions_dir, round_id, client_id)
                if source is HistorySource.RECONSTRUCTIONS
                else _load_tensor(deltas_dir, round_id, client_id)
            )
            record = CompressionRecord(round_id, client_id, f"history_seed_{source.value}")
            protocol._recons_history.commit(
                history_tensor.to(dtype=torch.float16), record, Access.TEMPORAL_TOO
            )


def replace_latest_reconstruction(
    protocol: BaseProtocol,
    raw_delta: torch.Tensor,
    round_id: int,
    client_id: int,
) -> None:
    """Replace the history entry just committed by simulate_compression with raw delta."""
    history = protocol._recons_history
    replacement = raw_delta.detach().to(device="cpu", dtype=history.dtype)
    replaced = False
    for ledger in (history._server.get(client_id), history._temporal.get(client_id)):
        if ledger is None:
            continue
        assert ledger and ledger[-1].round_id == round_id and ledger[-1].client_id == client_id, (
            f"Latest history entry for client {client_id} is not round {round_id}."
        )
        ledger[-1] = replace(ledger[-1], tensor=replacement)
        replaced = True
    assert replaced, (
        f"simulate_compression did not commit reconstruction history for round {round_id}, "
        f"client {client_id}."
    )


def main() -> None:
    """Replay cached client-round deltas through one protocol and save compression records."""
    args = _parse_args()
    assert args.start_round <= args.end_round, "start-round must not exceed end-round."
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

    if args.replace_commits_with_raw:
        message = (
            "RAW-COMMIT EXPERIMENT: simulated decoder outputs are recorded normally, but each "
            "newly committed reconstruction history entry is replaced with the raw client delta."
        )
        warnings.warn(message, stacklevel=1)
        print(f"\n{'!' * 88}\n{message}\n{'!' * 88}\n")

    set_global_seed(args.seed)
    cfg = _load_replay_config(
        args.reconstructions / "fl_config.json",
        args.protocol,
        args.name,
        args.records_root,
    )
    clients = args.clients if args.clients is not None else cfg.num_clients
    assert clients > 0, "clients must be positive."

    state_manager = _build_state_manager(cfg)
    protocol = create_protocol(args.protocol, state_manager.get_slices())
    client_ids = range(clients)
    seed_history(
        protocol,
        args.history_source,
        args.start_round,
        client_ids,
        args.reconstructions,
        args.deltas,
    )

    _prepare_records_dir(cfg)
    output_dir = cfg.records_dir

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
            if args.replace_commits_with_raw:
                replace_latest_reconstruction(protocol, delta, round_id, client_id)

    generated_path = output_dir / "compression_records.csv"
    print(f"\nRecords written to {generated_path}")


if __name__ == "__main__":
    main()
