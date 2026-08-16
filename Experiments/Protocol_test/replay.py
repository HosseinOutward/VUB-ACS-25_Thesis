from __future__ import annotations

import argparse
from dataclasses import replace
from enum import StrEnum
import json
from pathlib import Path
import sys
import warnings

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.FL_core.codec import (
    Access,
    CompressionRecord,
    DEFAULT_PROTOCOL_CACHE_DIR,
    create_protocol,
    simulate_compression,
)
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.utils import StateDictManager, prepare_records_dir, set_global_seed
from FL_code.run_fl import FLConfig


DEFAULT_PROTOCOL = "wz_cancer"
DEFAULT_RECONSTRUCTIONS = PROJECT_ROOT / "_/data/federated_learning/T_RD_reconst"
DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_RECORDS_ROOT = Path(__file__).parent / "exp_records"
# The delta dump carries its own fl_config.json, but it was written by the pre-protocol API and no
# longer validates; this one is the FL setup that produced those deltas, minus the replay-only fields.
DEFAULT_FL_CONFIG = Path(__file__).parent / "replay_fl_config.json"


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
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument(
        "--replace-commits-with-raw",
        action="store_true",
        help="After each simulated decode, replace the committed reconstruction history with the raw delta.",
    )
    parser.add_argument(
        "--protocol-cache-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PROTOCOL_CACHE_DIR,
        help="Reusable learned-model cache independent of FL update caches and replay round selection.",
    )
    parser.add_argument(
        "--protocol-cache-overwrite",
        action="store_true",
        help="Retrain and replace compatible entries in --protocol-cache-dir.",
    )
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def load_replay_tensor(directory: Path, round_id: int, client_id: int) -> torch.Tensor:
    """Load and validate one saved client delta or reconstruction tensor."""
    path = directory / f"round_{round_id}_client_{client_id}.pt"
    assert path.is_file(), f"Missing replay tensor: {path}."
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(tensor, torch.Tensor), f"Replay file does not contain a tensor: {path}."
    assert tensor.dtype == torch.float32 and tensor.device == torch.device("cpu"), (
        f"Expected a CPU float32 tensor in {path}; got {tensor.dtype} on {tensor.device}."
    )
    return tensor


def load_replay_config(
    config_path: Path,
    protocol: str,
    name: str,
    records_root: Path,
    seed: int,
) -> FLConfig:
    """Load the cached FL setup and set replay-only protocol and records fields."""
    assert config_path.is_file(), f"Missing replay FL configuration: {config_path}."
    cached_options = json.loads(config_path.read_text())
    return FLConfig.model_validate(
        cached_options | {
            "protocol": protocol,
            "run_name": name,
            "records_dir": records_root,
            "seed": seed,
        }
    )


def build_state_manager(cfg: FLConfig) -> StateDictManager:
    """Build the trainable-state flattener used by compression replay."""
    model = initialize_model(cfg, torch.device("cpu"))
    return StateDictManager(model)


def seed_history(
    protocol: _WZProtocol,
    source: HistorySource,
    start_round: int,
    client_ids: range,
    reconstructions_dir: Path,
    deltas_dir: Path,
) -> None:
    """Recreate every bounded protocol history immediately before the first replayed round."""
    histories = protocol.histories
    history_depth = max(history.max_per_client for history in histories)
    history_dtypes = {history.dtype for history in histories}
    assert len(history_dtypes) == 1, (
        f"Protocol {protocol.protocol_name_full!r} histories use mixed dtypes: {history_dtypes}."
    )
    history_dtype = next(iter(history_dtypes))
    assert history_depth > 0, (
        f"Protocol {protocol.protocol_name_full!r} does not retain reconstruction history."
    )
    assert start_round >= history_depth, (
        f"Round {start_round} has fewer than {history_depth} preceding rounds to seed."
    )

    for round_id in range(start_round - history_depth, start_round):
        for client_id in client_ids:
            history_tensor = (
                load_replay_tensor(reconstructions_dir, round_id, client_id)
                if source is HistorySource.RECONSTRUCTIONS
                else load_replay_tensor(deltas_dir, round_id, client_id)
            )
            record = CompressionRecord(round_id, client_id, f"history_seed_{source.value}")
            for history in histories:
                history.commit(
                    history_tensor.to(dtype=history_dtype), record, Access.TEMPORAL_TOO)


def replace_latest_reconstruction(
    protocol: _WZProtocol,
    raw_delta: torch.Tensor,
    round_id: int,
    client_id: int,
) -> None:
    """Replace the history entries just committed by simulate_compression with raw delta."""
    histories = protocol.histories
    history_dtypes = {history.dtype for history in histories}
    assert len(history_dtypes) == 1, (
        f"Protocol {protocol.protocol_name_full!r} histories use mixed dtypes: {history_dtypes}."
    )
    replacement = raw_delta.detach().to(device="cpu", dtype=next(iter(history_dtypes)))
    replaced = False
    for history in histories:
        assert history.dtype == replacement.dtype
        for ledger in (history._server.get(client_id), history._temporal.get(client_id)):
            if ledger is None or not ledger or ledger[-1].round_id != round_id:
                continue
            assert ledger[-1].client_id == client_id
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
    cfg = load_replay_config(
        args.fl_config,
        args.protocol,
        args.name,
        args.records_root,
        args.seed,
    )
    cfg.protocol_cache_dir = args.protocol_cache_dir
    cfg.protocol_cache_overwrite = args.protocol_cache_overwrite
    clients = args.clients if args.clients is not None else cfg.num_clients
    assert clients > 0, "clients must be positive."
    state_manager = build_state_manager(cfg)
    protocol = create_protocol(
        args.protocol,
        state_manager.get_slices(),
        cache_dir=cfg.protocol_cache_dir,
        cache_overwrite=cfg.protocol_cache_overwrite,
    )
    client_ids = range(clients)
    seed_history(
        protocol,
        args.history_source,
        args.start_round,
        client_ids,
        args.reconstructions,
        args.deltas,
    )

    prepare_records_dir(cfg)
    output_dir = cfg.records_dir

    for round_id in range(args.start_round, args.end_round + 1):
        for client_id in client_ids:
            delta = load_replay_tensor(args.deltas, round_id, client_id)
            assert delta.numel() == state_manager.param_count, (
                f"round {round_id}, client {client_id} has {delta.numel()} values; "
                f"the configured model has {state_manager.param_count}."
            )
            print(f"Simulating round {round_id}, client {client_id} with {args.protocol}")
            simulate_compression(
                protocol, delta, client_id, round_id, state_manager, output_dir, {}, (), ()
            )
            if args.replace_commits_with_raw:
                replace_latest_reconstruction(protocol, delta, round_id, client_id)

    generated_path = output_dir / "compression_records.csv"
    print(f"\nRecords written to {generated_path}")


if __name__ == "__main__":
    main()
