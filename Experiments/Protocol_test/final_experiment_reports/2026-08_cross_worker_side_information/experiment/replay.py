"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols.

Replay cached FL deltas for the cross-worker side-information report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Importing the package registers only the report's experimental round codecs.
import experiment_codecs  # noqa: E402, F401
from sparse_replay import (  # noqa: E402
    append_sparsification_record,
    shared_support_mask,
    sparsify_with_feedback,
)
from FL_code.FL_core.codec import (  # noqa: E402
    DEFAULT_PROTOCOL_CACHE_DIR,
    create_protocol,
    simulate_compression,
)
from FL_code.FL_core.utils import prepare_records_dir, set_global_seed  # noqa: E402
from Experiments.Protocol_test.replay import (  # noqa: E402
    HistorySource,
    build_state_manager,
    load_replay_config,
    load_replay_tensor,
    replace_latest_reconstruction,
    seed_history,
)


DEFAULT_PROTOCOL = "wz_cancer"
DEFAULT_RECONSTRUCTIONS = PROJECT_ROOT / "_/data/federated_learning/T_RD_reconst"
DEFAULT_DELTAS = PROJECT_ROOT / "_/data/federated_learning/raw_client_deltas"
DEFAULT_RECORDS_ROOT = REPORT_DIR / "results" / "records"
DEFAULT_FL_CONFIG = EXPERIMENT_DIR / "replay_fl_config.json"


def parse_args() -> argparse.Namespace:
    """Parse one report replay from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Directory name below --records-root.")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--start-round", type=int, default=12)
    parser.add_argument("--end-round", type=int, default=20)
    parser.add_argument("--clients", type=int)
    parser.add_argument(
        "--history-source",
        type=HistorySource,
        choices=tuple(HistorySource),
        default=HistorySource.RAW_DELTAS,
    )
    parser.add_argument("--reconstructions", type=Path, default=DEFAULT_RECONSTRUCTIONS)
    parser.add_argument("--deltas", type=Path, default=DEFAULT_DELTAS)
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--replace-commits-with-raw", action="store_true")
    parser.add_argument(
        "--protocol-cache-dir",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PROTOCOL_CACHE_DIR,
    )
    parser.add_argument("--protocol-cache-overwrite", action="store_true")
    parser.add_argument("--sparsify-keep", type=float)
    parser.add_argument("--sparsify-shared-support", action="store_true")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    """Replay the configured client rounds and save compression records."""
    args = parse_args()
    assert args.start_round <= args.end_round
    assert Path(args.name).name == args.name and args.name not in {".", ".."}
    assert args.sparsify_keep is None or 0.0 < args.sparsify_keep <= 1.0
    assert not (args.sparsify_shared_support and args.sparsify_keep is None), (
        "--sparsify-shared-support requires --sparsify-keep.")

    if args.history_source is HistorySource.RAW_DELTAS:
        warnings.warn(
            "Lossless-history experiment: seeded history contains raw deltas.", stacklevel=1)
    if args.replace_commits_with_raw:
        warnings.warn(
            "Raw-commit experiment: decoded history is replaced with raw inputs.", stacklevel=1)

    set_global_seed(args.seed)
    cfg = load_replay_config(
        args.fl_config, args.protocol, args.name, args.records_root, args.seed)
    cfg.protocol_cache_dir = args.protocol_cache_dir
    cfg.protocol_cache_overwrite = args.protocol_cache_overwrite
    client_ids = range(args.clients if args.clients is not None else cfg.num_clients)
    assert len(client_ids) > 0
    state_manager = build_state_manager(cfg)
    protocol = create_protocol(
        args.protocol,
        state_manager.get_slices(),
        cache_dir=cfg.protocol_cache_dir,
        cache_overwrite=cfg.protocol_cache_overwrite,
    )
    seed_history(
        protocol,
        args.history_source,
        args.start_round,
        client_ids,
        args.reconstructions,
        args.deltas,
    )
    prepare_records_dir(cfg)

    residuals = (
        {client_id: torch.zeros(state_manager.param_count) for client_id in client_ids}
        if args.sparsify_keep is not None else {}
    )
    for round_id in range(args.start_round, args.end_round + 1):
        deltas = {
            client_id: load_replay_tensor(args.deltas, round_id, client_id)
            for client_id in client_ids
        }
        assert all(delta.numel() == state_manager.param_count for delta in deltas.values())
        support = (
            shared_support_mask(
                {client: delta + residuals[client] for client, delta in deltas.items()},
                args.sparsify_keep,
            )
            if args.sparsify_keep is not None and args.sparsify_shared_support else None
        )
        for client_id, delta in deltas.items():
            sent = delta
            if args.sparsify_keep is not None:
                sent, residuals[client_id] = sparsify_with_feedback(
                    delta, residuals[client_id], args.sparsify_keep, support)
            reconstructed = simulate_compression(
                protocol, sent, client_id, round_id, state_manager, cfg.records_dir, {}, (), ())
            if args.sparsify_keep is not None:
                append_sparsification_record(
                    cfg.records_dir,
                    round_id,
                    client_id,
                    args.sparsify_keep,
                    sent,
                    delta,
                    reconstructed,
                    residuals[client_id],
                )
            if args.replace_commits_with_raw:
                replace_latest_reconstruction(protocol, sent, round_id, client_id)

    print(f"Records written to {cfg.records_dir / 'compression_records.csv'}")


if __name__ == "__main__":
    main()
