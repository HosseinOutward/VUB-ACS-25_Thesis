from __future__ import annotations

import argparse
from dataclasses import replace
from enum import StrEnum
import json
from pathlib import Path
import sys
from unittest.mock import patch
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
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.wz_quantizer import WZQuantizerCancer, WZcfgQuant, new_rnn_model
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
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--save-models",
        action="store_true",
        help="Save each client-round quantizer and retrained prior under the generated records directory.",
    )
    model_group.add_argument(
        "--load-models",
        type=Path,
        metavar="RECORDS_DIR",
        help="Load saved models from a prior replay records directory instead of training them.",
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


def _load_replay_config(
    config_path: Path,
    protocol: str,
    name: str,
    records_root: Path,
    seed: int,
) -> FLConfig:
    """Load the cached FL setup and set replay-only protocol and records fields."""
    assert config_path.is_file(), f"Missing replay FL configuration: {config_path}."
    cached_cfg = FLConfig.model_validate_json(config_path.read_text())
    return FLConfig(
        **(cached_cfg.model_dump() | {
            "protocol": protocol,
            "run_name": name,
            "records_dir": records_root,
            "seed": seed,
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
        if ledger is None or not ledger or ledger[-1].round_id != round_id:
            continue
        assert ledger[-1].client_id == client_id
        ledger[-1] = replace(ledger[-1], tensor=replacement)
        replaced = True
    assert replaced, (
        f"simulate_compression did not commit reconstruction history for round {round_id}, "
        f"client {client_id}."
    )


def simulate_with_models(
    protocol: _WZProtocol,
    delta: torch.Tensor,
    client_id: int,
    round_id: int,
    state_manager: StateDictManager,
    output_dir: Path,
    checkpoint_path: Path | None,
    load_models: bool,
) -> None:
    """Simulate one replay step while optionally saving or loading its learned models."""
    if checkpoint_path is None:
        simulate_compression(
            protocol, delta, client_id, round_id, state_manager, output_dir, {}, (), ()
        )
        return

    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if load_models else None
    )
    if load_models:
        assert isinstance(checkpoint, dict), f"Invalid model checkpoint: {checkpoint_path}."

    original_train_model = WZQuantizerCancer.train_model
    original_train_prior = PriorCalculator.make_trained_prior_model
    prior_model: EncoderDecoderLayeredRNN | None = None
    prior_side_info_size: int | None = None

    def train_or_load_quantizer(
        quantizer: WZQuantizerCancer,
        x_raw: torch.Tensor,
        side_information: list[torch.Tensor] | None,
    ) -> None:
        if checkpoint is None:
            original_train_model(quantizer, x_raw, side_information)
            return
        assert checkpoint["quantizer_side_info_size"] == quantizer.side_info_size, (
            f"Quantizer side-information size mismatch in {checkpoint_path}."
        )
        if quantizer.no_side_info:
            assert not side_information
            quantizer.side_info_list_used = []
        else:
            assert side_information
            quantizer.set_side_information(side_information)
        quantizer.vector_size = x_raw.numel()
        assert checkpoint["vector_size"] == quantizer.vector_size, (
            f"Quantizer vector size mismatch in {checkpoint_path}."
        )
        quantizer.coding_model.load_state_dict(checkpoint["quantizer_model"])
        quantizer.training_prior_stage_rates = tuple(checkpoint["training_prior_stage_rates"])
        quantizer.training_prior_rate = float(checkpoint["training_prior_rate"])

    def train_or_load_prior(
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_information: torch.Tensor,
        c_cfg: WZcfgQuant,
        batch_size: int = 50_000,
    ) -> EncoderDecoderLayeredRNN:
        nonlocal prior_model, prior_side_info_size
        prior_side_info_size = side_information.shape[1]
        if checkpoint is None:
            prior_model = original_train_prior(bins, soft_codes, side_information, c_cfg, batch_size)
            return prior_model
        assert checkpoint["prior_side_info_size"] == side_information.shape[1], (
            f"Prior side-information size mismatch in {checkpoint_path}."
        )
        prior_model = new_rnn_model(
            c_cfg.num_planes,
            c_cfg.bins_per_plane,
            side_information.shape[1],
            c_cfg.marginal_loss,
            c_cfg.hidden_dim,
            c_cfg.layers,
            c_cfg.shared_heads,
        )
        prior_model.load_state_dict(checkpoint["prior_model"])
        return prior_model

    with (
        patch.object(WZQuantizerCancer, "train_model", train_or_load_quantizer),
        patch.object(PriorCalculator, "make_trained_prior_model", train_or_load_prior),
    ):
        simulate_compression(
            protocol, delta, client_id, round_id, state_manager, output_dir, {}, (), ()
        )

    if checkpoint is None:
        quantizer = protocol.last_frozen_state[client_id]["f_quant"]
        assert (
            isinstance(quantizer, WZQuantizerCancer)
            and prior_model is not None
            and prior_side_info_size is not None
        )
        torch.save(
            {
                "quantizer_model": quantizer.coding_model.state_dict(),
                "quantizer_side_info_size": quantizer.side_info_size,
                "training_prior_stage_rates": quantizer.training_prior_stage_rates,
                "training_prior_rate": quantizer.training_prior_rate,
                "vector_size": quantizer.vector_size,
                "prior_model": prior_model.state_dict(),
                "prior_side_info_size": prior_side_info_size,
            },
            checkpoint_path,
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
        args.seed,
    )
    clients = args.clients if args.clients is not None else cfg.num_clients
    assert clients > 0, "clients must be positive."
    model_config = cfg.model_dump(mode="json", exclude={"records_dir", "run_name"}) | {
        "clients": clients,
        "history_source": args.history_source.value,
        "replace_commits_with_raw": args.replace_commits_with_raw,
    }
    load_models_dir = args.load_models / "models" if args.load_models else None
    if load_models_dir is not None:
        config_path = load_models_dir / "config.json"
        assert config_path.is_file(), f"Missing saved model configuration: {config_path}."
        saved_model_config = json.loads(config_path.read_text())
        differing_fields = sorted(
            field for field in saved_model_config.keys() | model_config.keys()
            if saved_model_config.get(field) != model_config.get(field)
        )
        assert not differing_fields, (
            f"Saved models in {load_models_dir} have an incompatible configuration; "
            f"different fields: {', '.join(differing_fields)}."
        )

    state_manager = _build_state_manager(cfg)
    protocol = create_protocol(args.protocol, state_manager.get_slices())
    assert isinstance(protocol, _WZProtocol) or not (args.save_models or args.load_models), (
        "Model save/load is only supported for learned WZ protocols."
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

    _prepare_records_dir(cfg)
    output_dir = cfg.records_dir
    models_dir = load_models_dir
    if args.save_models:
        models_dir = output_dir / "models"
        models_dir.mkdir()
        (models_dir / "config.json").write_text(json.dumps(model_config, indent=2))

    for round_id in range(args.start_round, args.end_round + 1):
        for client_id in client_ids:
            delta = _load_tensor(args.deltas, round_id, client_id)
            assert delta.numel() == state_manager.param_count, (
                f"round {round_id}, client {client_id} has {delta.numel()} values; "
                f"the configured model has {state_manager.param_count}."
            )
            print(f"Simulating round {round_id}, client {client_id} with {args.protocol}")
            checkpoint_path = (
                models_dir / f"round_{round_id}_client_{client_id}.pt"
                if models_dir is not None else None
            )
            if args.load_models:
                assert checkpoint_path is not None and checkpoint_path.is_file(), (
                    f"Missing model checkpoint: {checkpoint_path}."
                )
            simulate_with_models(
                protocol,
                delta,
                client_id,
                round_id,
                state_manager,
                output_dir,
                checkpoint_path,
                args.load_models is not None,
            )
            if args.replace_commits_with_raw:
                replace_latest_reconstruction(protocol, delta, round_id, client_id)

    generated_path = output_dir / "compression_records.csv"
    print(f"\nRecords written to {generated_path}")


if __name__ == "__main__":
    main()
