from __future__ import annotations

import argparse
import os
from datetime import timedelta
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from FL_code.FL_core.dataset import DATASET_NAMES, GROUP_FIELD, force_class_coverage
from FL_code.FL_core.models import BACKBONE_NAMES
from FL_code.FL_core.utils import assert_class_coverage_before_spawn, prepare_records_dir, write_fl_config_snapshot
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import start_processes
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator

from FL_code.FL_core.codec import DEFAULT_PROTOCOL_CACHE_DIR, parse_and_validate_protocol


class FLConfig(BaseModel):
    """Federated learning configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    protocol: str = Field(description="Compression protocol to run.")
    run_name: str | None = Field(None, description="Optional name for this run.")

    model_name: str = Field(
        "resnet18",
        description="Model backbone.",
        json_schema_extra={"flags": ["--model"], "choices": list(BACKBONE_NAMES)},
    )
    dataset_name: str = Field(
        "SVHN",
        description="Dataset name.",
        json_schema_extra={"flags": ["--dataset"], "choices": list(DATASET_NAMES)},
    )
    num_classes: int | None = Field(
        None,
        description="Number of classes, derived from the loaded labels.",
        json_schema_extra={"cli": False},
    )
    image_size: int = Field(32, description="Image resize and ViT input size.")

    debug_mode: bool = Field(
        False,
        description="Use a small, quick debug configuration.",
        json_schema_extra={"flags": ["--debug-mode", "--debug_mode"]},
    )

    num_clients: int = Field(5, description="Number of federated clients.")
    num_loader_workers: int = Field(3, description="Data-loader worker processes per client.")
    data_folder: Path = Field(Path("data"), description="Dataset directory.")
    rounds: int = Field(60, description="Number of federated rounds.")
    local_epochs: int = Field(5, description="Local training epochs per round.")
    batch_size: int = Field(500, description="Local training batch size.")
    lr: float = Field(1e-3, description="Client optimizer learning rate.")
    seed: int = Field(43, description="Random seed.")
    # FedAvgM: the aggregated delta is handed to a server-side torch SGD as a pseudo-gradient
    # (FedOpt, Reddi et al. 2021). Clients stay on plain SGD, so no client-private optimizer state
    # ever leaks into what is transmitted. Zero momentum is plain FedAvg.
    server_momentum: float = Field(0.0, description="FedAvgM momentum; zero selects plain FedAvg.")
    server_nesterov: bool = Field(False, description="Use Nesterov momentum in the server SGD.")

    # Random crop and flip during local training, and label smoothing in the loss. Both are
    # stateless: they shape the data and the objective, never the client optimizer.
    augment: bool = Field(False, description="Use random image augmentation during local training.")
    label_smoothing: float = Field(0.0, description="Label-smoothing factor.")
    grad_clip_norm: float = Field(
        0.0, description="Clip local gradients to this L2 norm; 0 disables clipping.")
    # Split the training set unevenly across clients instead of by the IID stride. The mechanism
    # follows the data: datasets carrying a client id (FEMNIST writers, Sent140 users) give each
    # client a whole real group, and the rest get a Dirichlet label skew. Named for the property
    # being asked for rather than for either mechanism, since both are non-IID and neither is
    # available everywhere.
    non_iid_partition: bool = Field(
        False,
        description="Use a non-IID client partition.",
        json_schema_extra={"flags": ["--non-iid"]},
    )
    # Concentration of that label skew, for the datasets with no client ids. 0.5 is the usual
    # benchmark setting; smaller leaves each client holding fewer classes.
    dirichlet_alpha: float = Field(0.5, description="Dirichlet concentration for label skew.")
    # Step applied to the aggregated update. It lives here and never on the client: the client
    # optimizer stays plain SGD, because anything stateful or scheduled there changes the deltas
    # the experiment is trying to measure.
    server_lr: float = Field(1.0, description="Step applied to the aggregated server update.")

    recalibrate_bn: bool = Field(True, description="Recalibrate batch normalization before evaluation.")
    bn_recalib_batches: int = Field(100, description="Batches used for batch-normalization recalibration.")
    client_eval_every_n_rounds: int = Field(1, description="Client evaluation frequency in rounds.")

    # Memory and performance optimizations
    channels_last: bool = Field(True, description="Use channels-last tensor layout.")
    fused_optimizer: bool = Field(True, description="Use the fused optimizer implementation.")
    mixed_precision: bool = Field(True, description="Use automatic mixed precision.")

    training_progress_bar: bool = Field(False, description="Show local training progress bars.")
    records_dir: Path = Field(Path("records"), description="Directory for experiment records.")
    dataset_fraction: float | None = Field(None, description="Fraction of each dataset to use.")

    master_addr: str = Field("localhost", description="Distributed-process master address.")
    master_port: str = Field("29500", description="Distributed-process master port.")

    protocol_cache_dir: Path | None = Field(
        DEFAULT_PROTOCOL_CACHE_DIR,
        description="Separate cache directory for learned protocol models.",
    )
    protocol_cache_overwrite: bool = Field(False, description="Retrain compatible cached protocol models.")

    debug_save_train_data: bool = Field(False, description="Save client updates for later replay.")
    debug_data_folder: Path = Field(Path("_/data/federated_learning"), description="Debug data directory.")
    debug_save_deltas: str = Field("client_deltas", description="Subdirectory for saved client deltas.")
    debug_save_recons: str | None = Field(None, description="Subdirectory for saved reconstructions.")
    debug_save_recons_rounds: int | None = Field(None, description="Optional reconstruction-save round limit.")
    debug_load_from_saved_data: bool = Field(False, description="Reuse saved client updates when available.")

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent changing the configured protocol after configuration creation."""
        assert not (
            name == "protocol" and "protocol" in self.__dict__ and self.__dict__["protocol"] != value
        ), "FLConfig.protocol is immutable after construction."
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def validate_explicit_configuration(self) -> FLConfig:
        """Reject contradictory options that would otherwise change behavior implicitly."""
        parse_and_validate_protocol(self.protocol)
        if self.debug_mode:
            object.__setattr__(self, "num_clients", 3)
            object.__setattr__(self, "local_epochs", 1)
            if self.dataset_fraction is None:
                object.__setattr__(self, "dataset_fraction", 0.1)
        assert not (
            self.debug_save_train_data and self.debug_load_from_saved_data
        ), "debug_save_train_data and debug_load_from_saved_data cannot both be enabled."
        assert self.protocol_cache_dir is not None or not self.protocol_cache_overwrite, (
            "protocol_cache_overwrite requires protocol_cache_dir.")
        assert self.dataset_fraction is None or 0.0 < self.dataset_fraction < 1.0, (
            "dataset_fraction must be None or a value in (0.0, 1.0).")
        assert self.client_eval_every_n_rounds > 0, "client_eval_every_n_rounds must be positive."
        from FL_code.FL_core.dataset import TEXT_CHECKPOINT, TEXT_DATASETS
        from FL_code.FL_core.models import BERT_CHECKPOINT, DISTILBERT_CHECKPOINT

        text_models = {"bert": BERT_CHECKPOINT, "bert_scratch": BERT_CHECKPOINT,
                       "distilbert": DISTILBERT_CHECKPOINT,
                       "distilbert_fresh": DISTILBERT_CHECKPOINT}
        # A randomly initialized encoder takes only its architecture from the checkpoint, so it has
        # no pretrained weights a mismatched vocabulary could scramble. It needs the token *ids* to
        # stay inside its embedding table, and every uncased WordPiece checkpoint here carries the
        # same 30,522 tokens, so any of these text sets is a legal partner for it.
        vocabulary_free = {"bert_scratch"}
        if (self.dataset_name.upper() in TEXT_DATASETS
                and self.model_name.lower() in text_models
                and self.model_name.lower() not in vocabulary_free):
            wanted = TEXT_CHECKPOINT[self.dataset_name.upper()]
            assert text_models[self.model_name.lower()] == wanted, (
                f"{self.dataset_name} is tokenized with {wanted}, which {self.model_name} does not "
                f"read; pair it with the model for that checkpoint."
            )
        # WikiText-2 is word-level next-token prediction, not WordPiece classification, so it pairs
        # with the LSTM and with nothing else - and the LSTM pairs with nothing but it.
        sequence = self.dataset_name.upper() == "WIKITEXT2"
        assert (self.model_name.lower() == "lstm_lm") == sequence, (
            f"lstm_lm and WIKITEXT2 must be used together; got {self.model_name} with "
            f"{self.dataset_name}."
        )
        assert sequence or (
            (self.model_name.lower() in text_models) == (self.dataset_name.upper() in TEXT_DATASETS)
        ), (
            f"{sorted(text_models)} must be paired with {sorted(TEXT_DATASETS)}; got "
            f"{self.model_name} with {self.dataset_name}."
        )
        # Neither split mechanism applies to a sequence target: it has a vocabulary rather than a
        # label set, so there is no label distribution to skew, and it carries no client ids.
        assert not (self.non_iid_partition and sequence), (
            f"{self.dataset_name} has sequence targets, which no non-IID partition can split; "
            "drop --non-iid."
        )
        # Augmentation is a per-image crop and flip, so it has nothing to act on here and would
        # otherwise be accepted and silently ignored.
        assert not (self.augment and (sequence or self.dataset_name.upper() in TEXT_DATASETS)), (
            f"augment applies to image batches; {self.dataset_name} is tokenized text."
        )
        assert self.grad_clip_norm >= 0.0, (
            f"grad_clip_norm must be non-negative; got {self.grad_clip_norm}."
        )
        assert self.dirichlet_alpha > 0, (
            f"dirichlet_alpha must be positive; got {self.dirichlet_alpha}."
        )
        assert self.image_size > 0 and self.image_size % 4 == 0, (
            f"image_size must be a positive multiple of 4 (the ViT patch size); got {self.image_size}."
        )
        assert 0.0 <= self.server_momentum < 1.0, (
            f"server_momentum must be in [0.0, 1.0); got {self.server_momentum}."
        )
        # torch.optim.SGD refuses this pairing, but only once the server has spawned its workers.
        assert not self.server_nesterov or self.server_momentum > 0.0, (
            "server_nesterov needs a positive server_momentum."
        )
        return self


def _argparse_type(annotation: Any) -> Any:
    """Return the argparse converter for a supported FLConfig field annotation."""
    if annotation is Any:
        return str
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = tuple(argument for argument in get_args(annotation) if argument is not type(None))
        assert len(non_none) == 1, f"CLI fields must have at most one non-None type: {annotation}."
        return _argparse_type(non_none[0])
    if origin is Literal:
        literal_values = get_args(annotation)
        assert literal_values, f"Literal CLI fields must define at least one choice: {annotation}."
        return type(literal_values[0])
    if annotation in (str, int, float, Path):
        return annotation
    raise TypeError(f"Unsupported CLI field annotation: {annotation!r}.")


def _add_fl_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one CLI argument for each FLConfig field exposed as a command-line setting."""
    for name, field in FLConfig.model_fields.items():
        schema_extra = field.json_schema_extra
        assert schema_extra is None or isinstance(schema_extra, dict)
        extra: dict[str, Any] = schema_extra or {}
        if extra.get("cli", True) is False:
            continue

        flags = extra.get("flags", (f"--{name.replace('_', '-')}",))
        argument_kwargs: dict[str, Any] = {
            "dest": name,
            "help": field.description,
        }
        if field.is_required():
            argument_kwargs["required"] = True
        else:
            argument_kwargs["default"] = field.default

        annotation = field.annotation
        if annotation is bool:
            argument_kwargs["action"] = argparse.BooleanOptionalAction
        else:
            argument_kwargs["type"] = _argparse_type(annotation)

        choices = extra.get("choices")
        if choices is None and get_origin(annotation) is Literal:
            choices = get_args(annotation)
        if choices is not None:
            argument_kwargs["choices"] = choices
        parser.add_argument(*flags, **argument_kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser from FLConfig field metadata."""
    parser = argparse.ArgumentParser(
        description=FLConfig.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_fl_config_arguments(parser)
    return parser


def _worker(
    rank: int,
    world_size: int,
    cfg: FLConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor
) -> None:
    """Distributed worker with access to shared data tensors."""
    import sys
    import traceback as tb

    role = "Server" if rank == 0 else f"Client {rank - 1}"
    if cfg.debug_mode:
        assert cfg.num_clients == 3 and cfg.local_epochs == 1 and cfg.dataset_fraction is not None

    # Set up process group
    dist.init_process_group(backend="gloo", init_method='env://',
            timeout=timedelta(minutes=75), world_size=world_size, rank=rank)

    try:
        if rank == 0:
            from FL_code.FL_core.server import run_federated_server
            run_federated_server(cfg, rank, world_size, X_test, y_test)
        else:
            from FL_code.FL_core.client import run_federated_client
            run_federated_client(cfg, rank, world_size, X_train, y_train, X_test, y_test)
    except Exception:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[{role}] EXCEPTION OCCURRED", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        # Print formatted traceback with role prefix
        exc_type, exc_value, exc_tb = sys.exc_info()
        formatted_tb = tb.format_exception(exc_type, exc_value, exc_tb)

        for line in formatted_tb:
            for subline in line.rstrip().split('\n'):
                print(f"[{role}] {subline}", file=sys.stderr)

        print(f"{'='*70}\n", file=sys.stderr)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = FLConfig.model_validate(vars(args))

    if cfg.debug_mode:
        print('**************************************************')
        print('******************  DEBUG MODE  ******************')

    prepare_records_dir(cfg)

    # Set environment variables for address and port
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port

    # Precompute dataset and store in shared memory
    if cfg.dataset_fraction is not None:
        print(f"[Debug] Using {cfg.dataset_fraction*100:.1f}% of dataset for quick testing.")

    # This entry point spawns workers with features and labels only, and calls
    # `precompute_dataset_to_shared` without `max_groups`. Both halves of a group dataset therefore
    # break here, not just the non-IID one: `natural_partition` never receives the group ids, and
    # the IID twin would be drawn from every group rather than the `num_clients` largest, holding
    # far more data per client than the split it is meant to be compared against. A runner that
    # wants either has to pass `groups` and `max_groups` through itself.
    assert cfg.dataset_name.upper() not in GROUP_FIELD, (
        f"{cfg.dataset_name} carries a per-sample client id ({GROUP_FIELD[cfg.dataset_name.upper()]}) "
        "that this entry point forwards to neither the dataloaders nor the group cap; run it from a "
        "script that calls precompute_dataset_to_shared and create_dataloader directly."
    )

    from FL_code.FL_core.dataset import precompute_dataset_to_shared
    X_train, y_train, _ = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                    "train", torch.float32, cfg.dataset_fraction,
                                                    cfg.seed, cfg.image_size)
    X_test, y_test, _ = precompute_dataset_to_shared(cfg.dataset_name, cfg.data_folder,
                                                  "test", torch.float32, cfg.dataset_fraction,
                                                  cfg.seed, cfg.image_size)

    cfg.num_classes = int(max(y_train.max(), y_test.max())) + 1
    if cfg.dataset_fraction is not None:
        force_class_coverage(y_train, y_test, cfg)
    assert_class_coverage_before_spawn(y_train, y_test, cfg)

    if cfg.debug_save_recons is not None:
        write_fl_config_snapshot(cfg, cfg.debug_data_folder / cfg.debug_save_recons)

    print(f'[MAIN] {cfg.protocol}')

    # Start distributed processes with spawn semantics.
    start_processes(
        _worker,
        args=(cfg.num_clients + 1, cfg, X_train, y_train, X_test, y_test),
        nprocs=cfg.num_clients + 1,
        join=True,
        start_method="spawn",
    )
