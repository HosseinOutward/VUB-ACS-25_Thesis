from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import ast
import gzip
import json
import pickle
from pathlib import Path
import random
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm


from .models import FLModelTemplate, initialize_model
from .dataset import create_dataloader

if TYPE_CHECKING:
    from FL_code.run_fl import FLConfig

# Canonical key order for evaluate() results. Client and server exchange worker
# metrics as a flat tensor, so both sides must agree on this exact order.
EVAL_METRIC_KEYS: tuple[str, ...] = ("loss", "acc", "f1", "auc")


@dataclass(frozen=True, slots=True)
class ParsedConfigurableName:
    """A pipe-delimited selector produced by configuration and consumed by factories."""

    name: str
    options: dict[str, Any] | None
    option_tokens: tuple[str, ...]


def parse_configurable_name(raw_name: str, field_name: str) -> ParsedConfigurableName:
    """Parse a selector like name|flag|key=value into its name and option dictionary."""
    assert isinstance(raw_name, str), f"{field_name} must be a string; got {type(raw_name).__name__}."
    assert raw_name, f"{field_name} must be a non-empty string."
    assert raw_name == raw_name.strip(), (
        f"{field_name}={raw_name!r} must not contain leading or trailing whitespace.")

    parts = raw_name.split("|")
    name = parts[0]
    assert name, f"{field_name}={raw_name!r} must start with a non-empty name."
    assert name == name.strip(), f"{field_name} selector {name!r} must not contain surrounding whitespace."

    options: dict[str, Any] = {}
    for token in parts[1:]:
        assert token, f"{field_name}={raw_name!r} contains an empty option token."
        assert token == token.strip(), (
            f"{field_name} option {token!r} must not contain surrounding whitespace.")
        key, sep, value = token.partition("=")
        assert key, f"{field_name} option {token!r} must have a non-empty key."
        assert key not in options, f"{field_name} option {key!r} appears more than once."
        options[key] = ast.literal_eval(value) if sep else True
        assert options[key] is True or options[key] != "", (
            f"{field_name} option {token!r} must not have an empty value.")

    options = None if not options else options
    return ParsedConfigurableName(name=name, options=options, option_tokens=tuple(parts[1:]))



def create_training_progress_bar(
    iterable_or_total: Any,
    desc: str,
    disable: bool = False,
    leave: bool = False,
    position: int = 0
) -> tqdm:
    # try:
    #     ipython = get_ipython()  # type: ignore
    # except NameError:
    return tqdm(
        total=iterable_or_total,
        disable=disable,
        desc=desc,
        leave=leave,
        position=position,
        file=sys.stderr,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id: int = 0) -> torch.device:
    """Get GPU device with optimizations enabled."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        return device
    print(' ********* USING CPU ********* ')
    return torch.device("cpu")


def setup_fl_worker(
    cfg: Any,
    role: str,
    device_id: int,
    X_train: torch.Tensor | None,
    y_train: torch.Tensor | None,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    client_id: int | None = None,
    num_clients: int | None = None
) -> tuple[FLModelTemplate, torch.device, DataLoader, DataLoader | None, StateDictManager]:
    """
    Common setup for FL server/client workers.

    Returns:
        model, device, test_loader, train_loader (None for server), sd_manager
    """
    device = get_device(device_id)
    print(f"[{role}] Device: {device}")

    model = initialize_model(cfg, device)
    test_loader = create_dataloader(X_test, y_test, cfg, device, is_train=False)

    train_loader = None
    has_training_tensors = X_train is not None or y_train is not None
    has_client_partition = client_id is not None or num_clients is not None

    if has_training_tensors:
        if X_train is None or y_train is None:
            raise ValueError(f"{role}: X_train and y_train must be provided together.")
        if client_id is None or num_clients is None:
            raise ValueError(f"{role}: training tensors require client_id and num_clients.")
        train_loader = create_dataloader(
            X_train, y_train, cfg, device, is_train=True,
            client_id=client_id, num_clients=num_clients
        )
    elif has_client_partition:
        raise ValueError(f"{role}: client_id and num_clients require training tensors.")

    sd_manager = StateDictManager(model)

    return model, device, test_loader, train_loader, sd_manager


def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """Format metrics dict into a readable string."""
    p = f"{prefix} " if prefix else ""
    return f"{p}Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}"


@torch.no_grad()
def recalibrate_batchnorm(model: FLModelTemplate, loader: DataLoader, max_batches: int = 50) -> None:
    """Recalibrate BatchNorm running statistics (critical for FL)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()

    model.train()
    for i, (x, _) in enumerate(loader):
        x: torch.Tensor
        if i >= max_batches:
            break
        x = x.to(model.device, non_blocking=True)
        if x.ndim == 4 and model.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        model(x)
    model.eval()  # Set back to eval mode after recalibration
    torch.cuda.empty_cache()  # Clear cache to free memory after recalibration


def evaluate(model: FLModelTemplate, loader: DataLoader) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(model.device, non_blocking=True)
            y = y.to(model.device, non_blocking=True)

            if x.ndim == 4 and model.cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)

            # Get predictions and probabilities
            probs_P = torch.softmax(logits, dim=1)
            y_preds = logits.argmax(dim=1)

            # Move to CPU and store
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs_P.cpu().numpy())
            all_preds.append(y_preds.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    # -- metrics --
    # Calculate metrics using scikit-learn
    avg_loss = total_loss / len(y_true)
    accuracy = accuracy_score(y_true, y_pred)

    # F1 score (macro average for multiclass)
    num_classes = y_probs.shape[1]
    f1 = f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary', zero_division=0)

    # AUC calculation
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    else:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    metrics = {
        "loss": avg_loss,
        "acc": accuracy,
        "f1": f1,
        "auc": auc
    }
    assert tuple(metrics) == EVAL_METRIC_KEYS, (
        f"evaluate() metric keys {tuple(metrics)} diverged from EVAL_METRIC_KEYS {EVAL_METRIC_KEYS}; "
        "client/server exchange worker metrics positionally and would silently mislabel them."
    )
    return metrics


class StateDictManager:
    def __init__(self, model: nn.Module) -> None:
        self.keys: list[str] = []
        self.shapes: list[torch.Size] = []
        self.numels: list[int] = []

        # Extract trainable parameters metadata
        for key, param in model.named_parameters():
            if param.requires_grad:
                self.keys.append(key)
                self.shapes.append(param.size())
                self.numels.append(param.numel())

        self.param_count = sum(self.numels)

        # State entries the workers never train via backprop: BatchNorm buffers
        # (running_mean, running_var, num_batches_tracked) plus any frozen params.
        # These are never transmitted in either direction; each node recomputes
        # them from its own local data (BN recalibration). Kept explicit here,
        # unused for now, until their handling is decided.
        tracked = set(self.keys)
        self.non_backprop_state_keys: list[str] = [key for key in model.state_dict() if key not in tracked]

    def flatten(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_list = []
        for key in self.keys:
            param = state_dict[key]
            flat_list.append(param.cpu().detach().reshape(-1))
        return torch.cat(flat_list)

    def unflatten(self, flat_vector: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        assert flat_vector.numel() == self.param_count, (
            f"Flat trainable vector has {flat_vector.numel()} values; expected {self.param_count}."
        )
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        offset = 0

        for key, shape, numel in zip(self.keys, self.shapes, self.numels):
            param_flat = flat_vector[offset:offset + numel]
            state_dict[key] = param_flat.view(shape)
            offset += numel

        return state_dict

    def load_trainable_state(
        self,
        model: nn.Module,
        trainable_state: dict[str, torch.Tensor]
    ) -> None:
        """Load exactly the trainable parameters tracked by this manager."""
        parameters = dict(model.named_parameters())
        expected_keys = set(self.keys)
        received_keys = set(trainable_state.keys())
        if received_keys != expected_keys:
            missing = sorted(expected_keys - received_keys)
            extra = sorted(received_keys - expected_keys)
            raise KeyError(f"Trainable state keys mismatch. Missing={missing}, extra={extra}.")

        with torch.no_grad():
            for key in self.keys:
                target = parameters[key]
                value = trainable_state[key]
                assert value.shape == target.shape, (
                    f"Trainable state shape mismatch for {key}: "
                    f"got {tuple(value.shape)}, expected {tuple(target.shape)}."
                )
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def get_slices(self) -> list[slice]:
        slices = []
        offset = 0
        for numel in self.numels:
            slices.append(slice(offset, offset + numel))
            offset += numel
        return slices

    def clone_trainable(self, state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, state_dict[k].cpu().detach().clone()) for k in self.keys)

    def compute_delta(
        self,
        new_state: dict[str, torch.Tensor],
        old_state: dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((k, new_state[k] - old_state[k]) for k in self.keys)

    def apply_delta_inplace(
        self,
        model: nn.Module,
        delta: dict[str, torch.Tensor]
    ) -> None:
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for key in self.keys:
                target = parameters[key]
                value = delta[key]
                assert value.shape == target.shape, (
                    f"Delta shape mismatch for {key}: "
                    f"got {tuple(value.shape)}, expected {tuple(target.shape)}."
                )
                target.add_(value.to(device=target.device, dtype=target.dtype))


def _prepare_records_dir(cfg: FLConfig) -> None:
    """Create the numbered run directory and save configuration snapshots."""
    run_name = cfg.run_name if cfg.run_name is not None else cfg.protocol

    records_root = Path(cfg.records_dir)
    records_root.mkdir(exist_ok=True, parents=True)
    existing_names = {path.name for path in records_root.iterdir()}
    run_num = 1
    while any(name == f"run{run_num}" or name.startswith(f"run{run_num}_") for name in existing_names):
        run_num += 1

    run_dir = records_root / f"run{run_num}_{run_name}"
    run_dir.mkdir()
    cfg.records_dir = run_dir

    write_fl_config_snapshot(cfg, run_dir)


def write_fl_config_snapshot(cfg: FLConfig, save_dir: Path) -> None:
    """Write the FL configuration snapshot used by records and debug replay."""
    save_dir.mkdir(exist_ok=True, parents=True)
    with (save_dir / "fl_config.json").open("w") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2)


def assert_debug_fl_config_matches(cfg: FLConfig, save_dir: Path) -> None:
    """Assert that saved debug data was produced with the same non-protocol FL configuration."""
    config_path = save_dir / "fl_config.json"
    with config_path.open() as f:
        saved_config = json.load(f)

    # Fields that cannot affect the saved training data, so replays may differ in them.
    ignored_fields = {
        "protocol",
        "codec",
        "run_name",
        "rounds",
        "records_dir",
        "master_port",
        "master_addr",
        "num_loader_workers",
        "training_progress_bar",
        "debug_save_train_data",
        "debug_load_from_saved_data",
        "debug_data_folder",
        "debug_save_deltas",
        "debug_save_recons",
    }
    unknown_fields = ignored_fields - (type(cfg).model_fields.keys() | {"codec"})
    assert not unknown_fields, f"Unknown debug config ignored field(s): {tuple(sorted(unknown_fields))}."

    saved_config = {field: value for field, value in saved_config.items() if field not in ignored_fields}
    current_config = cfg.model_dump(mode="json", exclude=ignored_fields)

    assert saved_config == current_config, (
        f"Debug data in {save_dir} was saved with a different FLConfig: "
        f"{_format_config_mismatch(saved_config, current_config)}"
    )


def _format_config_mismatch(saved_config: dict[str, Any], current_config: dict[str, Any]) -> str:
    differing_fields = sorted(
        field
        for field in saved_config.keys() | current_config.keys()
        if saved_config.get(field) != current_config.get(field)
    )
    return "; ".join(
        f"{field}: saved={saved_config.get(field)!r}, current={current_config.get(field)!r}"
        for field in differing_fields
    )


def _assert_class_coverage_before_spawn(y_train: torch.Tensor, y_test: torch.Tensor, cfg: FLConfig) -> None:
    """Assert every evaluated split has all classes required by AUC metrics."""
    assert cfg.num_classes is not None, "cfg.num_classes must be set before class-coverage validation."
    expected_classes = set(range(cfg.num_classes))
    test_classes = set(torch.unique(y_test).tolist())
    assert test_classes == expected_classes, (
        f"Test split classes do not match model classes: got {sorted(test_classes)}, "
        f"expected {sorted(expected_classes)}."
    )

    perm = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(cfg.seed))
    for client_id in range(cfg.num_clients):
        client_classes = set(torch.unique(y_train[perm[client_id::cfg.num_clients]]).tolist())
        assert client_classes == expected_classes, (
            f"Client {client_id} train split classes do not match model classes: "
            f"got {sorted(client_classes)}, expected {sorted(expected_classes)}."
        )


def _get_obj_storage_size(obj: Any) -> float:
    """Return the in-memory payload size for tensor-like compression objects in MB."""
    return _obj_storage_bytes(obj) / (1024 ** 2)


def _obj_storage_bytes(obj: Any) -> int | float:
    res: int | float | None = None
    if isinstance(obj, torch.Tensor):
        res = obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        res = obj.nbytes
    elif isinstance(obj, (list, tuple)):
        res = sum(_obj_storage_bytes(x) for x in obj)
    elif isinstance(obj, Mapping):
        res = sum(_obj_storage_bytes(v) for v in obj.values())
    elif hasattr(obj, '_dtype') and hasattr(obj, '__len__'):
        res = len(obj) * (obj._dtype.bitwidth // 8)
    elif isinstance(obj, bytes):
        res = len(obj)
    elif obj is None:
        res = 1
    assert res is not None
    return res

def make_serializable(item: Any) -> Any:
    """Normalize payload values that pickle should not store as-is."""
    if isinstance(item, torch.Tensor):
        assert item.device.type == 'cpu'
        return item
    if hasattr(item, '_dtype') and hasattr(item, '__len__'):
        return np.array(item, dtype=np.dtype(str(item._dtype)))
    if isinstance(item, (np.integer, np.floating)):
        return item.item()
    if isinstance(item, OrderedDict):
        return OrderedDict((key, make_serializable(value)) for key, value in item.items())
    if isinstance(item, Mapping):
        return {key: make_serializable(value) for key, value in item.items()}
    # NamedTuples (e.g. PreprocessMetadata) must survive the round trip as their own type.
    if isinstance(item, tuple) and hasattr(item, '_fields'):
        return type(item)(*(make_serializable(x) for x in item))
    if isinstance(item, (list, tuple)):
        return [make_serializable(x) for x in item]
    return item


def compress_data_list(data_list: Any) -> bytes:
    """Compress data using pickle and gzip."""
    return gzip.compress(pickle.dumps(
        make_serializable(data_list), protocol=pickle.HIGHEST_PROTOCOL), compresslevel=1,)


def decompress_data_list(compressed_data: bytes) -> Any:
    """Decompress data."""
    return pickle.loads(gzip.decompress(compressed_data))
