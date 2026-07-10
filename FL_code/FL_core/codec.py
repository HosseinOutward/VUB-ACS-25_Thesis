from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import csv
import tempfile
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import numpy as np

from FL_code.FL_core.utils import (
    ParsedConfigurableName,
    _get_obj_storage_size,
    compress_data_list,
    decompress_data_list,
    parse_configurable_name,
)

if TYPE_CHECKING:
    from .utils import StateDictManager


# --- History management --- #
class Access(Enum):
    """Which reconstructed-gradient history entries a process may inspect."""
    NONE = "none"
    TEMPORAL = "temporal"
    SERVER = "server"
    BOTH = "both"

@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One reconstructed delta committed by the codec after a successful decode."""
    tensor: torch.Tensor
    round_id: int
    client_id: int
    access: Access
    round_name: str | None


class ReconstructionHistory:
    """Owns reconstructed-gradient history and enforces decoder-side visibility."""

    def __init__(self, max_per_client: int | None, routine_accesses: Sequence[Access]) -> None:
        assert max_per_client != 0, 'to disable reconstruction history, set max_per_client to None'
        self.max_per_client: int = max_per_client if max_per_client else 0
        self._server: dict[int, list[HistoryEntry]] = {}
        self._temporal: dict[int, list[HistoryEntry]] = {}

        self._keep_server: bool = False
        self._keep_temporal: bool = False
        self.set_future_commit_access(tuple(routine_accesses))

        self.frozen: bool = self.max_per_client == 0

    def freeze(self) -> None:
        """Prevent further commits to the reconstruction history."""
        self.frozen = True

    def set_future_commit_access(self, routine_accesses: Sequence[Access]) -> None:
        """Discard ledgers that the repeating routine phase will never read."""
        routine_plan = tuple(routine_accesses)
        assert all(isinstance(access, Access) for access in routine_plan), (
            "Routine history access plan must contain Access values.")

        self._keep_server = any(access in (Access.SERVER, Access.BOTH) for access in routine_plan)
        self._server = {} if not self._keep_server else self._server
        self._keep_temporal = any(access in (Access.TEMPORAL, Access.BOTH) for access in routine_plan)
        self._temporal = {} if not self._keep_temporal else self._temporal

    def copy(self) -> ReconstructionHistory:
        """Return a state copy that reuses the same immutable history entries."""
        copied = type(self).__new__(type(self))
        copied.max_per_client = self.max_per_client
        copied._server = {
            client_id: entries.copy()
            for client_id, entries in self._server.items()
        }
        copied._temporal = {
            client_id: entries.copy()
            for client_id, entries in self._temporal.items()
        }
        copied._keep_server = self._keep_server
        copied._keep_temporal = self._keep_temporal
        copied.frozen = self.frozen
        return copied

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Commit a reconstructed tensor to every ledger allowed to retain it."""
        assert not self.frozen, "ReconstructionHistory is frozen and cannot be modified."
        assert reconst.dtype == torch.float16 and reconst.device == torch.device("cpu")
        if access is Access.NONE:
            return

        entry = HistoryEntry(
            tensor=reconst,
            round_id=record.round_id,
            client_id=record.client_id,
            access=access,
            round_name=record.round_name_full,
        )
        if self._keep_server:
            self._add_entry(self._server, entry)
        if access in (Access.TEMPORAL, Access.BOTH) and self._keep_temporal:
            self._add_entry(self._temporal, entry)

    def view(self, access: Access, client_id: int) -> dict[int, tuple[HistoryEntry, ...]]:
        """Return the history entries visible to the requested process."""
        assert access is not Access.BOTH, "Access.BOTH is not a valid view policy."

        if access is Access.NONE:
            return dict()

        if access is Access.TEMPORAL:
            assert client_id in self._temporal, (
                f"No temporal reconstruction history exists for client {client_id}.")
            return {client_id: tuple(self._temporal[client_id])}

        assert access is Access.SERVER, f"Unknown access policy: {access!r}."
        entries: dict[int, tuple[HistoryEntry, ...]] = {
            client_id: tuple(ledger)
            for client_id, ledger in self._server.items()
        }
        return entries

    def _add_entry(self, history: dict[int, list[HistoryEntry]], entry: HistoryEntry) -> None:
        ledger = history.setdefault(entry.client_id, [])
        ledger.append(entry)
        if len(ledger) > self.max_per_client:
            ledger.pop(0)


# --- Record Row Data --- #
class CompressionRecord:
    """Base metrics row for one client-round, produced by a concrete round codec."""

    round_id: int
    client_id: int

    round_name_full: str
    protocol_name_full: str | None = None

    model_size: int | None = None
    encode_seconds: float | None = None
    decode_seconds: float | None = None
    final_bytes: float | None = None
    global_eval_metrics: dict[str, float] | None = None
    worker_eval_metrics: dict[str, dict[str, float]] | None = None

    mse: float | None = None
    mape: float | None = None
    mspe_sqrt: float | None = None
    w_mean_of_vec: float | None = None
    wmape: float | None = None
    wmspe_sqrt: float | None = None

    def __init__(
        self,
        round_id: int,
        client_id: int,
        round_name_full: str,
    ) -> None:
        self.round_id: int = round_id
        self.client_id: int = client_id
        self.round_name_full: str = round_name_full

    def to_dict(self) -> dict[str, Any]:
        """Convert record attributes to a flat CSV-ready dictionary."""
        skipped_fields = {
            "global_eval_metrics", "worker_eval_metrics"}
        result: dict[str, Any] = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and key not in skipped_fields
        }
        # Add global eval metrics with prefix
        if self.global_eval_metrics is not None:
            for key, value in self.global_eval_metrics.items():
                result[f'global_eval_{key}'] = value

        # Add worker eval metrics with split prefix (e.g., train_loss, test_acc)
        if self.worker_eval_metrics is not None:
            for split, metrics in self.worker_eval_metrics.items():
                for metric_key, metric_value in metrics.items():
                    result[f'{split}_{metric_key}'] = metric_value

        return result

    def update_record_fields(self, **updates: Any) -> None:
        """Update unset record fields with known non-null values."""
        unknown_fields = {field_name for field_name in updates if not hasattr(self, field_name)}
        assert not unknown_fields, f"Unknown record fields: {tuple(unknown_fields)}."
        for field_name, value in updates.items():
            assert getattr(self, field_name) is None, (
                f"CompressionRecord.{field_name} is already set before update.")
            assert value is not None, f"CompressionRecord.{field_name} replacement must not be None."
            setattr(self, field_name, value)

    def append_record_to_csv(self, save_dir: Path) -> None:
        """Append record to CSV file, expanding the header when new metrics appear."""
        save_dir.mkdir(exist_ok=True, parents=True)

        csv_file = save_dir / "compression_records.csv"
        record_dict = self.to_dict()
        fieldnames = list(record_dict)
        rows_to_rewrite = None

        if csv_file.exists():
            with csv_file.open(newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or ())
                new_fieldnames = [key for key in record_dict if key not in fieldnames]
                if new_fieldnames:
                    fieldnames += new_fieldnames
                    rows_to_rewrite = [*reader, record_dict]

        if rows_to_rewrite is not None:
            with tempfile.NamedTemporaryFile(
                'w', newline='', dir=save_dir, prefix=f".{csv_file.stem}.",
                suffix=".tmp", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_rewrite)
            temp_path.replace(csv_file)
            return

        with csv_file.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record_dict)


# --- Round Codecs and Protocols --- #
class BaseRoundCodec:
    """One-round compressor selected by a protocol schedule."""

    record_class: ClassVar[type[CompressionRecord]]
    round_name: ClassVar[str]
    can_decode_where: ClassVar[Access]
    round_name_full: str

    def __init_subclass__(cls) -> None:
        """Require concrete round codecs to declare their record type and round acronym."""
        super().__init_subclass__()
        assert issubclass(cls.record_class, CompressionRecord)
        assert cls.round_name
        assert isinstance(cls.can_decode_where, Access)

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        **protocol_inputs: Any,
    ) -> None:
        self.round_name_full = round_name_full
        if cfg_options:
            self.validate_cfg(**cfg_options)
            self.options_to_config(**cfg_options)

    def create_r_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create this round codec's metrics record for one client-round compression."""
        if not issubclass(self.record_class, CompressionRecord):
            raise TypeError(
                "Don't use CompressionRecord directly, override create_r_record() in a concrete round codec to return a subclass.")
        return self.record_class(
            round_id=round_id,
            client_id=client_id,
            round_name_full=self.round_name_full,
        )

    def options_to_config(self, **options: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def validate_cfg(**options: Any) -> None:
        raise NotImplementedError

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        raise NotImplementedError

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        raise NotImplementedError


class BaseProtocol:
    """Protocol schedule that selects the round codec used for each training round."""

    warmup_round_codecs: ClassVar[tuple[str, ...]]
    routine_round_codecs: ClassVar[tuple[str, ...]]
    protocol_name: ClassVar[str]
    protocol_name_full: str
    _round_codec_classes: dict[str, type[BaseRoundCodec]]
    max_per_client_recons_history: ClassVar[int | None]
    _recons_history: ReconstructionHistory

    def __init_subclass__(cls) -> None:
        """Require concrete protocols to declare their warmup and routine round codec plans."""
        super().__init_subclass__()
        assert isinstance(cls.warmup_round_codecs, tuple)
        assert isinstance(cls.routine_round_codecs, tuple)
        assert cls.routine_round_codecs
        assert cls.protocol_name
        assert cls.max_per_client_recons_history != 0

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None
    ) -> None:
        self.protocol_name_full = protocol_name_full or type(self).protocol_name
        if options:
            self.options_to_config(**options)
        self._round_codec_classes = self._validate_round_codecs()
        self._recons_history = ReconstructionHistory(
            max_per_client=self.max_per_client_recons_history,
            routine_accesses=[h.can_decode_where for h in self._round_codec_classes.values()]
        )

    def options_to_config(self, **options: Any) -> Any:
        raise NotImplementedError

    def _get_curr_round_codec_name(self, round_id: int) -> tuple[type[BaseRoundCodec], ParsedConfigurableName, str]:
        if round_id < len(self.warmup_round_codecs):
            round_name_full = self.warmup_round_codecs[round_id]
        else:
            routine_idx = (round_id - len(self.warmup_round_codecs)) % len(self.routine_round_codecs)
            round_name_full = self.routine_round_codecs[routine_idx]
        parsed = parse_configurable_name(round_name_full, "round codec")

        rc_class = self._round_codec_classes[round_name_full]
        return rc_class, parsed, round_name_full

    def create_round_codec(
        self,
        rc_class: type[BaseRoundCodec],
        parsed: ParsedConfigurableName,
    ) -> BaseRoundCodec:
        raise NotImplementedError

    def _validate_round_codecs(self) -> dict[str, type[BaseRoundCodec]]:
        round_codec_classes: dict[str, type[BaseRoundCodec]] = {}
        for round_name_full in set(self.warmup_round_codecs + self.routine_round_codecs):
            parsed = parse_configurable_name(round_name_full, "round codec")
            round_codec_class = _single_named_subclass(BaseRoundCodec, "round_name", parsed.name)
            if parsed.options:
                round_codec_class.validate_cfg(**parsed.options)
            round_codec_classes[round_name_full] = round_codec_class
        return round_codec_classes


def create_protocol(protocol_name: str, sd_slices: Sequence[slice] | None = None) -> BaseProtocol:
    """Create a validated protocol schedule."""
    from FL_code.cancer_protocol import NewCancer
    parsed = parse_configurable_name(protocol_name, "protocol")
    return _single_named_subclass(BaseProtocol, "protocol_name", parsed.name)(parsed.options, protocol_name, sd_slices=sd_slices)


def _single_named_subclass(base_class: type, name_attr: str, name: str) -> type:
    def all_subclasses(parent: type) -> list[type]:
        return [
            subclass
            for direct_subclass in parent.__subclasses__()
            for subclass in (direct_subclass, *all_subclasses(direct_subclass))
        ]

    matches = [
        subclass for subclass in all_subclasses(base_class)
        if getattr(subclass, name_attr) == name]
    assert len(matches) == 1, (
        f"Expected one {base_class.__name__} subclass with {name_attr}={name!r}; got {len(matches)}.")
    return matches[0]


def parse_and_validate_protocol(protocol_name: str) -> str:
    """Validate a configured protocol name and return it unchanged."""
    create_protocol(protocol_name)
    return protocol_name


def record_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """Return distortion metrics for one reconstructed delta."""
    assert original.shape == reconstructed.shape, (
        f"Reconstruction shape mismatch: got {tuple(reconstructed.shape)}, expected {tuple(original.shape)}.")
    error = reconstructed - original
    mse = torch.mean(error ** 2).item()
    w_mean_of_vec = torch.abs(original).mean().item()
    assert w_mean_of_vec > 0, "Weight Gradients are suspiciously all zero."
    metrics: dict[str, float] = {
        "mse": mse,
        "mape": torch.mean(torch.abs(error) / (torch.abs(original) + 1e-8)).item() * 100,
        "mspe_sqrt": torch.sqrt(torch.mean(error ** 2 / (original ** 2 + 1e-8))).item() * 100,
        "w_mean_of_vec": w_mean_of_vec,
        "wmape": torch.mean(torch.abs(error)).item() / w_mean_of_vec * 100,
        "wmspe_sqrt": float(np.sqrt(mse)) / w_mean_of_vec * 100,
    }
    return metrics


def simulate_compression(
    protocol: BaseProtocol, delta_vec: torch.Tensor, client_id: int, round_id: int,
    sd_manager: StateDictManager, save_dir: Path,
    server_eval_metrics: dict[str, float], worker_eval_metrics: Sequence[float],
    metric_keys: Sequence[str]
) -> torch.Tensor:
    """Simulate client encoding and server decoding for one flattened delta."""
    model_size = sd_manager.param_count
    num_metrics = len(metric_keys)
    expected_len = num_metrics * 2
    assert len(worker_eval_metrics) == expected_len

    round_codec = protocol.create_round_codec(round_id, client_id)

    record = round_codec.create_r_record(round_id, client_id)

    start_time = time.perf_counter()
    payload = round_codec.encode(delta_vec, record)
    encode_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    reconstructed = round_codec.decode(payload, record)
    decode_seconds = time.perf_counter() - start_time
    assert reconstructed.numel() == model_size
    if not protocol._recons_history.frozen:
        protocol._recons_history.commit(
            reconstructed.detach().to(device="cpu", dtype=torch.float16),
            record,
            round_codec.can_decode_where,
        )

    record.update_record_fields(
        protocol_name_full=protocol.protocol_name_full,
        model_size=model_size,
        global_eval_metrics=server_eval_metrics,
        worker_eval_metrics={
            'train': {key: worker_eval_metrics[i] for i, key in enumerate(metric_keys)},
            'test': {key: worker_eval_metrics[i + num_metrics] for i, key in enumerate(metric_keys)},
        },
        final_bytes=_get_obj_storage_size(payload),
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
    )

    recons_error_and_metric = record_reconstruction_metrics(delta_vec, reconstructed)
    record.update_record_fields(**recons_error_and_metric)

    record.append_record_to_csv(save_dir)

    return reconstructed


# --- Demo Round Codec and Protocol --- #
class RCDemo(BaseRoundCodec):
    """Round codec that serializes raw float32 deltas with the shared gzip transport."""

    round_cfg: bool

    class RecordDemo(CompressionRecord):
        comp_report: int | None = None
        random_field: int | None = None
    record_class = RecordDemo
    round_name = "DemoRC"
    can_decode_where = Access.BOTH
    frozen_history: ReconstructionHistory

    def __init__(
        self,
        options: Mapping[str, Any] | None,
        round_name_full: str,
        frozen_history: ReconstructionHistory,
    ) -> None:
        super().__init__(options, round_name_full)
        self.frozen_history = frozen_history

    def create_r_record(self, round_id: int, client_id: int) -> RCDemo.RecordDemo:
        """Create a demo metrics record with a random demo-only field."""
        raw_record = super().create_r_record(round_id, client_id)
        assert isinstance(raw_record, RCDemo.RecordDemo)
        raw_record.random_field = int(np.random.randint(0, 100))
        return raw_record

    def options_to_config(self, compReport: bool) -> None:
        """Set whether demo records include the compression-report metric."""
        self.round_cfg = compReport

    @staticmethod
    def validate_cfg(compReport: bool) -> None:
        assert isinstance(compReport, bool), "DemoRC compReport option must be a boolean."

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> bytes:
        """Encode one raw float32 delta into f16, pickle then gzip."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        payload = compress_data_list(delta_vec.to(torch.float16))
        if self.round_cfg:
            record.update_record_fields(comp_report=1)
        return payload

    def decode(self, payload: bytes, record: CompressionRecord) -> torch.Tensor:
        """Decode one raw float32 delta from pickle plus gzip."""
        reconstructed = decompress_data_list(payload)
        return reconstructed


class DemoProtocol(BaseProtocol):
    """Protocol schedule that repeatedly uses the raw demo round codec."""
    warmup_round_codecs: ClassVar[tuple[str, ...]] = ()
    routine_round_codecs: ClassVar[tuple[str, ...]] = ("DemoRC|compReport",)
    protocol_name: ClassVar[str] = "demo"
    max_per_client_recons_history: ClassVar[int | None] = 10

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        rc_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        frozen_history = self._recons_history.copy()
        frozen_history.freeze()
        return rc_class(parsed.options, round_name_full, frozen_history=frozen_history)
