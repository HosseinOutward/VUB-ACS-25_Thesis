from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import csv
import tempfile
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import numpy as np

from FL_code.FL_core.utils import _get_obj_storage_size, compress_data_list, decompress_data_list

if TYPE_CHECKING:
    from .utils import StateDictManager


# --- History management --- #
class Access(Enum):
    """Which reconstructed-gradient history entries a process may inspect."""
    NONE = "none"
    TEMPORAL = "temporal"
    SERVER = "server"

@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One reconstructed delta committed by the codec after a successful decode."""
    tensor: torch.Tensor
    round_id: int
    client_id: int
    access: Access
    round_type: str | None


class ReconstructionHistory:
    """Owns reconstructed-gradient history and enforces decoder-side visibility."""

    def __init__(self, max_per_client: int) -> None:
        self.max_per_client: int = max_per_client
        self._server: dict[int, list[HistoryEntry]] = {}
        self._temporal: dict[int, list[HistoryEntry]] = {}
        self._keep_server: bool = True
        self._keep_temporal: bool = True

    def finish_warmup(self, routine_accesses: Sequence[Access]) -> None:
        """Discard ledgers that the repeating routine phase will never read."""
        routine_plan = tuple(routine_accesses)
        assert all(isinstance(access, Access) for access in routine_plan), (
            "Routine history access plan must contain Access values.")

        self._keep_server = Access.SERVER in routine_plan
        self._keep_temporal = Access.TEMPORAL in routine_plan

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Commit a reconstructed tensor to every ledger allowed to retain it."""
        assert reconst.dtype == torch.float16 and reconst.device == torch.device("cpu")
        if access is Access.NONE:
            return

        entry = HistoryEntry(
            tensor=reconst,
            round_id=record.round_id,
            client_id=record.client_id,
            access=access,
            round_type=getattr(record, "round_type", None),
        )
        if self._keep_server:
            self._add_entry(self._server, entry)
        if access is Access.TEMPORAL and self._keep_temporal:
            self._add_entry(self._temporal, entry)

    def view(self, access: Access, record: CompressionRecord) -> tuple[HistoryEntry, ...]:
        """Return the history entries visible to the requested process."""
        if access is Access.NONE:
            return ()

        if access is Access.TEMPORAL:
            assert record.client_id in self._temporal, (
                f"No temporal reconstruction history exists for client {record.client_id}.")
            return tuple(self._temporal[record.client_id])

        assert access is Access.SERVER, f"Unknown access policy: {access!r}."
        entries = [entry for ledger in self._server.values() for entry in ledger]
        return tuple(sorted(entries, key=lambda entry: (entry.round_id, entry.client_id)))

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

    round_acronym: ClassVar[str]
    round_cfg_acronym: str | None
    protocol_name: str | None = None

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

    def __init_subclass__(cls) -> None:
        """Require concrete record classes to declare their round acronym."""
        super().__init_subclass__()
        assert hasattr(cls, "round_acronym")

    def __init__(
        self,
        round_id: int,
        client_id: int,
        round_cfg_acronym: str | None,
    ) -> None:
        self.round_id: int = round_id
        self.client_id: int = client_id
        self.round_cfg_acronym: str | None = round_cfg_acronym

    def to_dict(self) -> dict[str, Any]:
        """Convert record attributes to a flat CSV-ready dictionary."""
        skipped_fields = {
            "global_eval_metrics", "worker_eval_metrics"}
        result: dict[str, Any] = {"round_acronym": self.round_acronym} | {
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

    def update_record_fields(self, updates: Mapping[str, Any]) -> None:
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
    round_acronym: ClassVar[str]
    round_cfg_acronym: ClassVar[str|None]

    def __init_subclass__(cls) -> None:
        """Require concrete round codecs to declare their record type and round acronym."""
        super().__init_subclass__()
        assert hasattr(cls, "record_class")
        assert hasattr(cls, "round_acronym")
        assert hasattr(cls, "round_cfg_acronym")

    def create_round_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create this round codec's metrics record for one client-round compression."""
        return self.record_class(
            round_id=round_id,
            client_id=client_id,
            round_cfg_acronym=self.round_cfg_acronym
        )

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        raise NotImplementedError

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        raise NotImplementedError


class BaseProtocol:
    """Protocol schedule that selects the round codec used for each training round."""

    warmup_round_codecs: ClassVar[tuple[type[BaseRoundCodec], ...]]
    routine_round_codecs: ClassVar[tuple[type[BaseRoundCodec], ...]]
    protocol_name: str

    def __init_subclass__(cls) -> None:
        """Require concrete protocols to declare their warmup and routine round codec plans."""
        super().__init_subclass__()
        assert hasattr(cls, "warmup_round_codecs")
        assert hasattr(cls, "routine_round_codecs")
        assert hasattr(cls, "protocol_name")
        assert cls.routine_round_codecs

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the round codec selected by this protocol schedule."""
        if round_id < len(self.warmup_round_codecs):
            rc_acronym = self.warmup_round_codecs[round_id]
        else:
            routine_idx = (round_id - len(self.warmup_round_codecs)) % len(self.routine_round_codecs)
            rc_acronym = self.routine_round_codecs[routine_idx]
        round_codec_class = rc_acronym
        return round_codec_class()


def create_protocol(protocol_name: str) -> BaseProtocol:
    """Create a validated protocol schedule."""
    parse_and_validate_protocol(protocol_name)
    return DemoProtocol()


def parse_and_validate_protocol(protocol_name: str) -> str:
    """Validate a protocol name and return its canonical protocol token."""
    raise NotImplemented
    assert isinstance(protocol_name, str), f"protocol must be a string; got {type(protocol_name).__name__}."
    assert protocol_name, "protocol must be a non-empty string."
    assert protocol_name == protocol_name.strip(), (
        f"protocol={protocol_name!r} must not contain leading or trailing whitespace.")
    assert protocol_name == "demo", f"Unknown protocol={protocol_name!r}."
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

    record = round_codec.create_round_record(round_id, client_id)

    start_time = time.perf_counter()
    payload = round_codec.encode(delta_vec, record)
    encode_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    reconstructed = round_codec.decode(payload, record)
    decode_seconds = time.perf_counter() - start_time
    assert reconstructed.numel() == model_size

    record.update_record_fields({
        "protocol_name": protocol.protocol_name,
        "model_size": model_size,
        "global_eval_metrics": server_eval_metrics,
        "worker_eval_metrics": {
            'train': {key: worker_eval_metrics[i] for i, key in enumerate(metric_keys)},
            'test': {key: worker_eval_metrics[i + num_metrics] for i, key in enumerate(metric_keys)},
        },

        "final_bytes": _get_obj_storage_size(payload),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
    })
    
    recons_error_and_metric = record_reconstruction_metrics(delta_vec, reconstructed)
    record.update_record_fields(recons_error_and_metric)

    record.append_record_to_csv(save_dir)

    return reconstructed


# --- Demo Round Codec and Protocol --- #
class RCDemo(BaseRoundCodec):
    """Round codec that serializes raw float32 deltas with the shared gzip transport."""

    class RecordDemo(CompressionRecord):
        round_acronym = "gzip"
        compression_method: str | None = None
    record_class = RecordDemo
    round_acronym = record_class.round_acronym
    round_cfg_acronym = None

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> bytes:
        """Encode one raw float32 delta into f16, pickle then gzip."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        payload = compress_data_list(delta_vec.to(torch.float16))
        record.update_record_fields({"compression_method": "gzip_level_1"})
        return payload

    def decode(self, payload: bytes, record: CompressionRecord) -> torch.Tensor:
        """Decode one raw float32 delta from pickle plus gzip."""
        reconstructed = decompress_data_list(payload)
        return reconstructed


class DemoProtocol(BaseProtocol):
    warmup_round_codecs= ()
    routine_round_codecs= (RCDemo,)
    protocol_name: str = "demo"
