from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import csv
import pickle
import gzip
import tempfile
from typing import TYPE_CHECKING, Any, Protocol

import torch
import numpy as np

if TYPE_CHECKING:
    from .utils import StateDictManager


def _get_obj_storage_size(obj: Any) -> int:
    """Return the in-memory payload size for tensor-like compression objects."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(_get_obj_storage_size(x) for x in obj)
    if isinstance(obj, Mapping):
        return sum(_get_obj_storage_size(v) for v in obj.values())
    if hasattr(obj, '_dtype') and hasattr(obj, '__len__'):
        return len(obj) * (obj._dtype.bitwidth // 8)
    if isinstance(obj, bytes):
        return len(obj)
    if obj is None:
        return 1
    raise TypeError(f"Unsupported object type: {type(obj)}")


def get_obj_compressed_size(obj: Any, with_compression: bool = True) -> int:
    """Return the compressed serialized size, or raw tensor-like storage size."""
    if not with_compression or isinstance(obj, bytes):
        return _get_obj_storage_size(obj)
    return len(compress_data_list(obj))


def make_serializable(item: Any) -> Any:
    """Normalize payload values that pickle should not store as-is."""
    if isinstance(item, torch.Tensor):
        return item.cpu()
    if hasattr(item, '_dtype') and hasattr(item, '__len__'):
        return np.array(item, dtype=np.dtype(str(item._dtype)))
    if isinstance(item, (np.integer, np.floating)):
        return item.item()
    if isinstance(item, OrderedDict):
        return OrderedDict((key, make_serializable(value)) for key, value in item.items())
    if isinstance(item, Mapping):
        return {key: make_serializable(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [make_serializable(x) for x in item]
    return item


def compress_data_list(data_list: Any) -> bytes:
    """Compress data using pickle and gzip."""
    # Level 1: gzip here only estimates payload size on the server's critical path,
    # and higher levels barely shrink float payloads while costing much more time.
    return gzip.compress(
        pickle.dumps(make_serializable(data_list), protocol=pickle.HIGHEST_PROTOCOL),
        compresslevel=1,
    )


def decompress_data_list(compressed_data: bytes) -> Any:
    """Decompress data."""
    return pickle.loads(gzip.decompress(compressed_data))


# --- Compression Record --- #
class Access(Enum):
    """Which reconstructed-gradient history entries a process may inspect."""

    NONE = "none"
    SHARED = "shared"
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
        self._shared: dict[int, list[HistoryEntry]] = {}
        self._keep_server: bool = True
        self._keep_shared: bool = True

    def finish_warmup(self, routine_accesses: Sequence[Access]) -> None:
        """Discard ledgers that the repeating routine phase will never read."""
        routine_plan = tuple(routine_accesses)
        assert all(isinstance(access, Access) for access in routine_plan), (
            "Routine history access plan must contain Access values.")
        self._set_retention(
            server=Access.SERVER in routine_plan,
            shared=Access.SHARED in routine_plan,
        )

    def _set_retention(self, *, server: bool, shared: bool) -> None:
        self._keep_server = server
        self._keep_shared = shared
        if not server:
            self._server.clear()
        if not shared:
            self._shared.clear()

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
        if access is Access.SHARED and self._keep_shared:
            self._add_entry(self._shared, entry)

    def view(self, access: Access, record: CompressionRecord) -> tuple[HistoryEntry, ...]:
        """Return the history entries visible to the requested process."""
        if access is Access.NONE:
            return ()
        if access is Access.SHARED:
            assert record.client_id in self._shared, (
                f"No shared reconstruction history exists for client {record.client_id}."
            )
            return tuple(self._shared[record.client_id])
        assert access is Access.SERVER, f"Unknown access policy: {access!r}."
        entries = [entry for ledger in self._server.values() for entry in ledger]
        return tuple(sorted(entries, key=lambda entry: (entry.round_id, entry.client_id)))

    def latest_for_client(self, access: Access, record: CompressionRecord) -> HistoryEntry:
        """Return the latest visible entry for the record's client."""
        assert access is Access.SERVER, "Client-specific latest lookup is only defined for server history."
        assert record.client_id in self._server and self._server[record.client_id], (
            f"No server reconstruction history exists for client {record.client_id}."
        )
        return self._server[record.client_id][-1]

    def seed(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Insert externally prepared side information for diagnostics or experiments."""
        tensor = reconst.detach().to(device="cpu", dtype=torch.float16)
        self.commit(tensor, record, access)

    def tensor_ledgers(self, access: Access) -> list[list[torch.Tensor]]:
        """Return a tensor-only snapshot of one internal ledger for diagnostics."""
        assert access in {Access.SERVER, Access.SHARED}, f"Cannot snapshot access policy: {access!r}."
        history = self._server if access is Access.SERVER else self._shared
        max_client_id = max(history, default=-1)
        return [[entry.tensor for entry in history.get(client_id, [])] for client_id in range(max_client_id + 1)]

    def _add_entry(self, history: dict[int, list[HistoryEntry]], entry: HistoryEntry) -> None:
        ledger = history.setdefault(entry.client_id, [])
        ledger.append(entry)
        if len(ledger) > self.max_per_client:
            ledger.pop(0)


# --- Compression Record --- #
class CompressionRecord:
    """Record for compression metrics. Stores all attributes for CSV export."""

    def __init__(self, round_id: int, client_id: int, method: str = "base") -> None:
        self.round_id: int = round_id
        self.client_id: int = client_id

        self.codec_class_used: str = method

        self.compressed_bytes: float | None = None
        self.basic_raw_bytes: float | None = None
        self.compression_ratio: float | None = None

        self.global_eval_metrics: dict[str, float] = {}
        self.worker_eval_metrics: dict[str, dict[str, float]] = {}

        self.entropy_real_rate: float | None = None
        self.model_size: int | None = None
        self.mse: float | None = None
        self.mape: float | None = None
        self.mspe_sqrt: float | None = None
        self.w_mean_of_vec: float | None = None
        self.wmape: float | None = None
        self.wmspe_sqrt: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record attributes to a flat CSV-ready dictionary."""
        skipped_fields = {
            "global_eval_metrics", "worker_eval_metrics"}
        result = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and key not in skipped_fields
        }
        # Add global eval metrics with prefix
        for key, value in self.global_eval_metrics.items():
            result[f'global_eval_{key}'] = value

        # Add worker eval metrics with split prefix (e.g., train_loss, test_acc)
        for split, metrics in self.worker_eval_metrics.items():
            for metric_key, metric_value in metrics.items():
                result[f'{split}_{metric_key}'] = metric_value

        return result

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


# --- Compression Codecs --- #
class BaseCodec:
    """Base codec: raw float32 through the shared pickle+gzip transport.

    Subclasses override `_compress`/`_decompress`; the reported sizes always
    include the generic serialization, so this is a gzip baseline, not a
    bit-exact identity rate.
    """

    OPTION_ORDER: tuple[str, ...] = ()

    def __init__(self, codec_name: str = "base") -> None:
        self.codec_name = codec_name

    @staticmethod
    def validate_codec_tokens(option_tokens: Sequence[str]) -> None:
        """Validate base codec name options."""
        assert not option_tokens, f"base codec does not accept options: {option_tokens!r}."

    @classmethod
    def create_from_codec_name(
        cls,
        codec_name: str,
        protocol_name: str,
        option_tokens: Sequence[str],
        sd_manager: StateDictManager | None,
    ) -> BaseCodec:
        """Create a base codec from a validated codec name."""
        assert protocol_name == "base"
        assert not option_tokens
        return cls(codec_name=codec_name)

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a metrics record for one client-round compression."""
        return CompressionRecord(round_id, client_id, method=self.codec_name)

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        """Encode one flattened model delta into a transport payload."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device('cpu')
        record.basic_raw_bytes = get_obj_compressed_size(delta_vec, with_compression=False) / (1024 ** 2)

        payload_content = self._compress(delta_vec, record)
        payload = compress_data_list(payload_content)

        record.compressed_bytes = len(payload) / (1024 ** 2)
        record.compression_ratio = record.basic_raw_bytes / record.compressed_bytes
        assert record.model_size is not None and record.model_size > 0, "CompressionRecord.model_size is bad."
        record.entropy_real_rate = record.compressed_bytes * (1024**2) * 8 / record.model_size

        return payload

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        """Decode one transport payload into a reconstructed delta vector."""
        payload_content = decompress_data_list(payload)
        res = self._decompress(payload_content, record)
        assert res.dtype == torch.float32 and res.device == torch.device('cpu')
        return res

    # Methods to be overridden by subclasses
    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        """Compress a delta vector before generic payload serialization."""
        return delta_vec

    # Methods to be overridden by subclasses
    def _decompress(self, payload_content: Any, record: CompressionRecord) -> torch.Tensor:
        """Reconstruct a delta vector from decoded payload content."""
        return payload_content


class BasicCompressionCodec(BaseCodec):
    """Basic compression: float16 + gzip. Extends BaseCodec."""

    @staticmethod
    def validate_codec_tokens(option_tokens: Sequence[str]) -> None:
        """Validate basic codec name options."""
        assert not option_tokens, f"basic codec does not accept options: {option_tokens!r}."

    @classmethod
    def create_from_codec_name(
        cls,
        codec_name: str,
        protocol_name: str,
        option_tokens: Sequence[str],
        sd_manager: StateDictManager | None,
    ) -> BasicCompressionCodec:
        """Create a basic compression codec from a validated codec name."""
        assert protocol_name == "basic"
        assert not option_tokens
        return cls(codec_name=codec_name)

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a metrics record for basic float16 compression."""
        return CompressionRecord(round_id, client_id, method=self.codec_name)

    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        delta_fp16 = delta_vec.to(torch.float16)
        return delta_fp16
    
    def _decompress(self, payload_content: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        return payload_content.to(torch.float32)


# --- Codec Name Parsing --- #
def parse_and_validate_codec_name(codec_name: str) -> tuple[str, tuple[str, ...]]:
    """Validate a codec name and return protocol plus ordered option tokens."""
    assert isinstance(codec_name, str), f"codec must be a string; got {type(codec_name).__name__}."
    assert codec_name, "codec must be a non-empty string."
    assert codec_name == codec_name.strip(), f"codec={codec_name!r} must not contain leading or trailing whitespace."
    tokens = codec_name.split("|")
    assert all(token != "" for token in tokens), f"codec={codec_name!r} contains an empty protocol or option token."
    protocol_name, *option_tokens = tokens
    option_tokens = tuple(option_tokens)

    protocol_class = get_protocol_class(protocol_name)
    protocol_class.validate_codec_tokens(option_tokens)

    return protocol_name, option_tokens


def get_protocol_class(protocol_name: str) -> type[BaseCodec]:
    """Return the protocol class selected by a codec-name protocol token."""
    if protocol_name == "base":
        return BaseCodec
    if protocol_name == "basic":
        return BasicCompressionCodec
    if protocol_name == "split":
        from FL_code.other_protocols.n_split_protocol import NSplitCodec
        return NSplitCodec
    if protocol_name == "cancer":
        from FL_code.cancer_protocol import CancerCodec
        return CancerCodec
    assert False, f"Unknown codec protocol={protocol_name!r}."


def create_codec(codec_name: str, sd_manager: StateDictManager | None) -> BaseCodec:
    """Create a codec from a protocol-owned, validated codec name."""
    protocol_name, option_tokens = parse_and_validate_codec_name(codec_name)
    protocol_class = get_protocol_class(protocol_name)
    return protocol_class.create_from_codec_name(codec_name, protocol_name, option_tokens, sd_manager)


def record_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    record: CompressionRecord,
) -> None:
    """Populate distortion metrics for one reconstructed delta."""
    assert original.shape == reconstructed.shape, (
        f"Reconstruction shape mismatch: got {tuple(reconstructed.shape)}, expected {tuple(original.shape)}."
    )
    error = reconstructed - original
    record.mse = torch.mean(error ** 2).item()
    record.mape = torch.mean(torch.abs(error) / (torch.abs(original) + 1e-8)).item() * 100
    record.mspe_sqrt = torch.sqrt(torch.mean(error ** 2 / (original ** 2 + 1e-8))).item() * 100

    record.w_mean_of_vec = torch.abs(original).mean().item()
    if record.w_mean_of_vec == 0.0:
        record.wmape = None
        record.wmspe_sqrt = None
        return

    record.wmape = torch.mean(torch.abs(error)).item() / record.w_mean_of_vec * 100
    record.wmspe_sqrt = float(np.sqrt(record.mse)) / record.w_mean_of_vec * 100


def simulate_compression(
    codec: BaseCodec, delta_vec: torch.Tensor, client_id: int, round_id: int,
    model_size: int, save_dir: Path,
    server_eval_metrics: dict[str, float], worker_eval_metrics: Sequence[float],
    metric_keys: list[str]) -> torch.Tensor:
    """Simulate client encoding and server decoding for one flattened delta."""
    # Create record for this compression operation
    record = codec.create_record(round_id, client_id)
    record.model_size = model_size
    record.global_eval_metrics = server_eval_metrics

    # Restructure worker metrics to match server metrics structure
    num_metrics = len(metric_keys)
    expected_len = num_metrics * 2
    assert len(worker_eval_metrics) == expected_len, (
        f"worker_eval_metrics must contain {expected_len} "
        f"values for {num_metrics} metric keys; "
        f"got {len(worker_eval_metrics)}."
    )
    train_metrics = {key: worker_eval_metrics[i] for i, key in enumerate(metric_keys)}
    test_metrics = {key: worker_eval_metrics[i + num_metrics] for i, key in enumerate(metric_keys)}
    record.worker_eval_metrics = {'train': train_metrics, 'test': test_metrics}

    # Encode (client-side simulation)
    payload = codec.encode(delta_vec, record)
    
    # Decode (server-side)
    reconstructed = codec.decode(payload, record)
    assert reconstructed.numel() == model_size, (
        f"{codec.__class__.__name__} reconstructed {reconstructed.numel()} values; expected {model_size}."
    )
    record_reconstruction_metrics(delta_vec, reconstructed, record)

    record.append_record_to_csv(save_dir)

    return reconstructed
