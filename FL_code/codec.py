from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from pathlib import Path
import csv
import pickle
import gzip

import torch
import numpy as np

if TYPE_CHECKING:
    from FL_code.utils import StateDictManager


def get_obj_compressed_size(obj: Any, with_compression: bool = True) -> int:
    """Return the serialized storage size estimate for tensors and payload objects."""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(get_obj_compressed_size(x, with_compression=False) for x in obj)
    if isinstance(obj, dict):
        return sum(get_obj_compressed_size(v, with_compression=False) for v in obj.values())
    if hasattr(obj, '_dtype') and hasattr(obj, '__len__'):
        return len(obj) * (obj._dtype.bitwidth // 8)
    if isinstance(obj, bytes):
        return len(obj)
    if obj is None:
        return 1
    raise TypeError(f"Unsupported object type: {type(obj)}")


def make_serializable(item: Any) -> Any:
    """Convert tensors and coder objects into pickle-friendly values."""
    if isinstance(item, np.ndarray):
        return item
    if isinstance(item, (np.integer, np.floating)):
        return item.item()
    if isinstance(item, (int, float, str, bytes)):
        return item
    if isinstance(item, torch.Tensor):
        return item.cpu()
    if isinstance(item, OrderedDict):
        return OrderedDict((key, make_serializable(value)) for key, value in item.items())
    if isinstance(item, Mapping):
        return {key: make_serializable(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return [make_serializable(x) for x in item]
    if hasattr(item, '_dtype') and hasattr(item, '__len__'):
        numpy_dtype = np.dtype(str(item._dtype))
        return np.array(item, dtype=numpy_dtype)
    if item is None:
        return None
    raise TypeError(f"Unsupported type for serialization: {type(item)}.")


def compress_data_list(data_list: Any) -> bytes:
    """Compress data using pickle and gzip."""
    serializable_list = make_serializable(data_list)

    pickled_data = pickle.dumps(serializable_list, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = gzip.compress(pickled_data, compresslevel=6)
    return compressed_data


def decompress_data_list(compressed_data: bytes) -> Any:
    """Decompress data."""
    decompressed_data = gzip.decompress(compressed_data)
    data_list = pickle.loads(decompressed_data)
    return data_list


# --- Compression Record --- #
class CompressionRecord:
    """Record for compression metrics. Stores all attributes for CSV export."""

    def __init__(self, round_id: int, client_id: int, method: str = "identity") -> None:
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
        self._og_delta_vec: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary using class attributes."""
        result = {
            'round_id': self.round_id,
            'client_id': self.client_id,
            'codec_class_used': self.codec_class_used,

            'wmape': self.wmape,
            'wmspe_sqrt': self.wmspe_sqrt,
            'mse': self.mse,
            'mape': self.mape,
            'mspe_sqrt': self.mspe_sqrt,

            'w_mean_of_vec': self.w_mean_of_vec,
            'model_size': self.model_size,

            'compressed_bytes': self.compressed_bytes,
            'basic_raw_bytes': self.basic_raw_bytes,
            'compression_ratio': self.compression_ratio,
            'entropy_real_rate': self.entropy_real_rate,
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
        """Append record to CSV file."""
        save_dir.mkdir(exist_ok=True, parents=True)

        csv_file = save_dir / "compression_records.csv"
        record_dict = self.to_dict()

        # Check if file exists to determine if we need to write headers
        file_exists = csv_file.exists()

        with csv_file.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record_dict)


# --- Compression Codecs --- #
class IdentityCodec:
    OPTION_ORDER: tuple[str, ...] = ()

    def __init__(self, codec_name: str = "identity") -> None:
        self.codec_name = codec_name

    @staticmethod
    def validate_codec_tokens(option_tokens: Sequence[str]) -> None:
        """Validate identity codec name options."""
        assert not option_tokens, f"identity codec does not accept options: {option_tokens!r}."

    @classmethod
    def create_from_codec_name(
        cls,
        codec_name: str,
        protocol_name: str,
        option_tokens: Sequence[str],
        sd_manager: StateDictManager | None,
    ) -> IdentityCodec:
        """Create an identity codec from a validated codec name."""
        assert protocol_name == "identity"
        assert not option_tokens
        return cls(codec_name=codec_name)

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a metrics record for one client-round compression."""
        return CompressionRecord(round_id, client_id, method=self.codec_name)

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        """Encode one flattened model delta into a transport payload."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device('cpu')
        record.basic_raw_bytes = get_obj_compressed_size(compress_data_list(delta_vec), with_compression=False) / (1024 ** 2)

        payload_content = self._compress(delta_vec, record)
        payload = compress_data_list(payload_content)

        record.compressed_bytes = get_obj_compressed_size(payload, with_compression=False) / (1024 ** 2)
        record.compression_ratio = record.basic_raw_bytes / record.compressed_bytes
        assert record.model_size is not None and record.model_size > 0, "CompressionRecord.model_size is bad."
        record.entropy_real_rate = record.compressed_bytes * (1024**2) * 8 / record.model_size

        record._og_delta_vec = delta_vec.clone()

        return payload

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        """Decode one transport payload and populate reconstruction metrics."""
        payload_content = decompress_data_list(payload)
        res = self._decompress(payload_content, record)
        assert res.dtype == torch.float32 and res.device == torch.device('cpu')

        delta_vec = record._og_delta_vec
        assert delta_vec is not None, "CompressionRecord is missing original delta vector for metric calculation."
        record._og_delta_vec = None

        record.mse = torch.mean((res - delta_vec) ** 2).item()
        record.mape = torch.mean(torch.abs(res - delta_vec) / (torch.abs(delta_vec) + 1e-8)).item() * 100
        record.mspe_sqrt = torch.sqrt(torch.mean(
            (res - delta_vec) ** 2 / (delta_vec ** 2 + 1e-8))).item() * 100

        record.w_mean_of_vec = torch.abs(delta_vec).mean().item()
        w = record.w_mean_of_vec
        assert w != 0.0, "CompressionRecord weighted metrics require a non-zero mean absolute delta vector."
        record.wmape = torch.mean(torch.abs(res - delta_vec)).item()/w * 100
        record.wmspe_sqrt = float(np.sqrt(record.mse))/w * 100

        return res

    # Methods to be overridden by subclasses
    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        """Compress a delta vector before generic payload serialization."""
        return delta_vec

    # Methods to be overridden by subclasses
    def _decompress(self, payload_content: Any, record: CompressionRecord) -> torch.Tensor:
        """Reconstruct a delta vector from decoded payload content."""
        return payload_content


class BasicCompressionCodec(IdentityCodec):
    """Basic compression: float16 + gzip. Extends IdentityCodec."""

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
    protocol_name, option_tokens = protocol_name, tuple(option_tokens)

    protocol_class = get_protocol_class(protocol_name)
    protocol_class.validate_codec_tokens(option_tokens)

    return protocol_name, option_tokens


def get_protocol_class(protocol_name: str) -> type[IdentityCodec]:
    """Return the protocol class selected by a codec-name protocol token."""
    if protocol_name == "identity":
        return IdentityCodec
    if protocol_name == "basic":
        return BasicCompressionCodec
    if protocol_name == "split":
        from FL_code.other_protocols.n_split_protocol import NSplitCodec
        return NSplitCodec
    if protocol_name == "cancer":
        from FL_code.cancer_protocol import CancerCodec
        return CancerCodec
    assert False, f"Unknown codec protocol={protocol_name!r}."


def create_codec(codec_name: str, sd_manager: StateDictManager | None) -> IdentityCodec:
    """Create a codec from a protocol-owned, validated codec name."""
    protocol_name, option_tokens = parse_and_validate_codec_name(codec_name)
    protocol_class = get_protocol_class(protocol_name)
    return protocol_class.create_from_codec_name(codec_name, protocol_name, option_tokens, sd_manager)


def simulate_compression(
    codec: IdentityCodec, delta_vec: torch.Tensor, client_id: int, round_id: int,
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

    record.append_record_to_csv(save_dir)

    return reconstructed
