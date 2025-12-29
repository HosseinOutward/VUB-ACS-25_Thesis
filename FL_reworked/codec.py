from __future__ import annotations
from typing import Any, Dict, OrderedDict
from pathlib import Path
import csv
import pickle
import gzip

import torch
import numpy as np

from FL_reworked.run_fl import FLConfig
from FL_reworked.utils import StateDictManager


def get_obj_size(obj):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(get_obj_size(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(get_obj_size(v) for k, v in obj.items())
    elif hasattr(obj, '_dtype') and hasattr(obj, '__len__'):
        return len(obj) * (obj._dtype.bitwidth // 8)
    elif isinstance(obj, bytes):
        return len(obj)
    elif obj is None:
        return 1
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")


def make_seriable(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, (np.uint8, np.uint16, np.uint32, np.uint64, np.float16)):
        return item.item()
    elif isinstance(item, (int, float, str, bytes)):
        return item
    elif isinstance(item, torch.Tensor):
        return item.cpu().numpy()
    elif isinstance(item, OrderedDict):
        return OrderedDict({k: make_seriable(v) for k, v in item.items()})
    elif isinstance(item, Dict):
        return {k: make_seriable(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [make_seriable(x) for x in item]
    elif hasattr(item, '_dtype') and hasattr(item, '__len__'):
        numpy_dtype = eval('np.' + str(item._dtype))
        return np.array(item, dtype=numpy_dtype)
    elif item is None:
        return None
    else:
        raise TypeError(f"Unsupported type for serialization: {type(item)}.")


def compress_data_list(data_list):
    """Compress data using pickle and gzip."""
    serializable_list = make_seriable(data_list)

    pickled_data = pickle.dumps(serializable_list, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = gzip.compress(pickled_data, compresslevel=6)
    return compressed_data


def decompress_data_list(compressed_data):
    """Decompress data."""
    decompressed_data = gzip.decompress(compressed_data)
    data_list = pickle.loads(decompressed_data)
    return data_list


# --- Compression Record --- #
class CompressionRecord:
    """Record for compression metrics. Stores all attributes for CSV export."""

    def __init__(self, round_id: int, client_id: int, method: str = "identity"):
        self.round_id: int = round_id
        self.client_id: int = client_id
        self.method = method
        self.compressed_bytes: int = 0
        self.raw_bytes: int = 0
        self.compression_ratio: float = 0.0
        self.global_eval_metrics: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary using class attributes."""
        result = {
            'round_id': self.round_id,
            'client_id': self.client_id,
            'method': self.method,
            'compressed_bytes': self.compressed_bytes,
            'raw_bytes': self.raw_bytes,
            'compression_ratio': self.compression_ratio,
        }
        # Add global eval metrics with prefix
        for key, value in self.global_eval_metrics.items():
            result[f'global_eval_{key}'] = value
        return result

    def save_to_csv(self, save_dir: str | None = None) -> None:
        """Append record to CSV file. If save_dir is None, skip saving."""
        if save_dir is None:
            return

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        csv_file = save_path / "compression_records.csv"
        record_dict = self.to_dict()

        # Check if file exists to determine if we need to write headers
        file_exists = csv_file.exists()

        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record_dict)


# --- Compression Codecs --- #
class IdentityCodec:
    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        return CompressionRecord(round_id, client_id, method="identity")

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device('cpu')
        record.raw_bytes = get_obj_size(delta_vec)

        payload_content = self._compress(delta_vec, record)
        payload = compress_data_list(payload_content)

        record.compressed_bytes = get_obj_size(payload)
        record.compression_ratio = record.compressed_bytes / record.raw_bytes if record.raw_bytes > 0 else 0.0
        return payload

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        payload_content = decompress_data_list(payload)
        res = self._decompress(payload_content, record)
        assert res.dtype == torch.float32 and res.device == torch.device('cpu')
        return res

    # Methods to be overridden by subclasses
    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        return delta_vec

    # Methods to be overridden by subclasses
    def _decompress(self, payload_content: Any, record: CompressionRecord) -> torch.Tensor:
        return payload_content


class BasicCompressionCodec(IdentityCodec):
    """Basic compression: float16 + gzip. Extends IdentityCodec."""

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        return CompressionRecord(round_id, client_id, method="basic")

    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        delta_fp16 = delta_vec.to(torch.float16)
        return delta_fp16
    
    def _decompress(self, payload_content: bytes, record: CompressionRecord) -> torch.Tensor:
        decompressed = decompress_data_list(payload_content)
        return decompressed.to(torch.float32)


def create_codec(fl_cfg:FLConfig, sd_manager:StateDictManager) -> IdentityCodec:
    """Create codec instance."""
    codec_name = fl_cfg.codec.lower()
    if codec_name == "identity":
        return IdentityCodec()
    elif codec_name == "basic":
        return BasicCompressionCodec()
    elif codec_name == "cancer":
        from FL_reworked.cancer_protocol import CancerCodec
        return CancerCodec(fl_cfg)
    elif codec_name == "cancer_data_prep":
        from FL_reworked.cancer_preprocess_protocol import CancerDataPrepCodec
        return CancerDataPrepCodec(fl_cfg, sd_manager.get_slices())
    else:
        raise NotImplementedError(f"Codec '{codec_name}' not implemented.")


def simulate_compression(
    codec: IdentityCodec,
    delta_vec: torch.Tensor,
    client_id: int,
    round_id: int,
    eval_metrics: Dict[str, float],
    save_dir: str | None = "compression_logs"
) -> torch.Tensor:
    # Create record for this compression operation
    record = codec.create_record(round_id, client_id)

    # Encode (client-side simulation)
    payload = codec.encode(delta_vec, record)
    
    # Decode (server-side)
    reconstructed = codec.decode(payload, record)

    record.save_to_csv(save_dir)

    return reconstructed
