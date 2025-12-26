from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import csv
import pickle
import gzip

import torch
import numpy as np



def get_obj_size(obj):
    """Get size of object in bytes."""
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


def compress_data_list(data_list):
    """Compress data using pickle and gzip."""
    if isinstance(data_list, torch.Tensor):
        data_list = data_list.cpu().numpy()
    
    pickled_data = pickle.dumps(data_list, protocol=pickle.HIGHEST_PROTOCOL)
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
        self.round_id = round_id
        self.client_id = client_id
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
        record.raw_bytes = get_obj_size(delta_vec)
        payload = self._compress(delta_vec, record)
        record.compressed_bytes = get_obj_size(payload)
        record.compression_ratio = record.compressed_bytes / record.raw_bytes if record.raw_bytes > 0 else 0.0
        return payload

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        return self._decompress(payload, record)

    # Methods to be overridden by subclasses
    def (self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        return delta_vec

    # Methods to be overridden by subclasses
    def _decompress(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        return payload


class BasicCompressionCodec(IdentityCodec):
    """Basic compression: float16 + gzip. Extends IdentityCodec."""

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        return CompressionRecord(round_id, client_id, method="basic")

    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> bytes:
        delta_fp16 = delta_vec.to(torch.float16)
        compressed = compress_data_list(delta_fp16)
        
        return compressed
    
    def _decompress(self, payload: bytes, record: CompressionRecord) -> torch.Tensor:
        decompressed = decompress_data_list(payload)
        decompressed = torch.from_numpy(decompressed)
        
        return decompressed.to(torch.float32)


def create_codec(codec_name: str, **kwargs) -> IdentityCodec:
    """Create codec instance."""
    if codec_name == "identity":
        return IdentityCodec()
    elif codec_name == "basic":
        return BasicCompressionCodec()
    elif codec_name == "cancer":
        from FL_reworked.cancer_protocol import CancerCodec
        return CancerCodec(**kwargs)
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
