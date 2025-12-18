from __future__ import annotations
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json
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
    # Convert to serializable format
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

# --- Compression Codec Framework --- #
class CompressionRecord(ABC):
    def __init__(self, round_id: int, client_id: int):
        self.round_id = round_id
        self.client_id = client_id
        self.compressed_bytes: int = None
        self.global_eval_metrics: Dict[str, float] = {}
    
    def set_compressed_bytes(self, compressed_bytes: int) -> None:
        """Set compressed bytes."""
        self.compressed_bytes = compressed_bytes
    
    def set_global_eval(self, eval_metrics: Dict[str, float]) -> None:
        """Set global evaluation metrics with 'global_eval_' prefix."""
        for key, value in eval_metrics.items():
            self.global_eval_metrics[f'global_eval_{key}'] = value
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        raise NotImplementedError
    
    def save_to_disk(self, save_dir: str = "compression_logs") -> None:
        """Save record to disk as JSON."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        filename = save_path / f"round_{self.round_id:03d}_client_{self.client_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class IdentityRecord(CompressionRecord):
    """Record for identity codec (no compression)."""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'round_id': self.round_id,
            'client_id': self.client_id,
            'method': 'identity',
            'compressed_bytes': self.compressed_bytes,
            **self.global_eval_metrics
        }


class BasicCompressionRecord(CompressionRecord):
    """Record for basic compression codec (float16 + gzip)."""
    
    def __init__(self, round_id: int, client_id: int):
        super().__init__(round_id, client_id)
        self.raw_bytes: int = 0
        self.compression_ratio: float = 0.0
    
    def set_raw_bytes(self, raw_bytes: int) -> None:
        """Set raw bytes and calculate compression ratio."""
        self.raw_bytes = raw_bytes
        if self.raw_bytes > 0:
            self.compression_ratio = self.compressed_bytes / self.raw_bytes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'round_id': self.round_id,
            'client_id': self.client_id,
            'method': 'basic',
            'compressed_bytes': self.compressed_bytes,
            'raw_bytes': self.raw_bytes,
            'compression_ratio': self.compression_ratio,
            **self.global_eval_metrics
        }


class FederatedCodec(ABC):
    """Base class for gradient compression codecs."""
    
    @abstractmethod
    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a record instance for this codec."""
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        """
        Encode delta vector and update record.
        
        Args:
            delta_vec: Flattened delta vector
            record: Record to update with metrics
            
        Returns:
            payload: Encoded data
        """
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        """
        Decode payload back to delta vector.
        
        Args:
            payload: Encoded data
            record: Record (for potential use in decoding)
            
        Returns:
            Reconstructed delta vector
        """
        raise NotImplementedError


class IdentityCodec(FederatedCodec):
    """No compression - just pass through."""
    
    def create_record(self, round_id: int, client_id: int) -> IdentityRecord:
        return IdentityRecord(round_id, client_id)
    
    def encode(self, delta_vec: torch.Tensor, record: IdentityRecord) -> torch.Tensor:
        """No compression, just return the vector."""
        compressed_bytes = get_obj_size(delta_vec)
        record.set_compressed_bytes(compressed_bytes)
        return delta_vec
    
    def decode(self, payload: torch.Tensor, record: IdentityRecord) -> torch.Tensor:
        """No decompression needed."""
        return payload


class BasicCompressionCodec(FederatedCodec):
    """Basic compression: float16 + gzip."""
    
    def create_record(self, round_id: int, client_id: int) -> BasicCompressionRecord:
        return BasicCompressionRecord(round_id, client_id)
    
    def encode(self, delta_vec: torch.Tensor, record: BasicCompressionRecord) -> bytes:
        """Compress using float16 and gzip."""
        # Record raw size
        raw_bytes = get_obj_size(delta_vec)
        record.set_raw_bytes(raw_bytes)
        
        # Convert to float16
        delta_fp16 = delta_vec.to(torch.float16)
        
        # Compress
        compressed = compress_data_list(delta_fp16)
        
        # Record compressed size
        compressed_bytes = get_obj_size(compressed)
        record.set_compressed_bytes(compressed_bytes)
        
        return compressed
    
    def decode(self, payload: bytes, record: BasicCompressionRecord) -> torch.Tensor:
        """Decompress and convert back to float32."""
        # Decompress
        decompressed = decompress_data_list(payload)
        
        # Convert back to tensor if needed
        if isinstance(decompressed, np.ndarray):
            decompressed = torch.from_numpy(decompressed)
        
        # Convert back to float32
        return decompressed.to(torch.float32)


def create_codec(codec_name: str, **kwargs) -> FederatedCodec:
    """Create codec instance."""
    if codec_name == "identity":
        return IdentityCodec()
    elif codec_name == "basic":
        return BasicCompressionCodec()
    else:
        raise NotImplementedError(f"Codec '{codec_name}' not implemented.")


def simulate_compression(
    codec: FederatedCodec,
    delta_vec: torch.Tensor,
    client_id: int,
    round_id: int,
    eval_metrics: Dict[str, float] = None
) -> torch.Tensor:
    """
    Entry point for compression simulation.
    
    Args:
        codec: Compression codec
        delta_vec: Flattened delta vector
        client_id: Client identifier
        round_id: Current round
        eval_metrics: Global evaluation metrics (optional)
        
    Returns:
        Tuple of (reconstructed delta vector, compression record)
    """
    # Create record for this compression operation
    record = codec.create_record(round_id, client_id)
    
    # Set global eval metrics if provided
    if eval_metrics is not None:
        record.set_global_eval(eval_metrics)
    
    # Encode (client-side simulation)
    payload = codec.encode(delta_vec, record)
    
    # Decode (server-side)
    reconstructed = codec.decode(payload, record)

    record.save_to_disk()

    return reconstructed

