from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path
import csv
import pickle
import gzip

import torch
import numpy as np

from FL_code.run_fl import FLConfig
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

    def save_to_csv(self, save_dir: Path | str | None = None) -> None:
        """Append record to CSV file. If save_dir is None, skip saving."""
        if save_dir is None:
            return

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        csv_file = save_path / "compression_records.csv"
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
    def __init__(self, fl_cfg: FLConfig) -> None:
        self.fl_cfg = fl_cfg

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a metrics record for one client-round compression."""
        return CompressionRecord(round_id, client_id, method="identity")

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

    def create_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create a metrics record for basic float16 compression."""
        return CompressionRecord(round_id, client_id, method="basic")

    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        delta_fp16 = delta_vec.to(torch.float16)
        return delta_fp16
    
    def _decompress(self, payload_content: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        return payload_content.to(torch.float32)


def create_codec(fl_cfg: FLConfig, sd_manager: StateDictManager | None) -> IdentityCodec:
    """Create codec instance."""
    codec_name = fl_cfg.codec.lower()

    codec: IdentityCodec | None = None

    if codec_name == "identity":
        codec = IdentityCodec(fl_cfg)
    elif codec_name == "basic":
        codec = BasicCompressionCodec(fl_cfg)
    elif codec_name == "split":
        from FL_code.other_protocols.n_split_protocol import NSplitCodec
        split_name = fl_cfg.run_name or fl_cfg.codec
        assert split_name.endswith("_split_codec") and split_name.split("_", 1)[0].isdigit(), (
            "Split codec requires an input name like '3_split_codec'."
        )
        codec = NSplitCodec(fl_cfg, int(split_name.split("_", 1)[0]))
    else:
        from FL_code.cancer_protocol import build_cancer_config_for_fl
        c_cfg = build_cancer_config_for_fl(fl_cfg)

        norm_slices = None
        if c_cfg.use_model_slices:
            assert sd_manager is not None, "StateDictManager is required when CancerConfig.use_model_slices is enabled."
            norm_slices = sd_manager.get_slices()
        quantizer_kwargs = {
            "norm_slices": norm_slices,
            "outlier_threshold": c_cfg.outlier_threshold if c_cfg.outlier_threshold is not None else False,
        }
        binary_prot = c_cfg.binary_protocol

        if codec_name == "debug_cancerwithboundcalc":
            from FL_code.experiments.rd_mspe_wz import CancerWithBoundCalc
            codec = CancerWithBoundCalc(fl_cfg, binary_prot, quantizer_kwargs)

        elif codec_name == "non_wz_learned_worker":
            from FL_code.other_protocols.SingleTypeCodecs import SingleTypeCodec
            codec = SingleTypeCodec('TMM', fl_cfg, binary_prot, quantizer_kwargs)
        elif codec_name == "non_wz_learned_server":
            from FL_code.other_protocols.SingleTypeCodecs import SingleTypeCodec
            codec = SingleTypeCodec('RMM', fl_cfg, binary_prot, quantizer_kwargs)

        elif codec_name == "temporal_only":
            from FL_code.other_protocols.SingleTypeCodecs import TemporalCodec
            codec = TemporalCodec(fl_cfg, binary_prot, quantizer_kwargs)
        elif codec_name == "retrain_only":
            from FL_code.other_protocols.SingleTypeCodecs import RetrainCodec
            codec = RetrainCodec(fl_cfg, binary_prot, quantizer_kwargs)

        elif codec_name == "cancer":
            from FL_code.cancer_protocol import CancerCodec
            codec = CancerCodec(fl_cfg, binary_prot, quantizer_kwargs)

    if codec is None:
        raise NotImplementedError(f"Codec '{codec_name}' not implemented.")

    return codec


def simulate_compression(
    codec: IdentityCodec, delta_vec: torch.Tensor, client_id: int, round_id: int,
    model_size: int | None = None, save_dir: Path | str | None = "compression_logs",
    server_eval_metrics: dict[str, float] | None = None, worker_eval_metrics: Sequence[float] | None = None,
    metric_keys: list[str] | None = None) -> torch.Tensor:
    """Simulate client encoding and server decoding for one flattened delta."""
    # Create record for this compression operation
    record = codec.create_record(round_id, client_id)
    record.model_size = model_size
    record.global_eval_metrics = server_eval_metrics or {}

    # Restructure worker metrics to match server metrics structure
    if worker_eval_metrics and metric_keys:
        num_metrics = len(metric_keys)
        train_metrics = {key: worker_eval_metrics[i] for i, key in enumerate(metric_keys)}
        test_metrics = {key: worker_eval_metrics[i + num_metrics] for i, key in enumerate(metric_keys)}
        record.worker_eval_metrics = {'train': train_metrics, 'test': test_metrics}

    # Encode (client-side simulation)
    payload = codec.encode(delta_vec, record)
    
    # Decode (server-side)
    reconstructed = codec.decode(payload, record)

    record.save_to_csv(save_dir)

    return reconstructed
