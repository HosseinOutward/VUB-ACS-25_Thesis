"""
Cancer Protocol
========================================================================================

Assumptions:
---------------
- Execution order: round 0 client 0, round 0 client 1, ..., round 1 client 0, ...

PHASE STRUCTURE:
----------------
The protocol has two sequential phase groups:

1. WARMUP PHASE: Initial rounds to bootstrap the protocol
   - Runs once at the start (e.g., rounds 0-4)
   - Uses a sequence of round types with decreasing bits per plane
   - Example: P(16bpp,3p), T(8bpp,3p), R(4bpp,3p), R(4bpp,3p), R(4bpp,3p)

2. ROUTINE PHASE: Repeating cycle after warmup
   - Repeats indefinitely (e.g., rounds 5+)
   - Example cycle: T(2bpp,3p), R(2bpp,3p), F(2bpp,3p), F(2bpp,3p), ...

ROUND TYPES:
------------
Each round has a type that determines quantizer training and side information usage:

'P' - PRETRAINED:
    - Uses a pre-trained marginal WZ-RNN model (no side information)
    - Used for cold start when no reconstruction history exists

'T' - TEMPORAL:
    - Trains a NEW quantizer using client-side information only (i.e. on the client side)
    - Training target: the current delta vector (directly on the gradient)
    - Training side-info: client's own past reconstructions done during previous temporal rounds

'R' - RETRAIN:
    - Trains a NEW quantizer using current gradient and client-side info
    - Training target: the last reconstruction from server side-info
    - Training side-info: all of client's own past reconstructions except for the one used for target

'F' - FROZEN:
    - NO training; reuses the last trained quantizer
"""
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from FL_reworked.cancer_quantizer import WZQuantizerCancer
from FL_reworked.codec import IdentityCodec, CompressionRecord, compress_data_list, decompress_data_list
from FL_reworked.run_fl import FLConfig


# ============================================================================
# Cancer Protocol Configuration and Records
# ============================================================================
@dataclass
class CancerConfig:
    """Configuration for Cancer protocol phases and WZ model."""
    # Phase info, (phase type, bins per plane (not bits), num planes)
    warmup_phase: Tuple[Tuple[str, int, int]] = (('P', 16, 3), ('T', 8, 3)) + (('R', 4, 3),) * 3
    routine_phase: Tuple[str] = (('T', 2, 3), ('R', 2, 3)) + (('F', 2, 3),) * 5
    max_side_info_count: int = 5
    pretrain_pth_dir: str = r'../data/pre_trained_pth/'

    train_epochs: int = 10
    reconst_ld: float = 400.0
    train_sample_size: int = 300_000
    lr: float = 1e-3
    lr_step: int = 40
    tau_rate: float = 10.0
    tau: float = 1.3


class CancerRecord(CompressionRecord):
    def __init__(self, round_id: int, client_id: int, method: str = "cancer",
                 phase: Optional[str] = None, round_type: Optional[str] = None,
                 bits_per_plane: Optional[int] = None, num_planes: Optional[int] = None):
        assert method == "cancer", "CancerRecord must be used by method 'cancer'"
        super().__init__(round_id, client_id, method)
        self.phase: Optional[str] = phase
        self.round_type: Optional[str] = round_type
        self.bits_per_plane: Optional[int] = bits_per_plane
        self.num_planes: Optional[int] = num_planes

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "phase": self.phase,
            "round_type": self.round_type,
            "bits_per_plane": self.bits_per_plane,
            "num_planes": self.num_planes,
        })
        return result


# ============================================================================
# Cancer Codec Implementation
# ============================================================================
class CancerCodec(IdentityCodec):
    def __init__(self, fl_cfg:FLConfig):
        super().__init__()
        self.fl_cfg = fl_cfg
        self.num_clients = fl_cfg.num_clients
        self.c_cfg = CancerConfig()

        # Per-client reconstruction histories
        self.srvr_past_reconst: List[List[torch.Tensor]] = [[] for _ in range(self.num_clients)]
        self.client_past_reconst: List[List[torch.Tensor]] = [[] for _ in range(self.num_clients)]

        # Frozen state for frozen phase
        self.frozen_quantizers: List[Optional[WZQuantizerCancer]] = [None] * self.num_clients

    def create_record(self, round_id: int, client_id: int) -> CancerRecord:
        cfg = self.c_cfg
        is_warmup = round_id < len(cfg.warmup_phase)
        temp = (round_id - len(cfg.warmup_phase)) % len(cfg.routine_phase)
        round_type, round_bpp, round_np = cfg.warmup_phase[round_id] if is_warmup else cfg.routine_phase[temp]
        phase = "warmup" if is_warmup else "routine"

        return CancerRecord(
            round_id=round_id, client_id=client_id, method="cancer",
            phase=phase, round_type=round_type, bits_per_plane=round_bpp, num_planes=round_np)

    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        round_type, round_bpp, round_np = record.round_type, record.bits_per_plane, record.num_planes
        client_idx = record.client_id

        # Train new quantizer if needed (P, T, R rounds)
        if round_type in ('P', 'T', 'R'):
            # Determine training side info and target based on round type
            if round_type == 'P':
                train_si, target_x = [], None
            elif round_type == 'T':
                train_si = [item
                            for cid, reconst_list in enumerate(self.srvr_past_reconst)
                            for item in (reconst_list[:-1] if cid == client_idx else reconst_list)]
                target_x = self.srvr_past_reconst[client_idx][-1]
                assert len(train_si) == min(record.round_id * self.num_clients + client_idx - 1,
                                            self.c_cfg.max_side_info_count * self.num_clients - 1)
            elif round_type == 'R':
                train_si = [item for reconst_list in self.client_past_reconst for item in reconst_list]
                target_x = delta_vec
            else:
                raise ValueError(f"Invalid round type for training: {round_type}")

            self.frozen_quantizers[client_idx] = WZQuantizerCancer(
                c_cfg=self.c_cfg, fl_cfg=self.fl_cfg, num_planes=round_np, bins_per_plane=round_bpp,
                train_x_vec=target_x, side_info_list=train_si, pretrained=(round_type == 'P')
            )

            gc.collect()
            torch.cuda.empty_cache()

        # Encode using current quantizer
        bins = self.frozen_quantizers[client_idx].encoding_process(delta_vec)
        res = compress_data_list(bins)
        return res

    def _decompress(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        client_idx = record.client_id
        bins = decompress_data_list(payload)
        bins = torch.from_numpy(bins)
        reconst = self.frozen_quantizers[client_idx].decoding_process(bins)

        # Update server-side history (always)
        self._update_history(self.srvr_past_reconst[client_idx], reconst)

        # Update client-side history (only for T and P rounds)
        if record.round_type in ('T', 'P'):
            self._update_history(self.client_past_reconst[client_idx], reconst)

        return reconst

    def _update_history(self, history: List[torch.Tensor], item: torch.Tensor):
        history.append(item.to(torch.float16))
        if len(history) > self.c_cfg.max_side_info_count:
            history.pop(0)


if __name__ == "__main__":
    import numpy as np
    import gzip

    num_clients = 5
    num_rounds = 15
    vector_size = 1_000_000

    codec = CancerCodec(FLConfig(num_clients=num_clients))
    base_vector = torch.normal(0.0, 1.0, size=(vector_size,))

    print(f"Clients={num_clients}, Rounds={num_rounds}, Vector size={vector_size:,}\n")

    mape_list, comp_vs_baseline_list = [], []

    for round_id in range(num_rounds):
        base_vector = base_vector + torch.normal(0.0, 0.1, size=(vector_size,))
        client_deltas = [base_vector + torch.normal(0.0, 0.1, size=(vector_size,)) for _ in range(num_clients)]

        if round_id == 0:
            initial_mape = np.mean([
                torch.mean(torch.abs(base_vector - delta) / (torch.abs(base_vector) + 1e-8)).item() * 100
                for delta in client_deltas
            ])
            print(f"Initial MAPE (base vs clients): {initial_mape:.2f}%\n")

        round_mape, round_comp_ratio = [], []

        for client_id in range(num_clients):
            print(f"C{client_id} -- ", end='', flush=True)
            delta = client_deltas[client_id]
            record = codec.create_record(round_id, client_id)

            # Cancer compression
            compressed = codec.encode(delta, record)
            decompressed = codec.decode(compressed, record)

            # Baseline: float16 + gzip
            baseline_compressed = gzip.compress(delta.to(torch.float16).numpy().tobytes())

            # Metrics
            mape = torch.mean(torch.abs(delta - decompressed) / (torch.abs(delta) + 1e-8)).item() * 100
            comp_ratio = len(baseline_compressed) / len(compressed)

            round_mape.append(mape)
            round_comp_ratio.append(comp_ratio)

        avg_mape = np.mean(round_mape)
        avg_comp = np.mean(round_comp_ratio)
        mape_list.append(avg_mape)
        comp_vs_baseline_list.append(avg_comp)

        print(f"\nR{round_id:2d} [{record.phase[0]}|{record.round_type}|"
              f"{record.bits_per_plane}bpp|{record.num_planes}p] -- "
              f"MAPE={avg_mape:5.2f}%, compr_ratio(vs FP16+zip): {avg_comp:.2f}x")

    print(f"\n{'='*60}")
    print(f"Overall: MAPE={np.mean(mape_list):.2f}% | Comp vs baseline={np.mean(comp_vs_baseline_list):.2f}x")
    print(f"{'='*60}")
