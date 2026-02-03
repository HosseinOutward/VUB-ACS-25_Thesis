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

'M' - MARGINAL (optional, not in default config):
    - Similar to 'P' but trains a marginal model from scratch instead of loading pretrained
"""
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from cancer_quantizer import WZQuantizerCancer
from codec import IdentityCodec, CompressionRecord, get_obj_compressed_size
from prior_calculator import PriorCalculator
from run_fl import FLConfig, _DEBUG_FLAG


# ============================================================================
# Cancer Protocol Configuration and Records
# ============================================================================
@dataclass
class CancerConfig:
    """Configuration for Cancer protocol phases and WZ model."""
    # Phase info, (phase type, bins per plane (not bits), num planes)
    warmup_phase: Tuple[Tuple[str, int, int]] = (('P', 8, 3), ('T', 8, 3)) + (('R', 4, 3),) * 3
    routine_phase: Tuple[Tuple[str, int, int]] = (('T', 2, 3), ('T', 2, 3), ('R', 2, 3)) + (('F', 2, 3),) * 6

    max_side_info_count: int = 5
    pretrain_pth_dir: str = r'data/pre_trained_pth/' # ignored if train_marginal=True

    train_epochs: int = 70 if not _DEBUG_FLAG else 1
    reconst_ld: float = 300.0
    train_sample_size: int = 300_000 if not _DEBUG_FLAG else 100_000
    lr: float = 1e-3
    lr_step: int = 35
    tau: float = 1.3
    tau_rate: float = 10.0
    quantizer_train_repeats = 3
    prior_train_repeats = 3


class BinsCodecRecord(CompressionRecord):
    def __init__(self, round_id: int, client_id: int, bits_per_plane: int, method: str):
        super().__init__(round_id, client_id, method)
        self.bits_per_plane: Optional[int] = bits_per_plane
        self.prior_rate: Optional[float] = None
        self.marginal_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "bits_per_plane": self.bits_per_plane,
            "prior_rate": self.prior_rate,
            "marginal_rate": self.marginal_rate,
        })
        return result


class CancerRecord(BinsCodecRecord):
    def __init__(self, round_id: int, client_id: int, method: str = "cancer",
                 phase: Optional[str] = None, round_type: Optional[str] = None,
                 bits_per_plane: Optional[int] = None, num_planes: Optional[int] = None):
        assert method == "cancer", "CancerRecord must be used by method 'cancer'"
        super().__init__(round_id, client_id, bits_per_plane, method)
        self.phase: Optional[str] = phase
        self.round_type: Optional[str] = round_type
        self.num_planes: Optional[int] = num_planes
        self.encoder_decoder_size: Optional[int] = None
        self.meta_data_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "phase": self.phase,
            "round_type": self.round_type,
            "bits_per_plane": self.bits_per_plane,
            "num_planes": self.num_planes,
            "prior_rate": self.prior_rate,
            "marginal_rate": self.marginal_rate,
            "encoder_decoder_size": self.encoder_decoder_size,
            "meta_data_size": self.meta_data_size,
        })
        return result


# ============================================================================
# Cancer Codec Implementation
# ============================================================================
class CancerCodec(IdentityCodec):
    def __init__(self, fl_cfg: FLConfig, binary_prot=False, quantizer_kwargs=None):
        super().__init__()
        if quantizer_kwargs is None:
            quantizer_kwargs = {'norm_slices': False, 'outlier_threshold': False}

        self.fl_cfg = fl_cfg
        self.num_clients = fl_cfg.num_clients

        self.c_cfg = CancerConfig()
        if binary_prot:
            self.c_cfg.routine_phase = tuple((a[0],2,1) for a in self.c_cfg.routine_phase)

        self.quantizer_kwargs = quantizer_kwargs

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
            round_id=round_id, client_id=client_id, method="cancer", phase=phase,
            round_type=round_type, bits_per_plane=round_bpp, num_planes=round_np)

    def _train_quantizer_or_load(self, delta_vec: torch.Tensor, record: CancerRecord) -> None:
        """Train new quantizer or load pretrained if needed (P, T, R rounds)."""
        round_type, round_bpp, round_np = record.round_type, record.bits_per_plane, record.num_planes
        client_idx = record.client_id

        extra_si_for_prior = []
        force_marginal_loss = False
        if len(round_type) != 1:
            assert round_type[1] == "M" and round_type[0] in ['T', 'R']
            force_marginal_loss = True
            round_type = round_type[0]
            assert round_type not in ['P', 'M']

        # Determine training side info and target based on round type
        if round_type == 'P': # Pretrained
            train_si, target_x = None, None

        elif round_type == 'M': # Marginal
            train_si, target_x = None, delta_vec

        elif round_type == 'R': # Retrain
            train_si = [item
                        for cid, reconst_list in enumerate(self.srvr_past_reconst)
                        for item in (reconst_list[:-1] if cid == client_idx else reconst_list)]
            target_x = self.srvr_past_reconst[client_idx][-1]
            extra_si_for_prior = [target_x]
            assert len(train_si) == min(record.round_id * self.num_clients + client_idx - 1,
                                      self.c_cfg.max_side_info_count * self.num_clients - 1)

        elif round_type == 'T': # Temporal
            train_si = [item
                        for reconst_list in self.client_past_reconst
                        for item in reconst_list]
            target_x = delta_vec

        else:
            raise ValueError(f"Invalid round type for training: {round_type}")

        # Create a new quantizer instance
        quantizer = WZQuantizerCancer(
            c_cfg=self.c_cfg, fl_cfg=self.fl_cfg, num_planes=round_np, bins_per_plane=round_bpp,
            si_size=len(train_si) if train_si is not None else 0,
            marginal_loss=force_marginal_loss, **self.quantizer_kwargs,
            extra_si_for_prior = extra_si_for_prior
        )

        # Load pretrained weights or train the model
        if round_type != 'P':
            quantizer.train_model(target_x, train_si)
        else:
            weight_path = self.c_cfg.pretrain_pth_dir + f'bpp{round_bpp}_np{round_np}_pretrained_wzq_rnn.pth'
            quantizer.coding_model.load_state_dict(torch.load(weight_path), strict=False)
            quantizer.side_info_list_used = []  # Pretrained models are marginal

        self.frozen_quantizers[client_idx] = quantizer

        gc.collect()
        torch.cuda.empty_cache()

    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict:
        if record.round_type != 'F':
            self._train_quantizer_or_load(delta_vec, record)

        # Encode using current quantizer
        quantizer = self.frozen_quantizers[record.client_id]
        bins, prep_metadata = quantizer.encoding_process(delta_vec)

        # Build payload
        payload = self._build_payload(bins, prep_metadata, quantizer, record)

        # Add prior info to record for analysis
        prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)

        # Compute marginal prior for comparison
        m_prior = PriorCalculator.compute_marginal_prior(
            bins, quantizer.bins_per_plane, quantizer.num_planes)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(m_prior, bins, quantizer.num_planes)

        return payload

    def _build_payload(self, bins, prep_metadata, quantizer: WZQuantizerCancer, record: CancerRecord) -> dict:
        payload:Dict[str, Any] = {
            'payload_content': (bins, prep_metadata),
        }

        # Include model states for tracking overhead size
        if record.round_type in ['P', 'T']:
            # if it's a T round, clients sending the decoder state as well
            # for P rounds, decoder is sent to clients so they have reconstruction of non-side-info case
            payload['decoder_state'] = quantizer.coding_model.decoder.state_dict()
            record.encoder_decoder_size = get_obj_compressed_size(payload['decoder_state']) / (1024 ** 2)
        else:
            payload['encoder_state'] = quantizer.coding_model.encoder.state_dict()
            record.encoder_decoder_size = get_obj_compressed_size(payload['encoder_state']) / (1024 ** 2)

        record.meta_data_size = get_obj_compressed_size(prep_metadata) / (1024 ** 2)

        return payload

    def _decompress(self, payload: dict, record: CancerRecord) -> torch.Tensor:
        client_idx = record.client_id
        quantizer = self.frozen_quantizers[client_idx]
        reconst = quantizer.decoding_process(payload['payload_content'])

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

    num_clients = 5
    num_rounds = 15
    vector_size = 1_000_000

    # # Test with normal Cancer codec (default: no outlier handling, single slice)
    # print('Using Cancer codec (no preprocessing)...\n')
    # codec = CancerCodec(FLConfig(num_clients=num_clients))

    # Uncomment to test with preprocessing
    print('Using Cancer codec WITH vec_slices and outlier handling...\n')
    codec = CancerCodec(
        FLConfig(num_clients=num_clients),
        quantizer_kwargs = {'norm_slices': [slice(i, None, 3) for i in range(3)], 'outlier_threshold': False}
    )

    base_vector = torch.normal(0.0, 1.0, size=(vector_size,))

    print(f"Clients={num_clients}, Rounds={num_rounds}, Vector size={vector_size:,}\n")

    mape_list, mspe_sqrt_list, comp_ratio_list = [], [], []
    prior_rate_list, marginal_rate_list, entropy_real_rate_list = [], [], []

    for round_id in range(num_rounds):
        base_vector = base_vector + torch.normal(0.0, 0.1, size=(vector_size,))
        client_deltas = [base_vector + torch.normal(0.0, 0.1, size=(vector_size,)) for _ in range(num_clients)]

        if round_id == 0:
            initial_mape = np.mean([
                torch.mean(torch.abs(delta1 - delta2) / (torch.abs(base_vector) + 1e-8)).item() * 100
                for i, delta1 in enumerate(client_deltas) for delta2 in client_deltas[i+1:]
            ])
            initial_mspe_sqrt = np.mean([
                torch.sqrt(torch.mean(
                    (delta1 - delta2) ** 2 / (base_vector ** 2 + 1e-8)
                )).item() * 100
                for i, delta1 in enumerate(client_deltas) for delta2 in client_deltas[i+1:]
            ])
            print(f"Initial MAPE (clients vs clients): {initial_mape:.2f}%")
            print(f"Initial MSPE_sqrt (clients vs clients): {initial_mspe_sqrt:.2f}%\n")

        round_records = []

        for client_id in range(num_clients):
            print(f"C{client_id} -- ", end='', flush=True)
            delta = client_deltas[client_id]
            record = codec.create_record(round_id, client_id)
            record.model_size = vector_size

            # Cancer compression & decompression (metrics computed in encode/decode)
            compressed = codec.encode(delta, record)
            decompressed = codec.decode(compressed, record)

            round_records.append(record)

        # Use metrics from records
        avg_mape = np.mean([r.mape for r in round_records])
        avg_mspe_sqrt = np.mean([r.mspe_sqrt for r in round_records])
        avg_comp_ratio = np.mean([r.compression_ratio for r in round_records])
        avg_prior_rate = np.mean([r.prior_rate for r in round_records])
        avg_marginal_rate = np.mean([r.marginal_rate for r in round_records])
        avg_entropy_real_rate = np.mean([r.entropy_real_rate for r in round_records])

        mape_list.append(avg_mape)
        mspe_sqrt_list.append(avg_mspe_sqrt)
        comp_ratio_list.append(avg_comp_ratio)
        prior_rate_list.append(avg_prior_rate)
        marginal_rate_list.append(avg_marginal_rate)
        entropy_real_rate_list.append(avg_entropy_real_rate)

        # Print round summary using first client's record for phase info
        r = round_records[0]
        print(f"\nR{round_id:2d} [{r.phase[0]}|{r.round_type}|{r.bits_per_plane}bpp|{r.num_planes}p]")
        print(f"  MAPE={avg_mape:5.2f}% | MSPE_sqrt={avg_mspe_sqrt:5.2f}% | Comp_ratio={avg_comp_ratio:.2f}x")
        print(f"  Prior_rate={avg_prior_rate:.3f}bpp | Marginal_rate={avg_marginal_rate:.3f}bpp | Entropy_real_rate={avg_entropy_real_rate:.3f}bpp")

    print(f"\n{'='*70}")
    print(f"Overall Metrics:")
    print(f"  MAPE={np.mean(mape_list):.2f}% | MSPE_sqrt={np.mean(mspe_sqrt_list):.2f}%")
    print(f"  Comp_ratio={np.mean(comp_ratio_list):.2f}x")
    print(f"  Prior_rate={np.mean(prior_rate_list):.3f}bpp | Marginal_rate={np.mean(marginal_rate_list):.3f}bpp | Entropy_real_rate={np.mean(entropy_real_rate_list):.3f}bpp")
    print(f"{'='*70}")



