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
from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict

from FL_code.cancer_quantizer import WZQuantizerCancer
from FL_code.codec import IdentityCodec, CompressionRecord, get_obj_compressed_size
from FL_code.prior_calculator import PriorCalculator
from FL_code.run_fl import FLConfig


# ============================================================================
# Cancer Protocol Configuration and Records
# ============================================================================

class CancerConfig(BaseModel):
    """Configuration for Cancer protocol phases and WZ model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Phase info, (phase type, bins per plane (not bits), num planes)
    warmup_phase: tuple[tuple[str, int, int], ...] = (('P', 8, 3), ('T', 8, 3)) + (('R', 4, 3),) * 3
    routine_phase: tuple[tuple[str, int, int], ...] = (('T', 2, 3), ('T', 2, 3), ('R', 2, 3)) + (('F', 2, 3),) * 6

    max_side_info_count: int = 5
    pretrain_pth_dir: Path = Path('data/pre_trained_pth')  # ignored if train_marginal=True

    train_epochs: int = 70
    reconst_ld: float = 200.0
    train_sample_size: int = 300_000
    lr: float = 1e-3
    lr_step: int = 35
    tau: float = 1.3
    tau_rate: float = 10.0
    quantizer_train_repeats: int = 3
    prior_train_repeats: int = 3

    debug_save_codec_state: str = 'quantizer_state'
    debug_load_state: bool = False
    binary_protocol: bool = False
    mid_rate_protocol: bool = False
    use_model_slices: bool = True
    outlier_threshold: float | None = None


def build_cancer_config_for_fl(fl_cfg: FLConfig, binary_prot: bool | None = None) -> CancerConfig:
    """Create the Cancer protocol configuration implied by an FL configuration."""
    cfg = CancerConfig()
    codec_label = (fl_cfg.run_name or fl_cfg.codec).lower()

    if fl_cfg.debug_mode:
        cfg.train_epochs = 1
        cfg.train_sample_size = 100_000

    cfg.binary_protocol = "_binary" in codec_label
    cfg.mid_rate_protocol = "_mid_rate" in codec_label or "_mid" in codec_label
    cfg.use_model_slices = "_basic_norm" not in codec_label
    cfg.outlier_threshold = 1.6 if "_w_outlier" in codec_label else None
    cfg.debug_load_state = "_load_state" in codec_label

    if binary_prot is not None:
        cfg.binary_protocol = binary_prot

    if cfg.binary_protocol:
        cfg.routine_phase = tuple((phase_type, 2, 1) for phase_type, _, _ in cfg.routine_phase)
    if cfg.mid_rate_protocol:
        cfg.routine_phase = tuple((phase_type, 2, 2) for phase_type, _, _ in cfg.routine_phase)
    return cfg


class SideInformationTensor(BaseModel):
    """A side-information tensor with explicit ownership and data-flow metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tensor: torch.Tensor
    owner: Literal["server", "client"]
    round_id: int
    client_id: int


class SideInformationBundle:
    """Side information selected for one quantizer and prior-training decision."""

    def __init__(
        self,
        training: Sequence[SideInformationTensor] = (),
        prior: Sequence[SideInformationTensor] = (),
        target: torch.Tensor | None = None,
    ) -> None:
        self.training: tuple[SideInformationTensor, ...] = tuple(training)
        self.prior: tuple[SideInformationTensor, ...] = tuple(prior)
        self.target: torch.Tensor | None = target

    @property
    def training_tensors(self) -> list[torch.Tensor]:
        """Return tensors consumed by quantizer training."""
        return [recon.tensor for recon in self.training]

    @property
    def prior_tensors(self) -> list[torch.Tensor]:
        """Return tensors consumed only by prior estimation."""
        return [recon.tensor for recon in self.prior]

    def with_training_as_prior(self) -> SideInformationBundle:
        """Return a bundle that also exposes training reconstructions to prior estimation."""
        return SideInformationBundle(
            training=self.training,
            prior=(*self.prior, *self.training),
            target=self.target,
        )


class BinsCodecRecord(CompressionRecord):
    """Compression record with quantized-bin rate diagnostics."""

    def __init__(self, round_id: int, client_id: int, bins_per_plane: int | None, method: str) -> None:
        super().__init__(round_id, client_id, method)
        self.bins_per_plane: int | None = bins_per_plane
        self.prior_rate: float | None = None
        self.marginal_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "bins_per_plane": self.bins_per_plane,
            "prior_rate": self.prior_rate,
            "marginal_rate": self.marginal_rate,
        })
        return result


class CancerRecord(BinsCodecRecord):
    """Compression record for one Cancer protocol client-round."""

    def __init__(self, round_id: int, client_id: int, method: str = "cancer",
                 phase: str | None = None, round_type: str | None = None,
                 bits_per_plane: int | None = None, num_planes: int | None = None) -> None:
        assert method == "cancer", "CancerRecord must be used by method 'cancer'"
        super().__init__(round_id, client_id, bits_per_plane, method)
        self.phase: str | None = phase
        self.round_type: str | None = round_type
        self.num_planes: int | None = num_planes
        self.encoder_decoder_size: float | None = None
        self.meta_data_size: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "phase": self.phase,
            "round_type": self.round_type,
            "bins_per_plane": self.bins_per_plane,
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
    """Cancer protocol codec coordinating WZ quantizers and side-information histories."""

    def __init__(
        self,
        fl_cfg: FLConfig,
        binary_prot: bool = False,
        quantizer_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__(fl_cfg)
        if quantizer_kwargs is None:
            quantizer_kwargs = {'norm_slices': False, 'outlier_threshold': False}

        self.fl_cfg = fl_cfg
        self.num_clients = fl_cfg.num_clients

        self.c_cfg = build_cancer_config_for_fl(fl_cfg, binary_prot)

        self.quantizer_kwargs = quantizer_kwargs

        # Per-client reconstruction histories
        self.srvr_past_reconst: list[list[SideInformationTensor]] = [[] for _ in range(self.num_clients)]
        self.client_past_reconst: list[list[SideInformationTensor]] = [[] for _ in range(self.num_clients)]

        # Frozen state for frozen phase
        self.frozen_quantizers: list[WZQuantizerCancer | None] = [None] * self.num_clients

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
        assert record.round_type is not None
        assert record.bins_per_plane is not None
        assert record.num_planes is not None
        
        round_type = record.round_type
        client_idx = record.client_id

        remove_si = len(round_type) == 3
        assert not remove_si or round_type[2] == "M", "Three-letter round types must end with M."
        force_marginal_loss = len(round_type) != 1
        if force_marginal_loss:
            assert round_type[1] == "M" and round_type[0] in ("T", "R"), f"Invalid marginal round type: {round_type}"
            round_type = round_type[0]

        if round_type == "P": # Pretrained
            bundle = SideInformationBundle(
                prior=tuple(recon for history in self.srvr_past_reconst for recon in history))
        elif round_type == "R":
            target_recon = self.srvr_past_reconst[client_idx][-1]
            training_recons = tuple(
                recon
                for cid, history in enumerate(self.srvr_past_reconst)
                for recon in (history[:-1] if cid == client_idx else history)
            )
            bundle = SideInformationBundle(
                training=training_recons,
                prior=(target_recon,),
                target=target_recon.tensor,
            )
            assert len(bundle.training) == min(
                record.round_id * self.num_clients + client_idx - 1,
                self.c_cfg.max_side_info_count * self.num_clients - 1
            ), "Retrain side-information count does not match protocol schedule."
        elif round_type == "T": # Temporal
            training_recons = tuple(self.client_past_reconst[client_idx])
            bundle = SideInformationBundle(
                training=training_recons,
                prior=training_recons,
                target=delta_vec,
            )
        else:
            raise ValueError(f"Invalid round type for training: {round_type}")

        if remove_si:
            bundle = bundle.with_training_as_prior()

        quantizer = WZQuantizerCancer(
            c_cfg=self.c_cfg, fl_cfg=self.fl_cfg, num_planes=record.num_planes,
            bins_per_plane=record.bins_per_plane, si_size=len(bundle.training),
            marginal_loss=force_marginal_loss, **self.quantizer_kwargs,
            extra_si_for_prior=bundle.prior_tensors
        )

        # Load pretrained weights or train the model
        if self.c_cfg.debug_load_state:
            quantizer.coding_model.eval()
        elif round_type != "P":
            assert bundle.target is not None
            quantizer.train_model(bundle.target, bundle.training_tensors)
        else:
            weight_path = Path(self.c_cfg.pretrain_pth_dir) / (
                f"bpp{record.bins_per_plane}_np{record.num_planes}_pretrained_wzq_rnn.pth"
            )
            quantizer.coding_model.load_state_dict(torch.load(weight_path), strict=False)
            quantizer.side_info_list_used = []  # Pretrained models are marginal

        self.frozen_quantizers[client_idx] = quantizer

        gc.collect()
        torch.cuda.empty_cache()

    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict:
        if self.c_cfg.debug_load_state:
            codec_state_path = self.fl_cfg.debug_data_folder / self.c_cfg.debug_save_codec_state
            codec_state_path = codec_state_path / f'round_{record.round_id}_client_{record.client_id}.pt'
            if self.fl_cfg.debug_continue_from_saved_data and not codec_state_path.exists():
                self.c_cfg.debug_load_state = False
                if self.fl_cfg.debug_continue_then_save:
                    self.fl_cfg.debug_save_train_data = True

        if record.round_type != 'F':
            self._train_quantizer_or_load(delta_vec, record)

        quantizer = self.frozen_quantizers[record.client_id]

        if self.c_cfg.debug_load_state:
            print(f"Debug load state for R{record.round_id}C{record.client_id} -- loading quantizer state from disk")
            assert not self.fl_cfg.debug_save_train_data
            q_state = torch.load(codec_state_path)

            quantizer.coding_model.load_state_dict(q_state['coding_model'])
            quantizer.side_info_list_used = q_state['side_info_list_used']
            quantizer.extra_si_for_prior = q_state['extra_si_for_prior']
            prior = q_state['prior']

        bins, prep_metadata = quantizer.encoding_process(delta_vec)

        # Build payload
        payload = self._build_payload(bins, prep_metadata, quantizer, record)

        # Add prior info to record for analysis
        if not self.c_cfg.debug_load_state:
            prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)

        # Compute marginal prior for comparison
        m_prior = PriorCalculator.compute_marginal_prior(
            bins, quantizer.bins_per_plane, quantizer.num_planes)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(m_prior, bins, quantizer.num_planes)

        if self.fl_cfg.debug_save_train_data:
            codec_state_path = self.fl_cfg.debug_data_folder / self.c_cfg.debug_save_codec_state
            assert not codec_state_path.exists() or len(list(codec_state_path.iterdir())) != 0
            codec_state_path.mkdir(parents=True, exist_ok=True)
            codec_state_path = codec_state_path / f'round_{record.round_id}_client_{record.client_id}.pt'
            q_state = {
                'coding_model': quantizer.coding_model.state_dict(),
                'side_info_list_used': quantizer.side_info_list_used,
                'extra_si_for_prior': quantizer.extra_si_for_prior,
                'prior': prior,
            }
            torch.save(q_state, codec_state_path)

        return payload

    def _build_payload(
        self,
        bins: torch.Tensor,
        prep_metadata: tuple[torch.Tensor, tuple],
        quantizer: WZQuantizerCancer,
        record: CancerRecord
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        assert quantizer is not None, f"Missing quantizer for client {client_idx}."
        reconst = quantizer.decoding_process(payload["payload_content"])

        self._update_history(
            self.srvr_past_reconst[client_idx], reconst, owner="server",
            round_id=record.round_id, client_id=client_idx
        )

        assert record.round_type is not None
        if record.round_type[0] in ("T", "P"):
            self._update_history(
                self.client_past_reconst[client_idx], reconst, owner="client",
                round_id=record.round_id, client_id=client_idx
            )

        return reconst

    def _update_history(
        self,
        history: list[SideInformationTensor],
        item: torch.Tensor,
        owner: Literal["server", "client"],
        round_id: int,
        client_id: int
    ) -> None:
        history.append(SideInformationTensor(
            tensor=item.to(torch.float16), owner=owner, round_id=round_id, client_id=client_id
        ))
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
        print(f"\nR{round_id:2d} [{r.phase[0]}|{r.round_type}|{r.bins_per_plane}bpp|{r.num_planes}p]")
        print(f"  MAPE={avg_mape:5.2f}% | MSPE_sqrt={avg_mspe_sqrt:5.2f}% | Comp_ratio={avg_comp_ratio:.2f}x")
        print(f"  Prior_rate={avg_prior_rate:.3f}bpp | Marginal_rate={avg_marginal_rate:.3f}bpp | Entropy_real_rate={avg_entropy_real_rate:.3f}bpp")

    print(f"\n{'='*70}")
    print(f"Overall Metrics:")
    print(f"  MAPE={np.mean(mape_list):.2f}% | MSPE_sqrt={np.mean(mspe_sqrt_list):.2f}%")
    print(f"  Comp_ratio={np.mean(comp_ratio_list):.2f}x")
    print(f"  Prior_rate={np.mean(prior_rate_list):.3f}bpp | Marginal_rate={np.mean(marginal_rate_list):.3f}bpp | Entropy_real_rate={np.mean(entropy_real_rate_list):.3f}bpp")
    print(f"{'='*70}")
