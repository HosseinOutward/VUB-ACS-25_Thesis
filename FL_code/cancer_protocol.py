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

'S' - SAMPLED RETRAIN:
    - Implemented by SampledCancerCodec in sampled_cancer_protocol.py
    - Similar to R, but trains from all server reconstructions and updates only the
      encoder head from a sampled first-pass payload before coding the rest

'F' - FROZEN:
    - NO training; reuses the last trained quantizer

'M' - MARGINAL (optional, not in default config):
    - Similar to 'P' but trains a marginal model from scratch instead of loading pretrained
"""
from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict

from FL_code.cancer_quantizer import WZQuantizerCancer
from FL_code.codec import (
    Access,
    CompressionRecord,
    HistoryEntry,
    BaseCodec,
    ReconstructionHistory,
    get_obj_compressed_size,
)
from FL_code.prior_calculator import PriorCalculator

if TYPE_CHECKING:
    from FL_code.utils import StateDictManager


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
    sampled_round_fraction: float = 0.02
    sampled_round_min_count: int = 1_024
    sampled_round_max_count: int = 50_000
    sampled_head_train_epochs: int = 10
    sampled_head_lr: float = 1e-3

    debug_save_codec_state: str = 'quantizer_state'
    debug_load_state: bool = False
    binary_protocol: bool = False
    mid_rate_protocol: bool = False
    use_model_slices: bool = True
    outlier_threshold: float | None = None
    training_progress_bar: bool = False
    records_dir: Path = Path("records")
    tf32: bool = True
    fused_optimizer: bool = True
    mixed_precision: bool = True


_CANCER_VARIANT_TOKENS: Mapping[str, str] = MappingProxyType({
    "temporal_only": "temporal_only",
    "retrain_only": "retrain_only",
    "non_wz_worker": "non_wz_learned_worker",
    "non_wz_server": "non_wz_learned_server",
    "sampled": "sampled",
    "bound_calc": "debug_cancerwithboundcalc",
})
_CANCER_RATE_TOKENS = frozenset({"binary", "mid_rate"})
_CANCER_OPTION_ORDER: tuple[str, ...] = ("variant", "rate", "model_slices", "outlier")
_CLIENT_RECONSTRUCTION_ROUNDS = frozenset({"P", "T"})


def _cancer_option_category(token: str) -> str:
    if token in _CANCER_VARIANT_TOKENS:
        return "variant"
    if token in _CANCER_RATE_TOKENS:
        return "rate"
    if token == "no_model_slices":
        return "model_slices"
    if token.startswith("outlier="):
        return "outlier"
    assert False, f"Unknown Cancer codec option token: {token!r}."


def _parse_outlier_threshold(token: str) -> float:
    _, _, raw_value = token.partition("=")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise AssertionError(f"Cancer codec option {token!r} must use a numeric threshold.") from exc
    assert threshold > 0, f"Cancer codec option {token!r} must be greater than 0."
    return threshold


class CancerRecord(CompressionRecord):
    """Compression record for one Cancer protocol client-round."""

    def __init__(self, round_id: int, client_id: int, method: str = "cancer",
                 phase: str | None = None, round_type: str | None = None,
                 bits_per_plane: int | None = None, num_planes: int | None = None) -> None:
        super().__init__(round_id, client_id, method)
        self.phase: str | None = phase
        self.round_type: str | None = round_type
        self.bins_per_plane: int | None = bits_per_plane
        self.num_planes: int | None = num_planes
        self.prior_rate: float | None = None
        self.marginal_rate: float | None = None
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
class CancerCodec(BaseCodec):
    """Cancer protocol codec coordinating WZ quantizers and side-information histories."""

    def __init__(
        self,
        c_cfg: CancerConfig | None = None,
        quantizer_kwargs: dict[str, Any] | None = None,
        codec_name: str = "cancer",
    ) -> None:
        super().__init__(codec_name)

        self.c_cfg = c_cfg if c_cfg is not None else CancerConfig()
        assert not self.c_cfg.debug_load_state, (
            "CancerConfig.debug_load_state requires FLConfig paths and is not supported by codec names."
        )
        if quantizer_kwargs is None:
            assert not self.c_cfg.use_model_slices, (
                "CancerCodec requires quantizer_kwargs['norm_slices'] when use_model_slices is enabled. "
                "Use create_codec(...) with a StateDictManager or add no_model_slices to the codec name."
            )
            quantizer_kwargs = {
                'norm_slices': None,
                'outlier_threshold': self.c_cfg.outlier_threshold
                if self.c_cfg.outlier_threshold is not None else False,
            }

        self.quantizer_kwargs = quantizer_kwargs

        self.reconstruction_history = ReconstructionHistory(self.c_cfg.max_side_info_count)
        self._history_warmup_finished: bool = False

        # Frozen state for frozen phase
        self.frozen_quantizers: list[WZQuantizerCancer | None] = []

    @staticmethod
    def validate_codec_tokens(option_tokens: Sequence[str]) -> None:
        """Validate Cancer protocol option tokens and their manual order."""
        seen_categories: set[str] = set()
        previous_order = -1
        for token in option_tokens:
            category = _cancer_option_category(token)
            assert category not in seen_categories, f"Cancer codec option category {category!r} appears more than once."
            seen_categories.add(category)

            order_index = _CANCER_OPTION_ORDER.index(category)
            expected = "|".join(_CANCER_OPTION_ORDER)
            assert order_index >= previous_order, (
                f"Cancer codec options must follow this manual order: {expected}; got {option_tokens!r}."
            )
            previous_order = order_index

            if category == "outlier":
                _parse_outlier_threshold(token)

    @classmethod
    def create_from_codec_name(
        cls,
        codec_name: str,
        protocol_name: str,
        option_tokens: Sequence[str],
        sd_manager: StateDictManager | None,
    ) -> BaseCodec:
        """Create a Cancer protocol codec variant from a validated codec name."""
        assert protocol_name == "cancer"

        c_cfg = CancerConfig()
        variant = "cancer"
        for token in option_tokens:
            if token in _CANCER_VARIANT_TOKENS:
                variant = _CANCER_VARIANT_TOKENS[token]
            elif token == "binary":
                c_cfg.binary_protocol = True
            elif token == "mid_rate":
                c_cfg.mid_rate_protocol = True
            elif token == "no_model_slices":
                c_cfg.use_model_slices = False
            elif token.startswith("outlier="):
                c_cfg.outlier_threshold = _parse_outlier_threshold(token)
            else:
                assert False, f"Unknown Cancer codec option token: {token!r}."

        if c_cfg.binary_protocol:
            c_cfg.routine_phase = tuple((phase_type, 2, 1) for phase_type, _, _ in c_cfg.routine_phase)
        if c_cfg.mid_rate_protocol:
            c_cfg.routine_phase = tuple((phase_type, 2, 2) for phase_type, _, _ in c_cfg.routine_phase)

        norm_slices = None
        if c_cfg.use_model_slices:
            assert sd_manager is not None, "StateDictManager is required when CancerConfig.use_model_slices is enabled."
            norm_slices = sd_manager.get_slices()
        quantizer_kwargs = {
            "norm_slices": norm_slices,
            "outlier_threshold": c_cfg.outlier_threshold if c_cfg.outlier_threshold is not None else False,
        }

        if variant == "cancer":
            return cls(c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "temporal_only":
            from FL_code.other_protocols.SingleTypeCodecs import TemporalCodec
            return TemporalCodec(c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "retrain_only":
            from FL_code.other_protocols.SingleTypeCodecs import RetrainCodec
            return RetrainCodec(c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "non_wz_learned_worker":
            from FL_code.other_protocols.SingleTypeCodecs import SingleTypeCodec
            return SingleTypeCodec('TMM', c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "non_wz_learned_server":
            from FL_code.other_protocols.SingleTypeCodecs import SingleTypeCodec
            return SingleTypeCodec('RMM', c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "sampled":
            from FL_code.sampled_cancer_protocol import SampledCancerCodec
            return SampledCancerCodec(c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        if variant == "debug_cancerwithboundcalc":
            from FL_code.experiments.rd_mspe_wz import CancerWithBoundCalc
            return CancerWithBoundCalc(c_cfg, quantizer_kwargs=quantizer_kwargs, codec_name=codec_name)
        assert False, f"Unknown Cancer codec variant: {variant!r}."

    def _ensure_client_state(self, client_id: int) -> None:
        while len(self.frozen_quantizers) <= client_id:
            self.frozen_quantizers.append(None)

    @staticmethod
    def _base_round_type(round_type: str) -> tuple[str, bool, bool]:
        """Return the executable round type and marginal-prior modifiers."""
        include_training_si_in_prior = len(round_type) == 3
        assert not include_training_si_in_prior or round_type[2] == "M", (
            "Three-letter round types must end with M."
        )

        force_marginal_loss = len(round_type) != 1
        if force_marginal_loss:
            assert round_type[1] == "M" and round_type[0] in ("T", "R"), (
                f"Invalid marginal round type: {round_type}"
            )
            return round_type[0], force_marginal_loss, include_training_si_in_prior
        return round_type, force_marginal_loss, include_training_si_in_prior

    def _reconstruction_access(self, record: CancerRecord) -> Access:
        """Return the visibility of the reconstruction produced by this round."""
        assert record.round_type is not None
        round_type = self._base_round_type(record.round_type)[0]
        if round_type in _CLIENT_RECONSTRUCTION_ROUNDS:
            return Access.SHARED
        if round_type in {"R", "S", "F"}:
            return Access.SERVER
        return Access.NONE

    def create_record(self, round_id: int, client_id: int) -> CancerRecord:
        cfg = self.c_cfg
        self._ensure_client_state(client_id)
        is_warmup = round_id < len(cfg.warmup_phase)
        if not is_warmup and not self._history_warmup_finished:
            _history_access = lambda round_type_s: {
                    "P": Access.SERVER, "R": Access.SERVER, "S": Access.SERVER, "T": Access.SHARED
                }.get(self._base_round_type(round_type_s)[0], Access.NONE)
            self.reconstruction_history.finish_warmup(
                [_history_access(round_type) for round_type, _, _ in cfg.routine_phase])
            self._history_warmup_finished = True

        temp = (round_id - len(cfg.warmup_phase)) % len(cfg.routine_phase)
        round_type, round_bpp, round_np = cfg.warmup_phase[round_id] if is_warmup else cfg.routine_phase[temp]
        phase = "warmup" if is_warmup else "routine"

        return CancerRecord(
            round_id=round_id, client_id=client_id, method=self.codec_name, phase=phase,
            round_type=round_type, bits_per_plane=round_bpp, num_planes=round_np)

    def _train_quantizer_or_load(self, delta_vec: torch.Tensor, record: CancerRecord) -> None:
        """Train new quantizer or load pretrained if needed (P, T, R rounds)."""
        assert record.round_type is not None
        assert record.bins_per_plane is not None
        assert record.num_planes is not None

        client_idx = record.client_id
        round_type, force_marginal_loss, include_training_si_in_prior = self._base_round_type(record.round_type)
        self._ensure_client_state(client_idx)

        training_recons: tuple[HistoryEntry, ...] = ()
        prior_recons: tuple[HistoryEntry, ...] = ()
        target: torch.Tensor | None = None

        if round_type == "P":
            prior_recons = self.reconstruction_history.view(Access.SERVER, record)
        elif round_type == "R":
            target_recon = self.reconstruction_history.latest_for_client(Access.SERVER, record)
            server_recons = self.reconstruction_history.view(Access.SERVER, record)
            training_recons = tuple(
                recon
                for recon in server_recons
                if recon is not target_recon
            )
            prior_recons = (target_recon,)
            target = target_recon.tensor
            client_count = max((entry.client_id for entry in server_recons), default=-1) + 1
            assert len(training_recons) == min(
                record.round_id * client_count + client_idx - 1,
                self.c_cfg.max_side_info_count * client_count - 1
            ), "Retrain side-information count does not match protocol schedule."
        elif round_type == "T":
            training_recons = self.reconstruction_history.view(Access.SHARED, record)
            prior_recons = training_recons
            target = delta_vec
        else:
            assert False, f"Invalid round type for training: {round_type}"

        if include_training_si_in_prior:
            prior_recons = (*prior_recons, *training_recons)

        quantizer = WZQuantizerCancer(
            c_cfg=self.c_cfg, num_planes=record.num_planes,
            bins_per_plane=record.bins_per_plane, si_size=len(training_recons),
            marginal_loss=force_marginal_loss or round_type == "P", **self.quantizer_kwargs,
            extra_si_for_prior=[recon.tensor for recon in prior_recons]
        )

        # Load pretrained weights or train the model
        if round_type != "P":
            assert target is not None
            quantizer.train_model(target, [recon.tensor for recon in training_recons])
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
        self._ensure_client_state(record.client_id)
        assert record.round_type is not None

        if record.round_type != "F":
            self._train_quantizer_or_load(delta_vec, record)

        quantizer = self.frozen_quantizers[record.client_id]
        assert quantizer is not None, f"Missing quantizer for client {record.client_id}."

        bins, prep_metadata = quantizer.encoding_process(delta_vec)

        # Build payload
        payload = self._build_payload(bins, prep_metadata, quantizer, record)

        prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)

        # Compute marginal prior for comparison
        m_prior = PriorCalculator.compute_marginal_prior(
            bins, quantizer.bins_per_plane, quantizer.num_planes)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(m_prior, bins, quantizer.num_planes)

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
        assert record.round_type is not None
        round_type = self._base_round_type(record.round_type)[0]
        if round_type in _CLIENT_RECONSTRUCTION_ROUNDS:
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
        self._ensure_client_state(client_idx)
        quantizer = self.frozen_quantizers[client_idx]
        assert quantizer is not None, f"Missing quantizer for client {client_idx}."
        reconst = quantizer.decoding_process(payload["payload_content"])
        self.reconstruction_history.commit(
            reconst.detach().to(device="cpu", dtype=torch.float16),
            record,
            self._reconstruction_access(record),
        )

        return reconst


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
        CancerConfig(),
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
