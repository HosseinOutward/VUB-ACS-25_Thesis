from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import (
    Access,
    BaseProtocol,
    BaseRoundCodec,
    CompressionRecord,
    HistoryEntry,
    ReconstructionHistory,
    record_reconstruction_metrics,
)
from FL_code.FL_core.utils import ParsedConfigurableName, compress_data_list, decompress_data_list
from FL_code.cancer_protocol.prior_code import PriorCalculator

from .wz_quantizer import DedupedWZQuantizerCancer, PreprocessMetadata, WZcfgQuant


def _compressed_size(obj: Any) -> float:
    """Return compressed payload size in MB."""
    return len(compress_data_list(obj)) / (1024 ** 2)


class CancerRecord(CompressionRecord):
    """Compression record for one NewCancer client-round."""

    bins_per_plane: int | None = None
    num_planes: int | None = None
    prior_rate: float | None = None
    prior_stage_rates: list[float] | None = None
    training_prior_rate: float | None = None
    training_prior_stage_rates: list[float] | None = None
    marginal_rate: float | None = None
    marginal_stage_rates: list[float] | None = None
    stage_mses: list[float] | None = None
    stage_wmapes: list[float] | None = None
    stage_wmspe_sqrts: list[float] | None = None
    encoder_size: float | None = None
    decoder_size: float | None = None
    meta_data_size: float | None = None


class _WZRoundCodec(BaseRoundCodec):
    record_class = CancerRecord

    round_name: ClassVar[str]  # needs to be set in subclasses
    can_decode_where: ClassVar[Access]  # needs to be set in subclasses

    quantizer: DedupedWZQuantizerCancer

    c_cfg: WZcfgQuant
    _original_delta_vec: torch.Tensor | None = None
    _prior_side_information: list[torch.Tensor] | None = None
    quantizer_class: type[DedupedWZQuantizerCancer] = DedupedWZQuantizerCancer

    def create_r_record(self, round_id: int, client_id: int) -> CancerRecord:
        raw_record = super().create_r_record(round_id, client_id)
        assert isinstance(raw_record, CancerRecord)
        raw_record.update_record_fields(
            bins_per_plane=self.c_cfg.bins_per_plane,
            num_planes=self.c_cfg.num_planes,
        )
        return raw_record

    def options_to_config(
        self,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Build the round's quantizer config; extra name tokens override WZcfgQuant fields."""
        self.c_cfg = WZcfgQuant(
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            norm_slices=sd_slices,
            cache_dir=cache_dir,
            cache_overwrite=cache_overwrite,
            **quant_overrides,
        )

    @staticmethod
    def validate_cfg(
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        **quant_overrides: Any,
    ) -> None:
        assert isinstance(bins_per_plane, int) and bins_per_plane > 0
        assert isinstance(num_planes, int) and num_planes > 0
        assert cache_dir is None or isinstance(cache_dir, Path)
        assert isinstance(cache_overwrite, bool)
        # WZcfgQuant forbids extra fields, so a misspelled override token fails here.
        WZcfgQuant(bins_per_plane=bins_per_plane, num_planes=num_planes, **quant_overrides)

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        raise NotImplementedError("Subclasses must return the quantizer training data.")

    def set_retrained_prior_si(
        self, history: ReconstructionHistory, client_id: int
    ) -> None:
        """Store server-visible side information for conditional prior estimation."""
        self._prior_side_information = [
            entry.tensor
            for client_history in history.view(Access.SERVER_ONLY, client_id).values()
            for entry in client_history
        ]

    def build_quantizer(self, x: torch.Tensor | None, si: list[torch.Tensor] | None) -> DedupedWZQuantizerCancer:
        assert x is not None and si is not None
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        self.quantizer.train_model(x, si)
        gc.collect()
        torch.cuda.empty_cache()
        return quantizer

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        self._original_delta_vec = delta_vec

        bins, soft_codes, prep_metadata = self.quantizer.encoding_process(delta_vec)
        record.update_record_fields(
            encoder_size=_compressed_size(self.quantizer.coding_model.encoder_state_dict()),
            decoder_size=_compressed_size(self.quantizer.coding_model.decoder_state_dict()),
            meta_data_size=_compressed_size(prep_metadata),
        )

        self.add_prior_to_record(bins, soft_codes, prep_metadata, record)

        return {'payload_content': (bins, prep_metadata)}

    def add_prior_to_record(
        self,
        bins: torch.Tensor,
        soft_codes: torch.Tensor | None,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> None:
        """Compute and store cumulative and per-stage prior rates in the record."""
        assert soft_codes is not None, "Prior retraining needs the encoder's soft codes; estimator quantizers do not produce them."
        prior_cfg = self.c_cfg.model_copy(update={"marginal_loss": False})
        prior_side_information = (
            self.quantizer.side_info_list_used
            if self._prior_side_information is None
            else self._prior_side_information
        )
        assert prior_side_information is not None
        side_info = self.quantizer.format_side_info_tensor(
            metadata, prior_side_information)
        prior_model = PriorCalculator.make_trained_prior_model(
            bins.long(), soft_codes, side_info, prior_cfg)
        # The quantizer's calculator dedupes identical (bins, side info) rows; the priors it
        # returns are identical to the full pass because they are a pure function of that row.
        prior_tensor = self.quantizer.prior_calculator.compute_prior_from_network(
            prior_model, bins, side_info)
        prior_stage_rates = PriorCalculator.compute_plane_rates(prior_tensor, bins, self.c_cfg.num_planes)

        marginal_prior = PriorCalculator.compute_marginal_prior(bins, self.c_cfg.bins_per_plane, self.c_cfg.num_planes)
        marginal_stage_rates = PriorCalculator.compute_plane_rates(marginal_prior, bins, self.c_cfg.num_planes)
        record.update_record_fields(
            prior_stage_rates=list(prior_stage_rates),
            prior_rate=sum(prior_stage_rates),
            marginal_stage_rates=list(marginal_stage_rates),
            marginal_rate=sum(marginal_stage_rates),
        )
        assert self.quantizer.training_prior_stage_rates is not None
        assert self.quantizer.training_prior_rate is not None
        record.update_record_fields(
            training_prior_stage_rates=list(self.quantizer.training_prior_stage_rates),
            training_prior_rate=self.quantizer.training_prior_rate,
        )

    def add_stage_distortion_to_record(self, stage_reconstructions: torch.Tensor, record: CancerRecord) -> None:
        """Store distortion metrics for every decoder refinement stage when the source vector is available."""
        if self._original_delta_vec is None:
            return

        stage_metrics = [
            record_reconstruction_metrics(self._original_delta_vec, reconstruction)
            for reconstruction in stage_reconstructions
        ]
        record.update_record_fields(
            stage_mses=[metrics["mse"] for metrics in stage_metrics],
            stage_wmapes=[metrics["wmape"] for metrics in stage_metrics],
            stage_wmspe_sqrts=[metrics["wmspe_sqrt"] for metrics in stage_metrics],
        )
        self._original_delta_vec = None

    def encode(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        payload = self.get_encod_payload(delta_vec, record)
        return compress_data_list(payload)

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        # Codecs overriding this methid: S
        payload_dict = decompress_data_list(payload)
        stage_reconstructions = self.quantizer.decoding_stages_process(
            payload_dict["payload_content"])
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]


class F_RoundCodec(_WZRoundCodec):
    """Reuse a frozen quantizer, optionally advancing every inference-SI round slot."""

    round_name: ClassVar[str] = "F"
    can_decode_where: Access

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        f_quant: DedupedWZQuantizerCancer,
        h_access: Access,
        side_information: list[torch.Tensor] | None = None,
    ) -> None:
        self.can_decode_where = h_access
        super().__init__(cfg_options, round_name_full)
        assert not cfg_options
        self.c_cfg = f_quant.c_cfg
        self.quantizer = f_quant
        if side_information is not None:
            assert not f_quant.no_side_info, (
                "A no-SI frozen quantizer cannot refresh side information.")
            assert len(side_information) == f_quant.side_info_size
            assert f_quant.vector_size is not None
            assert all(value.numel() == f_quant.vector_size for value in side_information)
            self.quantizer = copy(f_quant)
            self.quantizer.side_info_list_used = list(side_information)

    def build_quantizer(self, x: torch.Tensor | None, si: list[torch.Tensor] | None) -> DedupedWZQuantizerCancer:
        raise NotImplementedError("F_RoundCodec does not support building a new quantizer. Use the frozen quantizer instead.")


class P_RoundCodec(_WZRoundCodec):
    """Load a pretrained marginal quantizer (for the first round)."""
    round_name: ClassVar[str] = "P"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO

    def build_quantizer(self, x: torch.Tensor | None, si: list[torch.Tensor] | None) -> DedupedWZQuantizerCancer:
        assert x is None and si is None
        self.c_cfg = self.c_cfg.model_copy(update={"marginal_loss": True})
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=0)

        weight_path = self.c_cfg.pretrain_pth_dir / (
            f'bpp{self.c_cfg.bins_per_plane}_np{self.c_cfg.num_planes}_pretrained_wzq_rnn.pth')
        quantizer.coding_model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True))
        quantizer.side_info_list_used = []
        return quantizer


class R_RoundCodec(_WZRoundCodec):
    """Train a quantizer on the existing server reconstructions, with training target being the most recent recon, and excluding it from the side information."""
    round_name: ClassVar[str] = "R"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        history = given_history.view(Access.SERVER_ONLY, client_id)
        x_entry = history[client_id][-1]
        assert x_entry.client_id == client_id
        train_si = [entry.tensor for c_h in history.values() for entry in c_h if entry is not x_entry]
        return x_entry.tensor, train_si


class S_RoundCodec(_WZRoundCodec):
    """Two-pass sampled-retraining round against the quantizer's train-inference mismatch.

    NON-FUNCTIONAL as of 2026-07-26: get_encod_payload calls quantizer.encode_subset,
    quantizer.decode_subset and coding_model.encoder_head_state_dict, none of which survived
    the quantizer simplification in 22689ae. Scheduling an S round raises AttributeError on the
    first call. No active protocol references it, so it is kept as a design record only.

    A seeded sample of the delta is coded with a frozen copy of the initial server-trained
    model; its reconstruction is reproducible on the server, so both sides can retrain the
    encoder head and decoder on it. The remainder is coded with the retrained model, and only
    the small retrained encoder head has to travel back to the clients.
    """

    class S_record(CancerRecord):
        """Compression record with first-pass sample diagnostics for an S round."""

        sampled_count: int | None = None
        sampled_fraction: float | None = None
        sampled_data_size: float | None = None
        sampled_symbol_rate: float | None = None
        sampled_mse: float | None = None
        sampled_wmspe_sqrt: float | None = None
        retrain_loss: float | None = None
        head_size_only: float | None = None

    record_class = S_record
    round_name: ClassVar[str] = "S"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    sample_fraction: ClassVar[float] = 0.10
    _sample_original: torch.Tensor | None = None

    sampling_quantizer: DedupedWZQuantizerCancer

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Select one latest server reconstruction as target and all visible reconstructions as side information."""
        history = given_history.view(Access.SERVER_ONLY, client_id)
        train_si = [entry.tensor for c_h in history.values() for entry in c_h]
        x_tensor = torch.stack([c_h[-1].tensor for c_h in history.values()])
        x_tensor = x_tensor[torch.randint(x_tensor.shape[0], (), device=x_tensor.device)]
        return x_tensor, train_si

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Encode a seeded sample with the initial model, retrain on its reconstruction, encode the rest."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        assert isinstance(record, self.S_record)

        delta_prep, metadata = self.quantizer.preprocess_x(delta_vec)
        self.sampling_quantizer = copy(self.quantizer)
        self.sampling_quantizer.coding_model = deepcopy(self.quantizer.coding_model)

        sample_count = int(delta_vec.numel() * self.sample_fraction)
        assert sample_count > 100_000, f"Sample count {sample_count} is too small for retraining."
        sample_seed = self._sample_seed(record, delta_vec.numel())
        sample_indices, _ = self._split_indices(sample_seed, delta_vec.numel(), sample_count)
        self._sample_original = delta_vec[sample_indices]

        sample_bins = self.sampling_quantizer.encode_subset(delta_prep, sample_indices)
        sample_recons_prep = self.sampling_quantizer.decode_subset(
            sample_bins, sample_indices)
        retrain_loss = self._retrain_from_reconstructed_sample(sample_recons_prep, sample_indices, metadata)

        payload = super().get_encod_payload(delta_vec, record)
        full_bins, _ = payload["payload_content"]
        full_bins[:, sample_indices] = sample_bins

        encoder_head_state = self.quantizer.encoder_head_state_dict()
        sampling_payload = {
            "seed": sample_seed,
            "count": sample_count,
            "new_head_state": encoder_head_state,
        }
        payload["sampling"] = sampling_payload

        sample_data_size = _compressed_size([sampling_payload, sample_bins])
        head_size = _compressed_size(encoder_head_state)
        assert record.encoder_size is not None
        record.encoder_size += head_size
        record.update_record_fields(
            sampled_count=sample_count,
            sampled_fraction=sample_count / delta_vec.numel(),
            sampled_data_size=sample_data_size,
            sampled_symbol_rate=sample_data_size * (1024 ** 2) * 8 / sample_count,
            retrain_loss=retrain_loss,
            head_size_only=head_size
        )
        return payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Decode the sample with the initial model and the remainder with the retrained one, then reorder."""
        assert isinstance(record, self.S_record) and self._sample_original is not None
        payload_dict = decompress_data_list(payload)
        sampling_payload = payload_dict["sampling"]
        full_bins, _ = payload_dict["payload_content"]
        sample_indices, _ = self._split_indices(
            int(sampling_payload["seed"]), full_bins.shape[1], int(sampling_payload["count"]))

        stage_reconstructions = self.quantizer.decoding_stages_process(payload_dict["payload_content"])
        sampling_stage_reconstructions = self.sampling_quantizer.decoding_stages_process(
            payload_dict["payload_content"])
        stage_reconstructions[:, sample_indices] = sampling_stage_reconstructions[:, sample_indices]
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        reconst = stage_reconstructions[-1]

        sample_error = reconst[sample_indices] - self._sample_original
        record.update_record_fields(
            sampled_mse=sample_error.square().mean().item(),
            sampled_wmspe_sqrt=(
                sample_error.square().mean().sqrt().item()
                / (self._sample_original.abs().mean().item() + 1e-8) * 100
            ),
        )
        return reconst

    def _retrain_from_reconstructed_sample(
        self,
        sample_recons_prep: torch.Tensor,
        sample_indices: torch.Tensor,
        metadata: PreprocessMetadata,
    ) -> float:
        """Retrain the encoder head and downstream decoder/prior on the reproducible sample."""
        x_sample = sample_recons_prep.unsqueeze(1)
        side_info = self.quantizer.side_info_tensor(metadata)[
            sample_indices.to(self.quantizer.device)
        ]
        return self.quantizer.retrain_model(
            x_sample,
            side_info,
            self.c_cfg,
            x_sample.float().square().mean().item() / 2 + 1e-8,
        )

    @staticmethod
    def _sample_seed(record: CancerRecord, vector_size: int) -> int:
        return int(
            (record.round_id * 1_000_003 + record.client_id * 9_176 + vector_size) % (2**31 - 1))

    @staticmethod
    def _split_indices(
        seed: int, vector_size: int, sample_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert 0 < sample_count <= vector_size
        generator = torch.Generator().manual_seed(seed)
        sample_indices = (
            torch.randperm(vector_size, generator=generator)[:sample_count].sort().values
        )
        sample_mask = torch.zeros(vector_size, dtype=torch.bool)
        sample_mask[sample_indices] = True
        return sample_indices, torch.arange(vector_size)[~sample_mask]


class T_RoundCodec(_WZRoundCodec):
    """Train on client side only recons as SI, and current raw delta as target, then send the quantizer to the server for F rounds."""
    round_name: ClassVar[str] = "T"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO

    si: list[torch.Tensor]
    # Further (target, side information) problems mixed into training only; see _OffsetTrainedRoundCodec.
    extra_training_pairs: Sequence[tuple[torch.Tensor, list[torch.Tensor]]] = ()

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        si_entries = given_history.view(Access.TEMPORAL_TOO, client_id)[client_id]
        assert si_entries[-1].client_id == client_id
        assert si_entries[-1].access == Access.TEMPORAL_TOO
        return None, [entry.tensor for entry in si_entries]

    def build_quantizer(self, x: torch.Tensor | None, si: list[torch.Tensor] | None) -> DedupedWZQuantizerCancer:
        assert x is None and si is not None
        self.si = si
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        return quantizer

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        # The worker trains here, so encode timing includes quantizer training by design.
        self.quantizer.train_model(delta_vec, self.si, self.extra_training_pairs)
        gc.collect()
        torch.cuda.empty_cache()
        payload = super().get_encod_payload(delta_vec, record)
        payload['quantizer_state'] = self.quantizer.coding_model.decoder_state_dict()
        return payload

class Oracle_RoundCodec(T_RoundCodec):
    """it has magical access to the current raw delta as target while still having access to the server reconstructions as side information. for study purposes only."""
    round_name: ClassVar[str] = "O"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        history = given_history.view(Access.SERVER_ONLY, client_id)
        train_si = [entry.tensor for c_h in history.values() for entry in c_h]
        return None, train_si


class _OffsetTrainedRoundCodec(T_RoundCodec):
    """Train one round on several time-shifted repetitions of its own training problem.

    For offset n the problem is rebuilt as if the round ran n rounds earlier: the target is the
    round rid-n history entry of this client, and the side information is the history_window
    slots preceding it, in the per-client slot order the round uses at rid. Offset 0 stays the
    real problem (current delta against the current window), so the coded vector, the payload,
    and the quantizer frozen for later F rounds are unchanged; the lagged offsets only widen the
    training set. The protocol supplies the deeper history these offsets read.
    """

    train_offsets: int
    history_window: int

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        train_offsets: int,
        history_window: int,
    ) -> None:
        super().__init__(cfg_options, round_name_full)
        assert train_offsets > 0 and history_window > 0
        self.train_offsets = train_offsets
        self.history_window = history_window

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return the current side information and store the lagged training problems."""
        history = given_history.view(self.can_decode_where, client_id)
        window = self.history_window
        assert all(len(ledger) == window + self.train_offsets for ledger in history.values()), (
            f"{self.round_name_full} needs {window + self.train_offsets} history entries per client; "
            f"got {[len(ledger) for ledger in history.values()]}.")

        def window_ending_at(stop: int | None) -> list[torch.Tensor]:
            start = -window if stop is None else stop - window
            return [entry.tensor for ledger in history.values() for entry in ledger[start:stop]]

        # ledger[-offset] is the round rid-offset entry, and float16 history would otherwise
        # normalize a whole training target in half precision.
        self.extra_training_pairs = [
            (history[client_id][-offset].tensor.float(), window_ending_at(-offset))
            for offset in range(1, self.train_offsets + 1)
        ]
        return None, window_ending_at(None)


class OM_RoundCodec(_OffsetTrainedRoundCodec):
    """O round trained on its own lagged repetitions as well as the current problem."""
    round_name: ClassVar[str] = "OM"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY


class TM_RoundCodec(_OffsetTrainedRoundCodec):
    """T round trained on its own lagged repetitions as well as the current problem."""
    round_name: ClassVar[str] = "TM"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO


class _WZProtocol(BaseProtocol):
    max_per_client_recons_history = 5

    warmup_round_codecs: ClassVar[tuple[str, ...]]
    routine_round_codecs: ClassVar[tuple[str, ...]]
    protocol_name: ClassVar[str]

    last_frozen_state: dict[int, dict[str, Any]]
    refresh_frozen_si: bool = True
    retrained_prior_uses_server_si: bool = True
    train_offsets: int = 3
    # Deeper ledger read only by offset-trained rounds; None unless one is scheduled.
    _offset_history: ReconstructionHistory | None

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
    ) -> None:
        super().__init__(
            options,
            protocol_name_full,
            sd_slices,
            cache_dir,
            cache_overwrite,
        )
        self.last_frozen_state = {}
        # Same entries as _recons_history, kept train_offsets rounds longer so a round can
        # rebuild its own problem as it stood in earlier rounds. Only the tensor pointers repeat.
        self._offset_history = None
        if any(issubclass(rc, _OffsetTrainedRoundCodec) for rc in self._round_codec_classes.values()):
            self._offset_history = ReconstructionHistory(
                max_per_client=self.max_per_client_recons_history + self.train_offsets,
                routine_accesses=self.routine_accesses(),
            )

    def histories(self) -> tuple[ReconstructionHistory, ...]:
        """Return the decoder-visible ledger and, when offset rounds are scheduled, the deeper one."""
        if self._offset_history is None:
            return super().histories()
        return (*super().histories(), self._offset_history)

    def _shifted_side_informations(
        self,
        frozen_state: Mapping[str, Any],
        client_id: int,
    ) -> list[torch.Tensor]:
        """Return the current history window in the frozen quantizer's SI slot order."""
        side_information = [
            entry.tensor
            for client_history in self._recons_history.view(
                frozen_state["h_access"], client_id
            ).values()
            for entry in client_history
        ]
        assert len(side_information) == frozen_state["f_quant"].side_info_size, (
            "Refreshed side information no longer matches the frozen quantizer.")
        return side_information

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        rc_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        if not issubclass(rc_class, _WZRoundCodec):
            return rc_class(parsed.options, round_name_full)

        if rc_class is F_RoundCodec:
            frozen_state = self.last_frozen_state.get(client_id)
            assert frozen_state is not None, f"No frozen quantizer state exists yet for client {client_id}."
            side_information = None
            if self.refresh_frozen_si:
                side_information = self._shifted_side_informations(frozen_state, client_id)

            codec = F_RoundCodec(
                None,
                round_name_full,
                frozen_state["f_quant"],
                frozen_state["h_access"],
                side_information,
            )
            if self.retrained_prior_uses_server_si:
                codec.set_retrained_prior_si(self._recons_history, client_id)
            return codec

        assert isinstance(parsed.options, dict)
        training_history = self._recons_history
        protocol_inputs: dict[str, Any] = {}
        if issubclass(rc_class, _OffsetTrainedRoundCodec):
            assert self._offset_history is not None
            training_history = self._offset_history
            protocol_inputs = {
                "train_offsets": self.train_offsets,
                "history_window": self.max_per_client_recons_history,
            }
        codec = rc_class(
            parsed.options | {
                "sd_slices": self.sd_slices,
                "cache_dir": self.cache_dir,
                "cache_overwrite": self.cache_overwrite,
            },
            round_name_full,
            **protocol_inputs,
        )
        assert isinstance(codec, _WZRoundCodec)

        t_x, t_si = codec.get_training_data(training_history, client_id)
        if self.retrained_prior_uses_server_si:
            codec.set_retrained_prior_si(self._recons_history, client_id)
        quantizer = codec.build_quantizer(t_x, t_si)

        self.last_frozen_state[client_id] = {
            'f_quant': quantizer, # already includes si used too
            'h_access': codec.can_decode_where,
        }
        return codec

    def config_snapshot(self) -> dict[str, Any]:
        """Return the WZ schedule, quantizer defaults, and protocol-side-information policy."""
        snapshot = super().config_snapshot()
        snapshot["quantizer_defaults"] = WZcfgQuant(
            bins_per_plane=1,
            num_planes=1,
        ).model_dump(
            mode="json",
            exclude={
                "bins_per_plane",
                "num_planes",
                "norm_slices",
                "cache_dir",
                "cache_overwrite",
            },
        )
        snapshot["refresh_frozen_si"] = self.refresh_frozen_si
        snapshot["retrained_prior_uses_server_si"] = self.retrained_prior_uses_server_si
        if self._offset_history is not None:
            snapshot["train_offsets"] = self.train_offsets
            snapshot["max_per_client_offset_history"] = self._offset_history.max_per_client
        return snapshot


class WZCancerProtocol(_WZProtocol):
    """WZ cancer schedule using identity warmup, training rounds, then frozen models with refreshed SI."""

    warmup_round_codecs: tuple[str, ...] = (
        "I",
        "I",
        f"O|bins_per_plane=8|num_planes=3",
        f"O|bins_per_plane=4|num_planes=3",
    )
    routine_round_codecs: tuple[str, ...] = (
        f"O|bins_per_plane=3|num_planes=3",
        "F",
        "F",
        "F",
        "F",
    )
    protocol_name: ClassVar[str] = "wz_cancer"

    def options_to_config(
        self,
        replace_O_with: str | None = None,
        refresh_frozen_si: bool = True,
        retrained_prior_uses_server_si: bool = True,
        train_at_round: int | None = None,
        train_offsets: int | None = None,
    ) -> None:
        """Configure prior SI overrides, optionally replace every scheduled Oracle round, and
        optionally collapse the schedule to one training round followed by frozen rounds only."""
        assert isinstance(refresh_frozen_si, bool), "refresh_frozen_si must be boolean."
        assert isinstance(retrained_prior_uses_server_si, bool), (
            "retrained_prior_uses_server_si must be boolean.")
        self.refresh_frozen_si = refresh_frozen_si
        self.retrained_prior_uses_server_si = retrained_prior_uses_server_si
        if train_offsets is not None:
            assert isinstance(train_offsets, int) and train_offsets > 0, (
                "train_offsets must be a positive integer count of lagged training problems.")
            self.train_offsets = train_offsets
        if replace_O_with is not None:
            self.warmup_round_codecs = tuple(
                rc.replace("O|", f"{replace_O_with}|")
                if rc.startswith("O|") else rc for rc in self.warmup_round_codecs)
            self.routine_round_codecs = tuple(
                rc.replace("O|", f"{replace_O_with}|")
                if rc.startswith("O|") else rc for rc in self.routine_round_codecs)
        if train_at_round is not None:
            offset_round_names = {rc.round_name for rc in _OffsetTrainedRoundCodec.__subclasses__()}
            needed_history = self.max_per_client_recons_history + (
                self.train_offsets
                if self.routine_round_codecs[0].partition("|")[0] in offset_round_names else 0)
            assert train_at_round >= needed_history, (
                f"train_at_round={train_at_round} leaves fewer than "
                f"{needed_history} preceding rounds to seed history.")
            # Warmup is indexed by round id, so padding it puts the single training round exactly on
            # train_at_round; the padding slots only fill index space and are never replayed.
            self.warmup_round_codecs = ("I",) * train_at_round + (self.routine_round_codecs[0],)
            self.routine_round_codecs = ("F",)
