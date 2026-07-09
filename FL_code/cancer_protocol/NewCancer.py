from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple

import torch
from pydantic import BaseModel, ConfigDict

from FL_code.FL_core.codec import (
    Access,
    BaseProtocol,
    BaseRoundCodec,
    CompressionRecord,
    HistoryEntry,
    ReconstructionHistory,
)
from FL_code.FL_core.utils import compress_data_list

from .NewQuant import WZQuantizerCancer
from .NewPrior import PriorCalculator


PhasePlan = tuple[tuple[str, int, int], ...]


def _round_codec_names(phase_plan: PhasePlan) -> tuple[str, ...]:
    """Translate a (round_type, bins, planes) phase plan into scheduled round-codec names."""
    names = []
    for round_type, bins_per_plane, num_planes in phase_plan:
        base, modifiers = round_type[:1], round_type[1:]
        assert base in {"P", "T", "R", "F"} and modifiers in {"", "M", "MM"}, (
            f"Unknown NewCancer round type: {round_type!r}."
        )
        tokens = [f"bins={bins_per_plane}", f"planes={num_planes}"]
        if modifiers:
            tokens.append("marginal")
        if modifiers == "MM":
            tokens.append("priorTrainingSI")
        names.append(f"NewCancer{base}|{'|'.join(tokens)}")
    return tuple(names)


def _compressed_size(obj: Any) -> float:
    """Return compressed payload size in MB."""
    return len(compress_data_list(obj)) / (1024 ** 2)


class NewCancerConfig(BaseModel):
    """Configuration shared by NewCancer round codecs and produced by NewCancerProtocol."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    warmup_phase: PhasePlan = (("P", 8, 3), ("T", 8, 3)) + (("R", 4, 3),) * 3
    routine_phase: PhasePlan = (("T", 2, 3), ("T", 2, 3), ("R", 2, 3)) + (("F", 2, 3),) * 6

    max_side_info_count: int = 5
    pretrain_pth_dir: Path = Path("FL_code/data/pre_trained_pth")

    train_epochs: int = 70
    reconst_ld: float = 200.0
    train_sample_size: int = 300_000
    lr: float = 1e-3
    lr_step: int = 35
    tau: float = 1.3
    quantizer_train_repeats: int = 3
    prior_train_repeats: int = 3

    use_model_slices: bool = True
    outlier_threshold: float | None = None
    training_progress_bar: bool = False
    tf32: bool = True
    fused_optimizer: bool = True
    mixed_precision: bool = True


class NewCancerRecord(CompressionRecord):
    """Compression record for one NewCancer client-round."""

    phase: str | None = None
    round_type: str | None = None
    bins_per_plane: int | None = None
    num_planes: int | None = None
    prior_rate: float | None = None
    marginal_rate: float | None = None
    encoder_decoder_size: float | None = None
    meta_data_size: float | None = None


class _QuantizerTrainingData(NamedTuple):
    """Round-selected tensors used to fit a quantizer and estimate its decoder prior."""

    training_recons: tuple[HistoryEntry, ...]
    prior_recons: tuple[HistoryEntry, ...]
    target: torch.Tensor | None


class _NewCancerRound(BaseRoundCodec):
    """Shared quantizer training, encoding, and decoding for the NewCancer round types."""

    record_class = NewCancerRecord
    round_name = "NewCancerBase"  # placeholder: only the concrete subclasses are schedulable
    history_access_needs = Access.NONE
    # Which coding-model half travels in the payload, and whether the round is inherently marginal.
    payload_state_side: ClassVar[Literal["encoder", "decoder"]]
    always_marginal: ClassVar[bool] = False

    phase: str
    c_cfg: NewCancerConfig
    quantizer_kwargs: dict[str, Any]
    history: ReconstructionHistory
    frozen_quantizers: dict[int, WZQuantizerCancer]
    bins_per_plane: int
    num_planes: int
    force_marginal_loss: bool
    include_training_si_in_prior: bool

    def __init__(
        self,
        options: Mapping[str, Any] | None,
        round_name_full: str,
        c_cfg: NewCancerConfig,
        quantizer_kwargs: dict[str, Any],
        history: ReconstructionHistory,
        frozen_quantizers: dict[int, WZQuantizerCancer],
        phase: str,
    ) -> None:
        assert options, f"{round_name_full} requires bins= and planes= options."
        self.phase = phase
        self.c_cfg = c_cfg
        self.quantizer_kwargs = quantizer_kwargs
        self.history = history
        self.frozen_quantizers = frozen_quantizers
        super().__init__(options, round_name_full)

    def options_to_config(
        self, *, bins: int, planes: int, marginal: bool = False, priorTrainingSI: bool = False
    ) -> None:
        """Apply structural round-codec options parsed from the schedule entry."""
        self.bins_per_plane = bins
        self.num_planes = planes
        self.force_marginal_loss = marginal
        self.include_training_si_in_prior = priorTrainingSI

    @classmethod
    def validate_cfg(cls, bins: int, planes: int, marginal: bool = False, priorTrainingSI: bool = False) -> None:
        """Validate NewCancer round-codec schedule options."""
        assert isinstance(bins, int) and bins > 1, "NewCancer round option bins must be an int greater than 1."
        assert isinstance(planes, int) and planes > 0, "NewCancer round option planes must be a positive int."
        assert marginal in (True, False) and priorTrainingSI in (True, False), (
            "NewCancer round options marginal and priorTrainingSI must be boolean.")
        assert not priorTrainingSI or marginal, (
            "NewCancer round option priorTrainingSI requires marginal.")

    def create_round_record(self, round_id: int, client_id: int) -> NewCancerRecord:
        """Create a NewCancer metrics record for one client-round compression."""
        record = super().create_round_record(round_id, client_id)
        assert isinstance(record, NewCancerRecord)
        record.phase = self.phase
        record.round_type = self._record_round_type()
        record.bins_per_plane = self.bins_per_plane
        record.num_planes = self.num_planes
        return record

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> dict[str, Any]:
        """Encode one delta vector with the round's Wyner-Ziv quantizer state."""
        assert isinstance(record, NewCancerRecord)
        self._prepare_quantizer(delta_vec, record)
        quantizer = self._quantizer_for(record)
        bins, prep_metadata = quantizer.encoding_process(delta_vec)
        state = getattr(quantizer.coding_model, self.payload_state_side).state_dict()
        payload = {"payload_content": (bins, prep_metadata), f"{self.payload_state_side}_state": state}
        record.encoder_decoder_size = _compressed_size(state)
        record.meta_data_size = _compressed_size(prep_metadata)

        prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)
        marginal = PriorCalculator.compute_marginal_prior(bins, quantizer.bins_per_plane, quantizer.num_planes)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(marginal, bins, quantizer.num_planes)
        return payload

    def decode(self, payload: dict[str, Any], record: CompressionRecord) -> torch.Tensor:
        """Decode one NewCancer payload using the round's retained quantizer state."""
        return self.frozen_quantizers[record.client_id].decoding_process(payload["payload_content"])

    def _prepare_quantizer(self, delta_vec: torch.Tensor, record: NewCancerRecord) -> None:
        training_data = self._select_training_data(delta_vec, record)
        prior_recons = training_data.prior_recons
        if self.include_training_si_in_prior:
            prior_recons += training_data.training_recons
        quantizer = WZQuantizerCancer(
            c_cfg=self.c_cfg,
            num_planes=self.num_planes,
            bins_per_plane=self.bins_per_plane,
            si_size=len(training_data.training_recons),
            marginal_loss=self.force_marginal_loss or self.always_marginal,
            extra_si_for_prior=[entry.tensor for entry in prior_recons],
            **self.quantizer_kwargs,
        )
        self._fit_quantizer(quantizer, training_data)
        self.frozen_quantizers[record.client_id] = quantizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _select_training_data(
        self,
        delta_vec: torch.Tensor,
        record: NewCancerRecord,
    ) -> _QuantizerTrainingData:
        raise NotImplementedError

    def _fit_quantizer(
        self,
        quantizer: WZQuantizerCancer,
        training_data: _QuantizerTrainingData,
    ) -> None:
        assert training_data.target is not None
        quantizer.train_model(training_data.target, [entry.tensor for entry in training_data.training_recons])

    def _quantizer_for(self, record: NewCancerRecord) -> WZQuantizerCancer:
        quantizer = self.frozen_quantizers.get(record.client_id)
        assert quantizer is not None, (
            f"No trained NewCancer quantizer for client {record.client_id}; "
            "F rounds require an earlier training round for this client."
        )
        return quantizer

    def _record_round_type(self) -> str:
        round_type = self.round_name.removeprefix("NewCancer")
        assert round_type in {"P", "T", "R", "F"}
        return round_type + "M" * (self.force_marginal_loss + self.include_training_si_in_prior)


class NewCancerPretrainedRound(_NewCancerRound):
    """Pretrained marginal Wyner-Ziv round used for cold start."""

    round_name = "NewCancerP"
    history_access_needs = Access.TEMPORAL
    payload_state_side: ClassVar[Literal["encoder", "decoder"]] = "decoder"
    always_marginal: ClassVar[bool] = True

    @classmethod
    def validate_cfg(cls, bins: int, planes: int, marginal: bool = False, priorTrainingSI: bool = False) -> None:
        """Validate pretrained round-codec schedule options."""
        super().validate_cfg(bins=bins, planes=planes, marginal=marginal, priorTrainingSI=priorTrainingSI)
        assert not marginal, f"{cls.round_name} rounds cannot use marginal-prior modifiers."

    def _select_training_data(
        self,
        delta_vec: torch.Tensor,
        record: NewCancerRecord,
    ) -> _QuantizerTrainingData:
        return _QuantizerTrainingData((), self.history.view(Access.SERVER, record), None)

    def _fit_quantizer(
        self,
        quantizer: WZQuantizerCancer,
        training_data: _QuantizerTrainingData,
    ) -> None:
        weight_path = self.c_cfg.pretrain_pth_dir / (
            f"bpp{self.bins_per_plane}_np{self.num_planes}_pretrained_wzq_rnn.pth"
        )
        assert weight_path.exists(), f"Missing NewCancer pretrained weights: {weight_path}."
        quantizer.coding_model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True),
            strict=False,
        )


class NewCancerTemporalRound(_NewCancerRound):
    """Temporal NewCancer round trained from client-visible reconstructions."""

    round_name = "NewCancerT"
    history_access_needs = Access.TEMPORAL
    payload_state_side: ClassVar[Literal["encoder", "decoder"]] = "decoder"

    def _select_training_data(
        self,
        delta_vec: torch.Tensor,
        record: NewCancerRecord,
    ) -> _QuantizerTrainingData:
        training_recons = self.history.view(Access.TEMPORAL, record)
        return _QuantizerTrainingData(training_recons, training_recons, delta_vec)


class NewCancerRetrainRound(_NewCancerRound):
    """Retrain NewCancer round trained from server-visible reconstructions."""

    round_name = "NewCancerR"
    history_access_needs = Access.SERVER
    payload_state_side: ClassVar[Literal["encoder", "decoder"]] = "encoder"

    def _select_training_data(
        self,
        delta_vec: torch.Tensor,
        record: NewCancerRecord,
    ) -> _QuantizerTrainingData:
        server_recons = self.history.view(Access.SERVER, record)
        target_recon = next(
            (entry for entry in reversed(server_recons) if entry.client_id == record.client_id),
            None,
        )
        assert target_recon is not None, (
            f"No server reconstruction history exists for client {record.client_id}.")
        training_recons = tuple(entry for entry in server_recons if entry is not target_recon)
        return _QuantizerTrainingData(training_recons, (target_recon,), target_recon.tensor)


class NewCancerFrozenRound(_NewCancerRound):
    """Frozen NewCancer round that reuses the client's last trained quantizer."""

    round_name = "NewCancerF"
    history_access_needs = Access.SERVER
    payload_state_side: ClassVar[Literal["encoder", "decoder"]] = "encoder"

    @classmethod
    def validate_cfg(cls, bins: int, planes: int, marginal: bool = False, priorTrainingSI: bool = False) -> None:
        """Validate frozen round-codec schedule options."""
        super().validate_cfg(bins=bins, planes=planes, marginal=marginal, priorTrainingSI=priorTrainingSI)
        assert not marginal, f"{cls.round_name} rounds cannot use marginal-prior modifiers."

    def _prepare_quantizer(self, delta_vec: torch.Tensor, record: NewCancerRecord) -> None:
        self._quantizer_for(record)


_DEFAULT_CFG: NewCancerConfig = NewCancerConfig()


class NewCancerProtocol(BaseProtocol):
    """Round-based reimplementation of the Cancer Wyner-Ziv protocol."""

    protocol_name: ClassVar[str] = "NewCancer"
    warmup_round_codecs: ClassVar[tuple[str, ...]] = _round_codec_names(_DEFAULT_CFG.warmup_phase)
    routine_round_codecs: ClassVar[tuple[str, ...]] = _round_codec_names(_DEFAULT_CFG.routine_phase)
    max_per_client_recons_history: ClassVar[int | None] = _DEFAULT_CFG.max_side_info_count
    # Variant subclasses replace every scheduled training round with this round type.
    training_round_type: ClassVar[str | None] = None

    c_cfg: NewCancerConfig
    sd_slices: Sequence[slice] | None
    _frozen_quantizers: dict[int, WZQuantizerCancer]

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
    ) -> None:
        self.c_cfg = NewCancerConfig()
        self.sd_slices = sd_slices
        self._frozen_quantizers = {}
        self._history_warmup_finished = False
        self.options_to_config(**dict(options or {}))
        super().__init__(None, protocol_name_full, sd_slices=sd_slices)
        assert sd_slices is not None or not self.c_cfg.use_model_slices, (
            "NewCancerProtocol requires StateDictManager slices when model slices are enabled."
        )

    def options_to_config(
        self,
        *,
        binary: bool = False,
        midRate: bool = False,
        noModelSlices: bool = False,
        outlier: float | None = None,
        pretrainDir: str | None = None,
        trainEpochs: int | None = None,
        maxHistory: int | None = None,
        temporalOnly: bool = False,
        retrainOnly: bool = False,
        nonWzWorker: bool = False,
        nonWzServer: bool = False,
        sampled: bool = False,
        bound_calc: bool = False,
    ) -> None:
        """Apply NewCancer protocol options and refresh the scheduled round-codec names."""
        assert all(
            isinstance(flag, bool)
            for flag in (
                binary, midRate, noModelSlices, temporalOnly, retrainOnly, nonWzWorker, nonWzServer,
                sampled, bound_calc,
            )
        ), "NewCancer protocol flag options must be boolean."
        assert not (binary and midRate), "NewCancer protocol cannot be both binary and midRate."
        assert not (sampled or bound_calc), (
            "Round-based NewCancer does not implement the old monolithic sampled/bound_calc Cancer variants yet."
        )

        selected_variants = [
            round_type
            for round_type, enabled in
            (("T", temporalOnly), ("R", retrainOnly), ("TMM", nonWzWorker), ("RMM", nonWzServer))
            if enabled
        ]
        assert len(selected_variants) <= 1, (
            "NewCancer protocol can use only one of temporalOnly, retrainOnly, nonWzWorker, nonWzServer.")
        assert not selected_variants or type(self).protocol_name == "cancer", (
            "NewCancer protocol variant flags must be selected through 'cancer'.")
        training_round = selected_variants[0] if selected_variants else self.training_round_type

        if outlier is not None:
            assert not isinstance(outlier, bool) and float(outlier) > 0, (
                "NewCancer option outlier must be a positive number.")
            self.c_cfg.outlier_threshold = float(outlier)
        if pretrainDir is not None:
            assert isinstance(pretrainDir, str), "NewCancer option pretrainDir requires a path value."
            self.c_cfg.pretrain_pth_dir = Path(pretrainDir)
        if trainEpochs is not None:
            assert not isinstance(trainEpochs, bool) and int(trainEpochs) > 0, (
                "NewCancer option trainEpochs must be a positive integer.")
            self.c_cfg.train_epochs = int(trainEpochs)
        if maxHistory is not None:
            assert not isinstance(maxHistory, bool) and int(maxHistory) > 0, (
                "NewCancer option maxHistory must be a positive integer.")
            self.c_cfg.max_side_info_count = int(maxHistory)
        if noModelSlices:
            self.c_cfg.use_model_slices = False

        warmup_phase, routine_phase = self.c_cfg.warmup_phase, self.c_cfg.routine_phase
        if training_round is not None:
            # Variants keep the pretrained cold start and frozen reuse rounds, replacing only training rounds.
            warmup_phase = tuple(
                (round_type if round_type == "P" else training_round, bins, planes)
                for round_type, bins, planes in warmup_phase
            )
            routine_phase = tuple(
                (round_type if round_type == "F" else training_round, bins, planes)
                for round_type, bins, planes in routine_phase
            )
        if binary or midRate:
            routine_phase = tuple(
                (round_type, 2, 1 if binary else 2) for round_type, _, _ in routine_phase
            )

        self.c_cfg.warmup_phase = warmup_phase
        self.c_cfg.routine_phase = routine_phase
        self.warmup_round_codecs = _round_codec_names(warmup_phase)
        self.routine_round_codecs = _round_codec_names(routine_phase)
        self.max_per_client_recons_history = self.c_cfg.max_side_info_count

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the scheduled round codec, wired to the protocol's config, history, and quantizer state."""
        warmup_length = len(self.warmup_round_codecs)
        if round_id >= warmup_length and not self._history_warmup_finished:
            self._recons_history.set_future_commit_access(tuple(
                self._round_codec_classes[routine_round_name].history_access_needs
                for routine_round_name in self.routine_round_codecs
            ))
            self._history_warmup_finished = True

        history = self._recons_history.copy()
        history.freeze()
        return super().create_round_codec(
            round_id,
            client_id,
            c_cfg=self.c_cfg,
            quantizer_kwargs={
                "norm_slices": self.sd_slices if self.c_cfg.use_model_slices else None,
                "outlier_threshold": self.c_cfg.outlier_threshold,
            },
            history=history,
            frozen_quantizers=self._frozen_quantizers,
            phase="warmup" if round_id < warmup_length else "routine",
        )


class CancerProtocol(NewCancerProtocol):
    """Default round-based Cancer protocol selector; variant flags are only accepted here."""

    protocol_name: ClassVar[str] = "cancer"


class TemporalOnlyCancerProtocol(NewCancerProtocol):
    """Cancer variant using temporal training rounds wherever training is scheduled."""

    protocol_name: ClassVar[str] = "temporal_only"
    training_round_type: ClassVar[str | None] = "T"


class RetrainOnlyCancerProtocol(NewCancerProtocol):
    """Cancer variant using server retraining rounds wherever training is scheduled."""

    protocol_name: ClassVar[str] = "retrain_only"
    training_round_type: ClassVar[str | None] = "R"


class NonWzWorkerCancerProtocol(NewCancerProtocol):
    """Cancer variant with worker-side marginal training and learned decoder prior."""

    protocol_name: ClassVar[str] = "non_wz_learned_worker"
    training_round_type: ClassVar[str | None] = "TMM"


class NonWzServerCancerProtocol(NewCancerProtocol):
    """Cancer variant with server-side marginal training and learned decoder prior."""

    protocol_name: ClassVar[str] = "non_wz_learned_server"
    training_round_type: ClassVar[str | None] = "RMM"
