from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from FL_code.FL_core.codec import (
    Access,
    BaseRoundCodec,
    CompressionRecord,
    record_reconstruction_metrics,
)
from FL_code.FL_core.utils import compress_data_list, decompress_data_list

from .brent_wz_models import EncoderDecoderLayeredRNN
from .history import HistoryVector, ReconstructionHistory
from .prior_code import PriorCalculator, PriorTrainingResult
from .wz_quantizer import (
    DedupedWZQuantizerCancer,
    PreprocessMetadata,
    SideInfoColumnsCache,
    SideInfoMatrix,
    WZcfgQuant,
)


@dataclass
class ElementMetrics:
    """Per-coordinate diagnostics produced by a WZ round and persisted by its record.

    A codec needing further per-coordinate arrays subclasses this with its own tensor fields and
    names the subclass in its record's `element_metrics_class`; every tensor field that is set is
    saved, so the subclass needs no override.
    """

    include_arrays: bool
    models: dict[str, Any] | None = None
    reconstruction: torch.Tensor | None = None
    bits: torch.Tensor | None = None
    bins: torch.Tensor | None = None
    errors: torch.Tensor | None = None

    def save(
        self,
        save_dir: Path,
        round_id: int,
        client_id: int,
        w_mean_of_vec: float,
    ) -> None:
        """Write this client-round's model states and available element arrays."""
        assert self.models is not None, "Element diagnostics are missing their model states."
        assert self.reconstruction is not None, (
            "Element diagnostics are missing the committed reconstruction.")
        elements_dir = save_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)
        stem = f"round_{round_id}_client_{client_id}"
        torch.save(self.models, elements_dir / f"{stem}_models.pt")
        arrays = {
            "recon": self.reconstruction,
            **{
                field.name: value
                for field in fields(self)
                if field.name != "reconstruction"
                and isinstance(value := getattr(self, field.name), torch.Tensor)
            },
        }
        np.savez(
            elements_dir / f"{stem}.npz",
            w_mean_of_vec=w_mean_of_vec,
            **{name: tensor.numpy() for name, tensor in arrays.items()},
        )


class CancerRecord(CompressionRecord):
    """Compression and optional element-level metrics for one WZ client-round."""

    element_metrics_class: ClassVar[type[ElementMetrics]] = ElementMetrics
    _element_metrics: ElementMetrics | None = None

    numerical_runtime: dict[str, Any] | None = None
    quantizer_training_provenance: dict[str, Any] | None = None
    bins_per_plane: int | None = None
    num_planes: int | None = None
    quantizer_prior_rate: float | None = None
    quantizer_prior_stage_rates: list[float] | None = None
    temporal_si_prior_rate: float | None = None
    server_si_prior_rate: float | None = None
    marginal_rate: float | None = None
    stage_mses: list[float] | None = None
    stage_wmapes: list[float] | None = None
    stage_wmspe_sqrts: list[float] | None = None
    encoder_size: float | None = None
    decoder_size: float | None = None
    meta_data_size: float | None = None
    coder_rate: float | None = None
    coder_stage_rates: list[float] | None = None
    coder_overhead_rate: float | None = None
    temporal_si_rans_rate: float | None = None
    temporal_si_rans_stage_rates: list[float] | None = None
    temporal_si_rans_overhead_rate: float | None = None
    server_si_rans_rate: float | None = None
    server_si_rans_stage_rates: list[float] | None = None
    server_si_rans_overhead_rate: float | None = None
    symbol_error_rate: float | None = None
    warm_parent_slots: list[int] | None = None
    warm_deployed_slot: int | None = None
    server_si_prior_parent_slots: list[int] | None = None
    server_si_prior_deployed_slot: int | None = None

    @property
    def element_metrics(self) -> ElementMetrics | None:
        """Return the optional element-level artifact owned by this record."""
        return self._element_metrics

    def enable_element_metrics(self, include_arrays: bool) -> None:
        """Attach an empty element-level artifact for the round codec to populate."""
        assert self._element_metrics is None, "Element diagnostics were already enabled."
        self._element_metrics = self.element_metrics_class(include_arrays)

    def save_artifacts(self, save_dir: Path) -> None:
        """Persist element-level diagnostics after the CSV metrics have been finalized."""
        if self._element_metrics is None:
            return
        assert self.w_mean_of_vec is not None, (
            "Element diagnostics must be saved after reconstruction metrics are set.")
        self._element_metrics.save(
            save_dir, self.round_id, self.client_id, self.w_mean_of_vec)
        self._element_metrics = None


class F_RoundCodec(BaseRoundCodec):
    """Non-instantiated schedule marker for protocol-level frozen codec reuse."""

    record_class = CancerRecord
    round_name: ClassVar[str] = "F"
    history_access: ClassVar[Access] = Access.NONE

    @staticmethod
    def validate_cfg(**options: Any) -> None:
        """Reject options because F only marks protocol-level codec reuse."""
        assert not options, f"F protocol behavior does not accept options: {tuple(options)}."


def compressed_size(obj: Any) -> float:
    """Return a compressed object's size in MiB."""
    return len(compress_data_list(obj)) / (1024**2)


@dataclass(frozen=True)
class ServerSIPriorTrainingPlan:
    """Protocol-selected initial lineages for a T codec's full-server-SI prior."""

    initial_states: tuple[Mapping[str, torch.Tensor] | None, ...]
    parent_slots: tuple[int, ...]
    attempts_per_parent: int

    def __post_init__(self) -> None:
        assert self.initial_states and self.attempts_per_parent > 0
        assert not self.parent_slots or len(self.parent_slots) == len(self.initial_states)


@dataclass(frozen=True)
class PriorEvaluation:
    """One prior model paired with the exact side information used to evaluate it."""

    model: EncoderDecoderLayeredRNN
    side_information: SideInfoMatrix | torch.Tensor


class _WZRoundCodec(BaseRoundCodec):
    """Shared single-quantizer round mechanics; children own training-data availability policy."""

    record_class = CancerRecord
    round_name: ClassVar[str]
    history_access: ClassVar[Access]
    quantizer_class: type[DedupedWZQuantizerCancer] = DedupedWZQuantizerCancer
    config_class: type[WZcfgQuant] = WZcfgQuant

    c_cfg: WZcfgQuant
    quantizer: DedupedWZQuantizerCancer
    dump_element_metrics: bool = False
    _original_delta_vec: torch.Tensor | None = None
    _quantizer_prior_model: EncoderDecoderLayeredRNN | None
    _side_info_columns_cache: SideInfoColumnsCache
    warm_parent_slots: tuple[int, ...] = ()
    joint_quantizer_state: dict[str, torch.Tensor] | None = None
    refit_priors_on_frozen: bool = False
    _fine_tune_native_prior: ClassVar[bool] = True
    transmit_prior_parameters: ClassVar[bool] = False

    def options_to_config(
        self,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None,
        dump_element_metrics: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Separate round diagnostics from options that configure the learned quantizer."""
        self.dump_element_metrics = dump_element_metrics
        self._quantizer_prior_model = None
        self._side_info_columns_cache = SideInfoColumnsCache()
        self.c_cfg = self.config_class(
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            norm_slices=sd_slices,
            **quant_overrides,
        )

    @classmethod
    def validate_cfg(
        cls,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None = None,
        dump_element_metrics: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Validate the round wrapper and quantizer options before a replay starts."""
        assert isinstance(bins_per_plane, int) and bins_per_plane > 0
        assert isinstance(num_planes, int) and num_planes > 0
        assert isinstance(dump_element_metrics, bool)
        cls.config_class(
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            **quant_overrides,
        )

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Return the side-information access facts and rate-field availability `config_snapshot` records.

        Transport mixins layer coder facts and resolved settings on top via `super()`; unused
        quantizer options are accepted and ignored so a full round-token options dict can be
        passed straight through the cooperative chain without filtering it first.
        """
        return {
            "quantizer_si_access": cls.history_access.value,
            "sw_oracle_search": False,
            "prior_weights_transmitted": cls.transmit_prior_parameters,
            "rate_fields_available": {
                "quantizer_prior_rate": True,
                "quantizer_prior_stage_rates": True,
                "temporal_si_prior_rate": cls.history_access is Access.TEMPORAL_TOO,
                "server_si_prior_rate": True,
                "marginal_rate": True,
                "coder_rate": False,
                "coder_stage_rates": False,
                "coder_overhead_rate": False,
                "symbol_error_rate": False,
            },
        }

    def create_r_record(self, round_id: int, client_id: int) -> CancerRecord:
        """Create the metrics row and optional per-element artifact for this round."""
        record = super().create_r_record(round_id, client_id)
        assert isinstance(record, CancerRecord)
        record.update_record_fields(
            bins_per_plane=self.c_cfg.bins_per_plane,
            num_planes=self.c_cfg.num_planes,
            numerical_runtime=self.c_cfg.numerical_policy.runtime(
                self.quantizer.device, mixed_precision=self.c_cfg.mixed_precision),
        )
        if self.dump_element_metrics:
            # Model states are dumped every round, but the per-coordinate arrays are several
            # hundred MB each, so only even rounds carry them. Analyses that read the arrays must
            # therefore expect half the rounds to have none.
            record.enable_element_metrics(include_arrays=round_id % 2 == 0)
        return record

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor | None, list[HistoryVector] | None]:
        """Return data available where this child trains its quantizer."""
        raise NotImplementedError("Subclasses must return the quantizer training data.")

    def refresh_side_information(
        self, side_information: Sequence[HistoryVector]
    ) -> None:
        """Replace a trained quantizer's decoder-visible SI for protocol-level reuse."""
        assert not self.quantizer.no_side_info, (
            "A no-SI quantizer cannot refresh side information.")
        assert self.quantizer.side_info_list_used is not None, (
            "Side information cannot be refreshed before quantizer training.")
        assert len(side_information) == self.quantizer.side_info_size
        assert self.quantizer.vector_size is not None
        assert all(value.numel() == self.quantizer.vector_size for value in side_information)
        self.quantizer.side_info_list_used = list(side_information)

    def build_quantizer(
        self,
        x: torch.Tensor | None,
        si: list[HistoryVector] | None,
        initial_states: Sequence[Mapping[str, torch.Tensor]] = (),
    ) -> DedupedWZQuantizerCancer:
        """Train the ordinary decoder-side quantizer from the supplied target and SI."""
        assert x is not None and si is not None
        self.quantizer = self.quantizer_class(
            c_cfg=self.c_cfg, si_size=len(si), side_info_columns_cache=self._side_info_columns_cache)
        if initial_states:
            self.quantizer.load_warm_start_states(initial_states)
        self.quantizer.train_model(x, si)
        gc.collect()
        torch.cuda.empty_cache()
        return self.quantizer

    def release_round_resources(self) -> None:
        """Release transient encoded side-information columns after this client-round."""
        self._side_info_columns_cache.clear()

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Quantize a raw delta, fine-tune its native prior, and attach rate metrics."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        if self.quantizer.training_provenance is not None:
            record.update_record_fields(quantizer_training_provenance=self.quantizer.training_provenance)
        self._original_delta_vec = delta_vec
        bins, soft_codes, metadata = self.quantizer.encoding_process(delta_vec)
        assert soft_codes is not None, (
            "Prior fine-tuning needs encoder soft codes; estimator quantizers do not produce them."
        )
        native_side_info = self.quantizer.side_info_tensor(metadata)
        if self.round_name_full != "F":
            self._quantizer_prior_model = self.quantizer.coding_model.copy_for_prior_retraining()
        assert self._quantizer_prior_model is not None
        quantizer_prior = self.quantizer.prior_calculator.compute_prior_from_network(
            self._quantizer_prior_model, bins, native_side_info)
        quantizer_prior_code_lengths = PriorCalculator.compute_code_lengths(
            quantizer_prior, bins, self.c_cfg.num_planes)
        quantizer_prior_stage_rates = quantizer_prior_code_lengths.mean(dim=1).tolist()

        if self._fine_tune_native_prior and (
            self.round_name_full != "F" or self.refit_priors_on_frozen
        ):
            PriorCalculator.fit_prior_model(
                bins.long(), soft_codes, native_side_info,
                self.c_cfg.model_copy(update={"marginal_loss": False}),
                self.quantizer.coding_model, (None,), self.c_cfg.prior_train_repeats,
            )
        native_prior = self.quantizer.prior_calculator.compute_prior_from_network(
            self.quantizer.coding_model, bins, native_side_info)
        native_prior_rate = PriorCalculator.compute_rate_from_prior_tensor(
            native_prior, bins, self.c_cfg.num_planes)
        assert not self.warm_parent_slots or self.quantizer.warm_deployed_parent is not None, (
            "Warm training completed without selecting a deployed parent.")
        updates: dict[str, Any] = {
            "encoder_size": compressed_size(self.quantizer.coding_model.encoder_state_dict()),
            "decoder_size": compressed_size(self.quantizer.coding_model.decoder_state_dict(self.transmit_prior_parameters)),
            "meta_data_size": compressed_size(metadata),
            "quantizer_prior_rate": sum(quantizer_prior_stage_rates),
            "quantizer_prior_stage_rates": quantizer_prior_stage_rates,
            **({
                "warm_parent_slots": list(self.warm_parent_slots),
                "warm_deployed_slot": self.warm_parent_slots[self.quantizer.warm_deployed_parent],
            } if self.warm_parent_slots and self.quantizer.warm_deployed_parent is not None else {}),
        }
        updates[
            "temporal_si_prior_rate"
            if self.history_access is Access.TEMPORAL_TOO else "server_si_prior_rate"
        ] = native_prior_rate
        record.update_record_fields(**updates)
        self.fit_additional_priors(bins, soft_codes, metadata, record)

        marginal_prior = PriorCalculator.compute_marginal_prior(
            bins, self.c_cfg.bins_per_plane, self.c_cfg.num_planes)
        record.update_record_fields(marginal_rate=PriorCalculator.compute_rate_from_prior_tensor(
            marginal_prior, bins, self.c_cfg.num_planes))
        self.collect_element_rates(quantizer_prior_code_lengths, bins, record)
        return {"payload_content": (bins, metadata)}

    def fit_additional_priors(
        self,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> None:
        """Fit optional priors supplied by mixins; the base codec has only its native prior."""

    def prior_evaluations(self, metadata: PreprocessMetadata) -> dict[Access, PriorEvaluation]:
        """Return each available prior paired with its matching normalized side information."""
        assert self.history_access in (Access.TEMPORAL_TOO, Access.SERVER_ONLY)
        return {self.history_access: PriorEvaluation(
            self.quantizer.coding_model, self.quantizer.side_info_tensor(metadata)
        )}

    def collect_element_rates(
        self,
        code_lengths: torch.Tensor,
        bins: torch.Tensor,
        record: CancerRecord,
    ) -> None:
        """Populate the optional element artifact with rate arrays and producing models."""
        artifact = record.element_metrics
        if artifact is None:
            return
        artifact.models = {
            "round_name_full": self.round_name_full,
            "c_cfg": self.c_cfg.model_dump(mode="json", exclude={"norm_slices"}),
            "numerical_runtime": record.numerical_runtime,
            "quantizer_training_provenance": record.quantizer_training_provenance,
            "native_prior_training_provenance": self.quantizer.coding_model.training_provenance,
            "quantizer_model_after_prior_finetuning": self.quantizer.coding_model.state_dict(),
            "quantizer_prior_model": self._quantizer_prior_model.state_dict(),
        }
        if artifact.include_arrays:
            artifact.bits = code_lengths.cpu()
            artifact.bins = bins.cpu()

    def add_stage_distortion_to_record(
        self, stage_reconstructions: torch.Tensor, record: CancerRecord
    ) -> None:
        """Store each refinement stage's distortion when the encoder target is available."""
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
        artifact = record.element_metrics
        if artifact is not None:
            artifact.reconstruction = stage_reconstructions[-1].cpu()
            if artifact.include_arrays:
                artifact.errors = (
                    (stage_reconstructions - self._original_delta_vec)
                    / stage_metrics[-1]["w_mean_of_vec"]
                ).cpu()
        self._original_delta_vec = None

    def decoded_reconstruction(
        self,
        bins: torch.Tensor,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> torch.Tensor:
        """Decode bins and collect stage distortion metrics."""
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]

    def encode(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        """Serialize the round payload after quantization and metric collection."""
        return compress_data_list(self.get_encod_payload(delta_vec, record))

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Decode all refinement stages and return the committed final reconstruction."""
        bins, metadata = decompress_data_list(payload)["payload_content"]
        return self.decoded_reconstruction(bins, metadata, record)


class ServerSIPriorMixin:
    """Add a protocol-initialized prior conditioned on full server-visible history."""

    _server_si_prior_side_information: list[HistoryVector]
    _server_si_prior_training_plan: ServerSIPriorTrainingPlan | None = None
    _server_si_prior_model: EncoderDecoderLayeredRNN | None = None
    server_si_prior_training_result: PriorTrainingResult | None = None

    def set_server_si_prior_training(
        self,
        history: ReconstructionHistory,
        client_id: int,
        plan: ServerSIPriorTrainingPlan | None,
    ) -> None:
        """Receive server SI and the protocol's explicit cold or warm initialization plan."""
        self._server_si_prior_side_information = [
            entry.value
            for client_history in history.view(Access.SERVER_ONLY, client_id).values()
            for entry in client_history
        ]
        self._server_si_prior_training_plan = plan

    def fit_additional_priors(
        self,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> None:
        """Fit or reuse the explicit full-server-SI prior and record its scalar rate."""
        super().fit_additional_priors(bins, soft_codes, metadata, record)
        assert self._server_si_prior_side_information
        side_info = self.quantizer.normalize_side_info_tensor(
            metadata, self._server_si_prior_side_information)
        if self.round_name_full != "F":
            plan = self._server_si_prior_training_plan
            assert plan is not None, "A trained T round requires a protocol server-SI-prior plan."
            self.server_si_prior_training_result = PriorCalculator.fit_prior_model(
                bins.long(), soft_codes, side_info,
                self.c_cfg.model_copy(update={"marginal_loss": False}),
                None, plan.initial_states, plan.attempts_per_parent,
            )
            self._server_si_prior_model = self.server_si_prior_training_result.model
            parent_fields = ({
                "server_si_prior_parent_slots": list(plan.parent_slots),
                "server_si_prior_deployed_slot": plan.parent_slots[
                    self.server_si_prior_training_result.deployed_parent],
            } if plan.parent_slots else {})
        elif self.refit_priors_on_frozen:
            assert self._server_si_prior_model is not None
            self.server_si_prior_training_result = PriorCalculator.fit_prior_model(
                bins.long(), soft_codes, side_info,
                self.c_cfg.model_copy(update={"marginal_loss": False}),
                self._server_si_prior_model, (None,), self.c_cfg.prior_train_repeats,
            )
            parent_fields = {}
        else:
            assert self._server_si_prior_model is not None, (
                "A frozen T round requires the server-SI prior fitted by its training round.")
            parent_fields = {}
        prior = self.quantizer.prior_calculator.compute_prior_from_network(
            self._server_si_prior_model, bins, side_info)
        record.update_record_fields(
            server_si_prior_rate=PriorCalculator.compute_rate_from_prior_tensor(
                prior, bins, self.c_cfg.num_planes),
            **parent_fields,
        )

    def prior_evaluations(self, metadata: PreprocessMetadata) -> dict[Access, PriorEvaluation]:
        """Add the full-server-SI prior to the codec's native prior evaluation."""
        evaluations = super().prior_evaluations(metadata)
        assert self._server_si_prior_model is not None
        evaluations[Access.SERVER_ONLY] = PriorEvaluation(
            self._server_si_prior_model,
            self.quantizer.normalize_side_info_tensor(
                metadata, self._server_si_prior_side_information),
        )
        return evaluations

    def collect_element_rates(
        self, code_lengths: torch.Tensor, bins: torch.Tensor, record: CancerRecord,
    ) -> None:
        """Include the additional server-SI prior in optional model artifacts."""
        super().collect_element_rates(code_lengths, bins, record)
        if record.element_metrics is not None:
            assert record.element_metrics.models is not None
            assert self._server_si_prior_model is not None
            record.element_metrics.models["server_si_prior_model"] = (
                self._server_si_prior_model.state_dict())


class _CurrentTargetWZRoundCodec(_WZRoundCodec):
    """WZ codec trained on the current raw target immediately before encoding."""

    si: list[HistoryVector]
    _send_quantizer_state: bool

    def extra_training_problems(
        self,
    ) -> Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]]:
        """Return further (target, SI) problems to train on; one problem per round by default.

        The pair of hooks an experiment overrides to widen the training set without changing what
        the decoder receives; the shipped rounds never do.
        """
        return ()

    def training_problem_weights(self) -> tuple[float, ...] | None:
        """Return one sampling weight per training problem, or None to sample every row alike."""
        return None

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[HistoryVector]]:
        """Return worker-local history; the raw target arrives during pre-encode training."""
        entries = given_history.view(self.history_access, client_id)[client_id]
        assert entries[-1].client_id == client_id
        assert entries[-1].access == self.history_access
        return None, [entry.value for entry in entries]

    def build_quantizer(
        self,
        x: torch.Tensor | None,
        si: list[HistoryVector] | None,
        initial_states: Sequence[Mapping[str, torch.Tensor]] = (),
    ) -> DedupedWZQuantizerCancer:
        """Create an untrained worker-side quantizer and retain its legal SI."""
        assert x is None and si is not None
        self.si = si
        self.quantizer = self.quantizer_class(
            c_cfg=self.c_cfg, si_size=len(si), side_info_columns_cache=self._side_info_columns_cache)
        if initial_states:
            self.quantizer.load_warm_start_states(initial_states)
        self._send_quantizer_state = False
        return self.quantizer

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Train on the current raw target before its symbols are encoded."""
        if self.round_name_full == "F":
            assert self.quantizer.side_info_list_used is not None
            assert self.quantizer.vector_size is not None
            assert self.quantizer.quantizer_prior_stage_rates is not None
            assert not self._send_quantizer_state
            return

        assert self.quantizer.side_info_list_used is None
        assert not self._send_quantizer_state
        self.quantizer.train_model(
            delta_vec, self.si, self.extra_training_problems(), self.training_problem_weights())
        assert self.quantizer.vector_size is not None
        assert self.quantizer.quantizer_prior_stage_rates is not None
        self.joint_quantizer_state = {
            name: value.detach().cpu().clone()
            for name, value in self.quantizer.coding_model.state_dict().items()
        }
        gc.collect()
        torch.cuda.empty_cache()
        self._send_quantizer_state = True

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Encode with a trained quantizer and transmit newly trained decoder parameters once."""
        assert self.quantizer.side_info_list_used is not None, (
            f"{self.round_name_full} must run pre_encode_training_steps before encoding.")
        assert self.quantizer.vector_size is not None
        assert self.quantizer.quantizer_prior_stage_rates is not None
        payload = super().get_encod_payload(delta_vec, record)
        if self._send_quantizer_state:
            payload["quantizer_state"] = self.quantizer.coding_model.decoder_state_dict(self.transmit_prior_parameters)
            self._send_quantizer_state = False
        return payload


class T_RoundCodec(ServerSIPriorMixin, _CurrentTargetWZRoundCodec):
    """Train on the raw target with temporal SI and also fit a server-SI reference prior."""

    round_name: ClassVar[str] = "T"
    history_access: ClassVar[Access] = Access.TEMPORAL_TOO
    transmit_prior_parameters: ClassVar[bool] = True


class Oracle_RoundCodec(_CurrentTargetWZRoundCodec):
    """Study-only upper bound trained on the current raw delta and server-visible SI."""

    round_name: ClassVar[str] = "O"
    history_access: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[HistoryVector]]:
        """Return server-visible history while leaving the raw target for pre-encode training."""
        history = given_history.view(self.history_access, client_id)
        return None, [
            entry.value for client_history in history.values() for entry in client_history]
