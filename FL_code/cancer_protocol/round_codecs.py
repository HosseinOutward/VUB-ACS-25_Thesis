from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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
from .prior_code import PriorCalculator
from .wz_quantizer import DedupedWZQuantizerCancer, PreprocessMetadata, WZcfgQuant

if TYPE_CHECKING:
    from .cancer_protocol import ReconstructionHistory


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
    coder_rate: float | None = None
    coder_stage_rates: list[float] | None = None
    coder_overhead_rate: float | None = None
    symbol_error_rate: float | None = None

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
    can_decode_where: ClassVar[Access] = Access.NONE

    @staticmethod
    def validate_cfg(**options: Any) -> None:
        """Reject options because F only marks protocol-level codec reuse."""
        assert not options, f"F protocol behavior does not accept options: {tuple(options)}."


def compressed_size(obj: Any) -> float:
    """Return a compressed object's size in MiB."""
    return len(compress_data_list(obj)) / (1024**2)


class _WZRoundCodec(BaseRoundCodec):
    """Shared single-quantizer round mechanics; children own training-data availability policy."""

    record_class = CancerRecord
    round_name: ClassVar[str]
    can_decode_where: ClassVar[Access]
    # States, without inferring it from a display label such as codec_id, whether the quantizer
    # trained on temporal (worker-only) or oracle (server-visible) side information.
    quantizer_training_access: ClassVar[str]
    quantizer_class: type[DedupedWZQuantizerCancer] = DedupedWZQuantizerCancer
    config_class: type[WZcfgQuant] = WZcfgQuant

    c_cfg: WZcfgQuant
    quantizer: DedupedWZQuantizerCancer
    dump_element_metrics: bool = False
    _original_delta_vec: torch.Tensor | None = None
    _prior_side_information: list[torch.Tensor] | None = None

    def options_to_config(
        self,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        dump_element_metrics: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Separate round diagnostics from options that configure the learned quantizer."""
        self.dump_element_metrics = dump_element_metrics
        self.c_cfg = self.config_class(
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            norm_slices=sd_slices,
            cache_dir=cache_dir,
            cache_overwrite=cache_overwrite,
            **quant_overrides,
        )

    @classmethod
    def validate_cfg(
        cls,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        dump_element_metrics: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Validate the round wrapper and quantizer options before a replay starts."""
        assert isinstance(bins_per_plane, int) and bins_per_plane > 0
        assert isinstance(num_planes, int) and num_planes > 0
        assert cache_dir is None or isinstance(cache_dir, Path)
        assert isinstance(cache_overwrite, bool)
        assert isinstance(dump_element_metrics, bool)
        cls.config_class(
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            **quant_overrides,
        )

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Return the oracle-access facts and rate-field availability `config_snapshot` records.

        Transport mixins layer coder facts and resolved settings on top via `super()`; unused
        quantizer options are accepted and ignored so a full round-token options dict can be
        passed straight through the cooperative chain without filtering it first.
        """
        return {
            "quantizer_training_access": cls.quantizer_training_access,
            "oracle_prior_granted": False,
            "sw_oracle_search": False,
            "coder_prior_si": None,
            # T/O's prior_stage_rates follow the protocol-wide retrained_prior_uses_server_si
            # policy recorded alongside this in config_snapshot; transport mixins override this
            # when they instead fix the coder prior to the exact decoder-legal SI.
            "prior_stage_rates_si": "protocol_retrained_prior_uses_server_si_policy",
            "rate_fields_available": {
                "prior_stage_rates": True,
                "marginal_stage_rates": True,
                "training_prior_stage_rates": True,
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
        )
        if self.dump_element_metrics:
            # Model states are dumped every round, but the per-coordinate arrays are several
            # hundred MB each, so only even rounds carry them. Analyses that read the arrays must
            # therefore expect half the rounds to have none.
            record.enable_element_metrics(include_arrays=round_id % 2 == 0)
        return record

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Return data available where this child trains its quantizer."""
        raise NotImplementedError("Subclasses must return the quantizer training data.")

    def set_retrained_prior_si(
        self, history: ReconstructionHistory, client_id: int
    ) -> None:
        """Store server-visible SI for the separate conditional entropy-prior estimate."""
        self._prior_side_information = [
            entry.tensor
            for client_history in history.view(Access.SERVER_ONLY, client_id).values()
            for entry in client_history
        ]

    def refresh_side_information(self, side_information: Sequence[torch.Tensor]) -> None:
        """Replace a trained quantizer's decoder-visible SI for protocol-level reuse."""
        assert not self.quantizer.no_side_info, (
            "A no-SI quantizer cannot refresh side information.")
        assert self.quantizer.side_info_list_used is not None, (
            "Side information cannot be refreshed before quantizer training.")
        assert len(side_information) == self.quantizer.side_info_size
        assert self.quantizer.vector_size is not None
        assert all(
            value.numel() == self.quantizer.vector_size for value in side_information)
        self.quantizer.side_info_list_used = list(side_information)

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> DedupedWZQuantizerCancer:
        """Train the ordinary decoder-side quantizer from the supplied target and SI."""
        assert x is not None and si is not None
        self.quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        self.quantizer.train_model(x, si)
        gc.collect()
        torch.cuda.empty_cache()
        return self.quantizer

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Quantize a raw delta and attach rate and model-size metrics."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        self._original_delta_vec = delta_vec
        bins, soft_codes, metadata = self.quantizer.encoding_process(delta_vec)
        training_prior = self.quantizer.prior_calculator.compute_prior_from_network(
            self.quantizer.coding_model,
            bins,
            self.quantizer.side_info_tensor(metadata),
        )
        training_prior_stage_rates = list(self.quantizer.prior_calculator.compute_plane_rates(
            training_prior, bins, self.c_cfg.num_planes))
        record.update_record_fields(
            encoder_size=compressed_size(self.quantizer.coding_model.encoder_state_dict()),
            decoder_size=compressed_size(self.quantizer.coding_model.decoder_state_dict()),
            meta_data_size=compressed_size(metadata),
            training_prior_rate=sum(training_prior_stage_rates),
            training_prior_stage_rates=training_prior_stage_rates,
        )
        self.add_prior_to_record(bins, soft_codes, metadata, record)
        return {"payload_content": (bins, metadata)}

    def conditional_code_lengths(
        self,
        quantizer: DedupedWZQuantizerCancer,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
    ) -> tuple[torch.Tensor, EncoderDecoderLayeredRNN]:
        """Train the entropy prior on decoder SI and return each realized symbol's length."""
        prior_cfg = c_cfg.model_copy(update={"marginal_loss": False})
        prior_model = PriorCalculator.make_trained_prior_model(
            bins.long(), soft_codes, side_info, prior_cfg)
        prior = quantizer.prior_calculator.compute_prior_from_network(
            prior_model, bins, side_info)
        return (
            PriorCalculator.compute_code_lengths(prior, bins, prior_cfg.num_planes),
            prior_model,
        )

    def add_prior_to_record(
        self,
        bins: torch.Tensor,
        soft_codes: torch.Tensor | None,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> None:
        """Estimate conditional-coder and marginal rates independently of quantization."""
        assert soft_codes is not None, (
            "Prior retraining needs encoder soft codes; estimator quantizers do not produce them.")
        prior_si = (
            self.quantizer.side_info_list_used
            if self._prior_side_information is None
            else self._prior_side_information
        )
        assert prior_si is not None
        side_info = self.quantizer.normalize_side_info_tensor(metadata, prior_si)
        code_lengths, prior_model = self.conditional_code_lengths(
            self.quantizer, bins, soft_codes, side_info, self.c_cfg)
        prior_stage_rates = code_lengths.mean(dim=1).tolist()
        self.collect_element_rates(code_lengths, bins, prior_model, record)

        marginal_prior = PriorCalculator.compute_marginal_prior(
            bins, self.c_cfg.bins_per_plane, self.c_cfg.num_planes)
        marginal_stage_rates = list(PriorCalculator.compute_plane_rates(
            marginal_prior, bins, self.c_cfg.num_planes))
        record.update_record_fields(
            prior_stage_rates=prior_stage_rates,
            prior_rate=sum(prior_stage_rates),
            marginal_stage_rates=marginal_stage_rates,
            marginal_rate=sum(marginal_stage_rates),
        )

    def collect_element_rates(
        self,
        code_lengths: torch.Tensor,
        bins: torch.Tensor,
        prior_model: EncoderDecoderLayeredRNN,
        record: CancerRecord,
    ) -> None:
        """Populate the optional element artifact with rate arrays and producing models."""
        artifact = record.element_metrics
        if artifact is None:
            return
        artifact.models = {
            "round_name_full": self.round_name_full,
            "c_cfg": self.c_cfg.model_dump(mode="json", exclude={"norm_slices"}),
            "coding_model": self.quantizer.coding_model.state_dict(),
            "prior_model": prior_model.state_dict(),
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

    def encode(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        """Serialize the round payload after quantization and metric collection."""
        return compress_data_list(self.get_encod_payload(delta_vec, record))

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Decode all refinement stages and return the committed final reconstruction."""
        stage_reconstructions = self.quantizer.decoding_stages_process(
            decompress_data_list(payload)["payload_content"])
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]


class T_RoundCodec(_WZRoundCodec):
    """Train at the worker on its raw delta with worker-local temporal SI."""

    round_name: ClassVar[str] = "T"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO
    quantizer_training_access: ClassVar[str] = "temporal"
    si: list[torch.Tensor]
    _send_quantizer_state: bool

    def extra_training_problems(
        self,
    ) -> Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]]:
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
    ) -> tuple[None, list[torch.Tensor]]:
        """Return worker-local history; the raw target arrives during pre-encode training."""
        entries = given_history.view(self.can_decode_where, client_id)[client_id]
        assert entries[-1].client_id == client_id
        assert entries[-1].access == self.can_decode_where
        return None, [entry.tensor for entry in entries]

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> DedupedWZQuantizerCancer:
        """Create an untrained worker-side quantizer and retain its legal SI."""
        assert x is None and si is not None
        self.si = si
        self.quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        self._send_quantizer_state = False
        return self.quantizer

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Train on the current raw target before its symbols are encoded."""
        if self.round_name_full == "F":
            assert self.quantizer.side_info_list_used is not None
            assert self.quantizer.vector_size is not None
            assert self.quantizer.training_prior_stage_rates is not None
            assert not self._send_quantizer_state
            return

        assert self.quantizer.side_info_list_used is None
        assert not self._send_quantizer_state
        self.quantizer.train_model(
            delta_vec, self.si, self.extra_training_problems(), self.training_problem_weights())
        assert self.quantizer.vector_size is not None
        assert self.quantizer.training_prior_stage_rates is not None
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
        assert self.quantizer.training_prior_stage_rates is not None
        payload = super().get_encod_payload(delta_vec, record)
        if self._send_quantizer_state:
            payload["quantizer_state"] = self.quantizer.coding_model.decoder_state_dict()
            self._send_quantizer_state = False
        return payload


class Oracle_RoundCodec(T_RoundCodec):
    """Study-only upper bound trained on the current raw delta and server-visible SI."""

    round_name: ClassVar[str] = "O"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY
    quantizer_training_access: ClassVar[str] = "oracle"

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return server-visible history while leaving the raw target for pre-encode training."""
        history = given_history.view(self.can_decode_where, client_id)
        return None, [
            entry.tensor for client_history in history.values() for entry in client_history]
