from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from copy import copy
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


def compressed_size(obj: Any) -> float:
    """Return a compressed object's size in MiB."""
    return len(compress_data_list(obj)) / (1024**2)


class _WZRoundCodec(BaseRoundCodec):
    """Shared single-quantizer round mechanics; children own training-data availability policy."""

    record_class = CancerRecord
    round_name: ClassVar[str]
    can_decode_where: ClassVar[Access]
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

    def frozen_round_codec(
        self,
        frozen_class: type[F_RoundCodec],
        round_name_full: str,
        side_information: list[torch.Tensor] | None,
    ) -> BaseRoundCodec:
        """Create a frozen replay of the model this training round produced.

        The scheduled frozen class is passed in rather than hardcoded, so a protocol that schedules
        an F subclass gets that subclass instead of silently getting the base one.
        """
        return frozen_class(
            None,
            round_name_full,
            self.quantizer,
            self.can_decode_where,
            side_information,
            source_round_codec=self,
        )

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Quantize a raw delta and attach rate and model-size metrics."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        self._original_delta_vec = delta_vec
        bins, soft_codes, metadata = self.quantizer.encoding_process(delta_vec)
        record.update_record_fields(
            encoder_size=compressed_size(self.quantizer.coding_model.encoder_state_dict()),
            decoder_size=compressed_size(self.quantizer.coding_model.decoder_state_dict()),
            meta_data_size=compressed_size(metadata),
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
        assert self.quantizer.training_prior_stage_rates is not None
        assert self.quantizer.training_prior_rate is not None
        record.update_record_fields(
            training_prior_stage_rates=list(self.quantizer.training_prior_stage_rates),
            training_prior_rate=self.quantizer.training_prior_rate,
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


class F_RoundCodec(_WZRoundCodec):
    """Reuse a frozen quantizer, optionally advancing its decoder-side SI window."""

    round_name: ClassVar[str] = "F"
    can_decode_where: Access
    source_round_codec: _WZRoundCodec | None

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        frozen_quantizer: DedupedWZQuantizerCancer,
        history_access: Access,
        side_information: list[torch.Tensor] | None = None,
        source_round_codec: _WZRoundCodec | None = None,
    ) -> None:
        # Checked before super(), which would otherwise validate and apply the options this round
        # does not accept, leaving the assertion below with nothing left to catch.
        assert not cfg_options, f"F round does not accept options: {tuple(cfg_options or ())}."
        self.can_decode_where = history_access
        super().__init__(cfg_options, round_name_full)
        self.c_cfg = frozen_quantizer.c_cfg
        self.dump_element_metrics = False
        self.quantizer = frozen_quantizer
        # `conditional_code_lengths` delegates to this codec, so the call binds to *its* attributes,
        # not to the frozen round's. The base implementation takes everything as an argument and so
        # is unaffected; an override reading `self` would silently price the training round's
        # problem. Anchor the state both rounds must agree on: the quantizer configuration, minus
        # the norm slicing, which the conditional coder never reads.
        assert source_round_codec is None or (
            source_round_codec.c_cfg.model_dump(exclude={"norm_slices"})
            == frozen_quantizer.c_cfg.model_dump(exclude={"norm_slices"})
        ), (f"{round_name_full} replays a quantizer configured differently from the round that "
            "produced it; its conditional coder would be priced against the wrong configuration.")
        self.source_round_codec = source_round_codec
        if side_information is None:
            return
        assert not frozen_quantizer.no_side_info, (
            "A no-SI frozen quantizer cannot refresh side information.")
        assert len(side_information) == frozen_quantizer.side_info_size
        assert frozen_quantizer.vector_size is not None
        assert all(
            value.numel() == frozen_quantizer.vector_size for value in side_information)
        self.quantizer = copy(frozen_quantizer)
        self.quantizer.side_info_list_used = list(side_information)

    def conditional_code_lengths(
        self,
        quantizer: DedupedWZQuantizerCancer,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
    ) -> tuple[torch.Tensor, EncoderDecoderLayeredRNN]:
        """Preserve the producing training codec's conditional-coder behavior."""
        if self.source_round_codec is None:
            return super().conditional_code_lengths(
                quantizer, bins, soft_codes, side_info, c_cfg)
        return self.source_round_codec.conditional_code_lengths(
            quantizer, bins, soft_codes, side_info, c_cfg)

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> DedupedWZQuantizerCancer:
        """Reject training because this round exists only to replay a frozen model."""
        raise NotImplementedError("F_RoundCodec cannot build a new quantizer.")


class P_RoundCodec(_WZRoundCodec):
    """Load the pretrained marginal quantizer used before useful history exists."""

    round_name: ClassVar[str] = "P"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, None]:
        """Return no training data: this round loads a pretrained model instead of fitting one."""
        return None, None

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> DedupedWZQuantizerCancer:
        """Load the configured pretrained state without fitting it in this round."""
        assert x is None and si is None
        self.c_cfg = self.c_cfg.model_copy(update={"marginal_loss": True})
        self.quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=0)
        weight_path = self.c_cfg.pretrain_pth_dir / (
            f"bpp{self.c_cfg.bins_per_plane}_np{self.c_cfg.num_planes}_pretrained_wzq_rnn.pth")
        assert weight_path.is_file(), (
            f"{self.round_name_full} needs a pretrained marginal quantizer at {weight_path}; "
            "one file is required per (bins_per_plane, num_planes) pair.")
        # The checkpoint carries no record of the architecture it was trained with, so a config
        # change since it was written surfaces here as an unreadable wall of shape mismatches.
        try:
            self.quantizer.coding_model.load_state_dict(
                torch.load(weight_path, map_location="cpu", weights_only=True))
        except RuntimeError as error:
            raise RuntimeError(
                f"{weight_path} was trained with a different architecture than the current "
                f"hidden_dim={self.c_cfg.hidden_dim}, layers={self.c_cfg.layers}, "
                f"shared_heads={self.c_cfg.shared_heads}; retrain it or pin those fields."
            ) from error
        self.quantizer.side_info_list_used = []
        return self.quantizer


class R_RoundCodec(_WZRoundCodec):
    """Train at the server on the latest reconstruction as target and earlier history as SI."""

    round_name: ClassVar[str] = "R"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the latest own reconstruction and all other server-visible entries."""
        history = given_history.view(self.can_decode_where, client_id)
        target = history[client_id][-1]
        assert target.client_id == client_id
        side_info = [
            entry.tensor
            for client_history in history.values()
            for entry in client_history
            if entry is not target
        ]
        return target.tensor, side_info


class T_RoundCodec(_WZRoundCodec):
    """Train at the worker on its raw delta with worker-local temporal SI."""

    round_name: ClassVar[str] = "T"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO
    si: list[torch.Tensor]

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
        """Return worker-local temporal history; the current raw target arrives during encode."""
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
        return self.quantizer

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Train when the raw delta reaches the worker, then transmit decoder parameters."""
        self.quantizer.train_model(
            delta_vec, self.si, self.extra_training_problems(), self.training_problem_weights())
        gc.collect()
        torch.cuda.empty_cache()
        payload = super().get_encod_payload(delta_vec, record)
        payload["quantizer_state"] = self.quantizer.coding_model.decoder_state_dict()
        return payload


class Oracle_RoundCodec(T_RoundCodec):
    """Study-only upper bound trained on the current raw delta and server-visible SI."""

    round_name: ClassVar[str] = "O"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return every server-visible history entry while leaving the oracle target for encode."""
        history = given_history.view(self.can_decode_where, client_id)
        return None, [
            entry.tensor for client_history in history.values() for entry in client_history]
