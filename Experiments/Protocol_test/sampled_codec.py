"""Sampled current-target WZ rounds; schedules and weight transmission remain protocol-owned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from FL_code.FL_core.codec import Access, record_reconstruction_metrics
from FL_code.FL_core.utils import compress_data_list, decompress_data_list
from FL_code.cancer_protocol.bin_coder.rans_coder import HEADER, ConditionalPrior, RansConfig, decode, encode
from FL_code.cancer_protocol.bin_coder.rans_mixin import RansRoundMixin
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH, PriorCalculator
from FL_code.cancer_protocol.round_codecs import CancerRecord, Oracle_RoundCodec, PriorEvaluation, _WZRoundCodec, compressed_size
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, PreprocessMetadata, WZcfgQuant, SideInfoMatrix, _SideInfoRows, select_si_rows


class SampleTrainingConfig(BaseModel):
    """Sample-fit settings shared by SO's raw-label oracle and S's decoded-label fitting."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_size: int = Field(default=3_600_000, ge=1, strict=True)
    sample_fit_epochs: int | None = Field(default=28, ge=1, strict=True)
    sample_fit_lr: float | None = Field(default=0.0006807383623518464, gt=0)
    sample_fit_tau: float | None = Field(default=0.30, gt=0)
    sample_fit_batch_size: int | None = Field(default=7_500, ge=1, strict=True)
    sample_train_encoder: bool = True
    sample_train_prior: bool = True
    sample_train_backbones: bool = True
    sample_hard_inputs: bool = False


class SamplingConfig(SampleTrainingConfig):
    """S acquisition settings added to the sample-fit settings shared with SO."""

    sample_extra_planes: int = Field(default=1, ge=0, strict=True)
    sample_rans_lanes: int = Field(default=16, ge=1, le=2**20, strict=True)
    sample_training: Literal["all", "prefix"] = "all"
    sample_format: Literal["lwzq", "fp16"] = "lwzq"
    sample_noise_multiplier: float = Field(default=0.0, ge=0)


@dataclass(frozen=True)
class SamplePayload:
    """Worker-produced sample symbols and coordinate seed, decoded before the replacement fit."""

    vector_size: int
    sample_size: int
    seed: int
    symbols: torch.Tensor | bytes
    metadata: PreprocessMetadata
    format: Literal["lwzq", "fp16"] = "lwzq"

    def indices(self) -> torch.Tensor:
        """Regenerate distinct coordinates in layer order without transmitting an index array."""
        return torch.from_numpy(np.sort(np.random.default_rng(self.seed).choice(
            self.vector_size, self.sample_size, replace=False)))


class SampledRecord(CancerRecord):
    """SO/SOR and S/SR metrics, with sample acquisition rates normalized by full gradient size."""

    sample_count: int | None = None
    sample_num_planes: int | None = None
    sample_bootstrap: bool | None = None
    sample_wmape: float | None = None
    sample_mape: float | None = None
    sample_prior_rate: float | None = None
    sample_extra_prior_rate: float | None = None
    sample_coder_rate: float | None = None
    sample_extra_coder_rate: float | None = None
    sample_coder_overhead_rate: float | None = None
    sample_metadata_size: float | None = None
    sample_format: str | None = None
    sample_training_planes: int | None = None
    sample_fit_epochs: int | None = None
    sample_fit_lr: float | None = None
    sample_fit_tau: float | None = None
    sample_training_batch_size: int | None = None
    sample_training_encoder: bool | None = None
    sample_training_prior: bool | None = None
    sample_training_backbones: bool | None = None
    sample_training_hard_inputs: bool | None = None
    sample_noise_std: float | None = None
    normal_quantizer_prior_stage_rates: list[float] | None = None
    replaced_sample_prior_stage_rates: list[float] | None = None


class _SampleQuantizer(DedupedWZQuantizerCancer):
    """Ordinary WZ fitting with supplied normalization and decoder-side sample replacement."""

    normalization_factors: torch.Tensor | None = None
    sample_indices: torch.Tensor | None = None
    sample_reconstruction: torch.Tensor | None = None
    trained_num_planes: int
    fit_planes: int | None = None
    fit_epochs: int | None = None
    fit_lr: float | None = None
    fit_tau: float | None = None
    fit_batch_size: int | None = None
    fit_encoder: bool = True
    fit_prior: bool = True
    fit_backbones: bool = True
    fit_hard_inputs: bool = False
    fit_noise_std: float = 0.0
    _fit_machine_planes: int | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trained_num_planes = self.num_planes

    def _new_coding_model(self, si_size: int) -> EncoderDecoderLayeredRNN:
        config = self.c_cfg
        if self._fit_machine_planes is not None:
            self.c_cfg = config.model_copy(update={"num_planes": self._fit_machine_planes})
        try:
            model = super()._new_coding_model(si_size)
        finally:
            self.c_cfg = config
        model.num_planes = config.num_planes
        self._set_training_parameters(model)
        return model

    def _set_training_parameters(self, model: EncoderDecoderLayeredRNN) -> None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_((self.fit_encoder or not name.startswith(("encoder.", "binner")))
                                     and (self.fit_prior or not name.startswith(("conditionalRNN.", "conditionalPrior")))
                                     and (self.fit_backbones or not name.startswith(("encoder.", "decoder.", "conditionalRNN."))))

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        config = self.c_cfg
        self._set_training_parameters(self.coding_model)
        self.c_cfg = config.model_copy(update={"train_batch_size": self.fit_batch_size or config.train_batch_size})
        if self.fit_planes is not None:
            # Retain all heads for strict warm-state loading; train and validate only the deployed prefix.
            self._fit_machine_planes = config.num_planes
            self.c_cfg = self.c_cfg.model_copy(update={"num_planes": self.fit_planes})
            self.coding_model.num_planes = self.fit_planes
        try:
            super()._fit_prepared(x_prep, side_info)
        finally:
            self.c_cfg = config
            self.coding_model.num_planes = config.num_planes
            self._fit_machine_planes = None

    def training_config(self) -> WZcfgQuant:
        """Apply sample-only overrides after O's warm optimizer configuration."""
        config = super().training_config()
        return config.model_copy(update={key: value for key, value in (
            ("train_epochs", self.fit_epochs), ("lr", self.fit_lr), ("tau", self.fit_tau)) if value is not None})

    def compute_loss(
        self, model: EncoderDecoderLayeredRNN, training_input: tuple[torch.Tensor],
        si_batch: torch.Tensor, epoch: int, c_cfg: WZcfgQuant,
        sample_weights: torch.Tensor | None, wmspe_denom: float, perturb_generator: torch.Generator,
    ) -> torch.Tensor:
        """Reuse O's loss on the deployed prefix, optionally smoothing decoder-visible sample values."""
        if self.fit_noise_std:
            x, = training_input
            training_input = (x + self.fit_noise_std * torch.randn(
                x.shape, device=x.device, dtype=x.dtype, generator=perturb_generator),)
        planes = model.num_planes
        hard_inputs = model.hard_decoder_inputs
        model.hard_decoder_inputs = self.fit_hard_inputs
        model.num_planes = self.fit_planes or planes
        try:
            return super().compute_loss(model, training_input, si_batch, epoch, c_cfg,
                                        sample_weights, wmspe_denom, perturb_generator)
        finally:
            model.num_planes = planes
            model.hard_decoder_inputs = hard_inputs

    def preprocess_x(
        self, x_raw: torch.Tensor, skip_outliers: bool = False,
        normalization_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Use full-target normalization during the sample fit instead of estimating it again."""
        return super().preprocess_x(x_raw, skip_outliers, (
            self.normalization_factors if normalization_factors is None else normalization_factors))

    def decoding_stages_process(
        self, payload_content: tuple[torch.Tensor, PreprocessMetadata], batch_size: int = INFERENCE_BATCH,
    ) -> torch.Tensor:
        """Replace sampled coordinates before stage distortion and history are recorded."""
        stages = super().decoding_stages_process(payload_content, batch_size)
        if self.sample_indices is not None:
            assert self.sample_reconstruction is not None
            stages[:, self.sample_indices] = self.sample_reconstruction
        return stages


class _SampledOracleQuantizer(_SampleQuantizer):
    """SO's free raw-label training pool, retaining full-vector inference and normalization."""

    training_sample_size: int
    training_sample_indices: torch.Tensor | None = None

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        count = min(self.training_sample_size, x_prep.shape[0])
        self.training_sample_indices = (torch.arange(count) if count == x_prep.shape[0] else torch.from_numpy(np.sort(
            np.random.default_rng(int(torch.randint(0, 2**31, ()).item())).choice(x_prep.shape[0], count, replace=False))))
        # Match the raw-sample diagnostic's draw budget when the fixed pool is smaller than an epoch.
        rows = self.training_sample_indices.repeat((self.c_cfg.train_sample_size + count - 1) // count)
        super()._fit_prepared(x_prep[rows.to(x_prep.device)], select_si_rows(side_info, rows))


class _SampleFitRoundCodec(Oracle_RoundCodec):
    """Shared SO/S training configuration; each concrete codec owns sample availability and transport."""

    record_class: ClassVar[type[SampledRecord]] = SampledRecord
    quantizer: _SampleQuantizer
    sampling: SampleTrainingConfig

    def options_to_config(
        self, sample_size: int = 3_600_000, sample_fit_epochs: int | None = 28,
        sample_fit_lr: float | None = 0.0006807383623518464, sample_fit_tau: float | None = 0.30,
        sample_fit_batch_size: int | None = 7_500, sample_train_encoder: bool = True,
        sample_train_prior: bool = True, sample_hard_inputs: bool = False, sample_train_backbones: bool = True, **options: Any,
    ) -> None:
        """Configure sample fitting separately from ordinary quantizer and entropy-prior settings."""
        self.sampling = SampleTrainingConfig(sample_size=sample_size, sample_fit_epochs=sample_fit_epochs,
            sample_fit_lr=sample_fit_lr, sample_fit_tau=sample_fit_tau, sample_fit_batch_size=sample_fit_batch_size,
            sample_train_encoder=sample_train_encoder, sample_train_prior=sample_train_prior,
            sample_hard_inputs=sample_hard_inputs, sample_train_backbones=sample_train_backbones)
        super().options_to_config(**options)

    @classmethod
    def validate_cfg(cls, **options: Any) -> None:
        """Validate the shared sample-fit contract and ordinary WZ options independently."""
        SampleTrainingConfig(**{key: value for key, value in options.items() if key in SampleTrainingConfig.model_fields})
        super().validate_cfg(**{key: value for key, value in options.items() if key not in SampleTrainingConfig.model_fields})

    def _configure_sample_fit(self) -> None:
        for field, option in (("fit_epochs", "sample_fit_epochs"), ("fit_lr", "sample_fit_lr"),
            ("fit_tau", "sample_fit_tau"), ("fit_batch_size", "sample_fit_batch_size"),
            ("fit_encoder", "sample_train_encoder"), ("fit_prior", "sample_train_prior"),
            ("fit_backbones", "sample_train_backbones"), ("fit_hard_inputs", "sample_hard_inputs")):
            setattr(self.quantizer, field, getattr(self.sampling, option))


class SampledOracleRoundCodec(_SampleFitRoundCodec):
    """SO fits free raw samples; oracle labels are neither transmitted nor substituted at decode."""

    round_name: ClassVar[str] = "SO"
    quantizer_class: ClassVar[type[_SampledOracleQuantizer]] = _SampledOracleQuantizer
    quantizer: _SampledOracleQuantizer

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Declare SO's raw-label privilege explicitly rather than reporting it as acquisition."""
        sampling = SampleTrainingConfig(**{key: value for key, value in options.items() if key in SampleTrainingConfig.model_fields})
        return super().codec_provenance(**{key: value for key, value in options.items() if key not in SampleTrainingConfig.model_fields}) | dict(
            sampling=sampling.model_dump(), sample_training_target="raw_sample_oracle",
            sample_availability="free_current_target_labels_at_fitter", sample_acquisition="none",
            sample_rate_accounting="free_oracle_labels; full_vector_new_model_symbols",
            sample_replacement=False)

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Fit ordinary WZ weights on the raw training pool without a sample acquisition stream."""
        self._configure_sample_fit()
        self.quantizer.training_sample_size = self.sampling.sample_size
        super().pre_encode_training_steps(delta_vec)

    def get_encod_payload(self, delta_vec: torch.Tensor, record: SampledRecord) -> dict[str, Any]:
        """Record free raw labels explicitly while transmitting the ordinary full-vector symbols."""
        payload = super().get_encod_payload(delta_vec, record)
        record.update_record_fields(sample_count=0 if self.round_name_full == "F" else min(self.sampling.sample_size, delta_vec.numel()),
                                    sample_format="raw_oracle", sample_prior_rate=0.0, sample_coder_rate=0.0,
                                    **({"sample_wmape": 0.0} if self.round_name_full != "F" else {}))
        return payload

    def release_round_resources(self) -> None:
        """Drop training coordinates after the round; frozen rounds reuse only fitted weights."""
        super().release_round_resources()
        self.quantizer.training_sample_indices = None


class SampledOracleRansRoundCodec(RansRoundMixin, SampledOracleRoundCodec):
    """SOR uses rANS for ordinary symbols while its raw sample labels remain an oracle privilege."""

    round_name: ClassVar[str] = "SOR"
    transport_prior_access: ClassVar[Access] = Access.SERVER_ONLY


class SampledRoundCodec(_SampleFitRoundCodec):
    """S fits decoded samples and accounts for their acquisition before ordinary symbol transport."""

    round_name: ClassVar[str] = "S"
    record_class: ClassVar[type[SampledRecord]] = SampledRecord
    quantizer_class: ClassVar[type[_SampleQuantizer]] = _SampleQuantizer
    quantizer: _SampleQuantizer
    uses_rans: ClassVar[bool] = False
    sampling: SamplingConfig
    normal_planes: int
    _previous_state: dict[str, torch.Tensor] | None
    _previous_planes: int | None
    _sample_payload: bytes | None
    _ordinary_mask: torch.Tensor | None
    _sample_rates: list[float]
    _sample_marginal_rate: float
    _sample_fields: dict[str, Any]

    def options_to_config(
        self, sample_size: int = 3_600_000, sample_extra_planes: int = 1,
        sample_rans_lanes: int = 16, sample_training: Literal["all", "prefix"] = "all",
        sample_format: Literal["lwzq", "fp16"] = "lwzq", sample_fit_epochs: int | None = 28,
        sample_fit_lr: float | None = 0.0006807383623518464, sample_fit_tau: float | None = 0.30,
        sample_noise_multiplier: float = 0.0, sample_fit_batch_size: int | None = 7_500,
        sample_train_encoder: bool = True, sample_train_prior: bool = True,
        sample_hard_inputs: bool = False, sample_train_backbones: bool = True, **options: Any,
    ) -> None:
        """Keep sampling options at the experimental codec, then configure ordinary WZ fitting."""
        sampling = SamplingConfig(sample_size=sample_size, sample_extra_planes=sample_extra_planes,
                                       sample_rans_lanes=sample_rans_lanes, sample_training=sample_training,
                                       sample_format=sample_format, sample_fit_epochs=sample_fit_epochs,
                                       sample_fit_lr=sample_fit_lr, sample_fit_tau=sample_fit_tau,
                                       sample_noise_multiplier=sample_noise_multiplier,
                                       sample_fit_batch_size=sample_fit_batch_size, sample_train_encoder=sample_train_encoder,
                                       sample_train_prior=sample_train_prior, sample_hard_inputs=sample_hard_inputs,
                                       sample_train_backbones=sample_train_backbones)
        super().options_to_config(**{key: getattr(sampling, key) for key in SampleTrainingConfig.model_fields}, **options)
        self.sampling = sampling
        self.normal_planes = self.c_cfg.num_planes
        self._previous_state = None
        self._previous_planes = None
        self._sample_payload = None
        self._ordinary_mask = None
        self._sample_rates = []
        self._sample_marginal_rate = 0.0
        self._sample_fields = {}

    @classmethod
    def validate_cfg(
        cls, sample_size: int = 3_600_000, sample_extra_planes: int = 1,
        sample_rans_lanes: int = 16, sample_training: Literal["all", "prefix"] = "all",
        sample_format: Literal["lwzq", "fp16"] = "lwzq", sample_fit_epochs: int | None = 28,
        sample_fit_lr: float | None = 0.0006807383623518464, sample_fit_tau: float | None = 0.30,
        sample_noise_multiplier: float = 0.0, sample_fit_batch_size: int | None = 7_500,
        sample_train_encoder: bool = True, sample_train_prior: bool = True,
        sample_hard_inputs: bool = False, sample_train_backbones: bool = True, **options: Any,
    ) -> None:
        """Validate sample settings alongside the ordinary WZ quantizer configuration."""
        SamplingConfig(sample_size=sample_size, sample_extra_planes=sample_extra_planes,
                       sample_rans_lanes=sample_rans_lanes, sample_training=sample_training,
                       sample_format=sample_format, sample_fit_epochs=sample_fit_epochs,
                       sample_fit_lr=sample_fit_lr, sample_fit_tau=sample_fit_tau,
                       sample_noise_multiplier=sample_noise_multiplier,
                       sample_fit_batch_size=sample_fit_batch_size, sample_train_encoder=sample_train_encoder,
                       sample_train_prior=sample_train_prior, sample_hard_inputs=sample_hard_inputs,
                       sample_train_backbones=sample_train_backbones)
        super().validate_cfg(**options)

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Declare causal sample acquisition, cold bootstrap, and the mixed-model rate contract."""
        sampling = SamplingConfig(**{key: value for key, value in options.items()
                                     if key in SamplingConfig.model_fields})
        facts = super().codec_provenance(**options)
        facts.update(
            sampling=sampling.model_dump(), sample_coordinate_policy="numpy_pcg64_sorted_choice_v1",
            sample_model="preceding_deployed_weights_or_decoder_history_bootstrap",
            sample_training_target="decoded_sample_only", sample_normalization="current_target_metadata",
            sample_weight_transmission="same_assumption_as_O",
            sample_rate_accounting="old_sample_all_planes_plus_new_unsampled_prefix; acquisition_at_stage_0",
        )
        return facts

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[HistoryVector] | None,
        initial_states: Sequence[Mapping[str, torch.Tensor]] = (),
    ) -> _SampleQuantizer:
        """Build one machine with ordinary and refinement planes; O owns its warm initialization."""
        self.c_cfg = self.c_cfg.model_copy(update={
            "num_planes": self.normal_planes + self.sampling.sample_extra_planes})
        return super().build_quantizer(x, si, initial_states)

    def set_previous_codec(self, previous: _WZRoundCodec | None) -> None:
        """Retain only preceding weights; the sample will use this round's decoder-visible SI."""
        if previous is None:
            return
        available = getattr(previous.quantizer, "trained_num_planes", previous.quantizer.num_planes)
        assert available >= self.c_cfg.num_planes, (
            f"S needs {self.c_cfg.num_planes} sample planes; preceding codec trained only {available}.")
        self._previous_state = {
            name: value.detach().cpu().clone() for name, value in previous.quantizer.coding_model.state_dict().items()}
        self._previous_planes = available

    def create_r_record(self, round_id: int, client_id: int) -> SampledRecord:
        """Report the ordinary stage count even though the machine retains refinement weights."""
        record = super().create_r_record(round_id, client_id)
        record.num_planes = self.normal_planes
        return record

    def encode_sample_indices(
        self, bins: torch.Tensor, pilot: _SampleQuantizer, metadata: PreprocessMetadata,
    ) -> torch.Tensor | bytes:
        """Transport sample symbols directly for S; SR overrides this with rANS."""
        return bins

    def decode_sample_indices(
        self, symbols: torch.Tensor | bytes, pilot: _SampleQuantizer, metadata: PreprocessMetadata,
    ) -> torch.Tensor:
        """Recover plain sample symbols before S trains its replacement model."""
        assert isinstance(symbols, torch.Tensor)
        return symbols

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Acquire a causal sample, call O's fit on its reconstruction, then restore full-vector use."""
        if self.round_name_full == "F":
            super().pre_encode_training_steps(delta_vec)
            return

        full_config, full_si = self.c_cfg, self.si
        pilot = self.quantizer_class(full_config.model_copy(update={
            "warm_training": False, "num_planes": self._previous_planes or full_config.num_planes}), len(full_si))
        bootstrap = self._previous_state is None
        if bootstrap:
            # This target is already at the decoder; no current raw target enters the bootstrap fit.
            pilot.train_model(full_si[-1].materialize(), full_si)
        else:
            pilot.coding_model.load_state_dict(self._previous_state)
            pilot.c_cfg = full_config.model_copy(update={"warm_training": False})
            pilot.coding_model.num_planes = full_config.num_planes
            pilot.vector_size = delta_vec.numel()
            pilot.side_info_list_used = list(full_si)
        self._previous_state = None
        self._previous_planes = None

        _, full_metadata = pilot.preprocess_x(delta_vec, skip_outliers=True)
        count = min(self.sampling.sample_size, delta_vec.numel())
        seed = int(torch.randint(0, 2**31, ()).item())
        indices = SamplePayload(delta_vec.numel(), count, seed, b"", full_metadata).indices()
        counts = [int(((indices >= part.start) & (indices < part.stop)).sum())
                  for part in full_config.norm_slices]
        occupied = [index for index, size in enumerate(counts) if size]
        factors = full_metadata.norm_factors[occupied]
        sizes = [counts[index] for index in occupied]
        boundaries = np.cumsum([0, *sizes]).tolist()
        sample_slices = tuple(slice(start, stop) for start, stop in zip(boundaries, boundaries[1:]))
        row_factors = torch.repeat_interleave(factors, torch.tensor(sizes), dim=0)
        sample_si = pilot.side_info_tensor(full_metadata)[indices].cpu()
        sample_si = sample_si * row_factors[:, :1] + row_factors[:, 1:]
        sampled_history = [HistoryVector(column) for column in sample_si.unbind(dim=1)]

        pilot.c_cfg = full_config.model_copy(update={"norm_slices": sample_slices, "warm_training": False})
        pilot.vector_size = count
        pilot.normalization_factors = factors
        pilot.side_info_list_used = sampled_history
        if self.sampling.sample_format == "fp16":
            normalized, metadata = pilot.preprocess_x(delta_vec[indices])
            values = normalized.cpu().half()
            assert torch.isfinite(values).all(), "Normalized FP16 sample overflowed."
            symbols = compress_data_list(values)
            self._sample_bits = 8 * len(symbols)
            self._sample_rates = [self._sample_bits / delta_vec.numel()]
            self._sample_marginal_rate = sum(self._sample_rates)
        else:
            bins, _, metadata = pilot.encoding_process(delta_vec[indices])
            prior = PriorCalculator.compute_prior_from_network(pilot.coding_model, bins, pilot.side_info_tensor(metadata))
            self._sample_rates = (PriorCalculator.compute_code_lengths(prior, bins, pilot.num_planes).sum(dim=1)
                                  / delta_vec.numel()).tolist()
            marginal = PriorCalculator.compute_marginal_prior(bins, pilot.bins_per_plane, pilot.num_planes)
            self._sample_marginal_rate = float(PriorCalculator.compute_code_lengths(
                marginal, bins, pilot.num_planes).sum() / delta_vec.numel())
            symbols = self.encode_sample_indices(bins, pilot, metadata)
        packet = SamplePayload(delta_vec.numel(), count, seed, symbols, metadata, self.sampling.sample_format)
        self._sample_payload = compress_data_list(packet)
        received: SamplePayload = decompress_data_list(self._sample_payload)
        noise_std = 0.0
        if received.format == "fp16":
            reconstruction = pilot.postprocess(decompress_data_list(received.symbols).float().flatten(), received.metadata)
            assert self.sampling.sample_noise_multiplier == 0, "Refinement-derived smoothing requires LWZQ samples."
        else:
            received_bins = self.decode_sample_indices(received.symbols, pilot, received.metadata)
            source_stages = pilot.decoding_stages_process((received_bins, received.metadata))
            reconstruction = source_stages[-1]
            if self.sampling.sample_noise_multiplier:
                assert len(source_stages) > 1
                # Refinement RMS is decoder-visible; true sample reconstruction error is not.
                noise_std = self.sampling.sample_noise_multiplier * float(
                    ((source_stages[-1] - source_stages[-2]) / row_factors[:, 0]).square().mean().sqrt())
        metrics = record_reconstruction_metrics(delta_vec[indices], reconstruction)
        self._sample_fields = dict(
            sample_count=count, sample_bootstrap=bootstrap, sample_format=received.format,
            sample_wmape=metrics["wmape"], sample_mape=metrics["mape"],
            sample_prior_rate=sum(self._sample_rates),
            sample_metadata_size=compressed_size((received.vector_size, received.sample_size, received.seed, metadata)),
            sample_training_planes=self.normal_planes if self.sampling.sample_training == "prefix" else pilot.num_planes,
            sample_fit_epochs=self.sampling.sample_fit_epochs or self.quantizer.training_config().train_epochs,
            sample_training_batch_size=self.sampling.sample_fit_batch_size or self.c_cfg.train_batch_size,
            sample_training_encoder=self.sampling.sample_train_encoder,
            sample_training_prior=self.sampling.sample_train_prior,
            sample_training_backbones=self.sampling.sample_train_backbones,
            sample_training_hard_inputs=self.sampling.sample_hard_inputs,
            sample_noise_std=noise_std,
        )
        if received.format == "lwzq":
            self._sample_fields |= dict(sample_num_planes=pilot.num_planes,
                                       sample_extra_prior_rate=sum(self._sample_rates[self.normal_planes:]))

        self.c_cfg = full_config.model_copy(update={"norm_slices": sample_slices})
        self.quantizer.c_cfg = self.c_cfg
        self.quantizer.normalization_factors = factors
        self.quantizer.fit_planes = self.normal_planes if self.sampling.sample_training == "prefix" else None
        self._configure_sample_fit()
        fit_config = self.quantizer.training_config()
        self._sample_fields |= dict(sample_fit_lr=fit_config.lr, sample_fit_tau=fit_config.tau)
        self.quantizer.fit_noise_std = noise_std
        self.si = sampled_history
        super().pre_encode_training_steps(reconstruction)

        self.si = full_si
        self.c_cfg = full_config.model_copy(update={"num_planes": self.normal_planes})
        self.quantizer.c_cfg = self.c_cfg
        self.quantizer.coding_model.num_planes = self.normal_planes
        self.quantizer.vector_size = delta_vec.numel()
        self.quantizer.side_info_list_used = list(full_si)
        self.quantizer.normalization_factors = None
        self.quantizer.quantizer_prior_stage_rates = self.quantizer.quantizer_prior_stage_rates[:self.normal_planes]
        self.quantizer.sample_indices = received.indices()
        self.quantizer.sample_reconstruction = reconstruction
        self._ordinary_mask = torch.ones(delta_vec.numel(), dtype=torch.bool)
        self._ordinary_mask[self.quantizer.sample_indices] = False
        self._side_info_columns_cache.clear()

    def get_encod_payload(self, delta_vec: torch.Tensor, record: SampledRecord) -> dict[str, Any]:
        """Reuse O's full inference and replace sampled symbol costs with their acquisition costs."""
        payload = super().get_encod_payload(delta_vec, record)
        assert record.quantizer_prior_stage_rates is not None
        normal_rates = list(record.quantizer_prior_stage_rates)
        replaced = [0.0] * self.normal_planes
        if self._ordinary_mask is not None:
            bins, metadata = payload["payload_content"]
            sampled_bins = bins[:, ~self._ordinary_mask]
            side_info = self.quantizer.side_info_tensor(metadata).rows(self.quantizer.sample_indices)
            weight = sampled_bins.shape[1] / delta_vec.numel()
            prior = PriorCalculator.compute_prior_from_network(self._quantizer_prior_model, sampled_bins, side_info)
            replaced = [rate * weight for rate in PriorCalculator.compute_plane_rates(prior, sampled_bins, self.normal_planes)]
            ordinary_bins = bins[:, self._ordinary_mask]
            rates = ([rate - removed for rate, removed in zip(normal_rates, replaced, strict=True)]
                     if ordinary_bins.shape[1] else [0.0] * self.normal_planes)
            # Every stage needs the already-received full sample to obtain these fitted weights.
            rates[0] += sum(self._sample_rates)
            record.quantizer_prior_stage_rates = rates
            record.quantizer_prior_rate = sum(rates)
            native = PriorCalculator.compute_prior_from_network(self.quantizer.coding_model, sampled_bins, side_info)
            removed_native = sum(PriorCalculator.compute_plane_rates(native, sampled_bins, self.normal_planes)) * weight
            record.server_si_prior_rate = ((record.server_si_prior_rate - removed_native)
                                          if ordinary_bins.shape[1] else 0.0) + sum(self._sample_rates)
            marginal_rate = 0.0
            if ordinary_bins.shape[1]:
                marginal = PriorCalculator.compute_marginal_prior(ordinary_bins, self.c_cfg.bins_per_plane, self.normal_planes)
                marginal_rate = float(PriorCalculator.compute_code_lengths(
                    marginal, ordinary_bins, self.normal_planes).sum() / delta_vec.numel())
            record.marginal_rate = marginal_rate + self._sample_marginal_rate
            payload["sample_payload"] = self._sample_payload
            if not self.uses_rans:
                payload["payload_content"] = (ordinary_bins, metadata)
        record.update_record_fields(
            normal_quantizer_prior_stage_rates=normal_rates, replaced_sample_prior_stage_rates=replaced,
            **(self._sample_fields or dict(sample_count=0, sample_num_planes=self.quantizer.trained_num_planes,
                                         sample_bootstrap=False, sample_prior_rate=0.0, sample_extra_prior_rate=0.0)),
        )
        return payload

    def decode(self, payload: bytes, record: SampledRecord) -> torch.Tensor:
        """Decode ordinary positions and let the quantizer restore the previously received sample."""
        bins, metadata = decompress_data_list(payload)["payload_content"]
        if self._ordinary_mask is not None:
            restored = torch.zeros((self.normal_planes, self.quantizer.vector_size), dtype=bins.dtype)
            restored[:, self._ordinary_mask] = bins
            bins = restored
        return self.decoded_reconstruction(bins, metadata, record)

    def release_round_resources(self) -> None:
        """Drop the acquired sample at this round boundary; F reuses only the deployed machine."""
        super().release_round_resources()
        self.quantizer.sample_indices = None
        self.quantizer.sample_reconstruction = None
        self._sample_payload = None
        self._ordinary_mask = None
        self._sample_rates.clear()
        self._sample_fields.clear()
        self._sample_marginal_rate = 0.0


class SampledRansRoundCodec(RansRoundMixin, SampledRoundCodec):
    """S with independent old-model sample and new-model ordinary rANS streams."""

    round_name: ClassVar[str] = "SR"
    transport_prior_access: ClassVar[Access] = Access.SERVER_ONLY
    uses_rans: ClassVar[bool] = True
    _sample_bits: int = 0
    _sample_extra_bits: int = 0
    _sample_overhead_bits: int = 0

    @staticmethod
    def _pilot_prior(pilot: _SampleQuantizer, metadata: PreprocessMetadata) -> ConditionalPrior:
        side_info = pilot.side_info_tensor(metadata)

        def prior(decoded: torch.Tensor, stage: int) -> torch.Tensor:
            context = torch.zeros((pilot.num_planes, decoded.shape[1]), dtype=decoded.dtype)
            context[:decoded.shape[0]] = decoded.cpu()
            return PriorCalculator.compute_prior_from_network(pilot.coding_model, context, side_info)[stage]

        return prior

    def encode_sample_indices(
        self, bins: torch.Tensor, pilot: _SampleQuantizer, metadata: PreprocessMetadata,
    ) -> bytes:
        """Code all sample planes; measure the refinement's stream growth separately."""
        config = RansConfig(lanes=min(self.sampling.sample_rans_lanes, bins.shape[1]),
                            precision=self.rans_config.precision)
        prior = self._pilot_prior(pilot, metadata)
        payload = encode(bins, prior, config)
        prefix = encode(bins[:self.normal_planes], prior, config)
        self._sample_bits = 8 * len(payload)
        self._sample_extra_bits = 8 * (len(payload) - len(prefix))
        self._sample_overhead_bits = 8 * (HEADER.size + 4 * config.lanes)
        return payload

    def decode_sample_indices(
        self, symbols: torch.Tensor | bytes, pilot: _SampleQuantizer, metadata: PreprocessMetadata,
    ) -> torch.Tensor:
        """Recover sample symbols using the preceding machine before it is retired."""
        assert isinstance(symbols, bytes)
        return decode(symbols, self._pilot_prior(pilot, metadata))

    def prior_evaluations(self, metadata: PreprocessMetadata) -> dict[Access, PriorEvaluation]:
        """Condition the normal transport only on positions whose symbols it transmits."""
        evaluations = super().prior_evaluations(metadata)
        if self._ordinary_mask is None:
            return evaluations
        indices = self._ordinary_mask.nonzero().flatten()
        return {access: PriorEvaluation(value.model, value.side_information.rows(indices))
                for access, value in evaluations.items()}

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> bytes:
        """Transmit the new prefix only outside the sample, accounting for both streams over N."""
        if self._ordinary_mask is None:
            return super().encode_staged_indices(bins, metadata)
        ordinary = bins[:, self._ordinary_mask]
        if ordinary.shape[1]:
            payload = super().encode_staged_indices(ordinary, metadata)
        else:
            payload = b""
            self._rans_metrics = {self.transport_prior_access: (0.0, [0.0] * self.normal_planes, 0.0)}
        size, count = bins.shape[1], ordinary.shape[1]
        for access, (_, rates, overhead) in self._rans_metrics.items():
            weighted = [rate * count / size for rate in rates]
            weighted[0] += self._sample_bits / size
            self._rans_metrics[access] = (
                sum(weighted), weighted, overhead * count / size + self._sample_overhead_bits / size)
        return payload

    def decode_staged_indices(self, payload: bytes, metadata: PreprocessMetadata) -> torch.Tensor:
        """Expand recovered ordinary symbols; sample reconstructions replace the unused positions."""
        if self._ordinary_mask is None:
            return super().decode_staged_indices(payload, metadata)
        bins = torch.zeros((self.normal_planes, self.quantizer.vector_size), dtype=torch.long)
        if self._ordinary_mask.any():
            bins[:, self._ordinary_mask] = super().decode_staged_indices(payload, metadata)
        return bins

    def get_encod_payload(self, delta_vec: torch.Tensor, record: SampledRecord) -> dict[str, Any]:
        """Expose sample rANS bytes and its sparse refinement cost alongside ordinary transport."""
        payload = super().get_encod_payload(delta_vec, record)
        size = delta_vec.numel()
        record.update_record_fields(
            sample_coder_rate=self._sample_bits / size,
            **(dict(sample_extra_coder_rate=self._sample_extra_bits / size,
                    sample_coder_overhead_rate=self._sample_overhead_bits / size)
               if self.sampling.sample_format == "lwzq" or self._ordinary_mask is None else {}),
        )
        return payload

    def release_round_resources(self) -> None:
        """Keep sample transport costs from leaking into frozen-round records."""
        super().release_round_resources()
        self._sample_bits = self._sample_extra_bits = self._sample_overhead_bits = 0
