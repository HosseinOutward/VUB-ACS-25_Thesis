"""Matched O versus practical S extra-plane fits from a retained decoder-visible model."""

from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import tarfile
from time import perf_counter
from typing import Any, ClassVar, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import PositiveInt
import torch

from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.live_plot import watch
from Experiments.Protocol_test.sampled_codec import SampledRansRoundCodec, SamplePayload, SampledRecord, _SampleQuantizer
from Experiments.Protocol_test.starting_model_comparison import _MeasuredOracleCodec, projected_state
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text
from FL_code.FL_core.codec import record_reconstruction_metrics
from FL_code.FL_core.explicit_cache import PickleCodec, TorchCodec, atomic_output, content_hash
from FL_code.FL_core.utils import decompress_data_list, payload_size_mib, set_global_seed
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.wz_quantizer import PreprocessMetadata, WZcfgQuant
from FL_code.cancer_protocol.wz_quantizer import select_si_rows

PlaneVariant = Literal[
    "current_e1_mse", "current_e1_mae", "residual_e1_mse", "residual_e1_mae",
    "current_e1_mse_received_prefix_bins", "current_e1_mae_received_prefix_bins",
    "current_e1_raw_mae_received_prefix_bins",
    "current_e1_mae_received_prefix_bins_final_raw_mae",
    "current_e1_mse_reencode_teacher_frozen_control",
    "current_e1_mae_reencode_teacher_trainable_encoder",
    "current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic",
    "received_fp16_training_message", "received_fp16_training_message_validated_split",
    "received_fp16_training_message_validated_split_hard_inputs",
    "received_fp16_training_message_validated_split_hard_categorical_prior",
    "received_fp16_training_message_validated_split_hard_transport_nll",
    "received_fp16_training_message_validated_split_hard_transport_nll_heads_only",
]


def _split_received_sample_rows(
    sample_slices: Sequence[slice], validation_fraction: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[slice, ...], tuple[slice, ...]]:
    """Split packet row positions per normalization slice, retaining singleton rows for fitting."""
    generator = np.random.default_rng(seed)
    train_positions: list[np.ndarray] = []
    validation_positions: list[np.ndarray] = []
    train_slices: list[slice] = []
    validation_slices: list[slice] = []
    train_cursor = validation_cursor = 0
    for sample_slice in sample_slices:
        positions = np.arange(sample_slice.start, sample_slice.stop)
        assert positions.size > 0, "Received sample normalization slices must be nonempty."
        shuffled = generator.permutation(positions)
        validation_count = min(max(1, round(positions.size * validation_fraction)), positions.size - 1)
        validation = np.sort(shuffled[:validation_count])
        training = np.sort(shuffled[validation_count:])
        train_positions.append(training)
        validation_positions.append(validation)
        train_slices.append(slice(train_cursor, train_cursor + training.size))
        validation_slices.append(slice(validation_cursor, validation_cursor + validation.size))
        train_cursor += training.size
        validation_cursor += validation.size
    return (
        torch.from_numpy(np.concatenate(train_positions)).long(),
        torch.from_numpy(np.concatenate(validation_positions)).long(),
        tuple(train_slices), tuple(validation_slices),
    )
AcquisitionDistortion = Literal["mse", "mae", "raw_mae"]
SampleFitMode = Literal["reencode_teacher", "received_prefix_bins"]
_PLANE_VARIANT_OPTIONS: dict[PlaneVariant, tuple[bool, AcquisitionDistortion]] = {
    "current_e1_mse": (False, "mse"),
    "current_e1_mae": (False, "mae"),
    "residual_e1_mse": (True, "mse"),
    "residual_e1_mae": (True, "mae"),
    "current_e1_mse_received_prefix_bins": (False, "mse"),
    "current_e1_mae_received_prefix_bins": (False, "mae"),
    "current_e1_raw_mae_received_prefix_bins": (False, "raw_mae"),
    "current_e1_mae_received_prefix_bins_final_raw_mae": (False, "mae"),
    "current_e1_mse_reencode_teacher_frozen_control": (False, "mse"),
    "current_e1_mae_reencode_teacher_trainable_encoder": (False, "mae"),
    "current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic": (False, "mae"),
    "received_fp16_training_message": (False, "mse"),
    "received_fp16_training_message_validated_split": (False, "mse"),
    "received_fp16_training_message_validated_split_hard_inputs": (False, "mse"),
    "received_fp16_training_message_validated_split_hard_categorical_prior": (False, "mse"),
    "received_fp16_training_message_validated_split_hard_transport_nll": (False, "mse"),
    "received_fp16_training_message_validated_split_hard_transport_nll_heads_only": (False, "mse"),
}
_PLANE_VARIANT_FIT: dict[PlaneVariant, tuple[SampleFitMode, bool]] = {
    "current_e1_mse_received_prefix_bins": ("received_prefix_bins", False),
    "current_e1_mae_received_prefix_bins": ("received_prefix_bins", False),
    "current_e1_raw_mae_received_prefix_bins": ("received_prefix_bins", False),
    "current_e1_mae_received_prefix_bins_final_raw_mae": ("received_prefix_bins", False),
    "current_e1_mse_reencode_teacher_frozen_control": ("reencode_teacher", False),
    "current_e1_mae_reencode_teacher_trainable_encoder": ("reencode_teacher", True),
    "current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic": ("received_prefix_bins", False),
}


class PlaneSweepPlan(BaseModel):
    """Fixed-target acquisition study consumed by the worker, reporter and detached supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    output_root: Path
    reference_root: Path
    comparison_root: Path | None = None
    fit_device: str = "cuda:0"
    seeds: tuple[int, ...] = (43, 44, 45, 46, 47)
    extra_planes: tuple[int, ...] = (0, 1, 2, 3, 6)
    retain_seeds: int = Field(default=3, ge=1, strict=True)
    sample_size: int = Field(default=3_600_000, ge=1, strict=True)
    sample_fit_epochs: int = Field(default=28, ge=1, strict=True)
    sample_fit_lr: float = Field(default=0.0006807383623518464, gt=0)
    sample_fit_tau: float = Field(default=0.30, gt=0)
    sample_fit_batch_size: int = Field(default=7_500, ge=1, strict=True)
    training_message_coordinate_seed: int = Field(default=494_155_588, ge=0, lt=2**31, strict=True)
    sample_validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    sample_validation_seed: int = Field(default=494_155_589, ge=0, lt=2**31, strict=True)
    refiner_fit_epochs: int = Field(default=140, ge=1, strict=True)
    refiner_fit_lr: float = Field(default=0.0028364098431326932, gt=0)
    refiner_fit_tau: float = Field(default=5.5, gt=0)
    refiner_fit_lr_step: int = Field(default=43, ge=1, strict=True)
    refiner_fit_batch_size: int = Field(default=6_250, ge=1, strict=True)
    refiner_fit_reconst_multiplier: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sample_refiner: bool = False
    sample_refiner_residual_anchor: bool = False
    sample_acquisition_distortion: AcquisitionDistortion = "mse"
    sample_final_distortion: Literal["mse", "mae", "raw_mae"] = "mse"
    sample_fit_mode: SampleFitMode = "reencode_teacher"
    sample_train_encoder: bool = True
    variant_modes: tuple[PlaneVariant, ...] | None = None
    budget_sample_sizes: tuple[PositiveInt, ...] | None = None
    include_budget_control: bool = True
    include_variant_controls: bool = True

    @model_validator(mode="after")
    def validate_grid(self) -> PlaneSweepPlan:
        """Keep the seed/plane catalog unique and the selected population explicit."""
        assert self.seeds and len(set(self.seeds)) == len(self.seeds) and min(self.seeds) >= 0
        assert self.extra_planes and len(set(self.extra_planes)) == len(self.extra_planes) and min(self.extra_planes) >= 0
        assert self.retain_seeds <= len(self.seeds)
        assert not self.sample_refiner_residual_anchor or self.sample_refiner, (
            "sample_refiner_residual_anchor requires sample_refiner=True.")
        assert not self.sample_refiner_residual_anchor or any(self.extra_planes), (
            "sample_refiner_residual_anchor requires at least one extra plane.")
        assert self.sample_acquisition_distortion == "mse" or self.sample_refiner, (
            "A non-MSE sample acquisition distortion requires sample_refiner=True.")
        assert self.sample_acquisition_distortion == "mse" or any(self.extra_planes) or any(
            _PLANE_VARIANT_OPTIONS[mode][1] != "mse" for mode in self.variant_modes or ()
        ), (
            "A non-MSE sample acquisition distortion requires at least one extra plane.")
        assert (self.sample_final_distortion == "mse"
                or (self.sample_final_distortion == "mae" and self.sample_refiner
                    and self.variant_modes == ("current_e1_mae_received_prefix_bins",)
                    and self.sample_train_encoder is False)
                or (self.sample_final_distortion == "raw_mae" and self.sample_refiner
                    and self.variant_modes == ("current_e1_mae_received_prefix_bins_final_raw_mae",)
                    and self.sample_train_encoder is False)), (
                        "Final MAE fitting requires its matching frozen received-prefix variant.")
        assert (not any(mode.endswith("SO_fit_only_diagnostic") for mode in self.variant_modes or ())
                or (self.sample_final_distortion == "mse" and self.sample_refiner
                    and self.variant_modes == ("current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic",)
                    and self.sample_train_encoder is False)), (
                        "The SO fit-label diagnostic requires normalized-MAE acquisition, frozen received-prefix bins, and final MSE.")
        if self.variant_modes is not None:
            training_message = self.variant_modes in (
                ("received_fp16_training_message",), ("received_fp16_training_message_validated_split",),
                ("received_fp16_training_message_validated_split_hard_inputs",),
                ("received_fp16_training_message_validated_split_hard_categorical_prior",),
                ("received_fp16_training_message_validated_split_hard_transport_nll",),
                ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",))
            assert len(set(self.variant_modes)) == len(self.variant_modes), "Sample variant modes must be unique."
            assert self.extra_planes == (0,), "Variant-mode plans use the single extra-plane S acquisition."
            assert self.sample_refiner or training_message, "Variant-mode plans require sample_refiner=True."
            assert not training_message or (not self.sample_refiner and self.sample_train_encoder), (
                "The FP16 training message requires a trainable full quantizer and no sample refiner.")
            assert not training_message or (
                self.sample_fit_epochs == 35 and self.sample_fit_lr == 0.0008509229529398079
                and self.sample_fit_tau == 0.5 and self.sample_fit_batch_size == 6_250
            ), "The FP16 training-message fit must match O's 35/550k/6250/LR/tau schedule."
            assert not any("validated_split" in mode for mode in self.variant_modes) or (
                self.sample_validation_fraction == 0.2 and self.sample_validation_seed == 494_155_589), (
                    "The validated FP16 split must use the declared 80/20 deterministic sample split.")
            assert (not any("received_prefix_bins" in mode for mode in self.variant_modes)
                    or (self.extra_planes == (0,) and self.sample_refiner and not self.sample_train_encoder)), (
                        "Received-prefix variants require one extra plane, a sample refiner, and a frozen encoder.")
            assert (not any(mode.endswith("frozen_control") for mode in self.variant_modes)
                    or (self.extra_planes == (0,) and self.sample_refiner)), (
                        "The frozen reencode control requires one extra plane and a sample refiner.")
        if self.budget_sample_sizes is not None:
            assert self.budget_sample_sizes and len(set(self.budget_sample_sizes)) == len(self.budget_sample_sizes)
            assert self.variant_modes and self.extra_planes == (0,)
            assert self.sample_refiner or self.variant_modes in (
                ("received_fp16_training_message",), ("received_fp16_training_message_validated_split",),
                ("received_fp16_training_message_validated_split_hard_inputs",),
                ("received_fp16_training_message_validated_split_hard_categorical_prior",),
                ("received_fp16_training_message_validated_split_hard_transport_nll",),
                ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",))
            assert self.include_budget_control or self.comparison_root is not None, (
                "Budget candidates without a fresh O case require comparison_root for their read-only O control.")
        return self

    @property
    def methods(self) -> tuple[str, ...]:
        """Name the full-target control and every practical sample refinement variant."""
        if self.budget_sample_sizes is not None:
            control = ("O",) if self.include_budget_control else ()
            return (*control, *(f"{mode}_n{size}" for mode in self.variant_modes for size in self.budget_sample_sizes))
        if self.variant_modes is not None:
            controls = ("O", "S_e0") if self.include_variant_controls else ()
            return (*controls, *self.variant_modes)
        return ("O", *(f"S_e{extra}" for extra in self.extra_planes))

    @property
    def cases(self) -> tuple[str, ...]:
        """Expose seed-matched case labels to the detached health supervisor."""
        return tuple(f"{method}_seed{seed}" for seed in self.seeds for method in self.methods)


def load_plan(path: Path) -> PlaneSweepPlan:
    """Resolve retained inputs and fresh output relative to the named JSON plan."""
    plan = PlaneSweepPlan.model_validate_json(path.read_text())
    return plan.model_copy(update={key: (path.parent / getattr(plan, key)).resolve()
                                   for key in ("output_root", "reference_root")}
                           | ({"comparison_root": (path.parent / plan.comparison_root).resolve()}
                              if plan.comparison_root is not None else {}))


class PlaneSweepRecord(SampledRecord):
    """Practical S rates with directly evaluated full-SI prior stages including acquisition."""

    server_si_prior_stage_rates: list[float] | None = None
    sample_fit_label_source: str | None = None
    sample_fit_label_available_to_receiver: bool | None = None
    sample_setup_deployable: bool | None = None
    sample_training_message_bytes: int | None = None
    sample_training_message_rate: float | None = None
    sample_fit_training_pool_rows: int | None = None
    sample_prior_refit_rows: int | None = None
    sample_prior_refit_target: str | None = None
    sample_prior_fit_context: str | None = None
    sample_fit_prior_distribution: str | None = None
    sample_fit_rate_objective: str | None = None
    sample_fit_deterministic_encoder: bool | None = None
    sample_fit_trainable_parameter_names: list[str] | None = None
    sample_fit_trainable_parameter_count: int | None = None
    sample_fit_trunk_state_hash_before: str | None = None
    sample_fit_trunk_state_hash_after: str | None = None
    sample_prior_heads_only: bool | None = None
    sample_prior_trainable_parameter_names: list[str] | None = None
    sample_prior_trainable_parameter_count: int | None = None
    sample_prior_trunk_state_hash_before: str | None = None
    sample_prior_trunk_state_hash_after: str | None = None
    sample_full_target_prior_refit_rows: int | None = None
    sample_fit_draws_per_epoch: int | None = None
    sample_fit_batches_per_epoch: int | None = None
    sample_fit_updates_per_accepted_candidate: int | None = None
    sample_fit_required_accepted_candidates: int | None = None
    sample_fit_max_attempts: int | None = None
    sample_prior_draws_per_epoch: int | None = None
    sample_prior_updates_per_candidate: int | None = None
    sample_prior_required_candidates: int | None = None
    sample_validation_train_rows: int | None = None
    sample_validation_rows: int | None = None
    sample_validation_fraction: float | None = None
    sample_validation_seed: int | None = None
    sample_validation_split_hash: str | None = None
    sample_fit_validation_best_epoch: int | None = None
    sample_fit_validation_best_score: float | None = None
    sample_fit_validation_metric: str | None = None
    sample_fit_validation_attempts: int | None = None
    sample_fit_validation_updates: int | None = None
    sample_fit_rd_trace_attempts: list[int] | None = None
    sample_fit_rd_trace_epochs: list[int] | None = None
    sample_fit_training_rd_trace: list[float] | None = None
    sample_fit_validation_rd_trace: list[float] | None = None
    sample_prior_validation_best_epoch: int | None = None
    sample_prior_validation_best_score: float | None = None
    sample_prior_validation_metric: str | None = None
    sample_prior_validation_attempts: int | None = None
    sample_prior_validation_updates: int | None = None
    sample_prior_state_hash_before_full_eval: str | None = None
    sample_prior_state_hash_after_full_eval: str | None = None


class _ReceivedFP16TrainingMessageQuantizer(_SampleQuantizer):
    """Repeat the received label pool so a small fit keeps O's full per-epoch draw budget."""

    fit_pool_rows: int | None = None
    fit_draws_per_epoch: int | None = None

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: Any) -> None:
        self.fit_pool_rows = x_prep.shape[0]
        self.fit_draws_per_epoch = min(self.c_cfg.train_sample_size, x_prep.shape[0])
        if x_prep.shape[0] < self.c_cfg.train_sample_size:
            rows = torch.arange(self.c_cfg.train_sample_size) % x_prep.shape[0]
            x_prep = x_prep[rows.to(x_prep.device)]
            side_info = select_si_rows(side_info, rows)
            self.fit_draws_per_epoch = self.c_cfg.train_sample_size
        super()._fit_prepared(x_prep, side_info)


class _ReceivedFP16ValidatedSplitQuantizer(_SampleQuantizer):
    """Fit only on the training partition, letting the optimizer draw each row at most once per epoch."""

    fit_pool_rows: int | None = None
    fit_draws_per_epoch: int | None = None

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: Any) -> None:
        self.fit_pool_rows = x_prep.shape[0]
        self.fit_draws_per_epoch = min(self.c_cfg.train_sample_size, x_prep.shape[0])
        super()._fit_prepared(x_prep, side_info)


class _MeasuredSampleCodec(SampledRansRoundCodec):
    """Real S/SR acquisition and transport with an additional measured prior-stage breakdown."""

    record_class: ClassVar[type[PlaneSweepRecord]] = PlaneSweepRecord
    _diagnostic_target: torch.Tensor | None = None
    _extra_counterfactual_reconstructions: tuple[torch.Tensor, torch.Tensor] | None = None

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Retain the raw target only long enough to measure a transmitted-symbol ablation."""
        self._diagnostic_target = delta_vec
        self._extra_conditional_entropy: list[float] = []
        self._extra_counterfactual_reconstructions = None
        super().pre_encode_training_steps(delta_vec)
        if self._extra_conditional_entropy:
            self._sample_fields["sample_extra_prefix_conditional_entropy_bps"] = self._extra_conditional_entropy
        counterfactuals = self._extra_counterfactual_reconstructions
        if counterfactuals is not None:
            assert self.quantizer.sample_indices is not None
            target = delta_vec[self.quantizer.sample_indices]
            self._sample_fields["sample_extra_final_prefix_shuffle_wmape"] = record_reconstruction_metrics(
                target, counterfactuals[0])["wmape"]
            self._sample_fields["sample_extra_final_prior_mode_wmape"] = record_reconstruction_metrics(
                target, counterfactuals[1])["wmape"]
        self._diagnostic_target = None
        self._extra_counterfactual_reconstructions = None

    def fit_additional_priors(self, bins: torch.Tensor, soft_codes: torch.Tensor,
                              metadata: PreprocessMetadata, record: PlaneSweepRecord) -> None:
        """Evaluate the native fitted prior on transmitted positions and include old-model sample cost."""
        super().fit_additional_priors(bins, soft_codes, metadata, record)
        assert self._ordinary_mask is not None
        ordinary = bins[:, self._ordinary_mask]
        si = self.quantizer.side_info_tensor(metadata).rows(self._ordinary_mask.nonzero().flatten())
        prior = PriorCalculator.compute_prior_from_network(self.quantizer.coding_model, ordinary, si)
        rates = (PriorCalculator.compute_code_lengths(prior, ordinary, self.normal_planes).sum(dim=1)
                 / bins.shape[1]).tolist()
        rates[0] += sum(self._sample_rates)
        record.server_si_prior_stage_rates = rates

    def encode_sample_indices(self, bins: torch.Tensor, pilot: _SampleQuantizer,
                              metadata: PreprocessMetadata) -> bytes:
        """Measure whether extra symbols improve decoding when their coordinate alignment is removed."""
        payload = super().encode_sample_indices(bins, pilot, metadata)
        if bins.shape[0] == self.normal_planes or self._diagnostic_target is None:
            return payload
        received_bins = self.decode_sample_indices(payload, pilot, metadata)
        side_info = pilot.side_info_tensor(metadata)
        for stage in range(self.normal_planes, received_bins.shape[0]):
            context = received_bins[:stage].T.to(torch.long)
            symbols = received_bins[stage].to(torch.long)
            _, groups = torch.unique(context, dim=0, return_inverse=True)
            group_count = int(groups.max().item()) + 1
            joint_counts = torch.bincount(groups * pilot.bins_per_plane + symbols,
                                          minlength=group_count * pilot.bins_per_plane).reshape(
                                              group_count, pilot.bins_per_plane)
            context_counts = joint_counts.sum(dim=1)
            probabilities = joint_counts.float() / context_counts[:, None]
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log2()).sum(dim=1)
            self._extra_conditional_entropy.append(
                float((entropy * context_counts / symbols.numel()).sum().item()))
        final_stage = received_bins.shape[0] - 1
        final_context = received_bins[:final_stage].T.to(torch.long)
        _, groups = torch.unique(final_context, dim=0, return_inverse=True)
        shuffled = received_bins.clone()
        generator = torch.Generator().manual_seed(0)
        for group in range(int(groups.max().item()) + 1):
            positions = (groups == group).nonzero().flatten()
            order = torch.randperm(positions.numel(), generator=generator)
            shuffled[final_stage, positions] = received_bins[final_stage, positions[order]]
        prior = PriorCalculator.compute_prior_from_network(pilot.coding_model, received_bins, side_info)
        mode = received_bins.clone()
        mode[final_stage] = prior[final_stage].argmax(dim=-1).to(mode.dtype)
        self._extra_counterfactual_reconstructions = (
            pilot.decoding_stages_process((shuffled, metadata))[-1],
            pilot.decoding_stages_process((mode, metadata))[-1],
        )
        return payload


class _ReceivedFP16TrainingMessageCodec(_MeasuredSampleCodec):
    """Fit the full S quantizer on received FP16 labels, then code every target coordinate."""

    _fine_tune_native_prior: ClassVar[bool] = False
    quantizer_class: ClassVar[type[_SampleQuantizer]] = _ReceivedFP16TrainingMessageQuantizer
    _received_training_indices: torch.Tensor | None = None
    _received_training_values: torch.Tensor | None = None
    _received_prior_fit_rows: int | None = None
    _received_prior_state_hash: str | None = None
    validated_split: ClassVar[bool] = False
    trace_fit_rd: ClassVar[bool] = False
    categorical_prior: ClassVar[bool] = False
    hard_transport_nll: ClassVar[bool] = False
    heads_only: ClassVar[bool] = False
    sample_validation_fraction: float = 0.2
    sample_validation_seed: int = 494_155_589
    _received_validation_counts: tuple[int, int] | None = None
    _received_validation_split_hash: str | None = None
    _received_main_validation_attempts: int = 0
    _received_prior_validation_attempts: int = 0
    _received_prior_selection_epoch: int | None = None
    _received_prior_selection_score: float | None = None
    _received_joint_trunk_state_hash_before: str | None = None
    _received_joint_trunk_state_hash_after: str | None = None
    _received_prior_trunk_state_hash_before: str | None = None
    _received_prior_trunk_state_hash_after: str | None = None
    _received_joint_trainable_parameter_names: list[str] | None = None
    _received_prior_trainable_parameter_names: list[str] | None = None

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Retain receiver-visible FP16 labels for the fit while disabling S reconstruction replacement."""
        assert self.sampling.sample_format == "fp16" and self.sampling.sample_training == "all"
        assert self.sampling.sample_train_encoder
        assert self.sampling.sample_train_backbones or self.heads_only, (
            "Only the explicit heads-only candidate may freeze recurrent fitting backbones.")
        assert not self.heads_only or not self.sampling.sample_train_backbones, (
            "The heads-only candidate must freeze its recurrent fitting backbones.")
        assert self.sampling.sample_hard_inputs == self.trace_fit_rd, (
            "Only the validated hard-input sibling may fit with straight-through hard decoder/prior inputs.")
        self.quantizer.fit_force_prior_softmax = self.categorical_prior
        self.quantizer.fit_hard_transport_nll = self.hard_transport_nll
        original_train = self.quantizer.train_model

        def fit_received_prior(x_raw: torch.Tensor, si_raw_list: Any, *args: Any, **kwargs: Any) -> None:
            quantizer = self.quantizer
            if not self.validated_split:
                original_train(x_raw, si_raw_list, *args, **kwargs)
                x_prep, metadata = quantizer.preprocess_x(x_raw)
                side_info = quantizer.normalize_side_info_tensor(metadata, quantizer.side_info_list_used or [])
                bins, soft_codes = quantizer.encode_prepared(x_prep, quantizer.c_cfg.train_batch_size)
                rows = torch.arange(quantizer.c_cfg.train_sample_size) % bins.shape[1]
                PriorCalculator.fit_prior_model(
                    bins[:, rows].long(), soft_codes[:, rows], select_si_rows(side_info, rows), quantizer.c_cfg,
                    quantizer.coding_model, (None,), quantizer.c_cfg.prior_train_repeats,
                )
                self._received_prior_fit_rows = bins.shape[1]
                return

            sample_packet: SamplePayload = decompress_data_list(self._sample_payload)
            sample_slices = quantizer.norm_slices
            si_values = torch.stack([value.materialize() for value in si_raw_list], dim=1)
            train_positions, validation_positions, train_slices, validation_slices = _split_received_sample_rows(
                sample_slices, self.sample_validation_fraction, self.sample_validation_seed)
            assert train_positions.numel() + validation_positions.numel() == x_raw.numel()
            assert not torch.isin(train_positions, validation_positions).any()
            assert torch.equal(torch.sort(torch.cat((train_positions, validation_positions))).values,
                               torch.arange(x_raw.numel()))
            training_x = x_raw[train_positions]
            validation_x = x_raw[validation_positions]
            training_si = [HistoryVector(si_values[train_positions, column]) for column in range(si_values.shape[1])]
            validation_si = [HistoryVector(si_values[validation_positions, column]) for column in range(si_values.shape[1])]
            full_sample_config = quantizer.c_cfg
            quantizer.normalization_factors = sample_packet.metadata.norm_factors
            quantizer.c_cfg = full_sample_config.model_copy(update={"norm_slices": tuple(train_slices)})
            quantizer.vector_size = training_x.numel()
            if self.trace_fit_rd:
                self._received_fit_rd_trace_attempts: list[int] = []
                self._received_fit_rd_trace_epochs: list[int] = []
                self._received_fit_training_rd_trace: list[float] = []
                self._received_fit_validation_rd_trace: list[float] = []
            train_prepared, train_metadata = quantizer.preprocess_x(training_x)
            training_side_info = quantizer.normalize_side_info_tensor(train_metadata, training_si)
            wmspe_denom = (train_prepared.float().square().mean().item() / 2) + 1e-8
            quantizer.c_cfg = full_sample_config.model_copy(update={"norm_slices": tuple(validation_slices)})
            quantizer.vector_size = validation_x.numel()
            validation_prepared, validation_metadata = quantizer.preprocess_x(validation_x)
            validation_side_info = quantizer.normalize_side_info_tensor(validation_metadata, validation_si)
            quantizer.c_cfg = full_sample_config.model_copy(update={"norm_slices": tuple(train_slices)})
            quantizer.vector_size = training_x.numel()

            def hard_rd_score(model: Any, prepared: torch.Tensor, side_info: torch.Tensor) -> float:
                total = torch.zeros((), device=prepared.device)
                for start in range(0, prepared.shape[0], 4_096):
                    stop = min(start + 4_096, prepared.shape[0])
                    x_batch, y_batch = prepared[start:stop], side_info[start:stop]
                    bins, _ = model.encode(x_batch, force_softmax=True)
                    hard_codes = [torch.nn.functional.one_hot(
                        symbols.squeeze(-1).long(), quantizer.bins_per_plane).float() for symbols in bins]
                    reconstructions = model.decode(hard_codes, y_batch)
                    priors = model.get_priors(hard_codes, y_batch, force_softmax=True)
                    rows = torch.arange(stop - start, device=x_batch.device)
                    for plane in range(model.num_planes):
                        distortion = torch.nn.functional.mse_loss(reconstructions[plane], x_batch)
                        rate = -torch.log(priors[plane][rows, bins[plane].squeeze(-1)] + 1e-12).mean()
                        total += ((quantizer.reconst_ld / wmspe_denom * distortion + rate)
                                  * (stop - start) / model.num_planes)
                return float(total / prepared.shape[0])

            def quantizer_validation_score(model: Any, epoch: int) -> float:
                if epoch == -1:
                    self._received_main_validation_attempts += 1
                validation_score = hard_rd_score(model, validation_prepared, validation_side_info)
                if self.trace_fit_rd:
                    self._received_fit_rd_trace_attempts.append(self._received_main_validation_attempts - 1)
                    self._received_fit_rd_trace_epochs.append(epoch)
                    self._received_fit_training_rd_trace.append(
                        hard_rd_score(model, train_prepared, training_side_info))
                    self._received_fit_validation_rd_trace.append(validation_score)
                return validation_score

            quantizer.selection_metric = quantizer_validation_score
            if self.heads_only:
                self._received_joint_trunk_state_hash_before = content_hash({
                    name: value for name, value in quantizer.coding_model.state_dict().items()
                    if name.startswith(("encoder.", "decoder.", "conditionalRNN."))
                })
            try:
                original_train(training_x, training_si, *args, **kwargs)
            finally:
                quantizer.selection_metric = None
            if self.heads_only:
                self._received_joint_trunk_state_hash_after = content_hash({
                    name: value for name, value in quantizer.coding_model.state_dict().items()
                    if name.startswith(("encoder.", "decoder.", "conditionalRNN."))
                })
                assert self._received_joint_trunk_state_hash_before == self._received_joint_trunk_state_hash_after, (
                    "Head-only received-sample fitting changed a frozen recurrent trunk.")
                self._received_joint_trainable_parameter_names = [
                    name for name, parameter in quantizer.coding_model.named_parameters() if parameter.requires_grad
                ]

            training_bins, training_soft_codes = quantizer.encode_prepared(
                train_prepared, quantizer.c_cfg.train_batch_size)
            validation_bins, validation_soft_codes = quantizer.encode_prepared(
                validation_prepared, quantizer.c_cfg.train_batch_size)
            validation_context = torch.nn.functional.one_hot(
                validation_bins.T.long(), quantizer.bins_per_plane).float()
            prior_fit_context = (
                torch.nn.functional.one_hot(training_bins.T.long(), quantizer.bins_per_plane)
                .permute(1, 0, 2).float()
                if self.trace_fit_rd else training_soft_codes
            )

            def prior_validation_score(model: Any, epoch: int) -> float:
                if epoch == -1:
                    self._received_prior_validation_attempts += 1
                total = torch.zeros((), device=validation_prepared.device)
                for start in range(0, validation_bins.shape[1], 4_096):
                    stop = min(start + 4_096, validation_bins.shape[1])
                    codes = tuple(
                        code.to(validation_side_info.device)
                        for code in validation_context[start:stop].unbind(dim=1)
                    )
                    priors = model.get_priors(codes, validation_side_info[start:stop], force_softmax=True)
                    rows = torch.arange(stop - start, device=validation_side_info.device)
                    for plane, prior in enumerate(priors):
                        symbols = validation_bins[plane, start:stop].long().to(validation_side_info.device)
                        total -= torch.log(prior[rows, symbols] + 1e-12).sum() / quantizer.num_planes
                return float(total / validation_bins.shape[1])

            prior_trunk_hash_before = content_hash({
                name: value for name, value in quantizer.coding_model.state_dict().items()
                if name.startswith(("encoder.", "decoder.", "conditionalRNN."))
            }) if self.heads_only else None
            prior_fit_options = {"prior_heads_only": True} if self.heads_only else {}
            prior_result = PriorCalculator.fit_prior_model(
                training_bins.long(), prior_fit_context, training_side_info, quantizer.c_cfg,
                quantizer.coding_model, (None,), quantizer.c_cfg.prior_train_repeats,
                selection_metric=prior_validation_score,
                **prior_fit_options,
            )
            if self.heads_only:
                prior_trunk_hash_after = content_hash({
                    name: value for name, value in quantizer.coding_model.state_dict().items()
                    if name.startswith(("encoder.", "decoder.", "conditionalRNN."))
                })
                assert prior_trunk_hash_before == prior_trunk_hash_after, (
                    "Head-only native prior fitting changed the frozen conditional-RNN trunk.")
                self._received_prior_trunk_state_hash_before = prior_trunk_hash_before
                self._received_prior_trunk_state_hash_after = prior_trunk_hash_after
                self._received_prior_trainable_parameter_names = [
                    name for name, _ in quantizer.coding_model.named_parameters()
                    if set(name.split(".")) & {"conditionalPrior", "conditionalPriors"}
                ]
            self._received_prior_fit_rows = training_bins.shape[1]
            self._received_prior_selection_epoch = prior_result.selection_epoch
            self._received_prior_selection_score = prior_result.selection_score
            sample_coordinates = sample_packet.indices()
            train_coordinates = sample_coordinates[train_positions]
            validation_coordinates = sample_coordinates[validation_positions]
            self._received_validation_counts = (train_positions.numel(), validation_positions.numel())
            self._received_validation_split_hash = content_hash((train_coordinates, validation_coordinates))
            self._sample_fields |= dict(
                sample_validation_train_rows=train_positions.numel(),
                sample_validation_rows=validation_positions.numel(),
                sample_validation_fraction=validation_positions.numel() / x_raw.numel(),
                sample_validation_seed=self.sample_validation_seed,
                sample_validation_split_hash=self._received_validation_split_hash,
            )

        self.quantizer.train_model = fit_received_prior  # type: ignore[method-assign]
        try:
            super().pre_encode_training_steps(delta_vec)
        finally:
            del self.quantizer.__dict__["train_model"]
        packet: SamplePayload = decompress_data_list(self._sample_payload)
        assert packet.format == "fp16" and self.quantizer.sample_indices is not None
        assert self._received_prior_fit_rows is not None
        if self.validated_split:
            assert isinstance(self.quantizer, _ReceivedFP16ValidatedSplitQuantizer)
            assert self._received_validation_counts is not None
            assert self._received_prior_fit_rows == self._received_validation_counts[0]
            assert self.quantizer.fit_pool_rows == self._received_validation_counts[0]
            assert self.quantizer.fit_draws_per_epoch == min(
                self.quantizer.c_cfg.train_sample_size, self._received_validation_counts[0])
        else:
            assert self._received_prior_fit_rows == packet.sample_size, (
                "The candidate prior must be retrained only on received FP16 sample rows.")
            assert isinstance(self.quantizer, _ReceivedFP16TrainingMessageQuantizer)
            assert self.quantizer.fit_pool_rows == packet.sample_size
            assert self.quantizer.fit_draws_per_epoch == self.quantizer.c_cfg.train_sample_size
        self._received_prior_state_hash = content_hash(
            self.quantizer.coding_model.copy_for_prior_retraining().state_dict())
        fit_config = self.quantizer.training_config()
        fit_draws = self.quantizer.fit_draws_per_epoch
        assert fit_draws is not None
        fit_batches = (fit_draws + fit_config.train_batch_size - 1) // fit_config.train_batch_size
        prior_draws = (min(self.quantizer.c_cfg.train_sample_size, self._received_prior_fit_rows)
                       if self.validated_split else self.quantizer.c_cfg.train_sample_size)
        prior_batches = (prior_draws + 50_000 - 1) // 50_000
        self._received_training_indices = self.quantizer.sample_indices.clone()
        self._received_training_values = self.quantizer.sample_reconstruction.clone()
        self.quantizer.sample_indices = None
        self.quantizer.sample_reconstruction = None
        self._ordinary_mask = None
        self._sample_fields |= dict(
            sample_training_planes=self.normal_planes,
            sample_fit_label_source="decoded_received_fp16_sample",
            sample_fit_label_available_to_receiver=True,
            sample_setup_deployable=True,
            sample_prior_refit_rows=self._received_prior_fit_rows,
            sample_prior_refit_target="received_fp16_training_message_only",
            sample_full_target_prior_refit_rows=0,
            sample_fit_training_pool_rows=self.quantizer.fit_pool_rows,
            sample_fit_draws_per_epoch=self.quantizer.fit_draws_per_epoch,
            sample_fit_batches_per_epoch=fit_batches,
            sample_fit_updates_per_accepted_candidate=fit_config.train_epochs * fit_batches,
            sample_fit_required_accepted_candidates=(fit_config.warm_retrain_repeats if fit_config.warm_training
                                                     else fit_config.quantizer_train_repeats),
            sample_fit_max_attempts=fit_config.max_training_attempts,
            sample_prior_draws_per_epoch=prior_draws,
            sample_prior_updates_per_candidate=fit_config.prior_retrain_epochs * prior_batches,
            sample_prior_required_candidates=fit_config.prior_train_repeats,
            sample_prior_state_hash_before_full_eval=self._received_prior_state_hash,
        )
        if self.validated_split:
            self._sample_fields |= dict(
                sample_fit_validation_best_epoch=self.quantizer.fit_selection_epoch,
                sample_fit_validation_best_score=self.quantizer.fit_selection_score,
                sample_fit_validation_metric="hard_symbol_normalized_mse_rd_plus_conditional_ce_nats",
                sample_prior_fit_context=("received_hard_symbols" if self.quantizer.c_cfg.prior_context == "hard"
                                          else "received_soft_codes"),
                sample_fit_validation_attempts=self._received_main_validation_attempts,
                sample_fit_validation_updates=(
                    self._received_main_validation_attempts * fit_config.train_epochs * fit_batches),
                sample_prior_validation_best_epoch=self._received_prior_selection_epoch,
                sample_prior_validation_best_score=self._received_prior_selection_score,
                sample_prior_validation_metric="hard_conditional_ce_nats",
                sample_prior_validation_attempts=self._received_prior_validation_attempts,
                sample_prior_validation_updates=(
                    self._received_prior_validation_attempts * fit_config.prior_retrain_epochs * prior_batches),
            )
            if self.trace_fit_rd:
                self._sample_fields |= dict(
                    sample_fit_prior_distribution=("categorical_softmax" if self.categorical_prior
                                                   else "gumbel_softmax"),
                    sample_fit_rd_trace_attempts=self._received_fit_rd_trace_attempts,
                    sample_fit_rd_trace_epochs=self._received_fit_rd_trace_epochs,
                    sample_fit_training_rd_trace=self._received_fit_training_rd_trace,
                    sample_fit_validation_rd_trace=self._received_fit_validation_rd_trace,
                )
            if self.heads_only:
                assert self._received_joint_trainable_parameter_names is not None
                assert self._received_prior_trainable_parameter_names is not None
                assert self._received_joint_trunk_state_hash_before is not None
                assert self._received_joint_trunk_state_hash_after is not None
                assert self._received_prior_trunk_state_hash_before is not None
                assert self._received_prior_trunk_state_hash_after is not None
                self._sample_fields |= dict(
                    sample_fit_trainable_parameter_names=self._received_joint_trainable_parameter_names,
                    sample_fit_trainable_parameter_count=sum(
                        parameter.numel() for name, parameter in self.quantizer.coding_model.named_parameters()
                        if name in self._received_joint_trainable_parameter_names
                    ),
                    sample_fit_trunk_state_hash_before=self._received_joint_trunk_state_hash_before,
                    sample_fit_trunk_state_hash_after=self._received_joint_trunk_state_hash_after,
                    sample_prior_heads_only=True,
                    sample_prior_trainable_parameter_names=self._received_prior_trainable_parameter_names,
                    sample_prior_trainable_parameter_count=sum(
                        parameter.numel() for name, parameter in self.quantizer.coding_model.named_parameters()
                        if name in self._received_prior_trainable_parameter_names
                    ),
                    sample_prior_trunk_state_hash_before=self._received_prior_trunk_state_hash_before,
                    sample_prior_trunk_state_hash_after=self._received_prior_trunk_state_hash_after,
                )
            if self.hard_transport_nll:
                self._sample_fields |= dict(
                    sample_fit_rate_objective="hard_transport_nll_with_expected_ce_encoder_gradient",
                    sample_fit_deterministic_encoder=True,
                )

    def fit_additional_priors(self, bins: torch.Tensor, soft_codes: torch.Tensor,
                              metadata: PreprocessMetadata, record: PlaneSweepRecord) -> None:
        """Measure the ordinary full-vector native prior without S's sampled-position mask."""
        super(_MeasuredSampleCodec, self).fit_additional_priors(bins, soft_codes, metadata, record)
        prior = PriorCalculator.compute_prior_from_network(
            self.quantizer.coding_model, bins, self.quantizer.side_info_tensor(metadata))
        rates = (PriorCalculator.compute_code_lengths(prior, bins, self.normal_planes).sum(dim=1)
                 / bins.shape[1]).tolist()
        record.server_si_prior_stage_rates = rates
        record.server_si_prior_rate = sum(rates)

    def get_encod_payload(self, delta_vec: torch.Tensor, record: PlaneSweepRecord) -> dict[str, Any]:
        """Attach the actual sample message and charge its complete packet once at stage zero."""
        assert not self._fine_tune_native_prior, "The received-sample candidate cannot refit priors on the full target."
        payload = super().get_encod_payload(delta_vec, record)
        assert self._sample_payload is not None
        prior_state_hash = content_hash(self.quantizer.coding_model.copy_for_prior_retraining().state_dict())
        assert prior_state_hash == self._received_prior_state_hash, (
            "Full-vector evaluation mutated the received-sample-trained prior.")
        record.sample_prior_state_hash_after_full_eval = prior_state_hash
        packet_rate = 8 * len(self._sample_payload) / delta_vec.numel()
        payload["sample_payload"] = self._sample_payload
        record.update_record_fields(
            sample_training_message_bytes=len(self._sample_payload),
            sample_training_message_rate=packet_rate,
        )
        record.sample_prior_rate = None
        record.sample_coder_rate = packet_rate
        for stages_name, total_name in (
            ("quantizer_prior_stage_rates", "quantizer_prior_rate"),
            ("server_si_prior_stage_rates", "server_si_prior_rate"),
            ("temporal_si_prior_stage_rates", "temporal_si_prior_rate"),
            ("server_si_rans_stage_rates", "server_si_rans_rate"),
            ("temporal_si_rans_stage_rates", "temporal_si_rans_rate"),
            ("coder_stage_rates", "coder_rate"),
        ):
            stages = getattr(record, stages_name, None)
            if stages is not None:
                adjusted = [rate + (packet_rate if stage == 0 else 0.0)
                            for stage, rate in enumerate(stages)]
                setattr(record, stages_name, adjusted)
                setattr(record, total_name, sum(adjusted))
        record.marginal_rate = (record.marginal_rate or 0.0) + packet_rate
        return payload

    def release_round_resources(self) -> None:
        """Drop retained fit-only labels after the training-message round completes."""
        super().release_round_resources()
        self._received_training_indices = None
        self._received_training_values = None
        self._received_prior_fit_rows = None
        self._received_prior_state_hash = None


class _ReceivedFP16ValidatedSplitCodec(_ReceivedFP16TrainingMessageCodec):
    """Use a receiver-only train/validation split for both quantizer and prior selection."""

    validated_split: ClassVar[bool] = True
    quantizer_class: ClassVar[type[_SampleQuantizer]] = _ReceivedFP16ValidatedSplitQuantizer


class _ReceivedFP16ValidatedHardInputsCodec(_ReceivedFP16ValidatedSplitCodec):
    """Fit validated received-sample labels with straight-through hard decoder/prior inputs."""

    trace_fit_rd: ClassVar[bool] = True


class _ReceivedFP16ValidatedHardCategoricalPriorCodec(_ReceivedFP16ValidatedHardInputsCodec):
    """Train hard-input received-message S with categorical prior probabilities in the joint objective."""

    categorical_prior: ClassVar[bool] = True


class _ReceivedFP16ValidatedHardTransportNLLCodec(_ReceivedFP16ValidatedHardCategoricalPriorCodec):
    """Fit deterministic encoder probabilities against the measured hard transport NLL."""

    hard_transport_nll: ClassVar[bool] = True


class _ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec(_ReceivedFP16ValidatedHardTransportNLLCodec):
    """Adapt only symbol, reconstruction and prior heads over fixed recurrent features."""

    heads_only: ClassVar[bool] = True


class _SOFitOnlyDiagnosticCodec(_MeasuredSampleCodec):
    """Isolate final-fit teacher corruption using receiver-unavailable raw labels."""

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Replace only final-fit labels after the real sample packet has been assembled."""
        assert self.sampling.sample_acquisition_distortion == "mae"
        assert self.sampling.sample_fit_mode == "received_prefix_bins"
        assert not self.sampling.sample_train_encoder and self.sampling.sample_final_distortion == "mse"
        original_train = self.quantizer.train_model

        def train_on_raw_sample_labels(x_raw: torch.Tensor, si_raw_list: Any, *args: Any,
                                       **kwargs: Any) -> None:
            assert self._sample_payload is not None, "The SO diagnostic must fit only after acquisition is serialized."
            packet: SamplePayload = decompress_data_list(self._sample_payload)
            assert packet.format == "lwzq"
            indices = packet.indices()
            assert x_raw.shape == indices.shape, "Final-fit teacher labels do not align with received sample coordinates."
            return original_train(delta_vec[indices], si_raw_list, *args, **kwargs)

        self.quantizer.train_model = train_on_raw_sample_labels  # type: ignore[method-assign]
        try:
            super().pre_encode_training_steps(delta_vec)
        finally:
            del self.quantizer.__dict__["train_model"]
        self._sample_fields |= dict(
            sample_fit_label_source="SO_fit_only_raw_current_target_at_sample_indices",
            sample_fit_label_available_to_receiver=False,
            sample_setup_deployable=False,
        )
def summarize(groups: dict[str, list[dict[str, Any]]], retain: int) -> dict[str, Any]:
    """Rank each method's seeds against the mean O curve, then average selected measurements."""
    individual = {(method, run["fit_seed"]): mean_points([run["record"]])
                  for method, group in groups.items() for run in group}
    valid = {key: point for key, point in individual.items()
             if np.all(np.diff(point["wmape_db"]) < 0) and np.all(np.diff(point["server_si_rans_stage_rates"]) > 0)}
    low = max(min(point["wmape_db"]) for point in valid.values()) if valid else float("nan")
    high = min(max(point["wmape_db"]) for point in valid.values()) if valid else float("nan")
    ranking_spans: dict[str, list[float] | None] = {}
    scores: dict[tuple[str, int], float] = {}
    oracle = mean_points([run["record"] for run in groups["O"]]) if groups.get("O") else None
    for method in groups:
        points = {key: point for key, point in valid.items() if key[0] == method}
        method_low = max(min(point["wmape_db"]) for point in points.values()) if points else float("nan")
        method_high = min(max(point["wmape_db"]) for point in points.values()) if points else float("nan")
        if oracle is not None and points:
            low = max(method_low, min(oracle["wmape_db"]))
            high = min(method_high, max(oracle["wmape_db"]))
            ranking_spans[method] = [low, high] if low < high else None
            if low < high:
                scores |= {key: gap for key, point in points.items()
                           if np.isfinite(gap := curve_gap(point, oracle, "server_si_rans_stage_rates"))}
        else:
            ranking_spans[method] = None
    selected = {method: sorted((run for run in group if (method, run["fit_seed"]) in scores),
                               key=lambda run: (scores[method, run["fit_seed"]], run["fit_seed"]))[:retain]
                for method, group in groups.items()}
    cohorts = {"untrimmed": {method: mean_points([run["record"] for run in group]) for method, group in groups.items()},
               "trimmed": {method: mean_points([run["record"] for run in group])
                           for method, group in selected.items() if len(group) == retain}}
    selection = [dict(method=method, seed=run["fit_seed"], gap_vs_all_o_percent=scores.get((method, run["fit_seed"])),
                      retained=run in selected[method], reason=("unrankable stage curve or no shared span with all-O mean"
                      if (method, run["fit_seed"]) not in scores else "retained" if run in selected[method] else "worst RD seed"))
                 for method, group in groups.items() for run in group]
    return dict(cohorts=cohorts, common_distortion_db=[low, high] if low < high else None,
                ranking_distortion_db=ranking_spans, selection=selection)


def publish(output_root: Path, destination: Path, *, complete: bool = False) -> None:
    """Publish complete directed matrices, sample overhead, selections and plots as results arrive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((output_root / "test_config.json").read_text())
    plan = PlaneSweepPlan.model_validate(manifest["plan"])
    groups: dict[str, list[dict[str, Any]]] = {method: [] for method in plan.methods}
    for path in sorted((output_root / "cases").glob("*/*/complete.json")):
        run = json.loads(path.read_text())
        assert run["identity"] == manifest["identity"], f"Identity mismatch: {path}"
        assert run["record"]["symbol_error_rate"] == 0, f"Symbol errors: {path}"
        groups[run["method"]].append(run)
    if plan.budget_sample_sizes is not None and not plan.include_budget_control:
        assert plan.comparison_root is not None
        comparison_manifest = json.loads((plan.comparison_root / "test_config.json").read_text())
        assert comparison_manifest["reference_manifest"]["target_hash"] == manifest["reference_manifest"]["target_hash"], (
            "Read-only O comparison target differs from the sample-message candidate.")
        assert comparison_manifest["baseline_state_hash"] == manifest["baseline_state_hash"], (
            "Read-only O comparison starting model differs from the sample-message candidate.")
        baseline_path, = (plan.reference_root / "baseline").glob("*.pt")
        baseline = TorchCodec().load(baseline_path)
        groups["O"] = []
        for seed in plan.seeds:
            path = plan.comparison_root / "cases" / f"O_seed{seed}" / "target" / "complete.json"
            run = json.loads(path.read_text())
            assert run["method"] == "O" and run["fit_seed"] == seed and run["record"]["symbol_error_rate"] == 0, (
                f"Invalid read-only O control: {path}")
            assert run["identity"] == comparison_manifest["identity"], f"Read-only O identity mismatch: {path}"
            weights_path = path.with_name("weights.pt")
            weights = TorchCodec().load(weights_path)
            projected_baseline = {key: baseline["state"][key] for key in weights["native"]}
            assert run["initial_state_hash"] == content_hash(projected_baseline), (
                f"Read-only O projected starting model differs: {path}")
            groups["O"].append(run)
    coverage = {method: sorted(run["fit_seed"] for run in group) for method, group in groups.items()}
    if complete:
        assert all(seeds == sorted(plan.seeds) for seeds in coverage.values()), f"Incomplete sweep: {coverage}"
    groups = {method: group for method, group in groups.items() if group}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "coverage.json").write_text(json.dumps(coverage, indent=2))
    if not groups:
        (destination / "report.md").write_text("Waiting for the first completed fit.\n")
        return
    result = summarize(groups, plan.retain_seeds)
    finished = all(seeds == sorted(plan.seeds) for seeds in coverage.values())
    variant_definitions = []
    for mode in plan.variant_modes or ():
        _, distortion = _PLANE_VARIANT_OPTIONS.get(mode, (plan.sample_refiner_residual_anchor,
                                                           plan.sample_acquisition_distortion))
        fit_mode, train_encoder = _PLANE_VARIANT_FIT.get(mode, (plan.sample_fit_mode, plan.sample_train_encoder))
        variant_definitions.append(
            f"{mode} = acquisition {distortion}, final S {fit_mode}, "
            f"sample encoder {'trainable' if train_encoder else 'frozen'}, "
            f"final distortion {plan.sample_final_distortion if fit_mode == 'received_prefix_bins' else 'mse'}")
    variant_definitions_text = "; ".join(variant_definitions)
    comparison_note = (f"O is imported read-only from {plan.comparison_root}; its target and starting-model hashes match this candidate."
                       if plan.budget_sample_sizes is not None and not plan.include_budget_control else "")
    lines = [f"# Practical S {'dedicated-refiner' if plan.sample_refiner else 'extra-plane'} sweep versus O", "", f"{'COMPLETE' if finished else 'INCOMPLETE'}: {coverage}", "", comparison_note,
             "Fixed SVHN client 0, round 5; trace/starting seed 43; fit/sample seeds vary. One trained round, no frozen window.",
             "All methods use the same retained starting weights and 150 decoder-visible SI columns. O uses the full current target; S fits decoded LWZQ samples only.",
             ((f"S uses per-case budget overrides (global default {plan.sample_size:,}), 3 ordinary planes plus the declared extra planes, "
               if plan.budget_sample_sizes is not None else
               f"S uses {plan.sample_size:,} positions, 3 ordinary planes plus the declared extra planes, ")
              + f"{plan.sample_fit_epochs} epochs, LR {plan.sample_fit_lr}, tau {plan.sample_fit_tau}, batch {plan.sample_fit_batch_size}."),
             "Sample symbols are rANS compressed with the retained model before fitting S; all acquisition bytes are charged from stage 1. Normalization and sample coordinates are transmitted metadata.",
             (f"Budget variants use {', '.join(f'{size:,}' for size in plan.budget_sample_sizes)} sampled positions. Each size uses the configured coordinate seed with an independent no-replacement draw; these subsets are not guaranteed to be nested."
              if plan.budget_sample_sizes is not None else ""),
             "Model distribution and prior availability retain O's existing assumption. Native transport is full-SI rANS for both O and S.",
             ("Primary pilot comparison is each S curve against the fresh paired O/seed43 curve, using full-vector full-SI rANS rates with all sample acquisition and metadata cost included. Sample WMAPE is a diagnostic only."
              if plan.seeds == (43,) and "O" in plan.methods else
              "This run contains only new S fits; compare against the explicitly named read-only reference records in the postprocessed study report. Full-vector full-SI rANS remains primary and sample WMAPE is diagnostic."
              if plan.seeds == (43,) else
              "Primary comparison is the aggregate full-vector full-SI rANS rate-distortion curve; all sample acquisition and metadata costs are included. Sample WMAPE is a diagnostic only."),
             (f"The starting three-plane prefix stays fixed and deterministic during acquisition-refiner training. Only the independent secondary network is fitted on the same baseline gradient used to train the starting model, with acquisition settings {plan.refiner_fit_epochs} epochs, LR {plan.refiner_fit_lr}, tau {plan.refiner_fit_tau}, lr_step {plan.refiner_fit_lr_step}, batch {plan.refiner_fit_batch_size}, and hard decoder inputs. Existing candidate/repeat selection remains unchanged; these settings do not imply independent cold restarts. S then fits only three ordinary planes on decoded samples using the distinct final-S settings above. Refiner fitting never uses the current raw target."
              if plan.sample_refiner else
              "S trains all ordinary plus extra planes, then deploys the three-plane prefix. Sample refinement therefore changes both acquired labels and the fit objective."),
             (f"Variant-mode pilot: {variant_definitions_text}. Normalized MAE uses normalized absolute error scaled by 2; raw_mae weights normalized error with per-slice source scales and the source-target raw mean absolute value."
              if plan.variant_modes is not None else
              f"Acquisition refiner settings: residual_anchor={plan.sample_refiner_residual_anchor}, distortion={plan.sample_acquisition_distortion}. MAE is normalized absolute error scaled by 2, not the exact raw WMAPE objective."),
             (("Received-FP16 validated-split S fits with straight-through hard decoder/prior inputs and categorical-softmax prior probabilities in its joint objective, using the deterministic 80% received-sample partition; checkpoints use heldout hard-symbol RD/conditional CE in nats on the remaining 20%. Per-epoch train/heldout hard-RD traces are diagnostic; heldout score alone selects checkpoints. It codes all target coordinates. The full compressed message packet is charged once at stage 1; encoder feedback is reported separately as downlink cost, and decoder weights are not counted as downlink. Prior weights are frozen during full-target evaluation."
               if plan.variant_modes == ("received_fp16_training_message_validated_split_hard_categorical_prior",) else
               "S fits deterministic encoder probabilities with a hard-symbol conditional NLL forward objective and expected-CE encoder gradient; decoder/prior inputs remain straight-through hard symbols. It uses only the deterministic 80% received FP16 training partition and heldout hard-symbol RD/conditional CE for checkpoint selection. It codes all target coordinates, charges the message packet once, and freezes the sample-fit prior during full-target evaluation."
               if plan.variant_modes == ("received_fp16_training_message_validated_split_hard_transport_nll",) else
               "The hard-transport NLL candidate is head-only: encoder, decoder and conditional-RNN trunks stay frozen while symbol, reconstruction and conditional-prior heads adapt. Native-prior fine-tuning also trains only conditional-prior heads. It retains the same deterministic 80/20 split, hard RD selection, packet accounting and all-coordinate coding."
               if plan.variant_modes == ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",) else
               "Received-FP16 validated-split S fits with straight-through hard decoder/prior inputs on the deterministic 80% received-sample partition and selects checkpoints by heldout hard-symbol RD/conditional CE in nats on the remaining 20%. Per-epoch train/heldout hard-RD traces are diagnostic; heldout score alone selects checkpoints. It codes all target coordinates. The full compressed message packet is charged once at stage 1; encoder feedback is reported separately as downlink cost, and decoder weights are not counted as downlink. Prior weights are frozen during full-target evaluation."
               if plan.variant_modes == ("received_fp16_training_message_validated_split_hard_inputs",) else
               "Received-FP16 validated-split S fits the encoder/decoder and prior only on the deterministic 80% received-sample partition, selecting checkpoints by heldout hard-symbol RD/conditional CE in nats on the remaining 20%. It codes all target coordinates. The full compressed message packet is charged once at stage 1; encoder feedback is reported separately as downlink cost, and decoder weights are not counted as downlink. Prior weights are frozen during full-target evaluation."
               if plan.variant_modes == ("received_fp16_training_message_validated_split",) else
               "Received-FP16 training-message S uses decoded packet values for the full quantizer fit and received-sample bins/SI for the prior fit, then codes all target coordinates. The full compressed message packet is charged once at stage 1; encoder feedback is reported separately as downlink cost, and decoder weights are not counted as downlink. Prior weights are frozen during full-target evaluation.")
              if plan.variant_modes in (("received_fp16_training_message",),
                                        ("received_fp16_training_message_validated_split",),
                                        ("received_fp16_training_message_validated_split_hard_inputs",),
                                        ("received_fp16_training_message_validated_split_hard_categorical_prior",),
                                        ("received_fp16_training_message_validated_split_hard_transport_nll",),
                                        ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",)) else ""),
             ("The reencode_teacher frozen control is a fit-path ablation: its teacher bins are regenerated from decoded teacher values and need not match deployment bins, while evaluation uses the unchanged deployed encoder."
              if plan.variant_modes and "current_e1_mse_reencode_teacher_frozen_control" in plan.variant_modes else ""),
             "Quantizer and conditional-prior fitting are separate; fitted priors use the ordinary warm schedule. No frozen-prior refit is measured.",
             f"Trim independently to the best {plan.retain_seeds}/{len(plan.seeds)} seeds per method by full-SI rANS equal-distortion gap against the mean O curve from all available O seeds; lower is better. The completed five-seed study uses all five O seeds as the reference.",
             f"Per-method ranking intervals (WMAPE dB): {result['ranking_distortion_db']}.",
             "Ranking intervals show each method's shared distortion span with the all-O mean curve; individual gaps use each pair's shared span. Aggregate matrix entries use each pair's shared distortion interval.",
             "Unrankable curves are flagged; insufficient valid curves leave a trimmed method unavailable. Missing comparisons are —, without extrapolation.",
             "Client exclusions: none; with/without symmetric client exclusions identical, 1/1 retained. Fit-seed trimming is separate selection per method against the all-O mean, not a matched-seed population estimate.",
             "Average cumulative rates and raw WMAPE before conversion to dB and curve comparisons. Every reverse matrix entry is recomputed; negative favors the row.", "",
             "| Method | Seed | Full-SI gap vs all-O mean (%) | Selected | Reason |", "|---|---:|---:|---|---|"]
    lines += [f"| {row['method']} | {row['seed']} | {row['gap_vs_all_o_percent'] if row['gap_vs_all_o_percent'] is not None else '—'} | {row['retained']} | {row['reason']} |"
              for row in result["selection"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    fields = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
              "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    for axis, (cohort, points) in zip(axes, result["cohorts"].items(), strict=True):
        lines += ["", f"## {cohort}", "", "Full-SI rANS (primary):", "", "```text",
                  matrix_text(points, "server_si_rans_stage_rates"), "```", "", "Native transport:", "", "```text",
                  matrix_text(points, "coder_stage_rates"), "```", "",
                  "| Method | Stage | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for method, point in points.items():
            axis.plot(point["server_si_rans_stage_rates"], point["wmape_db"], "o-", label=method)
            for stage in range(3):
                values = [point["marginal_rate"] if stage == 2 else None,
                          *(point[field][stage] if point[field] is not None else None for field in fields)]
                lines.append(f"| {method} | {stage + 1} | " + " | ".join("—" if value is None else f"{value:.6f}" for value in values)
                             + f" | {point['wmape_db'][stage]:.6f} ({point['stage_wmapes'][stage]:.6f}%) |")
        axis.set(title=cohort + (" (partial)" if not finished else ""), xlabel="Full-SI rANS (bits/float)", ylabel="WMAPE dB")
        if points:
            axis.legend()
    lines += ["", "## Acquisition and stage diagnostics", "",
              "Sample rates include all acquisition planes and are already included in stage rates. Extra-plane stream growth and included stream overhead are shown separately.", "",
              "The acquisition table reports every hard-decoded LWZQ stage before the replacement S fit. Occupancy and empirical entropy use the transmitted hard symbols; conditional prior cost is converted to bits per sampled symbol.", "",
              "| Method | Seed | Stage | Raw WMAPE % | Occupied bins | Empirical entropy | H(stage | prefix, earlier extras) | Conditional prior | Flag |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for method, group in groups.items():
        for run in group:
            record = run["record"]
            stage_wmapes = record.get("sample_stage_wmape")
            occupancies = record.get("sample_stage_occupied_bins")
            entropies = record.get("sample_stage_marginal_entropy_bps")
            prior_rates = record.get("sample_stage_prior_rate")
            conditional_entropies = record.get("sample_extra_prefix_conditional_entropy_bps")
            if not all(values is not None for values in (stage_wmapes, occupancies, entropies, prior_rates)):
                lines.append(f"| {method} | {run['fit_seed']} | — | — | — | — | — | — | acquisition diagnostics unavailable |")
                continue
            assert len({len(stage_wmapes), len(occupancies), len(entropies), len(prior_rates)}) == 1
            sample_count, vector_size = record["sample_count"], record["model_size"]
            for stage, (wmape, occupied, entropy, prior_rate) in enumerate(
                    zip(stage_wmapes, occupancies, entropies, prior_rates, strict=True), start=1):
                flags = "dead stage: one occupied bin" if occupied == 1 else "low entropy" if entropy < .1 else "none"
                prior_per_symbol = prior_rate * vector_size / sample_count
                conditional = None if stage < 4 or conditional_entropies is None else conditional_entropies[stage - 4]
                lines.append(f"| {method} | {run['fit_seed']} | {stage} | {wmape:.6f} | {occupied} | {entropy:.6f} | "
                             f"{'—' if conditional is None else f'{conditional:.6f}'} | {prior_per_symbol:.6f} | {flags} |")
    lines += ["", "### Final-extra-stage counterfactuals", "",
              "These ablations affect only the final extra-symbol stage, so they do not isolate the aggregate contribution of all extra stages. The prefix-preserving shuffle permutes that row only within exact discrete prefix and earlier-extra contexts, retaining within-context counts but breaking alignment with continuous SI. The prior-mode counterfactual replaces that final transmitted symbol with the fitted decoder prior's argmax conditioned on SI and earlier symbols. It measures reliance on the final transmitted symbol while also changing decoder inputs; neither is an RD-optimality claim.", "",
              "| Method | Seed | Actual sample WMAPE % | Final-stage prefix-conditioned shuffle % | Final-stage prior-mode % |", "|---|---:|---:|---:|---:|"]
    for method, group in groups.items():
        for run in group:
            record = run["record"]
            actual = record.get("sample_wmape")
            shuffle = record.get("sample_extra_final_prefix_shuffle_wmape")
            mode = record.get("sample_extra_final_prior_mode_wmape")
            lines.append(f"| {method} | {run['fit_seed']} | "
                         + " | ".join("—" if value is None else f"{value:.6f}" for value in (actual, shuffle, mode))
                         + " |")
    lines += ["", "### Acquisition and final-fit summary", "",
              "The acquisition measurements above describe the old model's sample stream. The RD curves and stage WMAPE below describe the newly fitted three-plane S codec.", "",
              "| Method | Seed | Final acquisition WMAPE % | Sample prior b/full-gradient-float | Sample rANS b/full-gradient-float | Extra rANS b/full-gradient-float | Sample overhead b/full-gradient-float | Final S curve flags |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for method, group in groups.items():
        for run in group:
            record = run["record"]
            flags = []
            if np.any(np.diff(record["stage_wmapes"]) >= 0):
                flags.append("nonmonotone final S distortion")
            if record["quantizer_prior_rate"] > record["marginal_rate"]:
                flags.append("joint prior exceeds marginal")
            flags += [f"stage {stage} stream growth {rate:.6f} bpf" for stage, rate in enumerate(record["coder_stage_rates"], 1)
                      if rate < .01 or rate > 1.05]
            values = [record.get(field) for field in ("sample_wmape", "sample_prior_rate", "sample_coder_rate",
                                                     "sample_extra_coder_rate", "sample_coder_overhead_rate")]
            lines.append(f"| {method} | {run['fit_seed']} | " + " | ".join("—" if value is None else f"{value:.6f}" for value in values)
                         + f" | {', '.join(flags) or 'none; zero symbol errors'} |")
    lines += ["", "## Matched fit-seed comparisons", "",
              "These individual gaps are diagnostics; aggregate comparisons use mean RD curves above.", "",
              "| Variant | Seed | Full-SI rate change versus same-seed O % |", "|---|---:|---:|"]
    controls = {run["fit_seed"]: run for run in groups.get("O", [])}
    for method, group in groups.items():
        if method == "O":
            continue
        for run in group:
            if run["fit_seed"] in controls:
                value = curve_gap(mean_points([run["record"]]), mean_points([controls[run["fit_seed"]]["record"]]), "server_si_rans_stage_rates")
                lines.append(f"| {method} | {run['fit_seed']} | {value:+.6f} |" if np.isfinite(value)
                             else f"| {method} | {run['fit_seed']} | — |")
    figure.tight_layout()
    figure.savefig(destination / "rd.png", dpi=160)
    figure.savefig(destination / "rd.pdf")
    plt.close(figure)
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    (destination / "summary.json").write_text(json.dumps(result, indent=2))
    (destination / "raw_records.json").write_text(json.dumps(groups, indent=2))


def run(plan: PlaneSweepPlan) -> None:
    """Fit real sample-acquisition variants sequentially, retaining payloads, weights and final reports."""
    reference = json.loads((plan.reference_root / "test_config.json").read_text())
    problem_path, = (plan.reference_root / "problem").glob("*.pkl")
    baseline_path, = (plan.reference_root / "baseline").glob("*.pt")
    problem, baseline = PickleCodec().load(problem_path), TorchCodec().load(baseline_path)
    assert content_hash(problem["target"]) == reference["target_hash"]
    assert content_hash(tuple(content_hash(value.materialize()) for value in problem["si"])) == reference["si_hash"]
    torch.cuda.set_device(torch.device(plan.fit_device))
    config = WZcfgQuant.model_validate(reference["plan"]["quantizer"]).model_copy(update={"num_planes": 3, "warm_training": True})
    assert plan.sample_refiner or 3 + max(plan.extra_planes) <= reference["plan"]["quantizer"]["num_planes"], "Retained model lacks requested sample planes."
    if plan.variant_modes in (("received_fp16_training_message",),
                              ("received_fp16_training_message_validated_split",),
                              ("received_fp16_training_message_validated_split_hard_inputs",),
                              ("received_fp16_training_message_validated_split_hard_categorical_prior",),
                              ("received_fp16_training_message_validated_split_hard_transport_nll",),
                              ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",)):
        assert (config.train_sample_size, config.train_batch_size, config.warm_retrain_repeats,
                config.max_training_attempts, config.prior_retrain_epochs, config.prior_train_repeats) == (
                    550_000, 6_250, 2, 8, 30, 3,
                ), "The reference O fit and prior budgets do not match the received-message candidate."
    root = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(root)): path.read_bytes() for directory in (root / "FL_code", root / "Experiments/Protocol_test")
               for path in sorted(directory.rglob("*.py"))}
    match = dict(plan=plan.model_dump(mode="json"), reference_manifest=reference,
                 baseline_state_hash=content_hash(baseline["state"]), sources=content_hash(sources),
                 semantics=("Received FP16 sample labels train a full-vector S model and sample-trained prior; all N current symbols are transmitted; complete sample packet charged once at stage 0"
                            if plan.variant_modes in (("received_fp16_training_message",),
                                                      ("received_fp16_training_message_validated_split",),
                                                      ("received_fp16_training_message_validated_split_hard_inputs",),
                                                      ("received_fp16_training_message_validated_split_hard_categorical_prior",),
                                                      ("received_fp16_training_message_validated_split_hard_transport_nll",),
                                                      ("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",)) else
                            "SR decoded LWZQ samples; preceding retained model; acquisition at stage 1; separate per-method RD seed trimming"),
                 refiner_training=(f"frozen deterministic three-plane baseline prefix; independent refiner fit on original baseline target using {plan.refiner_fit_epochs}/{plan.refiner_fit_lr}/{plan.refiner_fit_tau}/{plan.refiner_fit_lr_step}/{plan.refiner_fit_batch_size}; existing candidate/repeat selection; final-S fit {plan.sample_fit_epochs}/{plan.sample_fit_lr}/{plan.sample_fit_tau}/{plan.sample_fit_batch_size}"
                                   if plan.sample_refiner else None),
                 numerical_runtime=config.numerical_policy.runtime(torch.device(plan.fit_device), mixed_precision=config.mixed_precision))
    identity = content_hash(match)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    manifest = plan.output_root / "test_config.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["identity"] == identity, f"Identity mismatch: {manifest}"
    else:
        with atomic_output(manifest) as temporary:
            temporary.write_text(json.dumps(dict(identity=identity, **match), indent=2))
        with tarfile.open(plan.output_root / "sources.tar.gz", "w:gz") as archive:
            for name, data in sources.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    sample_pools: dict[int, tuple[int, str]] = {}
    for seed in plan.seeds:
        for method in plan.methods:
            directory = plan.output_root / "cases" / f"{method}_seed{seed}" / "target"
            completion = directory / "complete.json"
            if completion.exists():
                saved = json.loads(completion.read_text())
                assert saved["identity"] == identity, f"Identity mismatch: {completion}"
                if method != "O":
                    pool = (saved["sample_coordinate_seed"], saved["sample_indices_hash"])
                    previous_pool = sample_pools.setdefault(seed, pool)
                    assert previous_pool[0] == pool[0] and (plan.budget_sample_sizes is not None or previous_pool[1] == pool[1]), (
                        f"Unmatched sample coordinates: {completion}")
                continue
            options = config.model_dump(exclude={"norm_slices"}) | {"sd_slices": problem["slices"], "rans_lanes": 2048}
            if method != "O":
                if "_n" in method:
                    base_method, budget = method.rsplit("_n", 1)
                else:
                    base_method, budget = method, str(plan.sample_size)
                options |= {key: getattr(plan, key) for key in ("sample_fit_epochs", "sample_fit_lr", "sample_fit_tau", "sample_fit_batch_size")}
                options["sample_size"] = int(budget)
                residual_anchor, acquisition_distortion = _PLANE_VARIANT_OPTIONS.get(
                    base_method, (plan.sample_refiner_residual_anchor, plan.sample_acquisition_distortion))
                sample_fit_mode, sample_train_encoder = _PLANE_VARIANT_FIT.get(
                    base_method, (plan.sample_fit_mode, plan.sample_train_encoder))
                training_message = base_method.startswith("received_fp16_training_message")
                extra_planes = (0 if training_message else
                                1 if base_method in _PLANE_VARIANT_OPTIONS else int(base_method.removeprefix("S_e")))
                options |= dict(sample_extra_planes=extra_planes,
                                sample_format="fp16" if training_message else "lwzq", sample_rans_lanes=2048,
                                sample_refiner_residual_anchor=residual_anchor,
                                sample_acquisition_distortion=acquisition_distortion,
                                sample_final_distortion=plan.sample_final_distortion,
                                sample_fit_mode=sample_fit_mode, sample_train_encoder=sample_train_encoder)
                if training_message:
                    options |= dict(sample_refiner=False, sample_training="all", sample_noise_multiplier=0.0,
                                    sample_coordinate_seed=plan.training_message_coordinate_seed,
                                    sample_hard_inputs=base_method.startswith(
                                        "received_fp16_training_message_validated_split_hard_"),
                                    sample_train_backbones=(base_method !=
                                        "received_fp16_training_message_validated_split_hard_transport_nll_heads_only"),
                                    max_training_attempts=8,
                                    sample_fit_epochs=plan.sample_fit_epochs, sample_fit_lr=plan.sample_fit_lr,
                                    sample_fit_tau=plan.sample_fit_tau, sample_fit_batch_size=plan.sample_fit_batch_size)
                elif plan.sample_refiner:
                    options |= dict(sample_refiner=True, sample_training="prefix",
                                    sample_coordinate_seed=int(torch.randint(0, 2**31, (), generator=torch.Generator().manual_seed(seed)).item()))
            sample_codec = (_SOFitOnlyDiagnosticCodec if method.endswith("SO_fit_only_diagnostic") else
                            _ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec
                            if method.startswith("received_fp16_training_message_validated_split_hard_transport_nll_heads_only") else
                            _ReceivedFP16ValidatedHardTransportNLLCodec
                            if method.startswith("received_fp16_training_message_validated_split_hard_transport_nll") else
                            _ReceivedFP16ValidatedHardCategoricalPriorCodec
                            if method.startswith("received_fp16_training_message_validated_split_hard_categorical_prior") else
                            _ReceivedFP16ValidatedHardInputsCodec
                            if method.startswith("received_fp16_training_message_validated_split_hard_inputs") else
                            _ReceivedFP16ValidatedSplitCodec
                            if method.startswith("received_fp16_training_message_validated_split") else
                            _ReceivedFP16TrainingMessageCodec if method.startswith("received_fp16_training_message") else
                            _MeasuredSampleCodec)
            codec = _MeasuredOracleCodec(options, "OR") if method == "O" else sample_codec(options, "SR")
            if isinstance(codec, _ReceivedFP16ValidatedSplitCodec):
                codec.sample_validation_fraction = plan.sample_validation_fraction
                codec.sample_validation_seed = plan.sample_validation_seed
            set_global_seed(seed)
            codec.build_quantizer(None, list(problem["si"]))
            refiner_fields: dict[str, Any] = {}
            if method != "O":
                if plan.sample_refiner:
                    # The acquisition refiner keeps its existing teacher re-encode fit path; the selected mode only changes final S fitting.
                    previous = _MeasuredSampleCodec(options | {"sample_fit_mode": "reencode_teacher",
                                                                "sample_train_encoder": True,
                                                                "sample_final_distortion": "mse"}, "SR")
                    set_global_seed(seed)
                    previous.build_quantizer(None, list(problem["si"]))
                    acquisition = previous.quantizer
                    state = {key: (value.detach().cpu().clone() if key.startswith("refiner.") else baseline["state"][key])
                             for key, value in acquisition.coding_model.state_dict().items()}
                    acquisition.load_warm_start_state(state)
                    if acquisition.refinement_planes:
                        acquisition.fit_refiner_only = True
                        acquisition.fit_hard_inputs = True
                        base_reconst_ld = acquisition.c_cfg.reconst_ld
                        effective_reconst_ld = base_reconst_ld * plan.refiner_fit_reconst_multiplier
                        assert np.isfinite(effective_reconst_ld), "Scaled acquisition reconst_ld must remain finite."
                        acquisition.c_cfg = acquisition.c_cfg.model_copy(
                            update={"lr_step": plan.refiner_fit_lr_step, "reconst_ld": effective_reconst_ld})
                        for field, option in (("fit_epochs", "refiner_fit_epochs"), ("fit_lr", "refiner_fit_lr"),
                                              ("fit_tau", "refiner_fit_tau"), ("fit_batch_size", "refiner_fit_batch_size"),
                                              ):
                            setattr(acquisition, field, getattr(plan, option))
                        acquisition.fit_distortion = acquisition_distortion
                        assert acquisition.fit_distortion == acquisition_distortion
                        refiner_fields |= dict(
                            refiner_fit_reconst_multiplier=plan.refiner_fit_reconst_multiplier,
                            refiner_fit_reconst_ld_base=base_reconst_ld,
                            refiner_fit_reconst_ld_effective=effective_reconst_ld,
                        )
                        print(f"REFINER STARTED {method} seed={seed}", flush=True)
                        started = perf_counter()
                        acquisition.train_model(problem["baseline"], list(problem["si"]))
                        torch.cuda.synchronize()
                        refiner_fields["refiner_fit_seconds"] = perf_counter() - started
                        refined_state = acquisition.coding_model.state_dict()
                        assert all(torch.equal(value, refined_state[key]) for key, value in state.items()
                                   if not key.startswith("refiner.")), "Acquisition refiner changed the fixed prefix."
                        state = {key: value.detach().cpu().clone() for key, value in refined_state.items()}
                        refiner_fields |= dict(refiner_state_hash=content_hash(state), base_prefix_unchanged=True,
                                               refiner_training_target_hash=content_hash(problem["baseline"]))
                        print(f"REFINER COMPLETED {method} seed={seed} seconds={refiner_fields['refiner_fit_seconds']:.1f}", flush=True)
                        acquisition.side_info_list_used = None
                        previous._side_info_columns_cache.clear()
                else:
                    previous = _MeasuredOracleCodec(config.model_dump(exclude={"norm_slices"}) | {
                        "sd_slices": problem["slices"], "num_planes": reference["plan"]["quantizer"]["num_planes"],
                        "warm_training": False}, "OR")
                    previous.build_quantizer(None, list(problem["si"]))
                    previous.quantizer.coding_model.load_state_dict(baseline["state"])
                    state = projected_state(codec, baseline["state"], None)
                codec.set_previous_codec(previous)
                del previous
            else:
                state = projected_state(codec, baseline["state"], None)
            codec.quantizer.load_warm_start_state(state)
            record = codec.create_r_record(reference["plan"]["target_round"], reference["plan"]["target_client"])
            record.model_size = problem["target"].numel()
            set_global_seed(seed)
            print(f"STARTED {method} seed={seed}", flush=True)
            started = perf_counter()
            codec.pre_encode_training_steps(problem["target"])
            if method.startswith("received_fp16_training_message_validated_split_hard_"):
                assert codec.sampling.sample_hard_inputs and codec.quantizer.fit_hard_inputs
            torch.cuda.synchronize()
            if method != "O":
                assert codec.quantizer.fit_final_distortion == plan.sample_final_distortion
                assert codec.quantizer.c_cfg.reconst_ld == config.reconst_ld
            record.pre_encode_training_seconds = perf_counter() - started
            started = perf_counter()
            payload = codec.encode(problem["target"], record)
            torch.cuda.synchronize()
            record.encode_seconds = perf_counter() - started
            started = perf_counter()
            reconstruction = codec.decode(payload, record)
            torch.cuda.synchronize()
            record.decode_seconds = perf_counter() - started
            record.total_seconds = record.pre_encode_training_seconds + record.encode_seconds + record.decode_seconds
            record.final_bytes = payload_size_mib(payload)
            record.update_record_fields(**record_reconstruction_metrics(problem["target"], reconstruction))
            assert record.symbol_error_rate == 0, f"Symbol errors in {method}/seed{seed}"
            assert np.isclose(sum(record.server_si_prior_stage_rates), record.server_si_prior_rate, rtol=2e-5)
            assert codec.joint_quantizer_state is not None
            native = codec.quantizer.coding_model.state_dict()
            assert all(torch.equal(value, native[key]) for key, value in codec.joint_quantizer_state.items()
                       if not key.startswith("conditional")), "Prior fitting changed quantizer weights."
            weights = dict(joint=codec.joint_quantizer_state, native=native)
            if plan.sample_refiner and method != "O":
                weights["acquisition"] = state
            sample_fields: dict[str, Any] = {}
            if method != "O":
                packet: SamplePayload = decompress_data_list(codec._sample_payload)
                indices = packet.indices()
                if method.startswith("received_fp16_training_message"):
                    assert packet.format == "fp16" and not record.sample_bootstrap
                    assert isinstance(codec, _ReceivedFP16TrainingMessageCodec)
                    assert torch.equal(indices, codec._received_training_indices), (
                        f"Received training-message coordinates differ: {method}/seed{seed}")
                    assert codec.quantizer.sample_indices is None and codec.quantizer.sample_reconstruction is None
                    training_values = codec._received_training_values
                else:
                    assert packet.format == "lwzq" and not record.sample_bootstrap
                    assert torch.equal(indices, codec.quantizer.sample_indices), (
                        f"Sample payload coordinates differ: {method}/seed{seed}")
                    training_values = codec.quantizer.sample_reconstruction
                # Pilot construction consumes RNG draws before acquisition; compare the actual payload-owned pool.
                pool = (packet.seed, content_hash(indices))
                previous_pool = sample_pools.setdefault(seed, pool)
                assert previous_pool[0] == pool[0] and (plan.budget_sample_sizes is not None or previous_pool[1] == pool[1]), (
                    f"Unmatched sample coordinates: {method}/seed{seed}")
                sample_fields = dict(sample_coordinate_seed=pool[0], sample_indices_hash=pool[1],
                                     sample_payload_hash=content_hash(codec._sample_payload))
                if method.startswith("received_fp16_training_message"):
                    weights |= dict(sample_training_indices=indices, sample_training_values=training_values)
                else:
                    weights |= dict(sample_indices=indices, sample_reconstruction=training_values)
            directory.mkdir(parents=True, exist_ok=True)
            with atomic_output(directory / "weights.pt") as temporary:
                torch.save(weights, temporary)
            with atomic_output(directory / "payload.bin") as temporary:
                temporary.write_bytes(payload)
            record.append_record_to_csv(directory)
            result = dict(identity=identity, method=method, fit_seed=seed, initial_state_hash=content_hash(state),
                          acquisition_state_hash=(content_hash(state) if plan.sample_refiner else match["baseline_state_hash"])
                          if method != "O" else None,
                          refiner_fit_distortion=(acquisition_distortion if plan.sample_refiner and method != "O" else None),
                          sample_fit_reconst_ld=(codec.quantizer.c_cfg.reconst_ld if method != "O" else None),
                          sample_final_distortion=(plan.sample_final_distortion if method != "O" else None),
                          sample_refiner_residual_anchor=(residual_anchor if method != "O" else None),
                          sample_acquisition_distortion=(acquisition_distortion if method != "O" else None),
                          record=record.to_dict(), **sample_fields, **refiner_fields)
            with atomic_output(completion) as temporary:
                temporary.write_text(json.dumps(result, indent=2))
            print(f"COMPLETED {method} seed={seed} WMAPE={record.wmape:.6f}% rANS={record.coder_rate:.9f} seconds={record.total_seconds:.1f}", flush=True)
            codec.release_round_resources()
            del codec, weights, native, state, payload, reconstruction, record
            gc.collect()
            torch.cuda.empty_cache()
    publish(plan.output_root, plan.output_root / "final_analysis", complete=True)


def main() -> None:
    """Run a fresh plan or serve disposable plot refreshes for its detached supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.runner_pid is not None:
        assert args.control_dir is not None
        watch(args.path, args.runner_pid, args.control_dir, plot_module=__spec__.name,
              source_paths=(Path(__file__), Path(__file__).with_name("five_seed_tables.py"), Path(__file__).with_name("starting_model_tables.py")))
    elif args.out is not None:
        publish(args.path, args.out)
    else:
        run(load_plan(args.path))


if __name__ == "__main__":
    main()
