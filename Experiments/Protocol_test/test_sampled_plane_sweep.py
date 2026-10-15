"""Check charged acquisition and independent O/S seed trimming in the practical plane study."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from Experiments.Protocol_test.sampled_plane_sweep import (
    _MeasuredSampleCodec, _ReceivedFP16TrainingMessageCodec, _ReceivedFP16ValidatedSplitCodec,
    _ReceivedFP16ValidatedHardInputsCodec,
    _ReceivedFP16ValidatedHardCategoricalPriorCodec,
    _ReceivedFP16ValidatedHardTransportNLLCodec,
    _ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec,
    _SOFitOnlyDiagnosticCodec,
    PlaneSweepPlan, PlaneSweepRecord, _split_received_sample_rows, publish, run, summarize,
)
from Experiments.Protocol_test.starting_model_comparison import _MeasuredOracleCodec
from Experiments.Protocol_test.test_sampled_codec import FitInputs, fits, make_codec
from Experiments.Protocol_test.sampled_codec import RefinedEncoderDecoderLayeredRNN, SamplePayload, SamplingConfig
from FL_code.FL_core.codec import Access
from FL_code.FL_core.utils import decompress_data_list
from FL_code.FL_core.explicit_cache import PickleCodec, content_hash
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer, new_rnn_model
import FL_code.cancer_protocol.wz_quantizer as wz_quantizer_module


def test_variant_mode_plan_has_explicit_six_case_catalog() -> None:
    """The fresh acquisition-objective pilot exposes one O, one S0, and four S1 fits."""
    plan = PlaneSweepPlan(
        name="pilot", output_root=Path("out"), reference_root=Path("reference"),
        seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
        variant_modes=("current_e1_mse", "current_e1_mae", "residual_e1_mse", "residual_e1_mae"),
    )
    assert plan.methods == (
        "O", "S_e0", "current_e1_mse", "current_e1_mae", "residual_e1_mse", "residual_e1_mae",
    )
    assert plan.cases == tuple(f"{method}_seed43" for method in plan.methods)
    assert PlaneSweepPlan(name="legacy", output_root=Path("out"), reference_root=Path("reference")).methods == (
        "O", "S_e0", "S_e1", "S_e2", "S_e3", "S_e6",
    )


@pytest.mark.parametrize("updates", [
    dict(sample_refiner=False, variant_modes=("current_e1_mse",)),
    dict(sample_refiner=True, extra_planes=(0, 1), variant_modes=("current_e1_mse",)),
    dict(sample_refiner=True, extra_planes=(0,), variant_modes=("current_e1_mse", "current_e1_mse")),
    dict(sample_refiner=False, sample_refiner_residual_anchor=True),
    dict(sample_refiner=False, sample_acquisition_distortion="mae"),
    dict(sample_refiner=False, sample_acquisition_distortion="raw_mae"),
    dict(sample_refiner=True, extra_planes=(0,), sample_refiner_residual_anchor=True),
    dict(sample_refiner=True, extra_planes=(0,), sample_acquisition_distortion="mae"),
])
def test_invalid_variant_mode_plan_fails_structurally(updates: dict[str, Any]) -> None:
    """A variant sweep cannot silently omit its residual-refiner or control semantics."""
    with pytest.raises(ValidationError):
        PlaneSweepPlan(name="invalid", output_root=Path("out"), reference_root=Path("reference"), **updates)


@pytest.mark.parametrize("value", [0.0, -0.5, float("inf"), float("nan")])
def test_refiner_reconstruction_multiplier_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValidationError):
        PlaneSweepPlan(name="invalid", output_root=Path("out"), reference_root=Path("reference"),
                       refiner_fit_reconst_multiplier=value)


def test_budget_plan_expands_one_variant_without_requiring_s0(tmp_path: Path) -> None:
    """Budget pilot cases use explicit per-case sample counts and keep the full O control."""
    plan = PlaneSweepPlan(name="budget", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
                          seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                          variant_modes=("current_e1_mse",), budget_sample_sizes=(1_800_000, 2_400_000))

    assert plan.methods == ("O", "current_e1_mse_n1800000", "current_e1_mse_n2400000")
    assert plan.cases == ("O_seed43", "current_e1_mse_n1800000_seed43", "current_e1_mse_n2400000_seed43")


def test_received_fp16_training_message_plan_uses_explicit_budgets(tmp_path: Path) -> None:
    """The full-fit sample message is a distinct no-refiner variant with declared sample counts."""
    plan = PlaneSweepPlan(
        name="received-fp16", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message",), include_variant_controls=False,
        budget_sample_sizes=(20_000, 100_000),
    )
    assert plan.methods == ("O", "received_fp16_training_message_n20000", "received_fp16_training_message_n100000")


def test_received_fp16_validated_split_plan_fixes_receiver_only_selection(tmp_path: Path) -> None:
    """The bounded follow-up uses a fixed receiver-only 80/20 split and fresh budget cases."""
    plan = PlaneSweepPlan(
        name="received-fp16-validated", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message_validated_split",), include_variant_controls=False,
        budget_sample_sizes=(20_000, 100_000), include_budget_control=False,
        comparison_root=tmp_path / "read-only-o",
    )
    assert plan.sample_validation_fraction == 0.2
    assert plan.sample_validation_seed == 494_155_589
    assert plan.methods == (
        "received_fp16_training_message_validated_split_n20000",
        "received_fp16_training_message_validated_split_n100000",
    )


def test_received_fp16_validated_hard_inputs_plan_is_an_explicit_sibling(tmp_path: Path) -> None:
    """Hard-input fitting is a separate validated-split mode with the same sample budgets."""
    plan = PlaneSweepPlan(
        name="received-fp16-hard", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message_validated_split_hard_inputs",),
        include_variant_controls=False, budget_sample_sizes=(20_000, 100_000), include_budget_control=False,
        comparison_root=tmp_path / "read-only-o",
    )
    assert plan.methods == (
        "received_fp16_training_message_validated_split_hard_inputs_n20000",
        "received_fp16_training_message_validated_split_hard_inputs_n100000",
    )


def test_received_fp16_validated_categorical_prior_plan_is_explicit(tmp_path: Path) -> None:
    """The categorical prior fit is a separate hard-input sibling with the same budgets."""
    plan = PlaneSweepPlan(
        name="received-fp16-categorical", output_root=tmp_path / "output",
        reference_root=tmp_path / "reference", seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message_validated_split_hard_categorical_prior",),
        include_variant_controls=False, budget_sample_sizes=(20_000, 100_000), include_budget_control=False,
        comparison_root=tmp_path / "read-only-o",
    )
    assert plan.methods == (
        "received_fp16_training_message_validated_split_hard_categorical_prior_n20000",
        "received_fp16_training_message_validated_split_hard_categorical_prior_n100000",
    )


def test_received_fp16_hard_transport_nll_plan_is_explicit(tmp_path: Path) -> None:
    """The exact hard-transport objective is a separate two-size validated candidate."""
    plan = PlaneSweepPlan(
        name="received-fp16-hard-transport", output_root=tmp_path / "output",
        reference_root=tmp_path / "reference", seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message_validated_split_hard_transport_nll",),
        include_variant_controls=False, budget_sample_sizes=(20_000, 100_000), include_budget_control=False,
        comparison_root=tmp_path / "read-only-o",
    )
    assert plan.methods == (
        "received_fp16_training_message_validated_split_hard_transport_nll_n20000",
        "received_fp16_training_message_validated_split_hard_transport_nll_n100000",
    )


def test_received_fp16_hard_transport_nll_heads_only_plan_is_explicit(tmp_path: Path) -> None:
    """The capacity control keeps the hard-transport objective and fixed two-budget sample plan."""
    plan = PlaneSweepPlan(
        name="received-fp16-heads-only", output_root=tmp_path / "output",
        reference_root=tmp_path / "reference", seeds=(43,), extra_planes=(0,), retain_seeds=1,
        sample_fit_epochs=35, sample_fit_lr=0.0008509229529398079,
        sample_fit_tau=0.5, sample_fit_batch_size=6_250,
        variant_modes=("received_fp16_training_message_validated_split_hard_transport_nll_heads_only",),
        include_variant_controls=False, budget_sample_sizes=(20_000, 100_000), include_budget_control=False,
        comparison_root=tmp_path / "read-only-o",
    )
    assert plan.methods == (
        "received_fp16_training_message_validated_split_hard_transport_nll_heads_only_n20000",
        "received_fp16_training_message_validated_split_hard_transport_nll_heads_only_n100000",
    )


def test_received_sample_split_keeps_singletons_in_train_and_allows_empty_validation_slices() -> None:
    """A singleton stratum stays train-only while packet factors make empty validation slices safe."""
    train, validation, train_slices, validation_slices = _split_received_sample_rows(
        (slice(0, 1), slice(1, 11)), 0.2, 494_155_589)
    assert train.tolist() == sorted(train.tolist())
    assert 0 in train and not bool((validation == 0).any())
    assert train.numel() == 9 and validation.numel() == 2
    assert train_slices == (slice(0, 1), slice(1, 9))
    assert validation_slices == (slice(0, 0), slice(0, 2))

    codec, _, _ = make_codec(_ReceivedFP16TrainingMessageCodec, samples=13, extra=0)
    quantizer = codec.quantizer
    quantizer.c_cfg = quantizer.c_cfg.model_copy(update={"norm_slices": validation_slices})
    quantizer.vector_size = validation.numel()
    packet_factors = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
    quantizer.normalization_factors = packet_factors
    prepared, metadata = quantizer.preprocess_x(torch.tensor([3.0, 5.0]))
    side_info = quantizer.normalize_side_info_tensor(metadata, [HistoryVector(torch.tensor([2.0, 4.0]))])
    assert prepared.isfinite().all() and side_info[:].isfinite().all()
    torch.testing.assert_close(metadata.norm_factors, packet_factors)


@pytest.mark.parametrize(("codec_type", "hard_inputs", "heads_only"), [
    (_ReceivedFP16ValidatedSplitCodec, False, False), (_ReceivedFP16ValidatedHardInputsCodec, True, False),
    (_ReceivedFP16ValidatedHardCategoricalPriorCodec, True, False),
    (_ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec, True, True),
])
def test_validated_fp16_fit_uses_packet_factors_and_excludes_heldout_rows(
    fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
    codec_type: type[_ReceivedFP16ValidatedSplitCodec], hard_inputs: bool, heads_only: bool,
) -> None:
    """Validated fitting keeps train/validation rows disjoint and matches prior contexts to the mode."""
    original_fit = WZQuantizerCancer._fit_prepared
    main_scores: list[float] = []
    prior_scores: list[float] = []
    prior_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def capture_fit(quantizer: WZQuantizerCancer, x_prep: torch.Tensor, side_info: Any) -> None:
        if heads_only:
            quantizer._set_training_parameters(quantizer.coding_model)
        original_fit(quantizer, x_prep, side_info)
        if quantizer.selection_metric is not None:
            model = quantizer.coding_model
            was_training = model.training
            model.to(quantizer.device).eval()
            with torch.inference_mode():
                score = quantizer.selection_metric(model, -1)
            model.train(was_training)
            quantizer.fit_selection_epoch = -1
            quantizer.fit_selection_score = score
            main_scores.append(score)

    def fit_prior(
        bins: torch.Tensor, contexts: torch.Tensor, side_info: Any, _config: WZcfgQuant,
        model: Any, _states: tuple[Any, ...], _repeats: int, *, selection_metric: Any, **options: Any,
    ) -> Any:
        assert options == ({"prior_heads_only": True} if heads_only else {})
        prior_rows.append((bins.clone(), contexts.clone(), side_info[:].clone()))
        was_training = model.training
        model.to(side_info.device).eval()
        with torch.inference_mode():
            score = selection_metric(model, -1)
        model.train(was_training)
        model.to("cpu")
        prior_scores.append(score)
        return SimpleNamespace(selection_epoch=-1, selection_score=score)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture_fit)
    monkeypatch.setattr(PriorCalculator, "fit_prior_model", staticmethod(fit_prior))
    codec, target, _ = make_codec(codec_type, samples=13, extra=0)
    norm_slices = (slice(0, 3), slice(3, 64))
    codec.c_cfg = codec.c_cfg.model_copy(update={"norm_slices": norm_slices, "warm_training": True})
    codec.quantizer.c_cfg = codec.c_cfg
    codec.quantizer.load_warm_start_state(codec.quantizer.coding_model.state_dict())
    codec.sampling = SamplingConfig(
        sample_size=13, sample_extra_planes=0, sample_format="fp16", sample_training="all",
        sample_fit_epochs=35, sample_coordinate_seed=19, sample_hard_inputs=hard_inputs,
        sample_train_backbones=not heads_only,
    )
    if heads_only:
        codec._configure_sample_fit()
    codec.quantizer.fit_epochs = 35
    codec.pre_encode_training_steps(target)

    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    decoded = decompress_data_list(packet.symbols).float().flatten()
    train_rows, validation_rows, _, _ = _split_received_sample_rows(
        (slice(0, 1), slice(1, 13)), codec.sample_validation_fraction, codec.sample_validation_seed)
    assert train_rows.numel() == 11 and validation_rows.numel() == 2
    assert len(fits) == 2 and fits[-1].target.shape[0] == train_rows.numel()
    torch.testing.assert_close(fits[-1].target.flatten(), decoded[train_rows], rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(fits[-1].factors, packet.metadata.norm_factors)
    assert len(prior_rows) == 1
    assert prior_rows[0][0].shape[1] == prior_rows[0][2].shape[0] == train_rows.numel()
    if hard_inputs:
        expected_hard_contexts = torch.nn.functional.one_hot(
            prior_rows[0][0].T.long(), codec.quantizer.bins_per_plane).permute(1, 0, 2).float()
        torch.testing.assert_close(prior_rows[0][1], expected_hard_contexts)
    assert codec._sample_fields["sample_prior_fit_context"] == (
        "received_hard_symbols" if codec.quantizer.c_cfg.prior_context == "hard" else "received_soft_codes")
    if heads_only:
        assert codec._sample_fields["sample_training_backbones"] is False
        assert codec._sample_fields["sample_fit_trunk_state_hash_before"] == (
            codec._sample_fields["sample_fit_trunk_state_hash_after"])
        assert codec._sample_fields["sample_prior_trunk_state_hash_before"] == (
            codec._sample_fields["sample_prior_trunk_state_hash_after"])
        assert all(set(name.split(".")) & {"binner", "binners", "reconstructor", "reconstructors",
                                            "conditionalPrior", "conditionalPriors"}
                   for name in codec._sample_fields["sample_fit_trainable_parameter_names"])
        assert all(set(name.split(".")) & {"conditionalPrior", "conditionalPriors"}
                   for name in codec._sample_fields["sample_prior_trainable_parameter_names"])
        assert codec._sample_fields["sample_prior_heads_only"] is True
    assert codec.quantizer.fit_force_prior_softmax == codec.categorical_prior
    if codec.trace_fit_rd:
        assert codec._sample_fields["sample_fit_prior_distribution"] == (
            "categorical_softmax" if codec.categorical_prior else "gumbel_softmax")
    assert len(main_scores) == len(prior_scores) == 1
    assert np.isfinite(main_scores + prior_scores).all()
    assert codec._received_validation_counts == (train_rows.numel(), validation_rows.numel())
    assert codec._sample_fields["sample_validation_fraction"] == validation_rows.numel() / packet.sample_size
    assert codec._sample_fields["sample_fit_validation_best_epoch"] == -1
    assert codec._sample_fields["sample_prior_validation_best_epoch"] == -1


@pytest.mark.parametrize(("codec_type", "categorical_prior"), [
    (_ReceivedFP16ValidatedHardInputsCodec, False),
    (_ReceivedFP16ValidatedHardCategoricalPriorCodec, True),
    (_ReceivedFP16ValidatedHardTransportNLLCodec, True),
])
def test_sample_hard_inputs_use_straight_through_symbols_and_restore_model_flag(
    codec_type: type[_ReceivedFP16ValidatedHardInputsCodec], categorical_prior: bool,
) -> None:
    """Hard-input fitting preserves encoder gradients and opts into categorical p only by mode."""
    codec, _, _ = make_codec(codec_type, samples=13, extra=0)
    codec.sampling = codec.sampling.model_copy(update={"sample_hard_inputs": True})
    codec._configure_sample_fit()
    quantizer = codec.quantizer
    quantizer.fit_force_prior_softmax = codec.categorical_prior
    quantizer.fit_hard_transport_nll = codec.hard_transport_nll
    model = quantizer.coding_model
    model.train()
    decoder_inputs: list[torch.Tensor] = []
    prior_inputs: list[torch.Tensor] = []
    prior_softmax_flags: list[bool] = []
    decoder_hook = model.decoder.register_forward_pre_hook(lambda _module, inputs: decoder_inputs.append(inputs[0]))
    prior_hook = model.conditionalRNN.register_forward_pre_hook(lambda _module, inputs: prior_inputs.append(inputs[0]))
    original_get_priors = model.get_priors

    def capture_get_priors(codes: Any, y: torch.Tensor | None = None, tau: float = 1.0,
                           force_softmax: bool = False) -> list[torch.Tensor]:
        prior_softmax_flags.append(force_softmax)
        return original_get_priors(codes, y, tau, force_softmax)

    model.get_priors = capture_get_priors  # type: ignore[method-assign]
    try:
        loss = quantizer.compute_loss(
            model, (torch.randn(19, 1),), torch.randn(19, 2), 0,
            quantizer.training_config(), None, 1.0, torch.Generator().manual_seed(43),
        )
        loss.backward()
    finally:
        decoder_hook.remove()
        prior_hook.remove()
    assert quantizer.fit_hard_inputs and not model.hard_decoder_inputs
    assert prior_softmax_flags == [categorical_prior]
    assert decoder_inputs and prior_inputs
    symbols = decoder_inputs[0][..., :model.bins_per_plane]
    torch.testing.assert_close(symbols.sum(dim=-1), torch.ones_like(symbols[..., 0]))
    assert torch.isin(symbols, torch.tensor([0.0, 1.0])).all()
    encoder_grad = model.encoder.input_layers[0].weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0
    prior_symbols = prior_inputs[0][:, 1:, -model.bins_per_plane:]
    torch.testing.assert_close(prior_symbols.sum(dim=-1), torch.ones_like(prior_symbols[..., 0]))
    assert torch.isin(prior_symbols, torch.tensor([0.0, 1.0])).all()


def test_heads_only_joint_fit_changes_heads_and_preserves_every_recurrent_trunk() -> None:
    """The opt-in low-capacity joint adapter updates only bin, reconstruction and prior heads."""
    codec, _, _ = make_codec(_ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec, samples=13, extra=0)
    codec.sampling = codec.sampling.model_copy(update={
        "sample_hard_inputs": True, "sample_train_backbones": False,
    })
    codec._configure_sample_fit()
    quantizer = codec.quantizer
    quantizer.fit_force_prior_softmax = True
    quantizer.fit_hard_transport_nll = True
    model = quantizer.coding_model.train()
    quantizer._set_training_parameters(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all(set(name.split(".")) & {"binner", "binners", "reconstructor", "reconstructors",
                                        "conditionalPrior", "conditionalPriors"} for name in trainable_names)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()
              if name.startswith(("encoder.", "decoder.", "conditionalRNN."))}
    head_before = {name: value.detach().clone() for name, value in model.state_dict().items()
                   if name in trainable_names}
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.03)
    inputs, side_info = torch.randn(25, 1), torch.randn(25, 2)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = quantizer.compute_loss(
            model, (inputs,), side_info, 0, quantizer.training_config(), None, 1.0,
            torch.Generator().manual_seed(43),
        )
        loss.backward()
        optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert any(not torch.equal(value, after[name]) for name, value in head_before.items())


def test_native_prior_heads_only_updates_heads_without_changing_conditional_rnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in prior lineage trains only conditional output heads, leaving its feature trunk fixed."""
    codec, _, _ = make_codec(_ReceivedFP16ValidatedHardTransportNLLHeadsOnlyCodec, samples=13, extra=0)
    model = codec.quantizer.coding_model
    config = codec.quantizer.c_cfg.model_copy(update={"prior_retrain_epochs": 3, "prior_train_repeats": 1})
    bins = torch.randint(config.bins_per_plane, (config.num_planes, 16))
    contexts = torch.nn.functional.one_hot(bins.T, config.bins_per_plane).permute(1, 0, 2).float()
    side_info = torch.randn(16, 2)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimized_names: list[str] = []

    def train_heads_only(compute_loss: Any, candidate: Any, parameters: Any, training_inputs: Any,
                         rows: torch.Tensor, cfg: WZcfgQuant, _batch_size: int, **_kwargs: Any) -> float:
        params = tuple(parameters)
        parameter_ids = {id(parameter) for parameter in params}
        optimized_names.extend(name for name, parameter in candidate.named_parameters()
                               if id(parameter) in parameter_ids)
        optimizer = torch.optim.SGD(params, lr=0.2)
        candidate.train()
        for epoch in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(candidate, training_inputs, rows, epoch, cfg, None)
            loss.backward()
            optimizer.step()
        return float(loss.detach())

    monkeypatch.setattr(wz_quantizer_module, "wz_model_training_loop", train_heads_only)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", classmethod(lambda *_args: (0.1, 1.0)))
    result = PriorCalculator.fit_prior_model(
        bins, contexts, side_info, config, model, (None,), 1, batch_size=8, prior_heads_only=True,
    )
    assert optimized_names and all(
        set(name.split(".")) & {"conditionalPrior", "conditionalPriors"} for name in optimized_names
    )
    after = result.model.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in original.items()
               if name.startswith("conditionalRNN."))
    assert any(not torch.equal(value, after[name]) for name, value in original.items()
               if set(name.split(".")) & {"conditionalPrior", "conditionalPriors"})


def test_hard_transport_nll_has_exact_hard_forward_value_and_st_encoder_gradient() -> None:
    """The fit scalar is measured hard-symbol RD while its encoder receives expected-CE gradients."""
    codec, _, _ = make_codec(_ReceivedFP16ValidatedHardTransportNLLCodec, samples=13, extra=0)
    quantizer = codec.quantizer
    quantizer.fit_hard_inputs = True
    quantizer.fit_force_prior_softmax = True
    quantizer.fit_hard_transport_nll = True
    model = quantizer.coding_model
    model.train()
    inputs, side_info = torch.randn(19, 1), torch.randn(19, 2)
    config = quantizer.training_config().model_copy(update={"source_perturbation": "none", "reconst_ld": 72.0})
    actual = quantizer.compute_loss(
        model, (inputs,), side_info, 0, config, None, 1.0, torch.Generator().manual_seed(43),
    )
    actual.backward()
    assert model.encoder.input_layers[0].weight.grad is not None
    assert model.encoder.input_layers[0].weight.grad.abs().sum() > 0

    model.hard_decoder_inputs = True
    try:
        with torch.no_grad():
            reconstruction, bins, _, priors = model(
                inputs, side_info, tau=config.tau, force_prior_softmax=True, deterministic_encoder=True)
            expected = sum(
                config.reconst_ld * torch.nn.functional.mse_loss(reconstruction[plane], inputs)
                - torch.log(priors[plane][torch.arange(inputs.shape[0]), bins[plane].long()] + 1e-12).mean()
                for plane in range(model.num_planes)
            ) / model.num_planes
    finally:
        model.hard_decoder_inputs = False
    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=1e-6)
    assert not model.hard_decoder_inputs

    model.zero_grad(set_to_none=True)
    zero_distortion_config = config.model_copy(update={"reconst_ld": 0.0})
    rate_only = quantizer.compute_loss(
        model, (inputs,), side_info, 0, zero_distortion_config, None, 1.0,
        torch.Generator().manual_seed(43),
    )
    rate_only.backward()
    encoder_gradient = model.encoder.input_layers[0].weight.grad
    prior_gradients = [parameter.grad for name, parameter in model.named_parameters()
                       if name.startswith(("conditionalRNN.", "conditionalPrior", "conditionalPriors."))]
    assert encoder_gradient is not None and encoder_gradient.abs().sum() > 0
    assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in prior_gradients)


def test_deterministic_encoder_is_opt_in_and_matches_softmax_tau() -> None:
    """The new encoder mode is deterministic and leaves default Gumbel draws unchanged."""
    codec, _, _ = make_codec(_ReceivedFP16ValidatedHardTransportNLLCodec, samples=13, extra=0)
    model = codec.quantizer.coding_model
    model.train()
    inputs, side_info = torch.randn(19, 1), torch.randn(19, 2)
    torch.manual_seed(43)
    default_result = model(inputs, side_info, tau=0.5)
    torch.manual_seed(43)
    explicit_default = model(inputs, side_info, tau=0.5, deterministic_encoder=False)
    for default_part, explicit_part in zip(default_result, explicit_default, strict=True):
        for default_tensor, explicit_tensor in zip(default_part, explicit_part, strict=True):
            torch.testing.assert_close(default_tensor, explicit_tensor, rtol=0, atol=0)

    _, deterministic_codes = model.encode(inputs, tau=0.5, deterministic_encoder=True)
    rnn_out = model.encoder(inputs.unsqueeze(1).repeat(1, model.num_planes, 1))
    for plane, code in enumerate(deterministic_codes):
        logits = model.binner(rnn_out[:, plane]) if model.shared_encoder else model.binners[plane](rnn_out[:, plane])
        torch.testing.assert_close(code, torch.softmax(logits / 0.5, dim=-1))


def test_prior_softmax_forward_option_preserves_default_and_is_deterministic_for_fixed_context() -> None:
    """Prior sampling is unchanged by default and softmax p is deterministic at fixed hard context."""
    codec, _, _ = make_codec(_ReceivedFP16ValidatedHardCategoricalPriorCodec, samples=13, extra=0)
    model = codec.quantizer.coding_model
    model.train()
    values, side_info = torch.randn(19, 1), torch.randn(19, 2)
    torch.manual_seed(43)
    default_result = model(values, side_info, tau=0.5)
    torch.manual_seed(43)
    explicit_default = model(values, side_info, tau=0.5, force_prior_softmax=False)
    for default_part, explicit_part in zip(default_result, explicit_default, strict=True):
        for default_tensor, explicit_tensor in zip(default_part, explicit_part, strict=True):
            torch.testing.assert_close(default_tensor, explicit_tensor, rtol=0, atol=0)

    fixed_codes = [torch.nn.functional.one_hot(
        torch.randint(model.bins_per_plane, (19,)), model.bins_per_plane).float()
        for _ in range(model.num_planes)]
    torch.manual_seed(2)
    categorical_a = model.get_priors(fixed_codes, side_info, force_softmax=True)
    torch.manual_seed(99)
    categorical_b = model.get_priors(fixed_codes, side_info, force_softmax=True)
    for left, right in zip(categorical_a, categorical_b, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.manual_seed(2)
    gumbel_a = model.get_priors(fixed_codes, side_info, force_softmax=False)
    torch.manual_seed(99)
    gumbel_b = model.get_priors(fixed_codes, side_info, force_softmax=False)
    assert any(not torch.allclose(left, right) for left, right in zip(gumbel_a, gumbel_b, strict=True))


def test_received_prefix_plan_names_fit_modes_without_rerunning_references(tmp_path: Path) -> None:
    """The fixed-bin study runs only its two objectives and one matched frozen reencode control."""
    modes = ("current_e1_mse_received_prefix_bins", "current_e1_mae_received_prefix_bins",
             "current_e1_mse_reencode_teacher_frozen_control")
    plan = PlaneSweepPlan(name="received", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
                          seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                          sample_train_encoder=False, variant_modes=modes, include_variant_controls=False)

    assert plan.methods == modes
    assert plan.cases == tuple(f"{method}_seed43" for method in modes)


def test_raw_mae_plan_pairs_one_received_prefix_variant(tmp_path: Path) -> None:
    """The raw-domain acquisition pilot keeps the received-prefix final-S fit frozen."""
    mode = "current_e1_raw_mae_received_prefix_bins"
    plan = PlaneSweepPlan(
        name="raw-mae", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43, 44, 45), extra_planes=(0,), retain_seeds=3, sample_refiner=True,
        sample_train_encoder=False,
        variant_modes=(mode,), include_variant_controls=False,
    )

    assert plan.methods == (mode,)
    assert plan.cases == tuple(f"{mode}_seed{seed}" for seed in plan.seeds)


def test_final_raw_mae_plan_pairs_normalized_mae_acquisition(tmp_path: Path) -> None:
    """The final-loss ablation keeps normalized-MAE acquisition and its received prefix fixed."""
    mode = "current_e1_mae_received_prefix_bins_final_raw_mae"
    plan = PlaneSweepPlan(
        name="final-raw-mae", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43, 44, 45), extra_planes=(0,), retain_seeds=3, sample_refiner=True,
        sample_train_encoder=False, sample_final_distortion="raw_mae",
        variant_modes=(mode,), include_variant_controls=False,
    )

    assert plan.methods == (mode,)
    assert plan.cases == tuple(f"{mode}_seed{seed}" for seed in plan.seeds)


def test_final_normalized_mae_plan_pairs_received_prefix_fit(tmp_path: Path) -> None:
    """Final normalized MAE is confined to the current frozen received-prefix S candidate."""
    mode = "current_e1_mae_received_prefix_bins"
    plan = PlaneSweepPlan(
        name="final-mae", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
        sample_train_encoder=False, sample_final_distortion="mae",
        variant_modes=(mode,), include_variant_controls=False,
    )

    assert plan.methods == (mode,)
    assert plan.sample_final_distortion == "mae"


def test_so_fit_only_diagnostic_plan_is_explicit_and_nondeployable(tmp_path: Path) -> None:
    """The causal oracle control is fixed to the matching acquisition and frozen-bin setup."""
    mode = "current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic"
    plan = PlaneSweepPlan(
        name="so-fit-only", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
        seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
        sample_train_encoder=False, sample_acquisition_distortion="mae",
        variant_modes=(mode,), include_variant_controls=False,
    )

    assert plan.methods == (mode,)
    assert plan.sample_final_distortion == "mse"


@pytest.mark.parametrize("updates", [
    dict(sample_train_encoder=True),
    dict(sample_final_distortion="mae"),
])
def test_so_fit_only_diagnostic_rejects_mismatched_setup(tmp_path: Path, updates: dict[str, Any]) -> None:
    """The oracle label swap cannot accidentally alter acquisition, encoder or final loss."""
    mode = "current_e1_mae_received_prefix_bins_SO_fit_only_diagnostic"
    with pytest.raises(ValidationError):
        PlaneSweepPlan.model_validate({
            "name": "invalid-so-fit-only", "output_root": tmp_path / "output", "reference_root": tmp_path / "reference",
            "seeds": (43,), "extra_planes": (0,), "retain_seeds": 1, "sample_refiner": True,
            "sample_train_encoder": False, "sample_acquisition_distortion": "mae",
            "variant_modes": (mode,), "include_variant_controls": False,
            **updates,
        })


def test_so_fit_only_diagnostic_changes_only_final_fit_labels(fits: list[FitInputs]) -> None:
    """Raw labels feed final fitting while the transmitted teacher reconstruction remains unchanged."""
    codec, target, _ = make_codec(_SOFitOnlyDiagnosticCodec, samples=13, extra=1, refiner=True)
    codec.sampling = codec.sampling.model_copy(update={
        "sample_acquisition_distortion": "mae", "sample_fit_mode": "received_prefix_bins",
        "sample_final_distortion": "mse", "sample_train_encoder": False,
    })
    codec.quantizer.fit_epochs = 1
    codec.quantizer.fit_batch_size = 8
    codec.pre_encode_training_steps(target)

    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    indices = packet.indices()
    expected = torch.empty_like(target[indices])
    for index, part in enumerate(codec.quantizer.norm_slices):
        mask = (indices >= part.start) & (indices < part.stop)
        scale, center = packet.metadata.norm_factors[index]
        expected[mask] = (target[indices[mask]] - center) / scale
    assert torch.equal(fits[-1].target, expected.reshape(-1, 1).cpu())
    assert not torch.equal(codec.quantizer.sample_reconstruction, target[packet.indices()])
    assert codec._sample_fields["sample_fit_label_source"] == "SO_fit_only_raw_current_target_at_sample_indices"
    assert codec._sample_fields["sample_fit_label_available_to_receiver"] is False
    assert codec._sample_fields["sample_setup_deployable"] is False


def test_received_fp16_message_trains_all_planes_and_codes_all_coordinates(
    fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only decoded packet values train S; the returned machine codes every coordinate without restoration."""
    prior_fits: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fit_prior(bins: torch.Tensor, _soft_codes: torch.Tensor, side_info: Any, *_args: Any,
                  **_kwargs: Any) -> None:
        prior_fits.append((bins.clone(), side_info[:].clone()))

    monkeypatch.setattr(PriorCalculator, "fit_prior_model", fit_prior)
    codec, target, _ = make_codec(_ReceivedFP16TrainingMessageCodec, samples=13, extra=0)
    codec.c_cfg = codec.c_cfg.model_copy(update={
        "warm_training": True, "warm_retrain_repeats": 2, "max_training_attempts": 8,
    })
    codec.quantizer.c_cfg = codec.c_cfg
    codec.quantizer.load_warm_start_state(codec.quantizer.coding_model.state_dict())
    codec.sampling = SamplingConfig(sample_size=13, sample_extra_planes=0, sample_format="fp16",
                                    sample_training="all", sample_fit_epochs=35,
                                    sample_coordinate_seed=19)
    codec.quantizer.fit_epochs = 35
    codec.pre_encode_training_steps(target)

    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    received_labels = decompress_data_list(packet.symbols).float().reshape(-1, 1)
    assert len(fits) == 2 and fits[-1].planes == codec.normal_planes == 3
    repeated_rows = torch.arange(550_000) % packet.sample_size
    torch.testing.assert_close(fits[-1].target, received_labels[repeated_rows], rtol=1e-5, atol=2e-7)
    assert len(prior_fits) == 1 and prior_fits[0][0].shape[1] == prior_fits[0][1].shape[0] == 550_000
    torch.testing.assert_close(prior_fits[0][0], prior_fits[0][0][:, :13][:, repeated_rows], rtol=0, atol=0)
    torch.testing.assert_close(prior_fits[0][1], fits[-1].side_info[:13][repeated_rows].to(prior_fits[0][1].device),
                               rtol=0, atol=0)
    assert torch.equal(codec._received_training_indices, packet.indices())
    assert codec.quantizer.sample_indices is None and codec.quantizer.sample_reconstruction is None
    assert codec._ordinary_mask is None

    record = codec.create_r_record(5, 0)
    payload = codec.encode(target, record)
    transmitted = decompress_data_list(payload)
    assert transmitted["sample_payload"] == codec._sample_payload
    assert record.sample_fit_label_source == "decoded_received_fp16_sample"
    assert record.sample_fit_label_available_to_receiver and record.sample_setup_deployable
    assert record.sample_prior_rate is None
    assert record.sample_training_message_bytes == len(codec._sample_payload)
    assert record.sample_training_message_rate == pytest.approx(8 * len(codec._sample_payload) / target.numel())
    assert record.sample_coder_rate == record.sample_training_message_rate
    assert record.sample_prior_refit_target == "received_fp16_training_message_only"
    assert record.sample_prior_refit_rows == 13 and record.sample_full_target_prior_refit_rows == 0
    assert record.sample_fit_training_pool_rows == 13
    assert record.sample_fit_draws_per_epoch == 550_000 and record.sample_fit_batches_per_epoch == 88
    assert record.sample_fit_updates_per_accepted_candidate == 3_080
    assert record.sample_fit_required_accepted_candidates == 2 and record.sample_fit_max_attempts == 8
    assert record.sample_prior_draws_per_epoch == 550_000
    assert record.sample_prior_updates_per_candidate == 330 and record.sample_prior_required_candidates == 3
    assert len(prior_fits) == 1
    assert record.sample_prior_state_hash_before_full_eval == record.sample_prior_state_hash_after_full_eval
    assert record.coder_stage_rates[0] == pytest.approx(
        codec._rans_metrics[Access.SERVER_ONLY][1][0] + record.sample_training_message_rate)
    recovered = codec.decode_staged_indices(transmitted["payload_content"][0], transmitted["payload_content"][1])
    assert recovered.shape == (codec.normal_planes, target.numel())
    assert record.symbol_error_rate == 0
    codec.decode(payload, record)


def test_so_fit_only_diagnostic_preserves_received_sample_packet(fits: list[FitInputs]) -> None:
    """Matched teacher and oracle fits consume identical acquired symbols and sample output."""
    packets: list[SamplePayload] = []
    teachers: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for codec_type in (_MeasuredSampleCodec, _SOFitOnlyDiagnosticCodec):
        torch.manual_seed(431)
        codec, target, _ = make_codec(codec_type, samples=13, extra=1, refiner=True)
        codec.sampling = codec.sampling.model_copy(update={
            "sample_acquisition_distortion": "mae", "sample_fit_mode": "received_prefix_bins",
            "sample_final_distortion": "mse", "sample_train_encoder": False,
        })
        codec.quantizer.fit_epochs = 1
        fits.clear()
        codec.pre_encode_training_steps(target)
        packets.append(decompress_data_list(codec._sample_payload))
        teachers.append(codec.quantizer.sample_reconstruction.clone())
        targets.append(fits[-1].target.clone())

    assert torch.equal(packets[0].indices(), packets[1].indices())
    assert packets[0].symbols == packets[1].symbols
    assert torch.equal(packets[0].metadata.norm_factors, packets[1].metadata.norm_factors)
    assert torch.equal(teachers[0], teachers[1])
    assert not torch.equal(targets[0], targets[1])


def test_so_fit_only_diagnostic_record_fields_serialize(fits: list[FitInputs]) -> None:
    """Experiment-only record fields serialize the oracle label source and stay null on deployable S."""
    codec, target, _ = make_codec(_SOFitOnlyDiagnosticCodec, samples=13, extra=1, refiner=True)
    codec.sampling = codec.sampling.model_copy(update={
        "sample_acquisition_distortion": "mae", "sample_fit_mode": "received_prefix_bins",
        "sample_final_distortion": "mse", "sample_train_encoder": False,
    })
    codec.quantizer.fit_epochs = 1
    diagnostic_record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    codec.get_encod_payload(target, diagnostic_record)
    diagnostic = diagnostic_record.to_dict()
    assert diagnostic["sample_fit_label_source"] == "SO_fit_only_raw_current_target_at_sample_indices"
    assert diagnostic["sample_fit_label_available_to_receiver"] is False
    assert diagnostic["sample_setup_deployable"] is False

    deployable_record = PlaneSweepRecord(5, 0, "SR")
    serialized = deployable_record.to_dict()
    assert serialized.get("sample_fit_label_source") is None
    assert serialized.get("sample_fit_label_available_to_receiver") is None
    assert serialized.get("sample_setup_deployable") is None


@pytest.mark.parametrize(("filename", "seeds", "multiplier"), [
    ("s_received_raw_mae_acq_ld05_seed43_20261013.json", (43,), .5),
    ("s_received_raw_mae_acq_ld075_seed43_20261013.json", (43,), .75),
    ("s_received_raw_mae_acq_ld05_confirm_seeds44_47_20261013.json", (44, 45, 46, 47), .5),
])
def test_raw_mae_reconstruction_weight_plans_change_only_acquisition_lambda(
    filename: str, seeds: tuple[int, ...], multiplier: float,
) -> None:
    plan = PlaneSweepPlan.model_validate_json((Path(__file__).parent / "plans" / filename).read_text())

    assert plan.seeds == seeds and plan.retain_seeds == len(seeds)
    assert plan.variant_modes == ("current_e1_raw_mae_received_prefix_bins",)
    assert plan.sample_acquisition_distortion == "mse" and plan.sample_final_distortion == "mse"
    assert plan.refiner_fit_reconst_multiplier == multiplier
    assert (plan.sample_size, plan.sample_fit_epochs, plan.refiner_fit_epochs) == (3_600_000, 28, 140)


def test_final_raw_mae_requires_its_frozen_received_prefix_variant() -> None:
    """Raw final distortion cannot silently change other S fit paths."""
    with pytest.raises(ValidationError, match="matching frozen received-prefix variant"):
        PlaneSweepPlan(name="invalid", output_root=Path("out"), reference_root=Path("reference"),
                       seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                       sample_train_encoder=False, sample_final_distortion="raw_mae",
                       variant_modes=("current_e1_mae_received_prefix_bins",), include_variant_controls=False)


def test_final_normalized_mae_requires_received_prefix_variant() -> None:
    """Final normalized MAE cannot silently change another S fit path."""
    with pytest.raises(ValidationError, match="matching frozen received-prefix variant"):
        PlaneSweepPlan(name="invalid", output_root=Path("out"), reference_root=Path("reference"),
                       seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                       sample_train_encoder=False, sample_final_distortion="mae",
                       variant_modes=("current_e1_mse_received_prefix_bins",), include_variant_controls=False)


def test_received_prefix_plan_requires_frozen_encoder(tmp_path: Path) -> None:
    """Fixed-bin final fitting must leave the sample encoder unchanged."""
    with pytest.raises(ValidationError, match="frozen encoder"):
        PlaneSweepPlan(name="invalid", output_root=tmp_path / "output", reference_root=tmp_path / "reference",
                       seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                       variant_modes=("current_e1_mse_received_prefix_bins",), include_variant_controls=False)


def test_independent_o_and_variant_trimming() -> None:
    """Every variant and O discard their own two worst seeds over their method's common span."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for method, scales in (("O", [1, 2, 3, 4, 5]), ("S_e0", [5, 4, 3, 2, 1]), ("S_e2", [4, 5, 1, 2, 3])):
        groups[method] = [dict(fit_seed=seed, record=dict(stage_wmapes=[70., 55., 40.], marginal_rate=2.,
            server_si_rans_stage_rates=[.2 * scale, .25 * scale, .3 * scale]))
            for seed, scale in zip(range(43, 48), scales, strict=True)]
    result = summarize(groups, 3)
    expected = {"O": [43, 44, 45], "S_e0": [45, 46, 47], "S_e2": [45, 46, 47]}
    for method, seeds in expected.items():
        assert [row["seed"] for row in result["selection"] if row["method"] == method and row["retained"]] == seeds
        assert sum(row["retained"] for row in result["selection"] if row["method"] == method) == 3
        assert result["cohorts"]["untrimmed"][method]["observations"] == 5
        assert result["cohorts"]["trimmed"][method]["observations"] == 3
    o_seed_43 = next(row for row in result["selection"] if row["method"] == "O" and row["seed"] == 43)
    assert o_seed_43["gap_vs_all_o_percent"] == pytest.approx(-100 * 2 / 3)
    np.testing.assert_allclose(result["cohorts"]["trimmed"]["O"]["server_si_rans_stage_rates"], [.4, .9, 1.5])


def test_disjoint_methods_are_not_ranked_without_overlap_with_all_o() -> None:
    """A method outside the all-O distortion span stays unranked instead of being extrapolated."""
    groups = {method: [dict(fit_seed=seed, record=dict(stage_wmapes=errors, marginal_rate=3.,
              server_si_rans_stage_rates=[.2 * seed, .25 * seed, .3 * seed])) for seed in range(1, 6)]
              for method, errors in (("O", [80., 70., 60.]), ("S_e6", [40., 30., 20.]))}
    result = summarize(groups, 3)
    assert result["common_distortion_db"] is None
    assert result["cohorts"]["trimmed"]["O"]["observations"] == 3
    assert "S_e6" not in result["cohorts"]["trimmed"]
    assert all(not row["retained"] and "no shared span" in row["reason"]
               for row in result["selection"] if row["method"] == "S_e6")
    assert result["ranking_distortion_db"]["S_e6"] is None


def test_final_report_and_nonmonotone_seed(tmp_path: Path) -> None:
    """A failed stage remains in raw results, with its exclusion and both full matrices published."""
    plan = PlaneSweepPlan(name="toy", output_root=tmp_path, reference_root=tmp_path, extra_planes=(0,), seeds=(1, 2, 3), retain_seeds=2)
    (tmp_path / "test_config.json").write_text(json.dumps(dict(identity="toy", plan=plan.model_dump(mode="json"))))
    for method in plan.methods:
        for seed in plan.seeds:
            rates = [.2 * seed, .25 * seed, .3 * seed]
            record = dict(stage_wmapes=[70., 55., 40.] if seed < 3 else [70., 75., 40.], marginal_rate=3.,
                          quantizer_prior_rate=sum(rates), quantizer_prior_stage_rates=rates,
                          coder_stage_rates=rates, server_si_rans_stage_rates=rates, symbol_error_rate=0.,
                          sample_count=10, model_size=100, sample_wmape=37.0,
                          sample_stage_wmape=[80.0, 65.0, 50.0], sample_stage_occupied_bins=[2, 2, 1],
                          sample_stage_marginal_entropy_bps=[1.0, .9, 0.0], sample_stage_prior_rate=[.01, .02, .03],
                          sample_extra_prefix_conditional_entropy_bps=[.4],
                          sample_extra_final_prefix_shuffle_wmape=58.0,
                          sample_extra_final_prior_mode_wmape=44.0)
            directory = tmp_path / "cases" / f"{method}_seed{seed}" / "target"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text(json.dumps(dict(identity="toy", method=method, fit_seed=seed, record=record)))
    publish(tmp_path, tmp_path / "analysis", complete=True)
    result = json.loads((tmp_path / "analysis/summary.json").read_text())
    assert all(not row["retained"] and "unrankable" in row["reason"] for row in result["selection"] if row["seed"] == 3)
    assert all(point["observations"] == 2 for point in result["cohorts"]["trimmed"].values())
    report = (tmp_path / "analysis/report.md").read_text()
    assert report.count("```text") == 4 and "COMPLETE" in report and "nonmonotone final S distortion" in report
    assert "every hard-decoded LWZQ stage" in report
    assert "dead stage: one occupied bin" in report
    assert "H(stage | prefix, earlier extras)" in report
    assert "prefix-conditioned shuffle" in report and "prior-mode %" in report
    assert "changing decoder inputs" in report and "Final S curve flags" in report
    assert (tmp_path / "analysis/rd.pdf").exists()


@pytest.mark.parametrize("extra", [0, 1, 3, 6])
def test_measured_sample_stream_and_prior(extra: int, fits: list[FitInputs]) -> None:
    """Real rANS acquisition changes training labels and is included exactly once in every rate source."""
    codec, target, _ = make_codec(_MeasuredSampleCodec, samples=13, extra=extra)
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    delivered = codec.decode(codec.encode(target, record), record)
    assert record.sample_format == "lwzq" and record.sample_num_planes == 3 + extra
    assert record.sample_coder_rate > 0 and record.symbol_error_rate == 0
    assert sum(record.server_si_prior_stage_rates) == pytest.approx(record.server_si_prior_rate, rel=2e-5)
    assert record.server_si_prior_stage_rates[0] >= record.sample_prior_rate
    assert record.server_si_rans_stage_rates[0] >= record.sample_coder_rate
    if extra:
        assert len(record.sample_extra_prefix_conditional_entropy_bps) == extra
        assert record.sample_extra_final_prefix_shuffle_wmape is not None
        assert record.sample_extra_final_prior_mode_wmape is not None
    assert not delivered[codec.quantizer.sample_indices].equal(target[codec.quantizer.sample_indices])


@pytest.mark.parametrize("extra", [0, 1, 2, 3, 6])
def test_dedicated_refiner_codec_transmits_sample_tail_only(extra: int, fits: list[FitInputs]) -> None:
    """Refined S trains the ordinary prefix and sends extra symbols only in the charged sample stream."""
    codec, target, _ = make_codec(_MeasuredSampleCodec, samples=13, extra=extra, refiner=True)
    codec.sampling = codec.sampling.model_copy(update={"sample_training": "prefix", "sample_coordinate_seed": 43})
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    assert packet.seed == 43 and fits[-1].planes == 3
    assert codec.quantizer.trained_num_planes == 3 + extra
    assert isinstance(codec.quantizer.coding_model, RefinedEncoderDecoderLayeredRNN) is bool(extra)
    assert codec.quantizer.coding_model.num_planes == 3
    delivered = codec.decode(codec.encode(target, record), record)
    assert record.sample_num_planes == 3 + extra and record.sample_training_planes == 3
    assert record.symbol_error_rate == 0 and sum(record.server_si_prior_stage_rates) == pytest.approx(record.server_si_prior_rate)
    torch.testing.assert_close(delivered[codec.quantizer.sample_indices], codec.quantizer.sample_reconstruction)


def test_acquisition_refiner_optimizer_preserves_fixed_starting_prefix() -> None:
    """Real optimization changes the secondary network while every retained ordinary weight stays fixed."""
    codec, _, _ = make_codec(_MeasuredSampleCodec, extra=2, refiner=True)
    quantizer = codec.quantizer
    quantizer.fit_refiner_only = True
    quantizer.fit_hard_inputs = True
    quantizer._set_training_parameters(quantizer.coding_model)
    model = quantizer.coding_model
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    loss = quantizer.compute_loss(model, (torch.randn(19, 1),), torch.randn(19, 2), 0,
                                 quantizer.training_config(), None, 1., torch.Generator().manual_seed(43))
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    for name, value in before.items():
        if not name.startswith("refiner."):
            torch.testing.assert_close(value, after[name], rtol=0, atol=0)
    assert any(not torch.equal(value, after[name]) for name, value in before.items() if name.startswith("refiner.encoder."))
    quantizer._fit_machine_planes = 5
    quantizer.c_cfg = quantizer.c_cfg.model_copy(update={"num_planes": 3})
    candidate = quantizer._new_coding_model(2)
    candidate.load_state_dict(after)
    assert candidate.num_planes == 3 and candidate.base_num_planes == 3 and candidate.refinement_planes == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The detached sweep worker requires CUDA.")
@pytest.mark.parametrize("refiner", [False, True])
def test_worker_saves_paired_samples_and_resumes(tmp_path: Path, fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch, refiner: bool) -> None:
    """The full worker saves real pilot-owned coordinates across variants and resumes completed fits."""
    monkeypatch.setattr(_MeasuredOracleCodec, "_fine_tune_native_prior", False)
    _, target, history = make_codec(_MeasuredSampleCodec)
    reference = tmp_path / "reference"
    (reference / "problem").mkdir(parents=True)
    (reference / "baseline").mkdir()
    PickleCodec().save(dict(target=target, baseline=target + .5, si=history, slices=(slice(0, 32), slice(32, 64))), reference / "problem/target.pkl")
    torch.save(dict(state=new_rnn_model(4, 3, 2, False, 4, 1).state_dict()), reference / "baseline/state.pt")
    (reference / "test_config.json").write_text(json.dumps(dict(
        plan=dict(quantizer=dict(num_planes=4, bins_per_plane=3, hidden_dim=4, layers=1), target_round=5, target_client=0),
        target_hash=content_hash(target), si_hash=content_hash(tuple(content_hash(value.materialize()) for value in history)))))
    plan = PlaneSweepPlan(name="worker", output_root=tmp_path / "output", reference_root=reference,
                          seeds=(43,), extra_planes=(0, 1), retain_seeds=1, sample_size=13, sample_refiner=refiner,
                          refiner_fit_epochs=1, refiner_fit_lr=.01, refiner_fit_tau=.3,
                          refiner_fit_lr_step=43, refiner_fit_batch_size=8)
    run(plan)
    saved = [json.loads((plan.output_root / "cases" / f"S_e{extra}_seed43/target/complete.json").read_text()) for extra in (0, 1)]
    assert saved[0]["sample_coordinate_seed"] == saved[1]["sample_coordinate_seed"]
    assert saved[0]["sample_indices_hash"] == saved[1]["sample_indices_hash"]
    first_draw = int(torch.randint(0, 2**31, (), generator=torch.Generator().manual_seed(43)).item())
    assert (saved[0]["sample_coordinate_seed"] == first_draw) is refiner
    if refiner:
        assert saved[1]["base_prefix_unchanged"] and saved[1]["refiner_fit_seconds"] > 0
        assert saved[0]["record"].get("sample_extra_prefix_conditional_entropy_bps") is None
        record = saved[1]["record"]
        assert len(record["sample_extra_prefix_conditional_entropy_bps"]) == 1
        assert record["sample_extra_final_prefix_shuffle_wmape"] is not None
        assert record["sample_extra_final_prior_mode_wmape"] is not None
    before = len(fits)
    run(plan)
    assert len(fits) == before
    assert len(tuple((plan.output_root / "cases").glob("*/*/complete.json"))) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The detached sweep worker requires CUDA.")
def test_worker_applies_each_variant_to_the_acquisition_fit(
    tmp_path: Path, fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The six-case catalog pairs one O/S0 with all current/residual MSE/MAE acquisitions."""
    monkeypatch.setattr(_MeasuredOracleCodec, "_fine_tune_native_prior", False)
    _, target, history = make_codec(_MeasuredSampleCodec)
    reference = tmp_path / "reference"
    (reference / "problem").mkdir(parents=True)
    (reference / "baseline").mkdir()
    PickleCodec().save(dict(target=target, baseline=target + .5, si=history,
                            slices=(slice(0, 32), slice(32, 64))), reference / "problem/target.pkl")
    torch.save(dict(state=new_rnn_model(4, 3, 2, False, 4, 1).state_dict()), reference / "baseline/state.pt")
    (reference / "test_config.json").write_text(json.dumps(dict(
        plan=dict(quantizer=dict(num_planes=4, bins_per_plane=3, hidden_dim=4, layers=1),
                  target_round=5, target_client=0),
        target_hash=content_hash(target),
        si_hash=content_hash(tuple(content_hash(value.materialize()) for value in history)))))
    modes = ("current_e1_mse", "current_e1_mae", "residual_e1_mse", "residual_e1_mae")
    plan = PlaneSweepPlan(name="variants", output_root=tmp_path / "output", reference_root=reference,
                          seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_size=13,
                          sample_refiner=True, variant_modes=modes, refiner_fit_reconst_multiplier=.5,
                          refiner_fit_epochs=1,
                          refiner_fit_lr=.01, refiner_fit_tau=.3, refiner_fit_lr_step=43,
                          refiner_fit_batch_size=8)

    run(plan)

    completed = {
        method: json.loads((plan.output_root / "cases" / f"{method}_seed43/target/complete.json").read_text())
        for method in plan.methods
    }
    sample_cases = {method: result for method, result in completed.items() if method != "O"}
    assert len({(result["sample_coordinate_seed"], result["sample_indices_hash"])
                for result in sample_cases.values()}) == 1
    assert {method: (result["sample_refiner_residual_anchor"], result["sample_acquisition_distortion"])
            for method, result in sample_cases.items()} == {
                "S_e0": (False, "mse"), "current_e1_mse": (False, "mse"),
                "current_e1_mae": (False, "mae"), "residual_e1_mse": (True, "mse"),
                "residual_e1_mae": (True, "mae"),
            }
    base_reconst_ld = WZcfgQuant(num_planes=3, bins_per_plane=2).reconst_ld
    for result in sample_cases.values():
        assert result["sample_fit_reconst_ld"] == base_reconst_ld
    for result in sample_cases.values():
        if "refiner_fit_reconst_multiplier" in result:
            assert result["refiner_fit_reconst_multiplier"] == .5
            assert result["refiner_fit_reconst_ld_base"] == base_reconst_ld
            assert result["refiner_fit_reconst_ld_effective"] == base_reconst_ld * .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The detached sweep worker requires CUDA.")
def test_worker_applies_budget_per_variant_and_keeps_same_seed_draws_distinct(
    tmp_path: Path, fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different sample counts use the same coordinate seed but independently chosen subsets."""
    monkeypatch.setattr(_MeasuredOracleCodec, "_fine_tune_native_prior", False)
    _, target, history = make_codec(_MeasuredSampleCodec)
    reference = tmp_path / "reference"
    (reference / "problem").mkdir(parents=True)
    (reference / "baseline").mkdir()
    PickleCodec().save(dict(target=target, baseline=target + .5, si=history,
                            slices=(slice(0, 32), slice(32, 64))), reference / "problem/target.pkl")
    torch.save(dict(state=new_rnn_model(4, 3, 2, False, 4, 1).state_dict()), reference / "baseline/state.pt")
    (reference / "test_config.json").write_text(json.dumps(dict(
        plan=dict(quantizer=dict(num_planes=4, bins_per_plane=3, hidden_dim=4, layers=1),
                  target_round=5, target_client=0),
        target_hash=content_hash(target),
        si_hash=content_hash(tuple(content_hash(value.materialize()) for value in history)))))
    plan = PlaneSweepPlan(name="budget", output_root=tmp_path / "output", reference_root=reference,
                          seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_refiner=True,
                          variant_modes=("current_e1_mse",), budget_sample_sizes=(13, 23),
                          sample_fit_epochs=1, refiner_fit_epochs=1, refiner_fit_lr=.01,
                          refiner_fit_tau=.3, refiner_fit_lr_step=43, refiner_fit_batch_size=8)

    run(plan)

    completed = {
        method: json.loads((plan.output_root / "cases" / f"{method}_seed43/target/complete.json").read_text())
        for method in plan.methods
    }
    small, large = (completed[f"current_e1_mse_n{size}"] for size in (13, 23))
    assert small["record"]["sample_count"] == 13 and large["record"]["sample_count"] == 23
    assert small["sample_coordinate_seed"] == large["sample_coordinate_seed"]
    assert small["sample_indices_hash"] != large["sample_indices_hash"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The detached sweep worker requires CUDA.")
def test_worker_pairs_received_bin_and_frozen_reencode_fit_paths(
    tmp_path: Path, fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen reencode control repeats the same MSE acquisition and changes only final-S fit inputs."""
    monkeypatch.setattr(_MeasuredOracleCodec, "_fine_tune_native_prior", False)
    _, target, history = make_codec(_MeasuredSampleCodec)
    reference = tmp_path / "reference"
    (reference / "problem").mkdir(parents=True)
    (reference / "baseline").mkdir()
    PickleCodec().save(dict(target=target, baseline=target + .5, si=history,
                            slices=(slice(0, 32), slice(32, 64))), reference / "problem/target.pkl")
    torch.save(dict(state=new_rnn_model(4, 3, 2, False, 4, 1).state_dict()), reference / "baseline/state.pt")
    (reference / "test_config.json").write_text(json.dumps(dict(
        plan=dict(quantizer=dict(num_planes=4, bins_per_plane=3, hidden_dim=4, layers=1),
                  target_round=5, target_client=0),
        target_hash=content_hash(target),
        si_hash=content_hash(tuple(content_hash(value.materialize()) for value in history)))))
    modes = ("current_e1_mse_received_prefix_bins", "current_e1_mae_received_prefix_bins",
             "current_e1_mse_reencode_teacher_frozen_control")
    plan = PlaneSweepPlan(name="received", output_root=tmp_path / "output", reference_root=reference,
                          seeds=(43,), extra_planes=(0,), retain_seeds=1, sample_size=13,
                          sample_refiner=True, sample_train_encoder=False, variant_modes=modes,
                          include_variant_controls=False, sample_fit_epochs=1, refiner_fit_epochs=1,
                          refiner_fit_lr=.01, refiner_fit_tau=.3, refiner_fit_lr_step=43,
                          refiner_fit_batch_size=8)

    # The shared fit fixture intentionally changes each fit's head bias; reset its local call counter
    # so identical acquisition work remains identical across final-S variants.
    fit_prepared = WZQuantizerCancer._fit_prepared

    def stable_fit(self: WZQuantizerCancer, x: torch.Tensor, si: Any) -> None:
        fits.clear()
        fit_prepared(self, x, si)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", stable_fit)

    run(plan)

    completed = {
        method: json.loads((plan.output_root / "cases" / f"{method}_seed43/target/complete.json").read_text())
        for method in plan.methods
    }
    received = completed["current_e1_mse_received_prefix_bins"]
    control = completed["current_e1_mse_reencode_teacher_frozen_control"]
    mae = completed["current_e1_mae_received_prefix_bins"]
    assert received["acquisition_state_hash"] == control["acquisition_state_hash"]
    assert received["sample_coordinate_seed"] == control["sample_coordinate_seed"]
    assert received["sample_indices_hash"] == control["sample_indices_hash"]
    assert received["record"]["sample_fit_mode"] == "received_prefix_bins"
    assert control["record"]["sample_fit_mode"] == "reencode_teacher"
    assert mae["sample_acquisition_distortion"] == "mae"
    assert all(not run["record"]["sample_training_encoder"] for run in completed.values())
