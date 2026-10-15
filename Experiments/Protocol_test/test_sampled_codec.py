"""Check causal sample fitting, prefix transport, and protocol-owned S/SR reuse."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from Experiments.Protocol_test.sampled_codec import RefinedEncoderDecoderLayeredRNN, SampledRecord, SampledRoundCodec, SampledRansRoundCodec, SampledOracleRoundCodec, SampledOracleRansRoundCodec, SamplePayload, SamplingConfig
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, record_reconstruction_metrics
from FL_code.FL_core.utils import decompress_data_list, normalization_params
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import PreprocessMetadata, SideInfoMatrix, WZQuantizerCancer, _SideInfoRows, new_rnn_model, unique_row_groups


def test_sampled_record_accepts_prefix_conditional_diagnostics() -> None:
    """New acquisition diagnostics pass the record's declared-field boundary and export intact."""
    record = SampledRecord(1, 2, "SR")
    record.update_record_fields(
        sample_extra_prefix_conditional_entropy_bps=[0.2, 0.1],
        sample_extra_final_prefix_shuffle_wmape=40.0,
        sample_extra_final_prior_mode_wmape=36.0,
    )
    expected = {
        "round_id": 1, "client_id": 2, "round_name_full": "SR",
        "sample_extra_prefix_conditional_entropy_bps": [0.2, 0.1],
        "sample_extra_final_prefix_shuffle_wmape": 40.0,
        "sample_extra_final_prior_mode_wmape": 36.0,
    }
    assert {key: record.to_dict()[key] for key in expected} == expected


def test_residual_anchor_matches_prefix_for_partial_and_full_refiner() -> None:
    """Zero residual heads preserve every prefix stage and anchor all active extras to its final stage."""
    torch.manual_seed(23)
    model = RefinedEncoderDecoderLayeredRNN(
        1, 2, 3, 2, 1, 4, output_activation=torch.tanh,
        refinement_planes=2, use_refiner=True, residual_anchor=True,
    ).eval()
    x, si = torch.randn(19, 1), torch.randn(19, 2)
    _, codes = model.encode(x)

    model.num_planes = model.base_num_planes
    prefix = model.decode(codes[:model.base_num_planes], si)
    for plane_count in (model.base_num_planes + 1, model.base_num_planes + 2):
        model.num_planes = plane_count
        decoded = model.decode(codes[:plane_count], si)
        for actual, expected in zip(decoded[:model.base_num_planes], prefix, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(decoded[-1], prefix[-1], rtol=0, atol=0)


def test_residual_anchor_builds_the_sample_machine() -> None:
    """The configured S build path constructs and freezes the residual-anchored prefix machine."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=2, refiner=True, residual_anchor=True)
    model = codec.quantizer.coding_model
    assert isinstance(model, RefinedEncoderDecoderLayeredRNN) and model.residual_anchor
    codec.quantizer.fit_refiner_only = True
    codec.quantizer._set_training_parameters(model)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("refiner."))
    assert any(parameter.requires_grad for parameter in model.refiner.parameters())


def test_residual_anchor_head_learns_before_refiner_encoder_and_preserves_base() -> None:
    """A zero head gets a residual gradient immediately and passes gradients upstream after its first update."""
    torch.manual_seed(31)
    model = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 2, 1, 8, refinement_planes=2,
                                           use_refiner=True, residual_anchor=True).train()
    model.num_planes = model.base_num_planes + 1
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.refiner.parameters():
        parameter.requires_grad_(True)
    base_before = {name: value.detach().clone() for name, value in model.state_dict().items()
                   if not name.startswith("refiner.")}
    head = model.refiner.reconstructors[0]
    assert not head.weight.any() and not head.bias.any()
    optimizer = torch.optim.SGD(model.refiner.parameters(), lr=.1)
    x, si = torch.randn(32, 1), torch.randn(32, 2)
    target = torch.randn(32, 1)

    def backward() -> None:
        _, codes = model.encode(x, tau=.5)
        stages = model.decode(codes, si)
        (stages[-1] - target).square().mean().backward()

    backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in model.refiner.encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.refiner.encoder.parameters())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in base_before.items())


def test_residual_anchor_keeps_wire_shapes_and_is_part_of_codec_identity() -> None:
    """Anchoring changes decoder meaning while retaining the same transmitted parameter shapes."""
    anchored = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 2, 1, 4, refinement_planes=2,
                                              use_refiner=True, residual_anchor=True)
    ordinary = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 2, 1, 4, refinement_planes=2, use_refiner=True)
    assert set(anchored.encoder_state_dict()) == set(ordinary.encoder_state_dict())
    assert set(anchored.decoder_state_dict()) == set(ordinary.decoder_state_dict())
    assert {key: value.shape for key, value in anchored.state_dict().items()} == {
        key: value.shape for key, value in ordinary.state_dict().items()}
    with pytest.raises(ValidationError, match="requires sample_refiner=True"):
        SamplingConfig(sample_refiner_residual_anchor=True)
    protocol = create_protocol("wz_cancer", (slice(0, 32), slice(32, 64)))
    token = ("S|bins_per_plane=2|num_planes=3|sample_extra_planes=1|sample_refiner|"
             "sample_refiner_residual_anchor|sample_acquisition_distortion='mae'")
    protocol.set_schedule(5, (token, "F"))
    facts = protocol.config_snapshot()["round_codecs"][token]["codec_provenance"]
    assert facts["sampling"]["sample_refiner_residual_anchor"] is True
    assert facts["sampling"]["sample_acquisition_distortion"] == "mae"


def test_acquisition_distortion_is_pilot_only(
    fits: list["FitInputs"], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAE is applied to the acquisition pilot and final replacement fitting remains at MSE."""
    codec, target, _ = make_codec(SampledRoundCodec, samples=11, extra=1, refiner=True, residual_anchor=True)
    codec.sampling = codec.sampling.model_copy(update={"sample_acquisition_distortion": "mae"})
    seen: list[tuple[str, float | None, bool]] = []
    original = WZQuantizerCancer._fit_prepared

    def capture(self: WZQuantizerCancer, x: torch.Tensor,
                si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        seen.append((getattr(self, "fit_distortion", "mse"), getattr(self, "_fit_mae_denominator", None),
                     getattr(self, "residual_anchor", False)))
        original(self, x, si)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture)
    codec.pre_encode_training_steps(target)
    assert seen[0] == ("mae", pytest.approx(float(fits[0].target.abs().mean()) + 1e-8), True)
    assert seen[1] == ("mse", None, True)


def test_acquisition_mae_loss_changes_only_distortion_and_preserves_rate() -> None:
    """MAE uses the fixed WMAPE-aligned scale while keeping O's per-stage rate term unchanged."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=2)
    quantizer = codec.quantizer
    quantizer.fit_distortion = "mae"
    quantizer._fit_mae_denominator = 1.7
    model = quantizer.coding_model.train()
    inputs, side_info = torch.randn(17, 1), torch.randn(17, 2)
    config = quantizer.training_config()
    generator = torch.Generator().manual_seed(9)
    torch.manual_seed(17)
    actual = quantizer.compute_loss(model, (inputs,), side_info, 0, config, None, 1., generator)

    torch.manual_seed(17)
    tau = config.tau * np.exp((0 / config.train_epochs) * np.log(.1 / config.tau))
    recons, bins, soft_codes, priors = model(inputs, side_info, tau=tau)
    expected = torch.zeros(())
    rows = torch.arange(inputs.shape[0])
    for plane in range(model.num_planes):
        distortion = torch.nn.functional.l1_loss(recons[plane], inputs, reduction="none").flatten(1).mean(1)
        posterior = soft_codes[plane][rows, bins[plane]]
        prior = priors[plane][rows, bins[plane]]
        rate = torch.log((posterior + 1e-12) / (prior + 1e-12)).mean()
        expected += (2 * config.reconst_ld / 1.7) * distortion.mean() + rate
    expected /= model.num_planes
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_raw_mae_acquisition_uses_baseline_affine_scales_and_keeps_final_s_mse(
    fits: list["FitInputs"],
) -> None:
    """The acquisition objective derives its scales and denominator from decoder-visible baseline rows."""
    codec, current_target, history = make_codec(SampledRoundCodec, samples=11, extra=1, refiner=True)
    codec.sampling = SamplingConfig(sample_size=11, sample_extra_planes=1, sample_refiner=True,
                                   sample_acquisition_distortion="raw_mae")
    baseline = history[-1].materialize()

    codec.pre_encode_training_steps(current_target)

    assert len(fits) == 2
    assert fits[0].raw_denominator == pytest.approx(float(baseline.abs().mean()))
    assert fits[0].raw_row_scales is not None
    expected_scales = torch.empty_like(baseline).reshape(-1, 1)
    for vector_slice in codec.quantizer.norm_slices:
        scale, _ = normalization_params(baseline[vector_slice])
        expected_scales[vector_slice] = scale
    torch.testing.assert_close(fits[0].raw_row_scales.cpu(), expected_scales)
    assert not torch.isclose(torch.tensor(fits[0].raw_denominator), current_target.abs().mean())
    assert fits[1].raw_denominator is None and fits[1].raw_row_scales is None
    assert codec._sample_fields["sample_acquisition_distortion"] == "raw_mae"
    assert codec._sample_fields["sample_acquisition_weight_source"] == (
        "baseline_fit_target_norm_factors_and_raw_mean_abs")


def test_raw_mae_loss_matches_raw_absolute_error_and_preserves_refiner_capacity() -> None:
    """Per-row affine scales recover raw MAE while the existing factor two and rate stay unchanged."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, refiner=True)
    quantizer = codec.quantizer
    quantizer.fit_distortion = "raw_mae"
    quantizer._fit_mae_denominator = 1.7
    quantizer.fit_refiner_only = True
    model = quantizer.coding_model.train()
    quantizer._set_training_parameters(model)
    active_planes = model.num_planes
    assert active_planes == model.base_num_planes + model.refinement_planes
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("refiner."))
    assert any(parameter.requires_grad for parameter in model.refiner.parameters())

    inputs, side_info = torch.linspace(-1, 1, 8).reshape(-1, 1), torch.randn(8, 2)
    row_scales = torch.tensor([1., 1., 1., 1., 4., 4., 4., 4.]).reshape(-1, 1)
    centers = torch.tensor([3., 3., 3., 3., -2., -2., -2., -2.]).reshape(-1, 1)
    denominator = 1.7
    quantizer._fit_mae_denominator = denominator
    config = quantizer.training_config()
    torch.manual_seed(47)
    actual = quantizer.compute_loss(model, (inputs, row_scales), side_info, 0, config, None, 1.,
                                    torch.Generator().manual_seed(3))

    torch.manual_seed(47)
    tau = config.tau * np.exp((0 / config.train_epochs) * np.log(.1 / config.tau))
    reconstructions, bins, soft_codes, priors = model(inputs, side_info, tau=tau)
    expected = torch.zeros(())
    rows = torch.arange(inputs.shape[0])
    for plane in range(model.num_planes):
        normalized_abs = (reconstructions[plane] - inputs).abs().flatten(1).mean(1)
        raw_target = inputs * row_scales + centers
        raw_reconstruction = reconstructions[plane] * row_scales + centers
        raw_abs = (raw_reconstruction - raw_target).abs().flatten(1).mean(1)
        torch.testing.assert_close(row_scales.flatten() * normalized_abs, raw_abs)
        posterior = soft_codes[plane][rows, bins[plane]]
        prior = priors[plane][rows, bins[plane]]
        rate = torch.log((posterior + 1e-12) / (prior + 1e-12)).mean()
        expected += (2 * config.reconst_ld / denominator) * raw_abs.mean() + rate
    expected /= model.num_planes
    torch.testing.assert_close(actual, expected)
    assert model.num_planes == active_planes
    quantizer._fit_mae_denominator = denominator * 1e-5
    torch.manual_seed(47)
    scaled_units_loss = quantizer.compute_loss(
        model, (inputs, row_scales * 1e-5), side_info, 0, config, None, 1.,
        torch.Generator().manual_seed(3))
    torch.testing.assert_close(scaled_units_loss, actual)


def test_raw_mae_rows_stay_aligned_through_sampling_and_deduplication() -> None:
    """Scale weights participate in row identity, and duplicate counts preserve the weighted loss."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, refiner=True)
    quantizer = codec.quantizer
    quantizer.fit_distortion = "raw_mae"
    inputs = torch.tensor([[.1], [.1], [.2], [.2], [.2]])
    scales = torch.tensor([[1.], [3.], [2.], [2.], [2.]])
    side_info = torch.zeros((5, 2))
    quantizer._fit_raw_row_scales = scales
    training_inputs = quantizer._fit_training_inputs(inputs)
    sample_indices = torch.tensor([4, 1, 3, 0])
    sampled = tuple(values[sample_indices] for values in training_inputs)
    torch.testing.assert_close(sampled[1], scales[sample_indices])

    representatives, groups, collisions = unique_row_groups((*training_inputs, side_info), 32)
    assert collisions.numel() == 0
    _, inverse, counts = torch.unique(groups, return_inverse=True, return_counts=True)
    per_row_raw_loss = (inputs.flatten().abs() + .25) * scales.flatten()
    unique_loss = per_row_raw_loss[representatives]
    deduplicated_mean = (unique_loss * counts).sum() / counts.sum()
    torch.testing.assert_close(deduplicated_mean, per_row_raw_loss.mean())
    assert groups[0] != groups[1]
    assert groups[2] == groups[3] == groups[4]


def test_raw_mae_requires_affine_refiner_acquisition_without_outliers() -> None:
    """Raw-domain acquisition is limited to a refiner and affine preprocessing."""
    with pytest.raises(ValidationError, match="sample refiner"):
        SamplingConfig(sample_acquisition_distortion="raw_mae")
    SamplingConfig(sample_extra_planes=1, sample_refiner=True, sample_acquisition_distortion="raw_mae")
    with pytest.raises(ValueError, match="affine raw-error weighting"):
        SampledRoundCodec.validate_cfg(sample_extra_planes=1, sample_refiner=True,
                                       sample_acquisition_distortion="raw_mae", outlier_threshold=3.)


@dataclass
class FitInputs:
    """Captured normalized rows consumed by O's fitter while exercising real model inference."""

    target: torch.Tensor
    side_info: torch.Tensor
    planes: int
    factors: torch.Tensor | None
    raw_row_scales: torch.Tensor | None = None
    raw_denominator: float | None = None
    final_row_scales: torch.Tensor | None = None
    final_denominator: float | None = None
    final_distortion: str = "mse"


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FitInputs]]:
    """Replace only expensive fitting; preprocessing, networks, serialization and coding stay real."""
    captured: list[FitInputs] = []
    torch.manual_seed(19)

    def fit(self: WZQuantizerCancer, x: torch.Tensor, si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        captured.append(FitInputs(x.detach().cpu().clone(), si[:].cpu().clone(), self.num_planes,
                                 getattr(self, "normalization_factors", None),
                                 getattr(self, "_fit_raw_row_scales", None),
                                 getattr(self, "_fit_mae_denominator", None),
                                 getattr(self, "_fit_final_row_scales", None),
                                 getattr(self, "_fit_final_mae_denominator", None),
                                 getattr(self, "fit_final_distortion", "mse")))
        with torch.no_grad():
            # Different fits must have different reconstructions to expose missing sample replacement.
            models = (self.coding_model, self.coding_model.refiner) if hasattr(self.coding_model, "refiner") else (self.coding_model,)
            for model in models:
                heads = (model.reconstructor,) if model.shared_decoder else model.reconstructors
                for head in heads:
                    if head.bias.requires_grad:
                        head.bias.fill_(len(captured) / 7)
        self.quantizer_prior_stage_rates = tuple(1.0 for _ in range(self.num_planes))

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", fit)
    monkeypatch.setattr(SampledRoundCodec, "_fine_tune_native_prior", False)
    yield captured


def make_codec(
    kind: type[SampledRoundCodec], samples: int = 11, extra: int = 1,
    shared: bool = True, outliers: float | None = None, refiner: bool = False,
    residual_anchor: bool = False,
) -> tuple[SampledRoundCodec, torch.Tensor, list[HistoryVector]]:
    """Build a small two-layer problem with deterministic, nonconstant target and history."""
    target = torch.sin(torch.arange(64).float()) * 2 + 3
    history = [HistoryVector(torch.arange(64).float() / 20 + shift) for shift in (1, 2)]
    options = dict(bins_per_plane=3, num_planes=3, sd_slices=(slice(0, 32), slice(32, 64)),
                   hidden_dim=4, layers=1, sample_size=samples, sample_extra_planes=extra,
                   sample_refiner=refiner, sample_refiner_residual_anchor=residual_anchor,
                   shared_heads=shared, outlier_threshold=outliers)
    if kind is SampledRansRoundCodec:
        options["rans_lanes"] = 2
    codec = kind(options, kind.round_name)
    codec.build_quantizer(None, history)
    return codec, target, history


@pytest.mark.parametrize("kind", [SampledOracleRoundCodec, SampledOracleRansRoundCodec])
def test_sampled_oracle_fits_raw_labels_without_acquisition(
    kind: type[SampledOracleRoundCodec], fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SO keeps the raw-sample oracle's full normalization, ordinary planes and full-vector decode."""
    monkeypatch.setattr(SampledOracleRoundCodec, "_fine_tune_native_prior", False)
    _, target, history = make_codec(SampledRoundCodec)
    options = dict(bins_per_plane=3, num_planes=3, sd_slices=(slice(0, 32), slice(32, 64)),
                   hidden_dim=4, layers=1, sample_size=11, train_sample_size=11)
    if kind is SampledOracleRansRoundCodec:
        options["rans_lanes"] = 2
    codec = kind(options, kind.round_name)
    codec.build_quantizer(None, list(history))
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    indices = codec.quantizer.training_sample_indices
    assert len(fits) == 1 and fits[0].planes == codec.quantizer.num_planes == 3
    normalized, _ = codec.quantizer.preprocess_x(target)
    torch.testing.assert_close(fits[0].target, normalized[indices].cpu())
    assert codec.quantizer.sample_indices is None and codec.quantizer.sample_reconstruction is None
    payload = codec.encode(target, record)
    assert "sample_payload" not in decompress_data_list(payload)
    delivered = codec.decode(payload, record)
    assert record.sample_format == "raw_oracle" and record.sample_count == 11
    assert record.sample_wmape == record.sample_prior_rate == record.sample_coder_rate == 0.0
    assert not torch.equal(delivered[indices], target[indices])
    codec.release_round_resources()
    assert codec.quantizer.training_sample_indices is None
    codec.refresh_side_information(history)
    codec.round_name_full = "F"
    frozen = codec.create_r_record(6, 0)
    codec.pre_encode_training_steps(target)
    codec.decode(codec.encode(target, frozen), frozen)
    assert len(fits) == 1 and frozen.sample_count == 0 and frozen.sample_wmape is None


@pytest.mark.parametrize("token", ["SO", "SOR"])
def test_sampled_oracle_protocol_provenance(token: str) -> None:
    """Schedules resolve SO/SOR distinctly from S/SR and disclose free raw-label access."""
    protocol = create_protocol("wz_cancer", (slice(0, 32), slice(32, 64)))
    protocol.set_schedule(5, (f"{token}|bins_per_plane=2|num_planes=3|sample_size=11", "F"))
    snapshot = protocol.config_snapshot()
    assert "raw_sample_oracle" in str(snapshot) and "free_current_target_labels_at_fitter" in str(snapshot)


@pytest.mark.parametrize("kind", [SampledRoundCodec, SampledRansRoundCodec])
@pytest.mark.parametrize("samples", [1, 11, 64])
@pytest.mark.parametrize("shared", [False, True])
def test_sample_fit_and_delivered_reconstruction(
    kind: type[SampledRoundCodec], samples: int, shared: bool, fits: list[FitInputs], tmp_path: Path,
) -> None:
    """O trains on reconstructed samples with real normalization, then codes the complete vector."""
    codec, target, history = make_codec(kind, samples, shared=shared)
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    assert len(fits) == 2  # Decoder-history bootstrap, then the reconstructed current sample.
    assert fits[0].target.shape == (64, 1) and fits[1].target.shape == (samples, 1)
    assert fits[0].planes == fits[1].planes == 4
    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    indices = packet.indices()
    reconstructed_sample = codec.quantizer.sample_reconstruction.clone()
    assert indices.unique().numel() == samples
    assert not torch.equal(reconstructed_sample, target[indices])
    layer_factors = torch.tensor([normalization_params(target[part]) for part in (slice(0, 32), slice(32, 64))])
    factors = layer_factors[indices.ge(32).long()]
    torch.testing.assert_close(fits[1].target[:, 0], (reconstructed_sample - factors[:, 1]) / factors[:, 0])
    expected_si = torch.stack([value.materialize()[indices] for value in history], dim=1)
    torch.testing.assert_close(fits[1].side_info, (expected_si - factors[:, 1:]) / factors[:, :1])
    assert codec.quantizer.vector_size == 64
    assert codec.quantizer.side_info_size == len(history)
    assert codec.quantizer.num_planes == codec.quantizer.coding_model.num_planes == 3
    assert codec.quantizer.trained_num_planes == 4
    assert tuple(codec.c_cfg.norm_slices) == (slice(0, 32), slice(32, 64))

    payload = codec.encode(target, record)
    delivered = codec.decode(payload, record)
    torch.testing.assert_close(delivered[indices], reconstructed_sample, rtol=0, atol=0)
    full_bins, _, metadata = codec.quantizer.encoding_process(target)
    stages = codec.quantizer.decoding_stages_process((full_bins, metadata))
    torch.testing.assert_close(delivered, stages[-1], rtol=0, atol=0)
    assert record.stage_wmapes == pytest.approx([
        record_reconstruction_metrics(target, stage)["wmape"] for stage in stages])
    assert record.quantizer_prior_rate == pytest.approx(
        sum(record.normal_quantizer_prior_stage_rates) - sum(record.replaced_sample_prior_stage_rates)
        + record.sample_prior_rate)
    assert record.num_planes == len(record.stage_wmapes) == len(record.quantizer_prior_stage_rates) == 3
    record.append_record_to_csv(tmp_path)
    assert "sample_extra_prior_rate" in (tmp_path / "compression_records.csv").read_text()

    if kind is SampledRansRoundCodec:
        outer = decompress_data_list(payload)
        sample = decompress_data_list(outer["sample_payload"])
        normal_bytes = outer["payload_content"][0]
        assert record.coder_rate == pytest.approx(8 * (len(sample.symbols) + len(normal_bytes)) / 64)
        assert sum(record.coder_stage_rates) == pytest.approx(record.coder_rate)
        assert record.server_si_rans_rate == pytest.approx(record.coder_rate)
        assert record.sample_coder_rate == pytest.approx(8 * len(sample.symbols) / 64)
        assert record.symbol_error_rate == 0.0
        assert record.coder_rate >= record.sample_coder_rate
    else:
        normal_bins = decompress_data_list(payload)["payload_content"][0]
        assert normal_bins.shape == (3, 64 - samples)


@pytest.mark.parametrize("kind", [SampledRoundCodec, SampledRansRoundCodec])
@pytest.mark.parametrize("extra", [0, 2])
def test_only_sample_refinement_is_charged(
    kind: type[SampledRoundCodec], extra: int, fits: list[FitInputs],
) -> None:
    """Refinement costs scale with transmitted sample positions, independently of normal planes."""
    codec, target, _ = make_codec(kind, samples=13, extra=extra)
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    if kind is SampledRoundCodec:
        assert packet.symbols.shape == (3 + extra, 13)
    codec.decode(codec.encode(target, record), record)
    assert record.sample_extra_prior_rate == pytest.approx(sum(codec._sample_rates[3:]))
    assert record.sample_num_planes == 3 + extra
    if extra == 0:
        assert record.sample_extra_prior_rate == 0.0
        if kind is SampledRansRoundCodec:
            assert record.sample_extra_coder_rate == 0.0


@pytest.mark.parametrize("token", ["S", "SR"])
def test_protocol_freezes_and_retrains_without_codec_specific_schedule(token: str, fits: list[FitInputs]) -> None:
    """An arbitrary train/F/train/F schedule hands off retained weights and never resamples on F."""
    protocol = create_protocol("wz_cancer", (slice(0, 32), slice(32, 64)))
    assert isinstance(protocol, _WZProtocol)
    trained = f"{token}|bins_per_plane=3|num_planes=3|sample_size=9|layers=1|hidden_dim=4|rans_lanes=2" if token == "SR" else (
        f"{token}|bins_per_plane=3|num_planes=3|sample_size=9|layers=1|hidden_dim=4")
    protocol.set_schedule(5, (trained, "F", "F", trained, "F"))
    for round_id in range(5):
        protocol._recons_history.commit(
            torch.sin(torch.arange(64).float()) + 2 + round_id,
            CompressionRecord(round_id, 0, "I"), Access.TEMPORAL_TOO)
    previous: SampledRoundCodec | None = None
    weights: dict[str, torch.Tensor] = {}
    for round_id in range(5, 10):
        codec = protocol.create_round_codec(round_id, 0)
        assert isinstance(codec, SampledRoundCodec)
        is_frozen = codec.round_name_full == "F"
        if is_frozen:
            assert codec is previous
        target = torch.sin(torch.arange(64).float() + round_id) + 4
        record = codec.create_r_record(round_id, 0)
        before = len(fits)
        codec.pre_encode_training_steps(target)
        payload = codec.encode(target, record)
        delivered = codec.decode(payload, record)
        protocol.after_decode(delivered, record, codec)
        if is_frozen:
            assert len(fits) == before and record.sample_count == 0
            assert "sample_payload" not in decompress_data_list(payload)
            for key, value in weights.items():
                torch.testing.assert_close(codec.quantizer.coding_model.state_dict()[key], value, rtol=0, atol=0)
            if token == "SR":
                assert record.sample_coder_rate == record.sample_extra_coder_rate == 0.0
        else:
            assert record.sample_bootstrap is (previous is None)
            assert len(fits) - before == (2 if previous is None else 1)
            weights = {key: value.clone() for key, value in codec.quantizer.coding_model.state_dict().items()}
        codec.release_round_resources()
        assert codec.quantizer.sample_indices is None and codec._sample_payload is None
        previous = codec


@pytest.mark.parametrize("shared", [False, True])
def test_retained_machine_has_exact_prefix(shared: bool) -> None:
    """Prefix symbols, reconstructions and priors stay identical without removing stage weights."""
    torch.manual_seed(23)
    model = new_rnn_model(4, 3, 2, False, 4, 1, shared_heads=shared).eval()
    x, si = torch.randn(17, 1), torch.randn(17, 2)
    with torch.no_grad():
        bins4, codes4 = model.encode(x)
        stages4, priors4 = model.decode(codes4, si), model.get_priors(codes4, si)
        keys = set(model.state_dict())
        model.num_planes = 3
        bins3, codes3 = model.encode(x)
        for prefix, full in ((bins3, bins4), (model.decode(codes3, si), stages4), (model.get_priors(codes3, si), priors4)):
            for actual, expected in zip(prefix, full[:3], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert set(model.state_dict()) == keys


def test_sample_fit_uses_real_normalization_with_outliers(fits: list[FitInputs]) -> None:
    """The sample and normal decode both preserve the existing outlier metadata path."""
    codec, target, _ = make_codec(SampledRoundCodec, samples=17, outliers=0.5)
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    delivered = codec.decode(codec.encode(target, record), record)
    assert torch.isfinite(delivered).all()
    torch.testing.assert_close(delivered[codec.quantizer.sample_indices], codec.quantizer.sample_reconstruction)


def test_sample_seed_is_payload_owned(fits: list[FitInputs]) -> None:
    """Regenerating sample coordinates is independent of later fitting and global RNG draws."""
    codec, target, _ = make_codec(SampledRoundCodec)
    codec.pre_encode_training_steps(target)
    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    expected = packet.indices()
    torch.manual_seed(987)
    np.random.seed(987)
    torch.testing.assert_close(packet.indices(), expected, rtol=0, atol=0)


def test_preceding_weights_are_used_and_can_supply_a_shorter_prefix(
    fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sample acquisition uses the actual deployed weights, including when extra-plane counts shrink."""
    previous, target, _ = make_codec(SampledRoundCodec, extra=3, shared=False)
    previous.pre_encode_training_steps(target)
    record = previous.create_r_record(5, 0)
    previous.decode(previous.encode(target, record), record)
    expected = {key: value.clone() for key, value in previous.quantizer.coding_model.state_dict().items()}
    previous.release_round_resources()
    current, target, _ = make_codec(SampledRoundCodec, extra=1, shared=False)
    current.set_previous_codec(previous)
    original = current.decode_sample_indices

    def receive(symbols: torch.Tensor | bytes, pilot: WZQuantizerCancer, metadata: PreprocessMetadata) -> torch.Tensor:
        assert pilot.num_planes == pilot.coding_model.num_planes == 4
        for key, value in expected.items():
            torch.testing.assert_close(pilot.coding_model.state_dict()[key], value, rtol=0, atol=0)
        return original(symbols, pilot, metadata)

    monkeypatch.setattr(current, "decode_sample_indices", receive)
    count = len(fits)
    current.pre_encode_training_steps(target)
    assert len(fits) == count + 1
    assert current._sample_fields["sample_bootstrap"] is False
    assert current.quantizer.trained_num_planes == 4


@pytest.mark.parametrize("token", ["S", "SR"])
def test_protocol_records_sample_settings(token: str) -> None:
    """Ordinary protocol configuration records sample acquisition and its training boundary."""
    protocol = create_protocol("wz_cancer", (slice(0, 32), slice(32, 64)))
    trained = f"{token}|bins_per_plane=2|num_planes=3|sample_size=1000|sample_extra_planes=3"
    protocol.set_schedule(5, (trained, "F"))
    facts = protocol.config_snapshot()["round_codecs"][trained]["codec_provenance"]
    assert facts["sample_training_target"] == "decoded_sample_only"
    assert facts["sampling"]["sample_size"] == 1000
    assert facts["sampling"]["sample_extra_planes"] == 3


@pytest.mark.parametrize("kind", [SampledRoundCodec, SampledRansRoundCodec])
def test_fp16_sample_is_received_before_fit_and_charged(kind: type[SampledRoundCodec], fits: list[FitInputs]) -> None:
    """High-precision control fits only serialized sample values and charges their measured wire size."""
    codec, target, _ = make_codec(kind, samples=17, extra=3)
    codec.sampling = SamplingConfig(sample_size=17, sample_extra_planes=3, sample_format="fp16",
                                    sample_training="prefix", sample_fit_epochs=10)
    record = codec.create_r_record(5, 0)
    codec.pre_encode_training_steps(target)
    packet = decompress_data_list(codec._sample_payload)
    assert packet.format == "fp16" and decompress_data_list(packet.symbols).dtype == torch.float16
    assert codec.quantizer.fit_planes == 3 and codec.quantizer.training_config().train_epochs == 10
    factors = fits[-1].factors.repeat_interleave(torch.tensor([
        int((packet.indices() < 32).sum()), int((packet.indices() >= 32).sum())]), dim=0)
    torch.testing.assert_close(fits[-1].target[:, 0],
                               (codec.quantizer.sample_reconstruction - factors[:, 1]) / factors[:, 0])
    delivered = codec.decode(codec.encode(target, record), record)
    torch.testing.assert_close(delivered[packet.indices()], codec.quantizer.sample_reconstruction, rtol=0, atol=0)
    assert record.sample_wmape < .1
    assert record.sample_extra_prior_rate is None and record.sample_num_planes is None
    assert record.sample_prior_rate == pytest.approx(8 * len(packet.symbols) / target.numel())
    if kind is SampledRansRoundCodec:
        assert record.sample_coder_rate == record.sample_prior_rate
        assert record.sample_extra_coder_rate is None and record.symbol_error_rate == 0


def test_prefix_loss_omits_tail_gradients_without_removing_weights() -> None:
    """The deployed prefix gets O's exact objective and discarded independent heads get no gradient."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=3, shared=False)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    quantizer.fit_planes = 3
    inputs, si = torch.randn(13, 1), torch.randn(13, 2)
    config = quantizer.training_config()
    generator = torch.Generator().manual_seed(8)
    torch.manual_seed(42)
    actual = quantizer.compute_loss(model, (inputs,), si, 0, config, None, 1., generator)
    assert model.num_planes == 6 and len(model.binners) == 6
    model.num_planes = 3
    torch.manual_seed(42)
    expected = WZQuantizerCancer.compute_loss(model, (inputs,), si, 0, config, None, 1., generator)
    model.num_planes = 6
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert model.binners[0].weight.grad is not None
    assert all(head.weight.grad is None for head in model.binners[3:])


def test_decoder_only_fit_retains_warm_heads_and_restores_batch(fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh candidates retain warm-state shapes but freeze encoding and validate the deployed prefix."""
    original = WZQuantizerCancer._fit_prepared

    def prepared(self: WZQuantizerCancer, x: torch.Tensor, si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        if not getattr(self, "fit_encoder", True):
            candidate = self._new_coding_model(self.side_info_size)
            candidate.load_state_dict(self.coding_model.state_dict())
            assert candidate.num_planes == self.num_planes == 3 and len(candidate.binners) == 6
            assert self.train_batch_size == 4
            assert all(not p.requires_grad for p in candidate.encoder.parameters())
            assert all(not p.requires_grad for p in candidate.binners.parameters())
            assert all(p.requires_grad for p in candidate.decoder.parameters())
        original(self, x, si)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", prepared)
    codec, target, _ = make_codec(SampledRansRoundCodec, samples=17, extra=3, shared=False)
    codec.sampling = SamplingConfig(sample_size=17, sample_extra_planes=3, sample_training="prefix",
                                    sample_train_encoder=False, sample_fit_batch_size=4)
    codec.pre_encode_training_steps(target)
    assert fits[-1].planes == 3 and codec.quantizer.train_batch_size == 6250
    record = codec.create_r_record(5, 0)
    codec.decode(codec.encode(target, record), record)
    assert record.sample_training_encoder is False and record.sample_training_batch_size == 4
    assert record.symbol_error_rate == 0


@pytest.mark.parametrize("frozen", ["prior", "backbones"])
def test_sample_optimizer_preserves_frozen_parameters(frozen: str) -> None:
    """Actual optimization updates unfrozen heads while preserving the chosen baseline networks."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=3, shared=False)
    q = codec.quantizer
    q.fit_planes, q.fit_prior, q.fit_backbones = 3, frozen != "prior", frozen != "backbones"
    q._set_training_parameters(q.coding_model)
    before = {name: value.clone() for name, value in q.coding_model.state_dict().items()}
    optimizer = torch.optim.Adam(q.coding_model.parameters(), lr=.001)
    loss = q.compute_loss(q.coding_model, (torch.randn(13, 1),), torch.randn(13, 2), 0,
                          q.training_config(), None, 1., torch.Generator().manual_seed(7))
    loss.backward()
    optimizer.step()
    after = q.coding_model.state_dict()
    prefixes = (("conditionalRNN.", "conditionalPrior") if frozen == "prior"
                else ("encoder.", "decoder.", "conditionalRNN."))
    for name in before:
        if name.startswith(prefixes):
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("binner"))


def test_hard_forward_retains_soft_posterior_and_encoder_gradients() -> None:
    """Decoder and prior receive real symbols, while the posterior and backward path remain soft."""
    model = new_rnn_model(4, 3, 2, False, 4, 1).train()
    model.hard_decoder_inputs = True
    x, si = torch.randn(17, 1), torch.randn(17, 2)
    with patch.object(model, "get_priors", wraps=model.get_priors) as prior_call:
        reconstructed, bins, soft, priors = model(x, si, tau=.5)
    hard = [torch.nn.functional.one_hot(symbols, 3).float() for symbols in bins]
    for actual, expected in zip(reconstructed, model.decode(hard, si), strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(prior_call.call_args.kwargs["codes"], hard, strict=True):
        torch.testing.assert_close(actual, expected)
    assert any(((code > 0) & (code < 1)).any() for code in soft)
    sum(stage.square().mean() for stage in reconstructed).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


@pytest.mark.parametrize("kind", [SampledRoundCodec, SampledRansRoundCodec])
@pytest.mark.parametrize("epochs,lr,tau,batch", [(28, .0006807383623518464, .30, 7500), (None, None, None, None), (10, .0002, .7, 4)])
def test_sample_optimizer_overrides_leave_acquisition_and_prior_unchanged(
    kind: type[SampledRoundCodec], epochs: int | None, lr: float | None, tau: float | None, batch: int | None,
    fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overrides affect the received-sample fit and leave acquisition and native-prior configs intact."""
    codec, target, _ = make_codec(kind)
    assert codec.sampling.sample_fit_epochs == 28
    assert codec.sampling.sample_fit_lr == .0006807383623518464
    assert codec.sampling.sample_fit_tau == .30
    assert codec.sampling.sample_fit_batch_size == 7500
    inherited = codec.quantizer.training_config()
    options = codec.sampling.model_copy(update={"sample_fit_epochs": epochs, "sample_fit_lr": lr,
                                              "sample_fit_tau": tau, "sample_fit_batch_size": batch})
    codec.sampling = options
    configs: list[object] = []
    original = WZQuantizerCancer._fit_prepared

    def capture(self: WZQuantizerCancer, *args: object, **kwargs: object) -> None:
        configs.append(self.training_config())
        original(self, *args, **kwargs)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture)
    codec.pre_encode_training_steps(target)
    expected_epochs = epochs if epochs is not None else inherited.train_epochs
    expected_lr = lr if lr is not None else inherited.lr
    expected_tau = tau if tau is not None else inherited.tau
    assert (configs[-1].train_epochs, configs[-1].lr, configs[-1].tau) == (expected_epochs, expected_lr, expected_tau)
    assert configs[-1].train_batch_size == (batch if batch is not None else inherited.train_batch_size)
    assert (configs[0].train_epochs, configs[0].lr, configs[0].tau) == (inherited.train_epochs, inherited.lr, inherited.tau)
    assert codec.c_cfg.model_dump(exclude={"norm_slices", "num_planes"}) == inherited.model_dump(exclude={"norm_slices", "num_planes"})
    assert codec._sample_fields["sample_fit_lr"] == expected_lr
    assert codec._sample_fields["sample_fit_tau"] == expected_tau
    assert SamplingConfig().sample_size == 3_600_000
    assert SamplingConfig().sample_fit_tau == .30
    assert SamplingConfig().sample_fit_epochs == 28
    assert SamplingConfig().sample_fit_lr == .0006807383623518464
    assert SamplingConfig().sample_fit_batch_size == 7500
    assert kind.codec_provenance()["sampling"]["sample_size"] == 3_600_000
    assert kind.codec_provenance()["sampling"]["sample_fit_tau"] == .30
    assert kind.codec_provenance()["sampling"]["sample_fit_epochs"] == 28
    assert kind.codec_provenance()["sampling"]["sample_fit_lr"] == .0006807383623518464
    assert kind.codec_provenance()["sampling"]["sample_fit_batch_size"] == 7500


def test_received_prefix_config_and_provenance() -> None:
    """Fixed-bin fitting is an explicit, constrained sample-fit identity."""
    options = dict(sample_fit_mode="received_prefix_bins", sample_training="prefix", sample_format="lwzq",
                   sample_extra_planes=1, sample_train_encoder=False)
    config = SamplingConfig(**options)
    assert config.sample_fit_mode == "received_prefix_bins"
    facts = SampledRoundCodec.codec_provenance(**options)
    assert facts["sampling"]["sample_fit_mode"] == "received_prefix_bins"
    assert facts["sample_training_target"] == "decoded_sample_only"
    for invalid in (
        {**options, "sample_training": "all"},
        {**options, "sample_format": "fp16"},
        {**options, "sample_extra_planes": 0},
        {**options, "sample_train_encoder": True},
        {**options, "sample_train_prior": False},
        {**options, "sample_train_backbones": False},
    ):
        with pytest.raises(ValidationError):
            SamplingConfig(**invalid)
    with pytest.raises(ValueError, match="outlier_threshold=None"):
        SampledRoundCodec.validate_cfg(**options, outlier_threshold=2.0)

    raw_final = SamplingConfig(**options, sample_final_distortion="raw_mae")
    assert raw_final.sample_final_distortion == "raw_mae"
    mae_final = SamplingConfig(**options, sample_final_distortion="mae")
    assert mae_final.sample_final_distortion == "mae"
    with pytest.raises(ValidationError, match="received_prefix_bins"):
        SamplingConfig(sample_final_distortion="raw_mae")
    with pytest.raises(ValidationError, match="received_prefix_bins"):
        SamplingConfig(sample_final_distortion="mae")


def test_received_prefix_loss_metrics_and_candidate_use_fixed_symbols() -> None:
    """Loss, fit metrics, and candidate reconstruction never regenerate fixed training symbols."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=2, shared=False)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.fit_encoder, quantizer.fit_prior, quantizer.fit_backbones = False, True, True
    quantizer._set_training_parameters(model)
    x, si = torch.randn(13, 1), torch.randn(13, 2)
    bins = torch.stack([torch.arange(13).remainder(model.bins_per_plane + plane)
                        .remainder(model.bins_per_plane) for plane in range(3)], dim=1)
    quantizer.fit_received_prefix_bins = bins
    received = (x, bins)

    original_planes = model.num_planes
    model.num_planes = 3
    was_training = model.training
    model.eval()
    codes = [torch.nn.functional.one_hot(bins[:, plane], model.bins_per_plane).float() for plane in range(3)]
    reconstructions = model.decode(codes, si)
    priors = model.get_priors(codes, si, force_softmax=True)
    model.train(was_training)
    model.num_planes = original_planes
    config = quantizer.training_config()
    denominator = 1.7
    expected = sum(
        (config.reconst_ld / denominator) * torch.nn.functional.mse_loss(reconstructions[plane], x)
        - torch.log(priors[plane][torch.arange(x.shape[0]), bins[:, plane]] + 1e-12).mean()
        for plane in range(3)
    ) / 3
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with (patch.object(model, "encode", side_effect=AssertionError("fixed-bin fit re-encoded teacher labels")),
          patch.object(quantizer, "decode_prepared", return_value=reconstructions[-1])):
        actual = quantizer.compute_loss(model, received, si, 0, config, None, denominator,
                                        torch.Generator().manual_seed(11))
        metrics = quantizer.training_metrics(model, received, si)
        candidate_bins, candidate_reconstruction = quantizer._fit_candidate_data(model, x, si)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert "cumulative_rate=" in metrics
    torch.testing.assert_close(candidate_bins, bins.T)
    torch.testing.assert_close(candidate_reconstruction, reconstructions[-1])

    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    optimizer.zero_grad()
    actual.backward()
    optimizer.step()
    after = model.state_dict()
    for name in before:
        if name.startswith(("encoder.", "binner", "binners.")):
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
    assert any(not torch.equal(after[name], before[name]) for name in before
               if name.startswith(("decoder.", "reconstructor", "conditional")))


def test_received_prefix_rows_keep_bins_aligned_and_block_bad_dedup() -> None:
    """The generic row deduplicator sees fixed symbols alongside teacher values and decoder SI."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=2)
    quantizer = codec.quantizer
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    x = torch.zeros(4, 1)
    bins = torch.tensor([[0, 1, 2], [0, 1, 2], [1, 1, 2], [0, 1, 2]])
    quantizer.fit_received_prefix_bins = bins
    inputs = quantizer._fit_training_inputs(x)
    torch.testing.assert_close(inputs[1], bins)
    si = torch.zeros(4, 2)
    _, groups, _ = unique_row_groups((*inputs, si), quantizer.match_bits)
    assert groups[0] == groups[1] == groups[3]
    assert groups[2] != groups[0]


def test_received_prefix_final_raw_scales_stay_aligned_through_deduplication() -> None:
    """Final raw-loss scales share row sampling and deduplication with teacher and received bins."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1)
    quantizer = codec.quantizer
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.fit_final_distortion = "raw_mae"
    x = torch.tensor([[.1], [.1], [.2], [.2], [.2]])
    bins = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    scales = torch.tensor([[1.], [3.], [2.], [2.], [2.]])
    quantizer.fit_received_prefix_bins = bins
    quantizer._fit_final_row_scales = scales
    inputs = quantizer._fit_training_inputs(x)
    assert len(inputs) == 3
    sample_indices = torch.tensor([4, 1, 3, 0])
    for actual, expected in zip((value[sample_indices] for value in inputs), (x, bins, scales), strict=True):
        torch.testing.assert_close(actual, expected[sample_indices])
    _, groups, collisions = unique_row_groups((*inputs, torch.zeros(5, 2)), quantizer.match_bits)
    assert collisions.numel() == 0
    assert groups[0] != groups[1]
    assert groups[2] == groups[3] == groups[4]


def test_received_prefix_raw_mae_uses_affine_scales_and_preserves_prior_rate() -> None:
    """Final raw MAE is affine-equivalent, unit-invariant, and leaves actual-symbol prior CE intact."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False, refiner=True)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    model.to(quantizer.device)
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.fit_final_distortion = "raw_mae"
    quantizer.fit_encoder = False
    quantizer._set_training_parameters(model)
    x = torch.randn(8, 1, device=quantizer.device)
    side_info = torch.randn(8, 2, device=quantizer.device)
    bins = torch.arange(8, device=quantizer.device).remainder(model.bins_per_plane).unsqueeze(1).repeat(1, 3)
    scales = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4, 2e-4, 3e-4, 4e-4, 5e-4], device=quantizer.device).unsqueeze(1)
    denominator = 2.7e-4
    quantizer._fit_final_row_scales = scales
    quantizer._fit_final_mae_denominator = denominator
    training_input = (x, bins, scales)
    config = quantizer.training_config()
    model.eval()
    original_planes = model.num_planes
    model.num_planes = 3
    codes = [torch.nn.functional.one_hot(bins[:, plane], model.bins_per_plane).float() for plane in range(3)]
    decoded = model.decode(codes, side_info)
    priors = model.get_priors(codes, side_info, force_softmax=True)
    model.num_planes = original_planes
    model.train()
    rows = torch.arange(x.shape[0], device=x.device)
    expected = sum(
        2 * config.reconst_ld * (decoded[plane] - x).abs().flatten(1).mean(1).mul(scales.flatten()).mean()
        / denominator
        - torch.log(priors[plane][rows, bins[:, plane]] + 1e-12).mean()
        for plane in range(3)
    ) / 3
    actual = quantizer.compute_loss(model, training_input, side_info, 0, config, None, 1.,
                                    torch.Generator().manual_seed(5))
    torch.testing.assert_close(actual, expected)

    unit_scale = 1e-5
    quantizer._fit_final_row_scales = scales * unit_scale
    quantizer._fit_final_mae_denominator = denominator * unit_scale
    scaled = quantizer.compute_loss(model, (x, bins, scales * unit_scale), side_info, 0,
                                    config, None, 1., torch.Generator().manual_seed(5))
    torch.testing.assert_close(scaled, actual, rtol=1e-5, atol=1e-6)
    assert model.num_planes == 4 and model.use_refiner
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith(("encoder.", "binner", "binners.")))
    assert any(parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith(("decoder.", "reconstructor", "conditional")))


def test_received_prefix_normalized_mae_uses_weighted_teacher_error_and_prior_rate() -> None:
    """Final normalized MAE reuses acquisition scaling and counts deduplicated rows by multiplicity."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False, refiner=True)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    model.to(quantizer.device)
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.fit_final_distortion = "mae"
    quantizer.fit_encoder = False
    quantizer._set_training_parameters(model)
    x = torch.tensor([[-2.], [-.5], [1.], [4.]], device=quantizer.device)
    side_info = torch.randn(4, 2, device=quantizer.device)
    bins = torch.arange(4, device=quantizer.device).remainder(model.bins_per_plane).unsqueeze(1).repeat(1, 3)
    weights = torch.tensor([3., 1., 2., 4.], device=quantizer.device)
    denominator = float(x.abs().mean()) + 1e-8
    quantizer._fit_final_mae_denominator = denominator
    training_input = (x, bins)
    config = quantizer.training_config()
    model.eval()
    original_planes = model.num_planes
    model.num_planes = 3
    codes = [torch.nn.functional.one_hot(bins[:, plane], model.bins_per_plane).float() for plane in range(3)]
    decoded = model.decode(codes, side_info)
    priors = model.get_priors(codes, side_info, force_softmax=True)
    model.num_planes = original_planes
    model.train()
    rows = torch.arange(x.shape[0], device=x.device)
    expected = sum(
        2 * config.reconst_ld * ((decoded[plane] - x).abs().flatten(1).mean(1) * weights).sum()
        / weights.sum() / denominator
        - ((priors[plane][rows, bins[:, plane]] + 1e-12).log() * weights).sum() / weights.sum()
        for plane in range(3)
    ) / 3
    actual = quantizer.compute_loss(model, training_input, side_info, 0, config, weights, 1.,
                                    torch.Generator().manual_seed(5))
    torch.testing.assert_close(actual, expected)
    assert model.num_planes == 4 and model.use_refiner


def test_received_prefix_normalized_mae_uses_only_decoded_normalized_teacher(fits: list[FitInputs]) -> None:
    """The final MAE denominator comes from normalized decoded teacher rows, not raw current labels."""
    codec, current_target, _ = make_codec(SampledRoundCodec, samples=11, extra=1, shared=False)
    codec.sampling = SamplingConfig(sample_size=11, sample_extra_planes=1, sample_training="prefix",
                                   sample_fit_mode="received_prefix_bins", sample_train_encoder=False,
                                   sample_final_distortion="mae", sample_coordinate_seed=73)
    codec.pre_encode_training_steps(current_target)
    assert len(fits) == 2 and fits[0].final_distortion == "mse" and fits[1].final_distortion == "mae"
    teacher = codec.quantizer.sample_reconstruction
    assert teacher is not None
    normalized_teacher = fits[1].target
    assert fits[1].final_row_scales is None
    assert np.isclose(fits[1].final_denominator, float(normalized_teacher.float().abs().mean()) + 1e-8)
    assert not np.isclose(fits[1].final_denominator, float(current_target[codec.quantizer.sample_indices].abs().mean()))
    assert codec._sample_fields["sample_final_distortion_weight_source"] == "decoded_teacher_normalized_mean_abs"


def test_received_prefix_raw_mae_weights_only_decoded_teacher(
    fits: list[FitInputs],
) -> None:
    """The final raw denominator and row scales come from decoder-visible sample metadata and teacher."""
    codec, current_target, _ = make_codec(SampledRoundCodec, samples=11, extra=1, shared=False)
    codec.sampling = SamplingConfig(sample_size=11, sample_extra_planes=1, sample_training="prefix",
                                   sample_fit_mode="received_prefix_bins", sample_train_encoder=False,
                                   sample_final_distortion="raw_mae", sample_coordinate_seed=73)
    codec.pre_encode_training_steps(current_target)
    packet: SamplePayload = decompress_data_list(codec._sample_payload)
    assert len(fits) == 2 and fits[0].final_distortion == "mse" and fits[1].final_distortion == "raw_mae"
    factors = packet.metadata.norm_factors
    indices = packet.indices()
    sizes = [int(((indices >= part.start) & (indices < part.stop)).sum()) for part in codec.c_cfg.norm_slices]
    sizes = [size for size in sizes if size]
    expected_scales = torch.repeat_interleave(factors[:, :1], torch.tensor(sizes), dim=0)
    torch.testing.assert_close(fits[1].final_row_scales, expected_scales)
    teacher = codec.quantizer.sample_reconstruction
    assert teacher is not None
    tiny = torch.finfo(torch.float32).tiny
    assert fits[1].final_denominator == max(float(teacher.float().abs().mean()), tiny)
    assert not np.isclose(fits[1].final_denominator, float(current_target[indices].abs().mean()))
    assert codec._sample_fields["sample_final_distortion_weight_source"] == (
        "received_norm_factors_and_decoded_teacher_raw_mean_abs")


def test_final_raw_mae_keeps_acquisition_state_payload_bins_and_rates_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing only final-S distortion cannot alter the frozen acquisition pilot or its wire symbols."""
    captured: list[tuple[dict[str, torch.Tensor], str, str]] = []

    def capture_fit(self: WZQuantizerCancer, x: torch.Tensor,
                    si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        captured.append(({name: value.detach().cpu().clone()
                          for name, value in self.coding_model.state_dict().items()},
                         self.fit_mode, self.fit_final_distortion))
        self.quantizer_prior_stage_rates = tuple(1.0 for _ in range(self.num_planes))

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture_fit)
    outcomes: list[tuple[SamplePayload, tuple[float, ...], torch.Tensor, tuple[str, str]]] = []
    for final_distortion in ("mse", "mae", "raw_mae"):
        torch.manual_seed(229)
        codec, target, _ = make_codec(SampledRoundCodec, samples=11, extra=1, shared=False)
        codec.sampling = SamplingConfig(
            sample_size=11, sample_extra_planes=1, sample_training="prefix", sample_format="lwzq",
            sample_coordinate_seed=91, sample_fit_mode="received_prefix_bins", sample_train_encoder=False,
            sample_final_distortion=final_distortion,
        )
        # Exercise configuration before acquisition; the pilot must still force its prior MSE behavior.
        codec._configure_sample_fit()
        codec.pre_encode_training_steps(target)
        outcomes.append((decompress_data_list(codec._sample_payload), tuple(codec._sample_rates),
                         captured[-2][0], captured[-2][1:]))

    assert len(captured) == 6
    assert [captured[index][1:] for index in (0, 2, 4)] == [("reencode_teacher", "mse")] * 3
    for outcome in outcomes[1:]:
        assert all(torch.equal(outcomes[0][2][name], outcome[2][name]) for name in outcomes[0][2])
        first, second = outcomes[0][0], outcome[0]
        assert (first.vector_size, first.sample_size, first.seed, first.format) == (
            second.vector_size, second.sample_size, second.seed, second.format)
        torch.testing.assert_close(first.symbols, second.symbols, rtol=0, atol=0)
        assert outcomes[0][1] == outcome[1]
    assert [outcome[3] for outcome in outcomes] == [("reencode_teacher", "mse")] * 3


def test_received_prefix_raw_mae_keeps_refiner_metrics_and_candidate_rates_on_prefix() -> None:
    """Raw distortion does not expand candidate decode, prior rates, or fit metrics past the received prefix."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False, refiner=True)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    model.to(quantizer.device)
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.fit_final_distortion = "raw_mae"
    x, side_info = torch.randn(8, 1), torch.randn(8, 2)
    bins = torch.arange(8).remainder(model.bins_per_plane).unsqueeze(1).repeat(1, 3)
    scales = torch.full_like(x, 1e-5)
    quantizer.fit_received_prefix_bins = bins
    quantizer._fit_final_row_scales = scales
    quantizer._fit_final_mae_denominator = 1e-5
    inputs = tuple(value.to(quantizer.device) for value in (x, bins, scales))
    x, bins, scales = inputs
    side_info = side_info.to(quantizer.device)
    metrics = quantizer.training_metrics(model, inputs, side_info)
    candidate_bins, reconstruction = quantizer._fit_candidate_data(model, x, side_info)
    prior, marginal = quantizer._fit_candidate_rates(model, candidate_bins, side_info)
    stage_prior = quantizer._fit_prior_for_bins(model, candidate_bins, side_info)
    assert metrics.count("hard_mse=") == 1 and metrics.count("cumulative_rate=") == 1
    assert candidate_bins.shape == (3, 8) and reconstruction.numel() == x.numel()
    assert stage_prior.shape == (3, 8, model.bins_per_plane)
    assert np.isfinite(prior) and np.isfinite(marginal)
    assert model.num_planes == 4 and model.use_refiner


def test_received_prefix_refiner_capacity_is_scoped_to_the_deployed_prefix() -> None:
    """A full refiner model evaluates loss diagnostics and candidate rates on only its 3-plane prefix."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False, refiner=True)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    model.to(quantizer.device)
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    x, si = torch.randn(8, 1, device=quantizer.device), torch.randn(8, 2, device=quantizer.device)
    bins = torch.arange(8, device=quantizer.device).remainder(model.bins_per_plane).unsqueeze(1).repeat(1, 3)
    quantizer.fit_received_prefix_bins = bins
    assert model.num_planes == 4 and model.use_refiner

    metrics = quantizer.training_metrics(model, (x, bins), si)
    candidate_bins, reconstruction = quantizer._fit_candidate_data(model, x, si)
    prior, marginal = quantizer._fit_candidate_rates(model, candidate_bins, si)
    stage_prior = quantizer._fit_prior_for_bins(model, candidate_bins, si)
    assert metrics.count("hard_mse=") == 1 and metrics.count("cumulative_rate=") == 1
    assert candidate_bins.shape == (3, 8) and reconstruction.numel() == x.numel()
    assert stage_prior.shape == (3, 8, model.bins_per_plane)
    assert np.isfinite(prior) and np.isfinite(marginal)
    assert model.num_planes == 4 and model.use_refiner


def test_received_prefix_candidate_retry_uses_only_deployed_refiner_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual candidate acceptance loop evaluates a full refiner model on supplied prefix bins."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False, refiner=True)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    model.to(quantizer.device)
    quantizer.fit_mode, quantizer.fit_planes = "received_prefix_bins", 3
    quantizer.vector_size = 8
    x = torch.randn(8, 1, device=quantizer.device)
    si = torch.randn(8, 2, device=quantizer.device)
    bins = torch.arange(8, device=quantizer.device).remainder(model.bins_per_plane).unsqueeze(1).repeat(1, 3)
    quantizer.fit_received_prefix_bins = bins
    assert model.num_planes == 4 and model.use_refiner

    def train_loop(*args: object, **kwargs: object) -> float:
        training_inputs, side_info = args[3], args[4]
        assert isinstance(training_inputs, tuple) and torch.equal(training_inputs[1], bins)
        kwargs["metrics_fn"](model, training_inputs, side_info)
        return 1.0

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train_loop)

    def rates(candidate: RefinedEncoderDecoderLayeredRNN, candidate_bins: torch.Tensor,
              side_info: torch.Tensor) -> tuple[float, float]:
        assert candidate.num_planes == 3 and candidate_bins.shape == (3, 8)
        return 1.0, 2.0

    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(rates))
    fitted, loss = quantizer._train_finite_candidate(x, si, 1.0, None, 1, "fixed-prefix")
    assert fitted is model and loss == 1.0
    assert fitted.num_planes == 4 and fitted.use_refiner


def test_acquisition_is_invariant_to_final_fit_mode_and_encoder_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final-S fitting variants share the same acquisition pilot state, payload, and reported rate."""
    captured: list[tuple[dict[str, torch.Tensor], str, bool]] = []

    def capture_fit(self: WZQuantizerCancer, x: torch.Tensor,
                    si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        captured.append(({name: value.detach().cpu().clone()
                          for name, value in self.coding_model.state_dict().items()},
                         self.fit_mode, self.fit_encoder))
        self.quantizer_prior_stage_rates = tuple(1.0 for _ in range(self.num_planes))

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture_fit)
    outcomes: list[tuple[SamplePayload, tuple[float, ...], dict[str, torch.Tensor], str, bool]] = []
    for fit_mode, train_encoder in (("reencode_teacher", True), ("received_prefix_bins", False)):
        torch.manual_seed(29)
        codec, target, _ = make_codec(SampledRoundCodec, samples=11, extra=1)
        codec.sampling = SamplingConfig(
            sample_size=11, sample_extra_planes=1, sample_training="prefix", sample_format="lwzq",
            sample_coordinate_seed=91, sample_fit_mode=fit_mode, sample_train_encoder=train_encoder,
        )
        codec._configure_sample_fit()
        codec.pre_encode_training_steps(target)
        outcomes.append((decompress_data_list(codec._sample_payload), tuple(codec._sample_rates), captured[-2][0],
                         codec.quantizer.fit_mode, codec.quantizer.fit_encoder))

    assert len(captured) == 4
    assert [captured[index][1:] for index in (0, 2)] == [("reencode_teacher", True)] * 2
    assert all(torch.equal(outcomes[0][2][name], outcomes[1][2][name]) for name in outcomes[0][2])
    first_payload, second_payload = outcomes[0][0], outcomes[1][0]
    assert (first_payload.vector_size, first_payload.sample_size, first_payload.seed, first_payload.format) == (
        second_payload.vector_size, second_payload.sample_size, second_payload.seed, second_payload.format)
    assert torch.equal(first_payload.symbols, second_payload.symbols)
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][3:] == ("reencode_teacher", True)
    assert outcomes[1][3:] == ("received_prefix_bins", False)


def test_reencode_teacher_encoder_freeze_control_is_bitwise_stable() -> None:
    """The freeze-only MSE control still re-encodes labels but never changes encoder weights."""
    codec, _, _ = make_codec(SampledRoundCodec, extra=1, shared=False)
    quantizer = codec.quantizer
    model = quantizer.coding_model
    quantizer.fit_mode, quantizer.fit_planes, quantizer.fit_encoder = "reencode_teacher", 3, False
    quantizer._set_training_parameters(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    optimizer.zero_grad()
    loss = quantizer.compute_loss(model, (torch.randn(12, 1),), torch.randn(12, 2), 0,
                                 quantizer.training_config(), None, 1., torch.Generator().manual_seed(2))
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    for name in before:
        if name.startswith(("encoder.", "binner", "binners.")):
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
    assert any(not torch.equal(after[name], before[name]) for name in before
               if name.startswith(("decoder.", "reconstructor", "conditional")))


def test_received_prefix_bins_match_frozen_encoder_on_full_raw_target(
    fits: list[FitInputs], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both final losses preserve the base encoder and full-target symbols after received-prefix fitting."""
    captured_inputs: list[tuple[torch.Tensor, ...]] = []
    original = WZQuantizerCancer._fit_prepared

    def capture(self: WZQuantizerCancer, x: torch.Tensor,
                si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        if getattr(self, "fit_mode", "reencode_teacher") == "received_prefix_bins":
            captured_inputs.append(self._fit_training_inputs(x))
        original(self, x, si)

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", capture)
    outcomes: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, SamplePayload, str]] = []
    for final_distortion in ("mse", "raw_mae"):
        torch.manual_seed(770)
        codec, target, _ = make_codec(SampledRoundCodec, samples=11, extra=1, shared=False)
        codec.sampling = SamplingConfig(sample_size=11, sample_extra_planes=1, sample_training="prefix",
                                       sample_fit_mode="received_prefix_bins", sample_train_encoder=False,
                                       sample_final_distortion=final_distortion, sample_coordinate_seed=73)
        codec._previous_state = {name: value.detach().cpu().clone()
                                 for name, value in codec.quantizer.coding_model.state_dict().items()}
        codec._previous_planes = 4
        normalized = target.clone()
        for vector_slice in codec.c_cfg.norm_slices:
            scale, center = normalization_params(target[vector_slice])
            normalized[vector_slice] = (target[vector_slice] - center) / scale
        model = codec.quantizer.coding_model.eval()
        with torch.inference_mode():
            baseline_bins = torch.stack(model.encode(normalized.unsqueeze(-1), force_softmax=True)[0]).cpu()
        baseline_encoder = codec.quantizer.coding_model.encoder_state_dict()
        codec.pre_encode_training_steps(target)
        packet: SamplePayload = decompress_data_list(codec._sample_payload)
        indices = packet.indices()
        received_bins = packet.symbols
        assert isinstance(received_bins, torch.Tensor)
        final_inputs = captured_inputs[-1]
        torch.testing.assert_close(final_inputs[1], received_bins[:codec.normal_planes].T)
        if final_distortion == "raw_mae":
            assert len(final_inputs) == 3
            torch.testing.assert_close(final_inputs[2], fits[-1].final_row_scales)
        torch.testing.assert_close(received_bins[:codec.normal_planes].long(), baseline_bins[:codec.normal_planes, indices])
        final_encoder = codec.quantizer.coding_model.encoder_state_dict()
        assert final_encoder.keys() == baseline_encoder.keys()
        for name in baseline_encoder:
            torch.testing.assert_close(final_encoder[name], baseline_encoder[name], rtol=0, atol=0)
        final_bins, _, _ = codec.quantizer.encoding_process(target)
        torch.testing.assert_close(final_bins.long(), baseline_bins[:codec.normal_planes], rtol=0, atol=0)
        assert codec._sample_fields["sample_fit_mode"] == "received_prefix_bins"
        assert codec._sample_fields["sample_training_encoder"] is False
        outcomes.append((final_encoder, final_bins, baseline_bins, packet, final_distortion))

    first, second = outcomes
    assert first[4:] == ("mse",) and second[4:] == ("raw_mae",)
    assert first[0].keys() == second[0].keys()
    for name in first[0]:
        torch.testing.assert_close(first[0][name], second[0][name], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
