"""Check causal sample fitting, prefix transport, and protocol-owned S/SR reuse."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from Experiments.Protocol_test.sampled_codec import SampledRecord, SampledRoundCodec, SampledRansRoundCodec, SamplePayload, SamplingConfig
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, record_reconstruction_metrics
from FL_code.FL_core.utils import decompress_data_list, normalization_params
from FL_code.cancer_protocol.cancer_protocol import _WZProtocol
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.wz_quantizer import PreprocessMetadata, SideInfoMatrix, WZQuantizerCancer, _SideInfoRows, new_rnn_model


@dataclass
class FitInputs:
    """Captured normalized rows consumed by O's fitter while exercising real model inference."""

    target: torch.Tensor
    side_info: torch.Tensor
    planes: int
    factors: torch.Tensor | None


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FitInputs]]:
    """Replace only expensive fitting; preprocessing, networks, serialization and coding stay real."""
    captured: list[FitInputs] = []
    torch.manual_seed(19)

    def fit(self: WZQuantizerCancer, x: torch.Tensor, si: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        captured.append(FitInputs(x.detach().cpu().clone(), si[:].cpu().clone(), self.num_planes,
                                 getattr(self, "normalization_factors", None)))
        with torch.no_grad():
            # Different fits must have different reconstructions to expose missing sample replacement.
            if self.coding_model.shared_decoder:
                self.coding_model.reconstructor.bias.fill_(len(captured) / 7)
            else:
                for head in self.coding_model.reconstructors:
                    head.bias.fill_(len(captured) / 7)
        self.quantizer_prior_stage_rates = tuple(1.0 for _ in range(self.num_planes))

    monkeypatch.setattr(WZQuantizerCancer, "_fit_prepared", fit)
    monkeypatch.setattr(SampledRoundCodec, "_fine_tune_native_prior", False)
    yield captured


def make_codec(
    kind: type[SampledRoundCodec], samples: int = 11, extra: int = 1,
    shared: bool = True, outliers: float | None = None,
) -> tuple[SampledRoundCodec, torch.Tensor, list[HistoryVector]]:
    """Build a small two-layer problem with deterministic, nonconstant target and history."""
    target = torch.sin(torch.arange(64).float()) * 2 + 3
    history = [HistoryVector(torch.arange(64).float() / 20 + shift) for shift in (1, 2)]
    options = dict(bins_per_plane=3, num_planes=3, sd_slices=(slice(0, 32), slice(32, 64)),
                   hidden_dim=4, layers=1, sample_size=samples, sample_extra_planes=extra,
                   shared_heads=shared, outlier_threshold=outliers)
    if kind is SampledRansRoundCodec:
        options["rans_lanes"] = 2
    codec = kind(options, kind.round_name)
    codec.build_quantizer(None, history)
    return codec, target, history


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
