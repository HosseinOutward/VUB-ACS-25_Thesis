"""Check the dedicated refiner's prefix, training, causal priors and staged wire format."""

from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from Experiments.Protocol_test.sampled_codec import RefinedEncoderDecoderLayeredRNN
from FL_code.cancer_protocol.bin_coder.rans_coder import RansConfig, decode, encode
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("marginal", [False, True])
def test_refiner_preserves_ordinary_prefix_and_partial_refinement(shared: bool, marginal: bool) -> None:
    """Both switches preserve exact ordinary outputs and weights; partial tails retain their prefix."""
    torch.manual_seed(23)
    options = dict(input_dim=1, side_info_size=2, bins_per_plane=3, num_planes=3, layers=2, hidden_dim=5,
                   shared_encoder=shared, shared_decoder=shared, shared_priors=shared, marginal=marginal)
    model = RefinedEncoderDecoderLayeredRNN(**options, refinement_planes=2).eval()
    ordinary = EncoderDecoderLayeredRNN(**options).eval()
    ordinary.load_state_dict({key: value for key, value in model.state_dict().items()
                             if not key.startswith("refiner.")})
    x, y = torch.randn(17, 1), torch.randn(17, 2)
    expected = ordinary(x, y)
    keys = set(model.state_dict())
    model.use_refiner = True
    full = model(x, y)
    assert model.num_planes == 5
    assert not ({id(parameter) for name, parameter in model.named_parameters() if not name.startswith("refiner.")}
                & {id(parameter) for parameter in model.refiner.parameters()})
    for outputs, prefix in zip(full, expected, strict=True):
        assert len(outputs) == 5
        for actual, reference in zip(outputs[:3], prefix, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    soft_bins, soft_codes = model.encode(x, force_softmax=True)
    for actual, reference in zip(soft_bins, full[1], strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert any(((code > 0) & (code < 1)).any() for code in soft_codes)
    for planes in (4, 3):
        model.num_planes = planes
        for outputs, all_outputs in zip(model(x, y), full, strict=True):
            for actual, reference in zip(outputs, all_outputs[:planes], strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    model.use_refiner = False
    assert model.num_planes == 3 and set(model.state_dict()) == keys
    assert model.encoder_state_dict().keys() == ordinary.encoder_state_dict().keys()
    assert model.decoder_state_dict(True).keys() == ordinary.decoder_state_dict(True).keys()


@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_standard_quantizer_loss_trains_both_networks(hard: bool, shared: bool) -> None:
    """The existing joint loss updates independent base and refiner networks with either input policy."""
    torch.manual_seed(43)
    model = RefinedEncoderDecoderLayeredRNN(
        1, 2, 3, 3, 1, 5, shared_encoder=shared, shared_decoder=shared, shared_priors=shared,
        refinement_planes=2, use_refiner=True,
    ).train()
    model.hard_decoder_inputs = hard
    x, y = torch.randn(19, 1), torch.randn(19, 2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    cfg = WZcfgQuant(bins_per_plane=3, num_planes=5)
    loss = WZQuantizerCancer.compute_loss(model, (x,), y, 0, cfg, None, 1., torch.Generator())
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for prefix in ("encoder.", "decoder.", "conditionalRNN.",
                   "refiner.encoder.", "refiner.decoder.", "refiner.conditionalRNN."):
        assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
                   if name.startswith(prefix))


def test_frozen_prefix_uses_deployment_symbols_and_keeps_refiner_gradients() -> None:
    """Refiner fitting sees stable prefix symbols while its own encoder remains trainable."""
    torch.manual_seed(49)
    model = RefinedEncoderDecoderLayeredRNN(
        1, 2, 3, 3, 1, 5, refinement_planes=2, use_refiner=True,
    )
    model.hard_decoder_inputs = True
    model.frozen_prefix = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("refiner."))
    x, y = torch.randn(29, 1), torch.randn(29, 2)

    model.eval()
    deployment_bins, _ = model.encode(x)
    model.train()
    first_bins, _ = model.encode(x)
    second_bins, _ = model.encode(x)
    for stage in range(model.base_num_planes):
        torch.testing.assert_close(first_bins[stage], deployment_bins[stage], rtol=0, atol=0)
        torch.testing.assert_close(second_bins[stage], deployment_bins[stage], rtol=0, atol=0)

    _, codes = model.encode(x)
    first_prior = model.get_priors(codes, y)
    second_prior = model.get_priors(codes, y)
    for stage in range(model.base_num_planes):
        torch.testing.assert_close(first_prior[stage], second_prior[stage], rtol=0, atol=0)

    before = [parameter.detach().clone() for parameter in model.refiner.encoder.parameters()]
    optimizer = torch.optim.Adam(model.refiner.parameters(), lr=.01)
    optimizer.zero_grad()
    reconstructed, *_ = model(x, y)
    reconstructed[-1].square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.refiner.encoder.parameters())
    optimizer.step()
    assert any(not torch.equal(value, parameter) for value, parameter in zip(
        before, model.refiner.encoder.parameters(), strict=True))


@pytest.mark.parametrize("marginal", [False, True])
def test_refiner_priors_are_causal_and_rans_decodes(marginal: bool) -> None:
    """Future symbols cannot change a stage prior, and both ends reproduce the same five-stage stream."""
    torch.manual_seed(47)
    model = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 3, 1, 5, marginal=marginal,
                                          refinement_planes=2, use_refiner=True).eval()
    y = torch.randn(23, 2)
    bins, codes = model.encode(torch.randn(23, 1))
    reference = model.get_priors(codes, y)
    for stage in range(model.num_planes):
        changed = list(codes[:stage]) + [code.roll(1, dims=-1) for code in codes[stage:]]
        torch.testing.assert_close(model.get_priors(changed, y)[stage], reference[stage], rtol=0, atol=0)
    if marginal:
        for actual, expected in zip(model.get_priors(codes), model.get_priors(codes, y + 100), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert not torch.equal(model.get_priors(codes, y + 100)[3], reference[3])
    changed_prefix = [code.roll(1, dims=-1) for code in codes[:3]] + codes[3:]
    assert not torch.equal(model.get_priors(changed_prefix, y)[3], reference[3])

    def prior(decoded: torch.Tensor, stage: int) -> torch.Tensor:
        context = [F.one_hot(symbols.long(), 3).float() for symbols in decoded]
        return model.get_priors(context, y)[stage]

    symbols = torch.stack(bins)
    with torch.inference_mode():
        packet = encode(symbols, prior, RansConfig(lanes=2))
        torch.testing.assert_close(decode(packet, prior), symbols, rtol=0, atol=0)


def test_prior_copy_and_wire_states_cover_both_independent_phases() -> None:
    """Prior copies release both coders, and transmitted decoder weights reproduce all reconstructions."""
    torch.manual_seed(45)
    model = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 3, 1, 5,
                                          refinement_planes=2, use_refiner=True).eval()
    x, y = torch.randn(13, 1), torch.randn(13, 2)
    _, codes = model.encode(x)
    prior = model.copy_for_prior_retraining()
    assert prior.encoder is prior.decoder is prior.refiner.encoder is prior.refiner.decoder is None
    assert all("conditional" in name for name in prior.state_dict())
    assert {id(parameter) for parameter in prior.prior_parameters()} == {
        id(parameter) for parameter in prior.parameters()}
    for actual, expected in zip(prior.get_priors(codes, y), model.get_priors(codes, y), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    decoder = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 3, 1, 5,
                                            refinement_planes=2, use_refiner=True).eval()
    encoder_state, decoder_state = model.encoder_state_dict(), model.decoder_state_dict(True)
    assert not set(encoder_state) & set(decoder_state)
    assert set(encoder_state) | set(decoder_state) == set(model.state_dict())
    assert not decoder.load_state_dict(decoder_state, strict=False).unexpected_keys
    for actual, expected in zip(decoder.decode(codes, y), model.decode(codes, y), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(decoder.get_priors(codes, y), model.get_priors(codes, y), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_existing_prior_fitter_optimizes_and_installs_both_priors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The normal prior-fitting entry point receives both networks' parameters and preserves both coders."""
    torch.manual_seed(46)
    model = RefinedEncoderDecoderLayeredRNN(1, 2, 3, 3, 1, 5,
                                          refinement_planes=2, use_refiner=True).eval()
    y = torch.randn(19, 2)
    bins, codes = model.encode(torch.randn(19, 1))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    cfg = WZcfgQuant(bins_per_plane=3, num_planes=5, hidden_dim=5, layers=1, prior_retrain_epochs=1)

    def train(
        loss_fn: Callable[..., torch.Tensor], candidate: EncoderDecoderLayeredRNN,
        parameters: Iterator[nn.Parameter], inputs: tuple[torch.Tensor, torch.Tensor],
        side_info: torch.Tensor, config: WZcfgQuant, batch_size: int, *, label: str,
    ) -> float:
        candidate.train()
        optimizer = torch.optim.Adam(parameters, lr=.01)
        loss = loss_fn(candidate, inputs, side_info, 0, config, None)
        loss.backward()
        optimizer.step()
        return loss.item()

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda *args: (0., 1.)))
    result = PriorCalculator.fit_prior_model(torch.stack(bins), torch.stack(codes), y, cfg, model, (None,), 1)
    assert result.model is model
    after = model.state_dict()
    for prefix in ("conditionalRNN.", "refiner.conditionalRNN."):
        assert any(not torch.equal(value, before[name]) for name, value in after.items() if name.startswith(prefix))
    for name, value in after.items():
        if "conditional" not in name:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
