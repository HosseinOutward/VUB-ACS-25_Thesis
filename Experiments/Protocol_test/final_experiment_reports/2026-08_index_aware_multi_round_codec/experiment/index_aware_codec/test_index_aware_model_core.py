from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.wz_quantizer import WZQuantizerCancer, WZcfgQuant

from .index_aware_model import (
    HashedIndexConditioner,
    IndexAwareEncoderDecoderLayeredRNN,
)


def _conditioner() -> HashedIndexConditioner:
    conditioner = HashedIndexConditioner(
        hash_table_count=2,
        hash_dim=2,
        route_count=3,
        embedding_dim=4,
        hash_seed=7,
    )
    conditioner.set_vector_size(65_537)
    return conditioner


def _model() -> IndexAwareEncoderDecoderLayeredRNN:
    model = IndexAwareEncoderDecoderLayeredRNN(
        side_info_size=2,
        bins_per_plane=3,
        num_planes=2,
        layers=2,
        hidden_dim=5,
        shared_encoder=True,
        shared_decoder=True,
        hash_table_count=2,
        hash_dim=2,
        route_count=3,
        embedding_dim=4,
        hash_seed=7,
    )
    model.set_vector_size(8)
    return model


def _v2_conditioner(**updates: object) -> HashedIndexConditioner:
    options: dict[str, object] = {
        "hash_table_count": 2,
        "hash_dim": 3,
        "route_count": 4,
        "embedding_dim": 3,
        "hash_seed": 7,
        "routing_version": "v2",
        "routing_dense_warmup_fraction": 0.5,
        "routing_top_k": 2,
        "index_dropout": 0.0,
    }
    conditioner = HashedIndexConditioner(**(options | updates))
    conditioner.set_vector_size(2**24)
    return conditioner


def _indexed_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(4, dtype=torch.float32)[:, None]
    x = torch.cat([torch.linspace(-1, 1, 4)[:, None], indices], dim=1)
    y = torch.cat([torch.randn(4, 2), indices], dim=1)
    return x, y


def _assert_conditioner_gradients(model: IndexAwareEncoderDecoderLayeredRNN) -> None:
    parameters = (
        *(table.weight for table in model.index_conditioner.hash_embeddings),
        model.index_conditioner.router.weight,
        model.index_conditioner.embedding_bank.weight,
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


def test_hashing_is_deterministic_high_bit_aware_and_range_checked() -> None:
    first, second = _conditioner(), _conditioner()
    indices = torch.tensor([0, 1, 65_536])
    buckets = first.hash_indices(indices)
    assert torch.equal(buckets, second.hash_indices(indices))
    assert ((0 <= buckets) & (buckets < 65_536)).all()
    assert (buckets[0] != buckets[2]).all()
    assert first(indices).shape == (3, 4)

    for invalid in (
        torch.tensor([-1]),
        torch.tensor([65_537]),
        torch.tensor([1.5]),
        torch.tensor([torch.nan]),
    ):
        with pytest.raises(AssertionError):
            first(invalid)


def test_vector_size_must_be_bound_consistently() -> None:
    conditioner = HashedIndexConditioner(hash_table_count=1, hash_dim=1)
    with pytest.raises(AssertionError, match="Set vector_size"):
        conditioner(torch.tensor([0]))
    conditioner.set_vector_size(4)
    conditioner.set_vector_size(4)
    with pytest.raises(AssertionError, match="already bound"):
        conditioner.set_vector_size(5)
    with pytest.raises(AssertionError):
        HashedIndexConditioner(routing_temperature=0)


def test_routing_temperature_changes_dense_mixture() -> None:
    cold = HashedIndexConditioner(
        hash_table_count=1, hash_dim=2, route_count=3, embedding_dim=2,
        hash_seed=4, routing_temperature=0.25,
    )
    warm = HashedIndexConditioner(
        hash_table_count=1, hash_dim=2, route_count=3, embedding_dim=2,
        hash_seed=4, routing_temperature=2.0,
    )
    warm.load_state_dict(cold.state_dict())
    cold.set_vector_size(4)
    warm.set_vector_size(4)
    assert not torch.allclose(cold(torch.tensor([0, 1])), warm(torch.tensor([0, 1])))


def test_v2_raw_bits_are_injective_and_can_separate_adjacent_routes() -> None:
    conditioner = _v2_conditioner().eval()
    indices = torch.tensor([0, 1, 255, 256, 65_535, 65_536, 2**24 - 1])
    raw_features = conditioner.raw_index_features(indices)
    reconstructed = (
        raw_features.long() * (2 ** torch.arange(24))[None, :]
    ).sum(dim=1)
    torch.testing.assert_close(reconstructed, indices)
    assert raw_features.unique(dim=0).shape[0] == indices.numel()

    with torch.no_grad():
        for table in conditioner.hash_embeddings:
            table.weight.zero_()
        conditioner.index_router.weight.zero_()
        conditioner.index_router.bias.zero_()
        conditioner.index_router.weight[0, 0] = 1_000
        conditioner.index_router.weight[1, 0] = -1_000
        conditioner.embedding_bank.weight.zero_()
        conditioner.embedding_bank.weight[0, 0] = 1
        conditioner.embedding_bank.weight[1, 0] = -1
    adjacent = conditioner(torch.tensor([0, 1]), torch.zeros(2, 1))
    assert not torch.allclose(adjacent[0], adjacent[1])


def test_v2_content_routing_and_dense_to_top_k_schedule() -> None:
    conditioner = _v2_conditioner().eval()
    with torch.no_grad():
        conditioner.index_router.weight.zero_()
        conditioner.index_router.bias.zero_()
        conditioner.content_router.weight.zero_()
        conditioner.content_router.bias.zero_()
        conditioner.content_router.weight[:, 0] = torch.arange(4)
    indices = torch.tensor([3, 3])
    content = torch.tensor([[0.0], [2.0]])
    _, warmup = conditioner.condition(indices, content, routing_progress=0.0)
    _, final = conditioner.condition(indices, content, routing_progress=1.0)

    assert not torch.allclose(warmup.dense_probabilities[0], warmup.dense_probabilities[1])
    assert (warmup.routed_probabilities > 0).sum(dim=1).tolist() == [4, 4]
    assert (final.routed_probabilities > 0).sum(dim=1).tolist() == [2, 2]
    top_probabilities, top_indices = final.dense_probabilities.topk(2, dim=1)
    reference = torch.zeros_like(final.dense_probabilities).scatter(
        1, top_indices, top_probabilities)
    reference /= reference.sum(dim=1, keepdim=True)
    torch.testing.assert_close(final.routed_probabilities, reference)
    assert warmup.temperature == pytest.approx(2.0)
    assert final.temperature == pytest.approx(0.8)


def test_v2_index_dropout_is_disabled_when_conditioner_is_frozen() -> None:
    conditioner = _v2_conditioner(index_dropout=1.0).train()
    with torch.no_grad():
        conditioner.index_router.weight.zero_()
        conditioner.index_router.bias.copy_(torch.arange(4))
        conditioner.content_router.weight.zero_()
        conditioner.content_router.bias.zero_()
    indices = torch.tensor([2, 3])
    content = torch.zeros(2, 1)
    _, trainable = conditioner.condition(indices, content, routing_progress=0.0)
    conditioner.requires_grad_(False)
    _, frozen = conditioner.condition(indices, content, routing_progress=0.0)
    torch.testing.assert_close(trainable.dense_probabilities, torch.full((2, 4), 0.25))
    assert not torch.allclose(frozen.dense_probabilities, trainable.dense_probabilities)


def test_v2_routed_scale_stays_positive_and_bounded() -> None:
    conditioner = _v2_conditioner(
        routed_scale_initial=0.01,
        routed_scale_max=0.05,
    )
    torch.testing.assert_close(
        conditioner.effective_routed_scale(), torch.tensor(0.01)
    )
    with torch.no_grad():
        conditioner.routed_scale.fill_(100)
    assert 0 < conditioner.effective_routed_scale() <= 0.05


def test_v2_embedding_diversity_penalty_trains_correlated_bank() -> None:
    conditioner = _v2_conditioner()
    with torch.no_grad():
        column = torch.linspace(-0.03, 0.03, conditioner.route_count)[:, None]
        conditioner.embedding_bank.weight.copy_(column.repeat(1, conditioner.embedding_dim))
    loss = conditioner.embedding_diversity_loss()
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    gradient = conditioner.embedding_bank.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_v2_model_keeps_public_forward_and_shared_branch_gradients() -> None:
    model = IndexAwareEncoderDecoderLayeredRNN(
        side_info_size=2,
        bins_per_plane=3,
        num_planes=2,
        layers=2,
        hidden_dim=5,
        shared_encoder=True,
        shared_decoder=True,
        hash_table_count=2,
        hash_dim=3,
        route_count=4,
        embedding_dim=3,
        routing_version="v2",
        routing_top_k=2,
        index_dropout=0.0,
    ).train()
    model.set_vector_size(8)
    x, y = _indexed_inputs()
    outputs = model(x, y)
    assert len(outputs) == 4
    assert model.encoder.input_dim == 1 + 3
    assert model.decoder.input_dim == 3 + 2 + 3

    reconstruction, _, codes, priors = outputs
    loss = reconstruction[-1].square().mean() + codes[-1][:, 0].mean() + priors[-1][:, 0].mean()
    loss.backward()
    parameters = (
        *(table.weight for table in model.index_conditioner.hash_embeddings),
        model.index_conditioner.index_router.weight,
        model.index_conditioner.content_router.weight,
        model.index_conditioner.embedding_bank.weight,
        model.index_conditioner.routed_scale,
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


@pytest.mark.parametrize("branch", ["encoder", "decoder", "prior"])
def test_each_v2_branch_trains_the_shared_content_aware_conditioner(branch: str) -> None:
    model = IndexAwareEncoderDecoderLayeredRNN(
        side_info_size=2,
        bins_per_plane=3,
        num_planes=2,
        layers=2,
        hidden_dim=5,
        hash_table_count=2,
        hash_dim=3,
        route_count=4,
        embedding_dim=3,
        routing_version="v2",
        routing_dense_warmup_fraction=1.0,
        index_dropout=0.0,
    ).train()
    model.set_vector_size(8)
    x, y = _indexed_inputs()
    fixed_codes = [torch.softmax(torch.randn(4, 3), dim=1) for _ in range(2)]
    context = [code.detach().requires_grad_() for code in fixed_codes]
    if branch == "encoder":
        loss = model.encode(x, routing_progress=0.0)[1][0][:, 0].sum()
    elif branch == "decoder":
        loss = model.decode(fixed_codes, y, routing_progress=0.0)[-1].square().sum()
    else:
        loss = model.get_priors(
            context, y, force_softmax=True, routing_progress=0.0)[-1][:, 0].sum()
    loss.backward()
    parameters = (
        *(table.weight for table in model.index_conditioner.hash_embeddings),
        model.index_conditioner.index_router.weight,
        model.index_conditioner.content_router.weight,
        model.index_conditioner.embedding_bank.weight,
        model.index_conditioner.routed_scale,
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )
    if branch == "prior":
        assert all(code.grad is None for code in context)


def test_indexed_model_shapes_matching_coordinates_and_transmission_state() -> None:
    model = _model().eval()
    x, y = _indexed_inputs()
    reconstructions, bins, codes, priors = model(x, y)
    assert len(reconstructions) == len(bins) == len(codes) == len(priors) == 2
    assert all(value.shape == (4, 1) for value in reconstructions)
    assert all(value.shape == (4,) for value in bins)
    assert all(value.shape == (4, 3) for value in (*codes, *priors))
    assert model.encoder.input_dim == 5
    assert model.decoder.input_dim == 3 + 2 + 4
    assert model.conditionalRNN.input_dim == 3 + 2 + 4
    assert any(key.startswith("index_conditioner.") for key in model.encoder_state_dict())
    assert any(key.startswith("index_conditioner.") for key in model.decoder_state_dict())

    mismatched = y.clone()
    mismatched[:, -1] = mismatched[:, -1].roll(1)
    with pytest.raises(AssertionError, match="same local coordinates"):
        model(x, mismatched)


@pytest.mark.parametrize("branch", ["encoder", "decoder", "prior"])
def test_each_branch_trains_the_shared_conditioner(branch: str) -> None:
    model = _model().train()
    x, y = _indexed_inputs()
    fixed_codes = [torch.softmax(torch.randn(4, 3), dim=1) for _ in range(2)]
    context = [code.detach().requires_grad_() for code in fixed_codes]
    if branch == "encoder":
        loss = model.encode(x)[1][0][:, 0].sum()
    elif branch == "decoder":
        loss = model.decode(fixed_codes, y)[-1].square().sum()
    else:
        loss = model.get_priors(context, y, force_softmax=True)[-1][:, 0].sum()
    loss.backward()
    _assert_conditioner_gradients(model)
    if branch == "prior":
        assert all(code.grad is None for code in context)


def test_two_optimizer_steps_do_not_reuse_an_autograd_graph() -> None:
    model = _model().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = _indexed_inputs()
    for _ in range(2):
        optimizer.zero_grad()
        reconstructions, _, codes, priors = model(x, y)
        loss = reconstructions[-1].square().mean() + codes[-1][:, 0].mean() + priors[-1][:, 0].mean()
        loss.backward()
        _assert_conditioner_gradients(model)
        optimizer.step()


def test_base_prior_detaches_codes_but_not_side_information() -> None:
    model = EncoderDecoderLayeredRNN(
        input_dim=1,
        side_info_size=2,
        bins_per_plane=3,
        num_planes=2,
        layers=1,
        hidden_dim=4,
    ).train()
    codes = [torch.randn(3, 3, requires_grad=True) for _ in range(2)]
    side_info = torch.randn(3, 2, requires_grad=True)
    model.get_priors(codes, side_info, force_softmax=True)[-1][:, 0].sum().backward()
    assert model.encoder.input_dim == 1
    assert model.decoder.input_dim == 5
    assert all(reconstruction.shape == (3, 1) for reconstruction in model.decode(codes, side_info))
    assert all(code.grad is None for code in codes)
    assert side_info.grad is not None and side_info.grad.abs().sum() > 0


def test_quantizer_model_and_cache_class_hooks_keep_baseline_defaults() -> None:
    class HookedQuantizer(WZQuantizerCancer):
        made_models = 0

        def _new_coding_model(self, side_info_size: int) -> EncoderDecoderLayeredRNN:
            type(self).made_models += 1
            return super()._new_coding_model(side_info_size)

    quantizer = HookedQuantizer(
        WZcfgQuant(bins_per_plane=2, num_planes=2), si_size=1)
    assert HookedQuantizer.made_models == 1
    assert type(quantizer.coding_model) is EncoderDecoderLayeredRNN
    assert quantizer._cached_training_quantizer_class() is WZQuantizerCancer
