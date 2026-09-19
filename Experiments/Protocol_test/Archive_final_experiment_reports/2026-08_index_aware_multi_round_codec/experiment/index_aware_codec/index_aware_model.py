"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN


_HASH_A = (40503, 25173, 13849, 34283, 53653, 61787, 18521, 45647)
_HASH_B = (55399, 39409, 47251, 19463, 65267, 30653, 58901, 23719)


class RoutingInfo(NamedTuple):
    """Per-row routing tensors used for regularization and later diagnostics."""

    dense_probabilities: torch.Tensor
    routed_probabilities: torch.Tensor
    z_loss: torch.Tensor
    temperature: float


class HashedIndexConditioner(nn.Module):
    """Map decoder-known local coordinates to one routed feature shared by every model branch."""

    def __init__(
        self,
        hash_table_count: int = 6,
        hash_bucket_count: int = 65_536,
        hash_dim: int = 16,
        route_count: int = 48,
        embedding_dim: int = 16,
        hash_seed: int = 0,
        routing_temperature: float = 1.0,
        routing_version: Literal["v1", "v2"] = "v1",
        raw_index_bits: int = 24,
        raw_index_scale: float = 0.01,
        routing_temperature_start: float = 2.0,
        routing_temperature_end: float = 0.8,
        routing_dense_warmup_fraction: float = 0.1,
        routing_top_k: int = 2,
        index_dropout: float = 0.05,
        routed_scale_initial: float = 0.01,
        routed_scale_max: float = 0.05,
        content_routing: bool = True,
    ) -> None:
        super().__init__()
        assert 1 <= hash_table_count <= len(_HASH_A)
        assert hash_bucket_count == 2**16, (
            "The high-bit-aware index hash is defined for exactly 65,536 buckets.")
        assert hash_dim > 0 and route_count > 0 and embedding_dim > 0
        assert routing_temperature > 0
        assert routing_version in ("v1", "v2")
        assert raw_index_bits == 24 and raw_index_scale > 0
        assert routing_temperature_start > 0 and routing_temperature_end > 0
        assert 0 <= routing_dense_warmup_fraction <= 1
        assert 1 <= routing_top_k <= route_count
        assert 0 <= index_dropout <= 1
        assert 0 < routed_scale_initial < routed_scale_max

        self.routing_version: Literal["v1", "v2"] = routing_version
        self.hash_table_count: int = hash_table_count
        self.hash_bucket_count: int = hash_bucket_count
        self.hash_dim: int = hash_dim
        self.route_count: int = route_count
        self.embedding_dim: int = embedding_dim
        self.routing_temperature: float = routing_temperature
        self.raw_index_bits: int = raw_index_bits
        self.raw_index_scale: float = raw_index_scale
        self.routing_temperature_start: float = routing_temperature_start
        self.routing_temperature_end: float = routing_temperature_end
        self.routing_dense_warmup_fraction: float = routing_dense_warmup_fraction
        self.routing_top_k: int = routing_top_k
        self.index_dropout: float = index_dropout
        self.routed_scale_max: float = routed_scale_max
        self.content_routing: bool = content_routing
        self.vector_size: int | None = None

        generator = torch.Generator().manual_seed(hash_seed)
        self.register_buffer("_hash_a", torch.tensor(_HASH_A[:hash_table_count], dtype=torch.int64))
        self.register_buffer("_hash_b", torch.tensor(_HASH_B[:hash_table_count], dtype=torch.int64))
        self.register_buffer(
            "_hash_c",
            torch.randint(0, hash_bucket_count, (hash_table_count,), generator=generator),
        )
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(hash_bucket_count, hash_dim) for _ in range(hash_table_count)
        ])
        self.embedding_bank = nn.Embedding(route_count, embedding_dim)
        for table in self.hash_embeddings:
            nn.init.normal_(table.weight, std=0.02)
        nn.init.normal_(self.embedding_bank.weight, std=0.02)
        if routing_version == "v1":
            self.router = nn.Linear(hash_table_count * hash_dim, route_count)
            nn.init.normal_(self.router.weight, std=0.01)
            nn.init.zeros_(self.router.bias)
        else:
            self.index_router = nn.Linear(raw_index_bits + hash_table_count * hash_dim, route_count)
            self.content_router = nn.Linear(3, route_count)
            self.routed_scale = nn.Parameter(torch.logit(torch.tensor(
                routed_scale_initial / routed_scale_max
            )))
            nn.init.normal_(self.index_router.weight, std=0.01)
            nn.init.zeros_(self.index_router.bias)
            nn.init.normal_(self.content_router.weight, std=0.01)
            nn.init.zeros_(self.content_router.bias)

    def set_vector_size(self, vector_size: int) -> None:
        """Bind the conditioner to the exact valid local-coordinate range for one model vector."""
        assert vector_size > 0
        assert self.routing_version == "v1" or vector_size <= 2**self.raw_index_bits, (
            f"V2 index conditioning supports at most {2**self.raw_index_bits} coordinates, "
            f"got {vector_size}.")
        assert self.vector_size in (None, vector_size), (
            f"Index conditioner is already bound to vector size {self.vector_size}, not {vector_size}.")
        self.vector_size = vector_size

    def _integer_indices(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.vector_size is not None, "Set vector_size before using index conditioning."
        assert indices.ndim == 1, f"Expected one coordinate per row, got shape {tuple(indices.shape)}."
        if indices.is_floating_point():
            assert torch.isfinite(indices).all(), "Coordinates must be finite integers."
            assert torch.equal(indices, indices.trunc()), "Coordinates must be integral."
        else:
            assert indices.dtype != torch.bool, "Boolean coordinates are invalid."
        integer_indices = indices.to(torch.int64)
        assert ((integer_indices >= 0) & (integer_indices < self.vector_size)).all(), (
            f"Coordinates must satisfy 0 <= index < {self.vector_size}.")
        return integer_indices

    def hash_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Return deterministic high-bit-aware bucket IDs for each local coordinate and table."""
        integer_indices = self._integer_indices(indices)
        low = (integer_indices & 0xFFFF).unsqueeze(1)
        high = (integer_indices >> 16).unsqueeze(1)
        return ((low * self._hash_a) ^ (high * self._hash_b) ^ self._hash_c) & 0xFFFF

    def raw_index_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the exact 24-bit representation that makes every supported index distinct."""
        integer_indices = self._integer_indices(indices)
        return self._raw_integer_features(integer_indices)

    def _raw_integer_features(self, integer_indices: torch.Tensor) -> torch.Tensor:
        bit_positions = torch.arange(self.raw_index_bits, device=integer_indices.device)
        return ((integer_indices[:, None] >> bit_positions) & 1).to(self.embedding_bank.weight.dtype)

    def embedding_diversity_loss(self) -> torch.Tensor:
        """Penalize collapsed or correlated columns in the shared routed-embedding bank."""
        centered = self.embedding_bank.weight - self.embedding_bank.weight.mean(dim=0)
        covariance = centered.T @ centered / self.route_count
        initial_variance = 0.02**2
        target = torch.eye(
            self.embedding_dim,
            device=covariance.device,
            dtype=covariance.dtype,
        ) * initial_variance
        return ((covariance - target) / initial_variance).square().mean()

    def effective_routed_scale(self) -> torch.Tensor:
        """Return the bounded nonnegative scale applied to v2 routed embeddings."""
        assert self.routing_version == "v2"
        return self.routed_scale_max * self.routed_scale.sigmoid()

    def condition(
        self,
        indices: torch.Tensor,
        content: torch.Tensor | None,
        routing_progress: float = 1.0,
    ) -> tuple[torch.Tensor, RoutingInfo]:
        """Build one index feature and expose the routing tensors needed by the training loss."""
        assert 0 <= routing_progress <= 1
        integer_indices = self._integer_indices(indices)
        low = (integer_indices & 0xFFFF).unsqueeze(1)
        high = (integer_indices >> 16).unsqueeze(1)
        buckets = ((low * self._hash_a) ^ (high * self._hash_b) ^ self._hash_c) & 0xFFFF
        hash_features = torch.cat([
            table(buckets[:, table_index])
            for table_index, table in enumerate(self.hash_embeddings)
        ], dim=1)
        if self.routing_version == "v1":
            logits = self.router(hash_features)
            probabilities = F.softmax(logits / self.routing_temperature, dim=1)
            return probabilities @ self.embedding_bank.weight, RoutingInfo(
                probabilities,
                probabilities,
                torch.logsumexp(logits, dim=1).square(),
                self.routing_temperature,
            )

        raw_features = self._raw_integer_features(integer_indices)
        index_logits = self.index_router(torch.cat([
            raw_features * self.raw_index_scale,
            hash_features,
        ], dim=1))
        if content is None or not self.content_routing:
            content_logits = torch.zeros_like(index_logits)
        else:
            assert content.ndim == 2 and content.shape[0] == indices.shape[0] and content.shape[1] > 0
            content = content.to(index_logits.dtype)
            content_summary = torch.stack([
                content.mean(dim=1),
                content.square().mean(dim=1).add(1e-12).sqrt(),
                content.abs().amax(dim=1),
            ], dim=1)
            content_logits = self.content_router(content_summary)
        keep_index: torch.Tensor | None = None
        if self.training and self.routed_scale.requires_grad and self.index_dropout:
            keep_index = (
                torch.rand(index_logits.shape[0], 1, device=index_logits.device)
                >= self.index_dropout
            )
            index_logits = index_logits * keep_index
        logits = index_logits + content_logits
        temperature = self.routing_temperature_start * (
            self.routing_temperature_end / self.routing_temperature_start
        ) ** routing_progress
        dense_probabilities = F.softmax(logits / temperature, dim=1)
        routed_probabilities = dense_probabilities
        if (
            routing_progress >= self.routing_dense_warmup_fraction
            and self.routing_top_k < self.route_count
        ):
            top_probabilities, top_indices = dense_probabilities.topk(self.routing_top_k, dim=1)
            routed_probabilities = torch.zeros_like(dense_probabilities).scatter(
                1, top_indices, top_probabilities)
            routed_probabilities /= routed_probabilities.sum(dim=1, keepdim=True)
        routed_embedding = routed_probabilities @ self.embedding_bank.weight
        if keep_index is not None:
            routed_embedding = routed_embedding * keep_index
        feature = self.effective_routed_scale() * routed_embedding
        return feature, RoutingInfo(
            dense_probabilities,
            routed_probabilities,
            torch.logsumexp(logits, dim=1).square(),
            float(temperature),
        )

    def forward(
        self,
        indices: torch.Tensor,
        content: torch.Tensor | None = None,
        routing_progress: float = 1.0,
    ) -> torch.Tensor:
        """Return the shared index feature, optionally routed using branch-local content."""
        return self.condition(indices, content, routing_progress)[0]


class IndexAwareEncoderDecoderLayeredRNN(EncoderDecoderLayeredRNN):
    """Augment encoder, decoder, and conditional-prior inputs with one shared routed index feature."""

    def __init__(
        self,
        *,
        side_info_size: int,
        bins_per_plane: int,
        num_planes: int,
        layers: int,
        hidden_dim: int,
        shared_encoder: bool = False,
        shared_decoder: bool = False,
        shared_priors: bool = False,
        marginal: bool = False,
        output_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hash_table_count: int = 6,
        hash_bucket_count: int = 65_536,
        hash_dim: int = 16,
        route_count: int = 48,
        embedding_dim: int = 16,
        hash_seed: int = 0,
        routing_temperature: float = 1.0,
        routing_version: Literal["v1", "v2"] = "v1",
        raw_index_bits: int = 24,
        raw_index_scale: float = 0.01,
        routing_temperature_start: float = 2.0,
        routing_temperature_end: float = 0.8,
        routing_dense_warmup_fraction: float = 0.1,
        routing_top_k: int = 2,
        index_dropout: float = 0.05,
        routed_scale_initial: float = 0.01,
        routed_scale_max: float = 0.05,
        content_routing: bool = True,
    ) -> None:
        assert side_info_size > 0
        assert not marginal, "Index-aware priors require conditional side information."
        index_feature_dim = embedding_dim
        super().__init__(
            input_dim=1,
            encoder_input_dim=1 + index_feature_dim,
            side_info_size=side_info_size + index_feature_dim,
            bins_per_plane=bins_per_plane,
            num_planes=num_planes,
            layers=layers,
            hidden_dim=hidden_dim,
            shared_encoder=shared_encoder,
            shared_decoder=shared_decoder,
            shared_priors=shared_priors,
            marginal=False,
            output_activation=output_activation,
        )
        self.ordinary_side_info_size: int = side_info_size
        self.index_conditioner = HashedIndexConditioner(
            hash_table_count=hash_table_count,
            hash_bucket_count=hash_bucket_count,
            hash_dim=hash_dim,
            route_count=route_count,
            embedding_dim=embedding_dim,
            hash_seed=hash_seed,
            routing_temperature=routing_temperature,
            routing_version=routing_version,
            raw_index_bits=raw_index_bits,
            raw_index_scale=raw_index_scale,
            routing_temperature_start=routing_temperature_start,
            routing_temperature_end=routing_temperature_end,
            routing_dense_warmup_fraction=routing_dense_warmup_fraction,
            routing_top_k=routing_top_k,
            index_dropout=index_dropout,
            routed_scale_initial=routed_scale_initial,
            routed_scale_max=routed_scale_max,
            content_routing=content_routing,
        )

    def set_vector_size(self, vector_size: int) -> None:
        """Set the exclusive upper bound for coordinates accepted by every model branch."""
        self.index_conditioner.set_vector_size(vector_size)

    def _condition(
        self,
        indexed: torch.Tensor,
        ordinary_width: int,
        label: str,
        routing_progress: float,
    ) -> tuple[torch.Tensor, RoutingInfo]:
        assert indexed.ndim == 2 and indexed.shape[1] == ordinary_width + 1, (
            f"{label} must have shape [batch, {ordinary_width + 1}], got {tuple(indexed.shape)}.")
        feature, routing_info = self.index_conditioner.condition(
            indexed[:, -1], indexed[:, :ordinary_width], routing_progress)
        return torch.cat([indexed[:, :ordinary_width], feature], dim=1), routing_info

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return encoder and shared index-conditioner parameters in transmission precision."""
        return self._transmission_state_dict(
            self.state_dict(), ("encoder.", "binner", "index_conditioner."))

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return decoder, prior, and shared index-conditioner parameters in transmission precision."""
        return self._transmission_state_dict(
            self.state_dict(),
            ("decoder.", "reconstructor", "conditional", "index_conditioner."),
        )

    def encode(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        force_softmax: bool = False,
        routing_progress: float = 1.0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode normalized scalar-and-coordinate rows using the routed index feature."""
        augmented, _ = self._condition(x, 1, "Encoder input", routing_progress)
        return super().encode(augmented, tau=tau, force_softmax=force_softmax)

    def decode(
        self,
        codes: Sequence[torch.Tensor],
        y: torch.Tensor | None,
        routing_progress: float = 1.0,
    ) -> list[torch.Tensor]:
        """Decode symbols from ordinary side information plus its local coordinate."""
        assert y is not None, "The decoder requires indexed side information."
        augmented, _ = self._condition(
            y, self.ordinary_side_info_size, "Decoder side information", routing_progress)
        return super().decode(codes, augmented)

    def get_priors(
        self,
        codes: Sequence[torch.Tensor],
        y: torch.Tensor | None = None,
        tau: float = 1.0,
        force_softmax: bool = False,
        routing_progress: float = 1.0,
    ) -> list[torch.Tensor]:
        """Predict symbols from ordinary side information plus its local coordinate."""
        assert y is not None, "The conditional prior requires indexed side information."
        augmented, _ = self._condition(
            y, self.ordinary_side_info_size, "Prior side information", routing_progress)
        return super().get_priors(
            codes, augmented, tau=tau, force_softmax=force_softmax)

    def forward_with_routing(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        tau: float = 1.0,
        routing_progress: float = 1.0,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        tuple[RoutingInfo, RoutingInfo],
    ]:
        """Run all branches and return routing tensors for the joint quantizer loss."""
        assert y is not None, "The index-aware model requires indexed side information."
        assert x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0]
        assert torch.equal(x[:, -1], y[:, -1]), (
            "Encoder and decoder side information must use the same local coordinates.")
        encoder_input, encoder_routing = self._condition(
            x, 1, "Encoder input", routing_progress)
        decoder_input, decoder_routing = self._condition(
            y, self.ordinary_side_info_size, "Decoder side information", routing_progress)
        bins, codes = super().encode(encoder_input, tau=tau)
        reconstructed = super().decode(codes, decoder_input)
        priors = super().get_priors(codes, decoder_input, tau=tau)
        return reconstructed, bins, codes, priors, (encoder_routing, decoder_routing)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        tau: float = 1.0,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Run the compatible four-output indexed-model interface at the final routing schedule."""
        reconstructed, bins, codes, priors, _ = self.forward_with_routing(
            x, y, tau=tau, routing_progress=1.0)
        return reconstructed, bins, codes, priors
