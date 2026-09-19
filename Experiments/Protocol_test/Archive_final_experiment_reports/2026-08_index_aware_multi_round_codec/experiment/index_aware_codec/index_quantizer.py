"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator
import torch
import torch.nn.functional as F

from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH, PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer,
    PreprocessMetadata,
    WZQuantizerCancer,
    WZcfgQuant,
    _sample_mean,
)

from .index_aware_model import IndexAwareEncoderDecoderLayeredRNN


class IndexAwareWZcfgQuant(WZcfgQuant):
    """Configuration for the experimental coordinate-conditioned WZ quantizer."""

    _INDEX_CACHE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "index_cache_version",
        "hash_table_count",
        "hash_bucket_count",
        "hash_dim",
        "route_count",
        "embedding_dim",
        "hash_seed",
        "routing_temperature",
        "training_seed",
        "quantizer_train_repeats",
        "deduplicate_training",
        "training_match_bits",
    })
    _V2_CACHE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "raw_index_bits",
        "raw_index_scale",
        "routing_temperature_start",
        "routing_temperature_end",
        "routing_dense_warmup_fraction",
        "routing_top_k",
        "routing_balance_loss_weight",
        "routing_z_loss_weight",
        "embedding_diversity_loss_weight",
        "index_dropout",
        "routed_scale_initial",
        "current_problem_sampling_fraction",
        "routed_scale_max",
        "content_routing",
    })

    train_epochs: int = 180
    hidden_dim: int = 112
    layers: int = 8
    marginal_loss: Literal[False] = False
    index_cache_version: Literal[
        "index-side-information/v1", "index-side-information/v2"
    ] = "index-side-information/v2"
    hash_table_count: int = Field(default=8, ge=1, le=8)
    hash_bucket_count: Literal[65_536] = 65_536
    hash_dim: int = Field(default=24, gt=0)
    route_count: int = Field(default=64, gt=0)
    embedding_dim: int = Field(default=24, gt=0)
    hash_seed: int = Field(default=0, ge=0)
    routing_temperature: Literal[1.0] = 1.0
    training_seed: int = Field(default=43, ge=0, le=2**63 - 1)
    raw_index_bits: Literal[24] = 24
    raw_index_scale: float = Field(default=0.01, gt=0)
    routing_temperature_start: float = Field(default=2.0, gt=0)
    routing_temperature_end: float = Field(default=0.8, gt=0)
    routing_dense_warmup_fraction: float = Field(default=0.1, ge=0, le=1)
    routing_top_k: int = Field(default=2, gt=0)
    routing_balance_loss_weight: float = Field(default=1e-3, ge=0)
    routing_z_loss_weight: float = Field(default=1e-3, ge=0)
    embedding_diversity_loss_weight: float = Field(default=1e-4, ge=0)
    index_dropout: float = Field(default=0.05, ge=0, le=1)
    routed_scale_initial: float = Field(default=0.01, gt=0)
    current_problem_sampling_fraction: float = Field(default=0.75, gt=0, le=1)
    routed_scale_max: float = Field(default=0.05, gt=0)
    content_routing: bool = False

    @model_validator(mode="after")
    def validate_v2_routing(self) -> IndexAwareWZcfgQuant:
        """Validate relationships among the v2 routing controls."""
        assert self.routing_temperature_end <= self.routing_temperature_start
        assert self.routing_top_k <= self.route_count
        assert self.routed_scale_initial < self.routed_scale_max
        return self

    def training_cache_payload(self) -> dict[str, Any]:
        """Identify caches by every option that changes indexed quantizer training."""
        fields = self._INDEX_CACHE_FIELDS | (
            self._V2_CACHE_FIELDS
            if self.index_cache_version == "index-side-information/v2"
            else frozenset()
        )
        return super().training_cache_payload() | self.model_dump(include=fields)


class IndexAwareDedupedWZQuantizer(DedupedWZQuantizerCancer):
    """Deduplicating WZ quantizer that conditions every model branch on local coordinates."""

    candidate_failure_tolerance: ClassVar[int] = 8
    c_cfg: IndexAwareWZcfgQuant
    coding_model: IndexAwareEncoderDecoderLayeredRNN

    def __init__(self, c_cfg: IndexAwareWZcfgQuant, si_size: int) -> None:
        assert isinstance(c_cfg, IndexAwareWZcfgQuant)
        super().__init__(c_cfg, si_size)
        if c_cfg.index_cache_version == "index-side-information/v2":
            self.prior_calculator = PriorCalculator

    def _new_coding_model(self, side_info_size: int) -> EncoderDecoderLayeredRNN:
        """Create the indexed architecture used for initial, retry, and restored models."""
        model = IndexAwareEncoderDecoderLayeredRNN(
            num_planes=self.num_planes,
            bins_per_plane=self.bins_per_plane,
            side_info_size=side_info_size,
            marginal=self.marginal_loss,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            shared_encoder=self.shared_heads,
            shared_decoder=self.shared_heads,
            hash_table_count=self.c_cfg.hash_table_count,
            hash_bucket_count=self.c_cfg.hash_bucket_count,
            hash_dim=self.c_cfg.hash_dim,
            route_count=self.c_cfg.route_count,
            embedding_dim=self.c_cfg.embedding_dim,
            hash_seed=self.c_cfg.hash_seed,
            routing_temperature=self.c_cfg.routing_temperature,
            routing_version=(
                "v2"
                if self.c_cfg.index_cache_version == "index-side-information/v2"
                else "v1"
            ),
            raw_index_bits=self.c_cfg.raw_index_bits,
            raw_index_scale=self.c_cfg.raw_index_scale,
            routing_temperature_start=self.c_cfg.routing_temperature_start,
            routing_temperature_end=self.c_cfg.routing_temperature_end,
            routing_dense_warmup_fraction=self.c_cfg.routing_dense_warmup_fraction,
            routing_top_k=self.c_cfg.routing_top_k,
            index_dropout=self.c_cfg.index_dropout,
            routed_scale_initial=self.c_cfg.routed_scale_initial,
            routed_scale_max=self.c_cfg.routed_scale_max,
            content_routing=self.c_cfg.content_routing,
        )
        vector_size = self.__dict__.get("vector_size")
        if vector_size is not None:
            model.set_vector_size(vector_size)
        return model

    def _cached_training_quantizer_class(self) -> type[WZQuantizerCancer]:
        """Train the indexed class on cache misses instead of the baseline architecture."""
        return IndexAwareDedupedWZQuantizer

    def train_model(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[torch.Tensor] | None,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]] = (),
    ) -> None:
        """Train or restore the model and bind its exact coordinate domain."""
        super().train_model(x_raw, si_raw_list, extra_pairs)
        assert self.vector_size is not None
        self.coding_model.set_vector_size(self.vector_size)

    def encode_prepared(
        self,
        x_prep: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate every v2 coordinate separately so index identity is never deduplicated."""
        if self.c_cfg.index_cache_version == "index-side-information/v1":
            return super().encode_prepared(x_prep, batch_size)
        return WZQuantizerCancer.encode_prepared(self, x_prep, batch_size)

    def decode_stages_prepared(
        self,
        bins: torch.Tensor,
        side_info: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Evaluate every v2 coordinate separately during reconstruction."""
        if self.c_cfg.index_cache_version == "index-side-information/v1":
            return super().decode_stages_prepared(bins, side_info, batch_size)
        return WZQuantizerCancer.decode_stages_prepared(
            self, bins, side_info, batch_size
        )

    def _candidate_is_usable(
        self,
        bins: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> bool:
        """Reject indexed candidates whose refinement symbols collapse to constants."""
        return all(plane.unique().numel() > 1 for plane in bins)

    def _train_model_uncached(
        self,
        x_raw: torch.Tensor,
        si_raw_list: Sequence[torch.Tensor] | None,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]] = (),
    ) -> None:
        """Train reproducibly without changing the caller's process-wide random state."""
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.c_cfg.training_seed)
            self.coding_model = self._new_coding_model(self.side_info_size)
            super()._train_model_uncached(x_raw, si_raw_list, extra_pairs)

    def preprocess_x(
        self,
        x_raw: torch.Tensor,
        skip_outliers: bool = False,
        normalization_factors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PreprocessMetadata]:
        """Normalize scalar values and append their exact local coordinates."""
        values, metadata = super().preprocess_x(x_raw, skip_outliers, normalization_factors)
        assert self.vector_size is not None
        assert self.vector_size <= 2**24, (
            f"Float32 coordinate transport is exact only through 2**24 rows, got {self.vector_size}."
        )
        self.coding_model.set_vector_size(self.vector_size)
        coordinates = torch.arange(self.vector_size, device=values.device, dtype=values.dtype)[:, None]
        return torch.cat((values, coordinates), dim=1), metadata

    def normalize_side_info_tensor(
        self,
        source_metadata: PreprocessMetadata,
        side_information: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Normalize ordinary side information and append decoder-known coordinates."""
        assert self.vector_size is not None, "Vector size must be known before side information is formatted."
        assert all(value.numel() == self.vector_size for value in side_information), (
            f"Each side-information vector must contain {self.vector_size} values."
        )

        formatted = torch.zeros(
            (self.vector_size, max(len(side_information), 1) + 1),
            device=self.device,
            dtype=torch.float32,
        )
        for index, side_info in enumerate(side_information):
            values, _ = super().preprocess_x(
                side_info,
                skip_outliers=True,
                normalization_factors=source_metadata.norm_factors,
            )
            formatted[:, index] = values.squeeze(1)
        formatted[:, -1] = torch.arange(self.vector_size, device=self.device, dtype=torch.float32)
        self.coding_model.set_vector_size(self.vector_size)
        return formatted

    def _stack_training_pairs(
        self,
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append training problems while restarting local coordinates for every vector."""
        if not extra_pairs:
            return x_prep, side_info

        rows = x_prep.shape[0]
        stacked_x = torch.empty(
            (rows * (1 + len(extra_pairs)), x_prep.shape[1]),
            device=x_prep.device,
            dtype=x_prep.dtype,
        )
        stacked_si = torch.empty(
            (stacked_x.shape[0], side_info.shape[1]),
            device=side_info.device,
            dtype=side_info.dtype,
        )
        stacked_x[:rows], stacked_si[:rows] = x_prep, side_info
        del x_prep, side_info

        for index, (extra_x, extra_si) in enumerate(extra_pairs, start=1):
            assert len(extra_si) == self.side_info_size, (
                f"Extra training pair {index} has {len(extra_si)} side-information vectors; "
                f"the quantizer is built for {self.side_info_size}."
            )
            block = slice(index * rows, (index + 1) * rows)
            extra_prep, extra_metadata = self.preprocess_x(extra_x)
            stacked_x[block] = extra_prep
            stacked_si[block] = self.normalize_side_info_tensor(extra_metadata, extra_si)
        self._set_training_block_probabilities(len(extra_pairs))
        return stacked_x, stacked_si

    def _set_training_block_probabilities(self, extra_pair_count: int) -> None:
        """Prefer explicit M weights while retaining the legacy indexed sampling fallback."""
        if self.c_cfg.training_problem_weights is not None:
            super()._set_training_block_probabilities(extra_pair_count)
            return
        historical_fraction = 1 - self.c_cfg.current_problem_sampling_fraction
        assert historical_fraction > 0, (
            "Extra training pairs require current_problem_sampling_fraction < 1."
        )
        self.training_block_probabilities = (
            self.c_cfg.current_problem_sampling_fraction,
            *(historical_fraction / extra_pair_count for _ in range(extra_pair_count)),
        )

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: torch.Tensor) -> None:
        """Fit using value distortion while retaining coordinates as model inputs."""
        assert self.vector_size is not None
        wmspe_denom = (
            x_prep[: self.vector_size, :1].float().square().mean().item() / 2
        ) + 1e-8
        self.coding_model = self._train_finite_candidate_and_retrieve_best(
            x_prep, side_info, wmspe_denom
        )

        bins, _ = self.encode_prepared(x_prep[: self.vector_size], INFERENCE_BATCH)
        prior = self.prior_calculator.compute_prior_from_network(
            self.coding_model, bins, side_info[: self.vector_size]
        )
        self.training_prior_stage_rates = self.prior_calculator.compute_plane_rates(
            prior, bins, self.c_cfg.num_planes
        )
        self.training_prior_rate = sum(self.training_prior_stage_rates)

    @staticmethod
    def training_metrics(
        model: EncoderDecoderLayeredRNN,
        training_inputs: tuple[torch.Tensor, ...],
        side_info: torch.Tensor,
    ) -> str:
        """Summarize indexed hard-code distortion and cumulative rate."""
        x = training_inputs[0][:10_000]
        target = x[:, :1]
        y = side_info[: x.shape[0]]
        bins, codes = model.encode(x)
        reconstructions = model.decode(codes, y)
        priors = model.get_priors(codes, y)
        sample_indices = torch.arange(x.shape[0], device=x.device)
        plane_rates = [
            -torch.log2(prior[sample_indices, symbols] + 1e-12).mean().item()
            for prior, symbols in zip(priors, bins, strict=True)
        ]
        return (
            f"hard_mse={[round(F.mse_loss(reconstruction, target).item(), 6) for reconstruction in reconstructions]}, "
            f"cumulative_rate={[round(float(rate), 4) for rate in np.cumsum(plane_rates)]}"
        )

    @staticmethod
    def compute_loss(
        model: EncoderDecoderLayeredRNN,
        training_input: tuple[torch.Tensor],
        si_batch: torch.Tensor,
        epoch: int,
        c_cfg: WZcfgQuant,
        sample_weights: torch.Tensor | None,
        wmspe_denom: float,
        perturb_generator: torch.Generator,
    ) -> torch.Tensor:
        """Compute indexed RD loss without treating the coordinate as a reconstruction target."""
        indexed_x, = training_input
        target = indexed_x[:, :1]
        if c_cfg.source_perturbation != "none":
            perturbation = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=perturb_generator,
            ) * (1e-5 * target.abs().mean())
            if c_cfg.source_perturbation == "add":
                indexed_x = indexed_x.clone()
                indexed_x[:, :1] = target + perturbation
                target = indexed_x[:, :1]

        tau_progress = epoch / c_cfg.train_epochs
        tau = c_cfg.tau * np.exp(tau_progress * np.log(0.1 / c_cfg.tau))
        assert isinstance(model, IndexAwareEncoderDecoderLayeredRNN)
        assert isinstance(c_cfg, IndexAwareWZcfgQuant)
        routing_infos = ()
        if c_cfg.index_cache_version == "index-side-information/v2":
            routing_progress = epoch / max(c_cfg.train_epochs - 1, 1)
            reconstructions, bins, soft_codes, priors, routing_infos = (
                model.forward_with_routing(
                    indexed_x,
                    si_batch,
                    tau=tau,
                    routing_progress=routing_progress,
                )
            )
        else:
            reconstructions, bins, soft_codes, priors = model(
                indexed_x, si_batch, tau=tau)

        loss = torch.zeros((), device=indexed_x.device)
        sample_indices = torch.arange(indexed_x.shape[0], device=indexed_x.device)
        for plane in range(model.num_planes):
            distortion = _sample_mean(
                F.mse_loss(reconstructions[plane], target, reduction="none")
                .flatten(start_dim=1)
                .mean(dim=1),
                sample_weights,
            )
            posterior_prob = soft_codes[plane][sample_indices, bins[plane]]
            prior_prob = priors[plane][sample_indices, bins[plane]]
            rate = _sample_mean(
                torch.log((posterior_prob + 1e-12) / (prior_prob + 1e-12)),
                sample_weights,
            )
            loss += ((c_cfg.reconst_ld / wmspe_denom) * distortion + rate) / model.num_planes

        if routing_infos:
            balance_losses: list[torch.Tensor] = []
            z_losses: list[torch.Tensor] = []
            for routing_info in routing_infos:
                if sample_weights is None:
                    usage = routing_info.dense_probabilities.mean(dim=0)
                else:
                    usage = (
                        routing_info.dense_probabilities * sample_weights[:, None]
                    ).sum(dim=0) / sample_weights.sum()
                balance_losses.append(
                    model.index_conditioner.route_count
                    * (usage - 1 / model.index_conditioner.route_count).square().sum()
                )
                z_losses.append(_sample_mean(routing_info.z_loss, sample_weights))
            loss += c_cfg.routing_balance_loss_weight * torch.stack(balance_losses).mean()
            loss += c_cfg.routing_z_loss_weight * torch.stack(z_losses).mean()
            loss += (
                c_cfg.embedding_diversity_loss_weight
                * model.index_conditioner.embedding_diversity_loss()
            )
        return loss

    def index_artifact(self) -> dict[str, Any]:
        """Return compact learned index state without materializing per-coordinate embeddings."""
        assert self.vector_size is not None, "The quantizer must be trained before exporting index state."
        conditioner_state = {
            key: value.detach().cpu().clone()
            for key, value in self.coding_model.state_dict().items()
            if key.startswith("index_conditioner.")
        }
        assert conditioner_state, "The indexed model exposed no index_conditioner parameters."
        return {
            "schema_version": self.c_cfg.index_cache_version,
            "vector_size": self.vector_size,
            "config": self.c_cfg.training_cache_payload(),
            "routing_diagnostics": self.index_diagnostics(),
            "conditioner_state": conditioner_state,
        }

    def index_diagnostics(
        self, sample_size: int = 32_768
    ) -> dict[str, float | int | str]:
        """Summarize routed-index usage on a deterministic coordinate sample."""
        assert self.vector_size is not None, "The quantizer must be trained before routing analysis."
        assert sample_size > 0
        conditioner = self.coding_model.index_conditioner
        parameter = next(conditioner.parameters())
        count = min(sample_size, self.vector_size)
        coordinates = (
            torch.zeros(1, device=parameter.device, dtype=torch.long)
            if count == 1
            else torch.arange(count, device=parameter.device) * (self.vector_size - 1) // (count - 1)
        )

        was_training = conditioner.training
        conditioner.eval()
        try:
            with torch.inference_mode():
                _, routing_info = conditioner.condition(
                    coordinates, content=None, routing_progress=1.0
                )
                probabilities = routing_info.dense_probabilities.float()
                routed_probabilities = routing_info.routed_probabilities.float()
                loads = probabilities.mean(dim=0)
                top1_loads = torch.bincount(
                    routed_probabilities.argmax(dim=1), minlength=conditioner.route_count
                ).float() / count
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
                bank = conditioner.embedding_bank.weight.float()
                routed = routed_probabilities @ bank
                centered = routed - routed.mean(dim=0)

                singular_values = torch.linalg.svdvals(bank)
                singular_mass = singular_values.square()
                singular_mass /= singular_mass.sum().clamp_min(1e-12)
                bank_effective_rank = torch.exp(
                    -(singular_mass * singular_mass.clamp_min(1e-12).log()).sum()
                )

                route_count = conditioner.route_count
                if route_count > 1:
                    normalized_bank = F.normalize(bank, dim=1)
                    cosine = normalized_bank @ normalized_bank.T
                    off_diagonal = ~torch.eye(
                        route_count, device=bank.device, dtype=torch.bool
                    )
                    bank_abs_cosine = cosine[off_diagonal].abs().mean()
                    bank_distances = torch.pdist(bank)
                    bank_distance_mean = bank_distances.mean()
                    bank_distance_min = bank_distances.min()
                else:
                    bank_abs_cosine = bank_distance_mean = bank_distance_min = (
                        bank.new_zeros(())
                    )
        finally:
            conditioner.train(was_training)

        entropy_mean = entropy.mean()
        return {
            "scope": "index_only_final_schedule",
            "sample_size": count,
            "routing_temperature": routing_info.temperature,
            "effective_routed_scale": (
                float(conditioner.effective_routed_scale())
                if conditioner.routing_version == "v2" else 1.0
            ),
            "routing_entropy_normalized": float(
                entropy_mean / np.log(conditioner.route_count)
                if conditioner.route_count > 1 else 0
            ),
            "routing_effective_routes": float(entropy_mean.exp()),
            "routing_max_probability_mean": float(probabilities.max(dim=1).values.mean()),
            "routing_soft_load_min": float(loads.min()),
            "routing_soft_load_max": float(loads.max()),
            "routing_soft_load_cv": float(loads.std(unbiased=False) / loads.mean()),
            "routing_top1_active_routes": int((top1_loads > 0).sum()),
            "routing_top1_load_max": float(top1_loads.max()),
            "routing_top1_load_cv": float(top1_loads.std(unbiased=False) / top1_loads.mean()),
            "embedding_bank_effective_rank": float(bank_effective_rank),
            "embedding_bank_abs_cosine_mean": float(bank_abs_cosine),
            "embedding_bank_distance_mean": float(bank_distance_mean),
            "embedding_bank_distance_min": float(bank_distance_min),
            "routed_feature_rms_spread": float(centered.square().sum(dim=1).mean().sqrt()),
            "routed_feature_norm_mean": float(routed.norm(dim=1).mean()),
        }
