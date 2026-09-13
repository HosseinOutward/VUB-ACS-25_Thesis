"""Cheating reconstruction-ordered labeling for the oracle Slepian--Wolf codec."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
import torch

from .oracle_sw_mixin import OracleSWPayload
from .sw_coder import ConditionalPrior
from ..wz_quantizer import PreprocessMetadata


class IndexMappingPlan(BaseModel):
    """Per-stage original-symbol to transmitted-label permutations sent with a payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_to_label: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def validate_permutations(self) -> Self:
        """Require equally sized, non-empty permutations at every stage."""
        assert self.original_to_label
        alphabet = len(self.original_to_label[0])
        assert alphabet >= 2
        expected = list(range(alphabet))
        assert all(sorted(stage) == expected for stage in self.original_to_label)
        return self

    @property
    def description_bits(self) -> int:
        """Return the canonical JSON size when the mapping is transmitted explicitly."""
        return len(self.model_dump_json().encode()) * 8

    def map_indices(self, bins: torch.Tensor) -> torch.Tensor:
        """Map original staged symbols to their transmitted labels."""
        assert bins.shape[0] == len(self.original_to_label)
        mapped = torch.empty_like(bins)
        for stage, permutation in enumerate(self.original_to_label):
            labels = torch.tensor(permutation, dtype=torch.long, device=bins.device)
            mapped[stage] = labels[bins[stage].long()]
        return mapped

    def unmap_indices(self, labels: torch.Tensor) -> torch.Tensor:
        """Recover original staged symbols from transmitted labels."""
        assert labels.shape[0] == len(self.original_to_label)
        bins = torch.empty_like(labels)
        for stage, permutation in enumerate(self.original_to_label):
            inverse = torch.argsort(torch.tensor(permutation, dtype=torch.long, device=labels.device))
            bins[stage] = inverse[labels[stage].long()]
        return bins

    def map_prior(self, original_prior: torch.Tensor, stage: int) -> torch.Tensor:
        """Permute an original-symbol prior into transmitted-label order."""
        permutation = torch.tensor(self.original_to_label[stage], dtype=torch.long, device=original_prior.device)
        mapped = torch.empty_like(original_prior)
        mapped[:, permutation] = original_prior
        return mapped


class MappedOracleSWPayload(OracleSWPayload):
    """An oracle SW payload carrying the value-derived symbol-label permutation."""

    mapping: IndexMappingPlan

    @model_validator(mode="after")
    def validate_mapping_shape(self) -> Self:
        """Match the mapping dimensions to the accompanying SW rate plan."""
        assert len(self.mapping.original_to_label) == self.plan.num_stages
        assert all(len(stage) == self.plan.alphabet for stage in self.mapping.original_to_label)
        return self

    @property
    def wire_bits(self) -> int:
        """Include the value-derived label mapping that must accompany the syndrome."""
        return super().wire_bits + self.mapping.description_bits


class OracleValueOrderedIndexMixin:
    """Relabel staged symbols by reconstruction value along a Gray-code path."""

    _active_index_mapping: IndexMappingPlan

    def _value_ordered_mapping(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> IndexMappingPlan:
        """Derive a cheating per-stage mapping from forced-symbol reconstructions."""
        alphabet = self.quantizer.bins_per_plane
        gray_path = tuple(rank ^ (rank >> 1) for rank in range(alphabet))
        ordinary = self.quantizer.decoding_stages_process((bins, metadata))
        permutations: list[tuple[int, ...]] = []

        for stage in range(self.quantizer.num_planes):
            centroids: list[float] = []
            for symbol in range(alphabet):
                forced = bins.clone()
                forced[stage].fill_(symbol)
                reconstruction = self.quantizer.decoding_stages_process((forced, metadata))[stage]
                contribution = reconstruction if stage == 0 else reconstruction - ordinary[stage - 1]
                centroids.append(float(contribution.mean()))

            ordered_symbols = sorted(range(alphabet), key=centroids.__getitem__)
            original_to_label = [0] * alphabet
            for rank, symbol in enumerate(ordered_symbols):
                original_to_label[symbol] = gray_path[rank]
            permutations.append(tuple(original_to_label))

        return IndexMappingPlan(original_to_label=tuple(permutations))

    def _sw_context(self, metadata: PreprocessMetadata) -> tuple[torch.Tensor, ConditionalPrior]:
        """Adapt the quantizer's original-symbol prior to the active transmitted labels."""
        side_info, original_prior_provider = super()._sw_context(metadata)
        mapping = self._active_index_mapping

        def mapped_prior_provider(
            decoded_labels: torch.Tensor, decoder_side_info: torch.Tensor, stage: int
        ) -> torch.Tensor:
            original_bins = mapping.unmap_indices(decoded_labels)
            original_prior = original_prior_provider(original_bins, decoder_side_info, stage)
            return mapping.map_prior(original_prior, stage)

        return side_info, mapped_prior_provider

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> MappedOracleSWPayload:
        """Relabel original indices before invoking the ordinary oracle SW encoder."""
        mapping = self._value_ordered_mapping(bins, metadata)
        self._active_index_mapping = mapping
        payload = super().encode_staged_indices(mapping.map_indices(bins), metadata)
        return MappedOracleSWPayload(plan=payload.plan, syndrome=payload.syndrome, mapping=mapping)

    def decode_staged_indices(self, payload: OracleSWPayload, metadata: PreprocessMetadata) -> torch.Tensor:
        """Decode transmitted labels and invert their value-derived permutation."""
        assert isinstance(payload, MappedOracleSWPayload)
        self._active_index_mapping = payload.mapping
        decoded_labels = super().decode_staged_indices(payload, metadata)
        return payload.mapping.unmap_indices(decoded_labels)


__all__ = ["IndexMappingPlan", "MappedOracleSWPayload", "OracleValueOrderedIndexMixin"]
