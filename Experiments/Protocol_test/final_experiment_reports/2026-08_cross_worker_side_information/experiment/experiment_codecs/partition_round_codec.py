from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from FL_code.FL_core.codec import BaseRoundCodec, ReconstructionHistory
from FL_code.FL_core.utils import decompress_data_list
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH, PriorCalculator
from FL_code.cancer_protocol.round_codecs import (
    CancerRecord,
    ElementMetrics,
    F_RoundCodec,
    Oracle_RoundCodec,
    _WZRoundCodec,
    compressed_size,
)
from FL_code.cancer_protocol.wz_quantizer import PreprocessMetadata, WZcfgQuant

from .experimental_quantizer import PreparedWZQuantizerCancer


@dataclass
class CrossPartElementMetrics(ElementMetrics):
    """Element diagnostics extended with every part model's bits and error on every coordinate."""

    cross_bits: torch.Tensor | None = None
    cross_error: torch.Tensor | None = None


class CrossPartCancerRecord(CancerRecord):
    """Cancer record whose element artifact carries this experiment's cross-part arrays."""

    element_metrics_class: ClassVar[type[ElementMetrics]] = CrossPartElementMetrics


class OP_RoundCodec(Oracle_RoundCodec):
    """Partition the vector and fit one oracle quantizer per structural coordinate part."""

    round_name: ClassVar[str] = "OP"
    record_class = CrossPartCancerRecord
    quantizer_class: type[PreparedWZQuantizerCancer] = PreparedWZQuantizerCancer

    _frozen: bool = False
    cross_evaluate: bool = False
    _client_id: int | None = None
    split_mask_path: Path
    split_mask: torch.Tensor
    part_quantizers: dict[int, PreparedWZQuantizerCancer]
    part_rows: dict[int, torch.Tensor]
    si: list[torch.Tensor]

    def options_to_config(
        self,
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        dump_element_metrics: bool = False,
        split_mask: str | None = None,
        cross_evaluate: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Separate partition policy from the common round and quantizer configuration."""
        assert split_mask is not None
        self.split_mask_path = Path(split_mask)
        self.cross_evaluate = cross_evaluate
        super().options_to_config(
            bins_per_plane,
            num_planes,
            sd_slices,
            cache_dir,
            cache_overwrite,
            dump_element_metrics,
            **quant_overrides,
        )

    @staticmethod
    def validate_cfg(
        bins_per_plane: int,
        num_planes: int,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        dump_element_metrics: bool = False,
        split_mask: str | None = None,
        cross_evaluate: bool = False,
        **quant_overrides: Any,
    ) -> None:
        """Validate the partition artifact and the independent quantizer configuration."""
        assert isinstance(cross_evaluate, bool)
        assert split_mask is not None and Path(split_mask).is_file(), (
            f"split_mask must name an existing .npy partition; got {split_mask!r}.")
        _WZRoundCodec.validate_cfg(
            bins_per_plane,
            num_planes,
            sd_slices,
            cache_dir,
            cache_overwrite,
            dump_element_metrics,
            **quant_overrides,
        )

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return oracle SI and retain the client selecting a worker-specific partition row."""
        self._client_id = client_id
        return super().get_training_data(given_history, client_id)

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> PreparedWZQuantizerCancer:
        """Load the partition and create one untrained quantizer for each label."""
        assert x is None and si is not None
        self.si = si
        stored = torch.from_numpy(np.load(self.split_mask_path))
        if stored.ndim == 2:
            assert self._client_id is not None
            assert self._client_id < stored.shape[0], (
                f"{self.split_mask_path} has {stored.shape[0]} partitions; "
                f"worker {self._client_id} has none.")
            stored = stored[self._client_id]
        assert stored.numel() == si[0].numel(), (
            f"{self.split_mask_path} covers {stored.numel()} coordinates; "
            f"the vector has {si[0].numel()}.")
        self.split_mask = stored.to(torch.long)
        self.part_rows = {
            int(label): self.split_mask == label
            for label in self.split_mask.unique().tolist()
        }
        assert len(self.part_rows) > 1, "A one-part partition codes nothing separately."
        assert all(rows.any() for rows in self.part_rows.values()), (
            "Every partition label must contain at least one coordinate.")
        self.part_quantizers = {
            part: self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
            for part in self.part_rows
        }
        self.quantizer = self.part_quantizers[min(self.part_quantizers)]
        return self.quantizer

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Normalize once, then train and encode each coordinate part independently."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        self._original_delta_vec = delta_vec
        for quantizer in self.part_quantizers.values():
            quantizer.set_side_information(self.si)
        whole = self.quantizer
        x_prep, metadata = whole.preprocess_x(delta_vec)
        model_si = whole.side_info_tensor(metadata)
        prior_si = whole.normalize_side_info_tensor(metadata, self.si)

        bins = torch.zeros(
            (self.c_cfg.num_planes, delta_vec.numel()),
            dtype=torch.uint8 if self.c_cfg.bins_per_plane <= 2**8 else torch.uint16,
        )
        soft = torch.zeros(
            (self.c_cfg.num_planes, delta_vec.numel(), self.c_cfg.bins_per_plane))
        for part, rows in sorted(self.part_rows.items()):
            quantizer = self.part_quantizers[part]
            if not self._frozen:
                quantizer.train_on_prepared(x_prep[rows], model_si[rows], int(rows.sum()))
                gc.collect()
                torch.cuda.empty_cache()
            part_bins, part_soft = quantizer.encode_prepared(
                x_prep[rows], INFERENCE_BATCH)
            bins[:, rows], soft[:, rows] = part_bins, part_soft
        for quantizer in self.part_quantizers.values():
            quantizer.vector_size = delta_vec.numel()

        artifact = record.element_metrics
        if self.cross_evaluate and artifact is not None and artifact.include_arrays:
            self.collect_cross_evaluation(
                x_prep, model_si, prior_si, metadata, delta_vec, record)

        record.update_record_fields(
            encoder_size=sum(
                compressed_size(quantizer.coding_model.encoder_state_dict())
                for quantizer in self.part_quantizers.values()),
            decoder_size=sum(
                compressed_size(quantizer.coding_model.decoder_state_dict())
                for quantizer in self.part_quantizers.values()),
            meta_data_size=compressed_size(metadata),
        )
        self.add_prior_to_record(bins, soft, metadata, record)
        return {
            "payload_content": (bins, metadata),
            "quantizer_state": {
                part: quantizer.coding_model.decoder_state_dict()
                for part, quantizer in sorted(self.part_quantizers.items())
            },
        }

    def collect_cross_evaluation(
        self,
        x_prep: torch.Tensor,
        model_si: torch.Tensor,
        prior_si: torch.Tensor,
        metadata: PreprocessMetadata,
        delta_vec: torch.Tensor,
        record: CancerRecord,
    ) -> None:
        """Evaluate every part model on every coordinate using separately trained entropy priors."""
        artifact = record.element_metrics
        assert isinstance(artifact, CrossPartElementMetrics) and artifact.include_arrays
        w_mean = delta_vec.abs().mean().item()
        assert w_mean > 0
        bits: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        for quantizer in dict(sorted(self.part_quantizers.items())).values():
            part_bins, part_soft = quantizer.encode_prepared(x_prep, INFERENCE_BATCH)
            lengths, _ = self.conditional_code_lengths(
                quantizer, part_bins, part_soft, prior_si, self.c_cfg)
            bits.append(lengths)
            normalized = quantizer.decode_stages_prepared(
                part_bins, model_si, INFERENCE_BATCH)
            reconstructions = torch.stack([
                quantizer.postprocess(stage, metadata) for stage in normalized])
            errors.append(
                ((reconstructions - delta_vec) / w_mean).abs())
            del part_bins, part_soft, lengths, normalized, reconstructions
            gc.collect()
            torch.cuda.empty_cache()
        artifact.cross_bits = torch.stack(bits).cpu()
        artifact.cross_error = torch.stack(errors).cpu()

    def add_prior_to_record(
        self,
        bins: torch.Tensor,
        soft_codes: torch.Tensor | None,
        metadata: PreprocessMetadata,
        record: CancerRecord,
    ) -> None:
        """Train one entropy prior per part and aggregate rates over the full vector."""
        assert soft_codes is not None
        prior_sources = (
            self.quantizer.side_info_list_used
            if self._prior_side_information is None
            else self._prior_side_information
        )
        assert prior_sources is not None
        side_info = self.quantizer.normalize_side_info_tensor(metadata, prior_sources)

        code_lengths = torch.zeros((self.c_cfg.num_planes, bins.shape[1]))
        prior_models = {}
        for part, rows in sorted(self.part_rows.items()):
            code_lengths[:, rows], prior_models[part] = self.conditional_code_lengths(
                self.part_quantizers[part],
                bins[:, rows],
                soft_codes[:, rows],
                side_info[rows],
                self.c_cfg,
            )
        stage_rates = code_lengths.mean(dim=1).tolist()
        artifact = record.element_metrics
        if artifact is not None:
            artifact.models = {
                "round_name_full": self.round_name_full,
                "c_cfg": self.c_cfg.model_dump(mode="json", exclude={"norm_slices"}),
                "coding_models": {
                    part: quantizer.coding_model.state_dict()
                    for part, quantizer in sorted(self.part_quantizers.items())
                },
                "prior_models": {
                    part: model.state_dict() for part, model in sorted(prior_models.items())
                },
            }
            if artifact.include_arrays:
                artifact.bits = code_lengths.cpu()
                artifact.bins = bins.cpu()

        marginal = PriorCalculator.compute_marginal_prior(
            bins, self.c_cfg.bins_per_plane, self.c_cfg.num_planes)
        marginal_rates = PriorCalculator.compute_plane_rates(
            marginal, bins, self.c_cfg.num_planes)
        record.update_record_fields(
            prior_stage_rates=stage_rates,
            prior_rate=sum(stage_rates),
            marginal_stage_rates=list(marginal_rates),
            marginal_rate=sum(marginal_rates),
        )
        for quantizer in self.part_quantizers.values():
            assert quantizer.training_prior_stage_rates is not None
        training_rates = [
            sum(
                self.part_quantizers[part].training_prior_stage_rates[stage]
                * int(rows.sum())
                for part, rows in sorted(self.part_rows.items())
            ) / bins.shape[1]
            for stage in range(self.c_cfg.num_planes)
        ]
        record.update_record_fields(
            training_prior_stage_rates=training_rates,
            training_prior_rate=sum(training_rates),
        )

    def frozen_round_codec(
        self,
        frozen_class: type[F_RoundCodec],
        round_name_full: str,
        side_information: list[torch.Tensor] | None,
    ) -> BaseRoundCodec:
        """Copy every part model into a frozen round and optionally refresh their shared SI.

        `frozen_class` is unused: this codec freezes by copying itself, because one F instance has
        to carry every part quantizer rather than the single one F is built around.
        """
        frozen = copy(self)
        frozen.round_name_full = round_name_full
        frozen._frozen = True
        frozen._original_delta_vec = None
        frozen.part_quantizers = {
            part: copy(quantizer) for part, quantizer in self.part_quantizers.items()
        }
        for quantizer in frozen.part_quantizers.values():
            quantizer.side_info_list_used = None
        frozen.quantizer = frozen.part_quantizers[min(frozen.part_quantizers)]
        if side_information is not None:
            frozen.si = list(side_information)
        return frozen

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Decode each coordinate part with its own quantizer and reassemble every stage."""
        bins, metadata = decompress_data_list(payload)["payload_content"]
        whole = self.quantizer
        side_info = whole.side_info_tensor(metadata)
        normalized = torch.zeros((self.c_cfg.num_planes, bins.shape[1]))
        for part, rows in sorted(self.part_rows.items()):
            normalized[:, rows] = self.part_quantizers[part].decode_stages_prepared(
                bins[:, rows], side_info[rows], INFERENCE_BATCH).to(normalized.dtype)
        stages = torch.stack([whole.postprocess(stage, metadata) for stage in normalized])
        self.add_stage_distortion_to_record(stages, record)
        return stages[-1]
