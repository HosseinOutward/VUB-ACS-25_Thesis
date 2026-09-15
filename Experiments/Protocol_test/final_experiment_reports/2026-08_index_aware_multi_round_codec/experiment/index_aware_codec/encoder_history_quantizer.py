"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer,
    PreprocessMetadata,
    WZQuantizerCancer,
    WZcfgQuant,
    _sample_mean,
)


class EncoderHistoryWZcfgQuant(WZcfgQuant):
    """Configure an OM/TM quantizer whose encoder also sees selected raw-source history columns."""

    encoder_side_information_indices: tuple[int, ...] = ()

    def training_cache_payload(self) -> dict[str, object]:
        """Include the selected encoder-history columns in the trained-model cache identity."""
        return super().training_cache_payload() | {
            "encoder_side_information_indices": self.encoder_side_information_indices
        }


class EncoderHistoryDedupedWZQuantizer(DedupedWZQuantizerCancer):
    """Feed selected decoder-known local history to the encoder without changing other inputs."""

    c_cfg: EncoderHistoryWZcfgQuant
    candidate_failure_tolerance: ClassVar[int] = 2

    def __init__(self, c_cfg: EncoderHistoryWZcfgQuant, si_size: int) -> None:
        assert c_cfg.encoder_side_information_indices
        assert min(c_cfg.encoder_side_information_indices) >= 0
        assert len(set(c_cfg.encoder_side_information_indices)) == len(
            c_cfg.encoder_side_information_indices
        )
        assert max(c_cfg.encoder_side_information_indices) < si_size
        super().__init__(c_cfg, si_size)

    def _new_coding_model(self, side_info_size: int) -> EncoderDecoderLayeredRNN:
        """Create the ordinary WZ model with a wider encoder input only."""
        return EncoderDecoderLayeredRNN(
            input_dim=1,
            encoder_input_dim=1 + len(self.c_cfg.encoder_side_information_indices),
            side_info_size=side_info_size,
            bins_per_plane=self.bins_per_plane,
            num_planes=self.num_planes,
            layers=self.layers,
            hidden_dim=self.hidden_dim,
            shared_encoder=self.shared_heads,
            shared_decoder=self.shared_heads,
            marginal=self.marginal_loss,
        )

    def _cached_training_quantizer_class(self) -> type[WZQuantizerCancer]:
        """Recreate the encoder-history architecture when populating a cache miss."""
        return EncoderHistoryDedupedWZQuantizer

    def _with_encoder_history(
        self, source: torch.Tensor, side_info: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (source, side_info[:, self.c_cfg.encoder_side_information_indices]), dim=1
        )

    def _stack_training_pairs(
        self,
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ordinary OM/TM problems, then add their aligned local history to the encoder."""
        source, stacked_side_info = super()._stack_training_pairs(
            x_prep, side_info, extra_pairs
        )
        return self._with_encoder_history(source, stacked_side_info), stacked_side_info

    def encoding_process(
        self, x_raw: torch.Tensor, batch_size: int = INFERENCE_BATCH
    ) -> tuple[torch.Tensor, torch.Tensor, PreprocessMetadata]:
        """Encode the current source with the decoder-aligned local raw-source history window."""
        source, metadata = self.preprocess_x(x_raw)
        encoder_input = self._with_encoder_history(source, self.side_info_tensor(metadata))
        bins, soft_codes = self.encode_prepared(encoder_input, batch_size)
        return bins, soft_codes, metadata

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: torch.Tensor) -> None:
        """Fit against the current value while treating the remaining columns as encoder context."""
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
        """Summarize reconstruction and rate without treating encoder history as a target."""
        encoder_input = training_inputs[0][:10_000]
        target = encoder_input[:, :1]
        y = side_info[: encoder_input.shape[0]]
        bins, codes = model.encode(encoder_input)
        reconstructions = model.decode(codes, y)
        priors = model.get_priors(codes, y)
        sample_indices = torch.arange(encoder_input.shape[0], device=encoder_input.device)
        plane_rates = [
            -torch.log2(prior[sample_indices, symbols] + 1e-12).mean().item()
            for prior, symbols in zip(priors, bins, strict=True)
        ]
        return (
            f"hard_mse={[round(F.mse_loss(value, target).item(), 6) for value in reconstructions]}, "
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
        """Compute ordinary WZ RD loss with local history used only as encoder context."""
        encoder_input, = training_input
        target = encoder_input[:, :1]
        if c_cfg.source_perturbation != "none":
            perturbation = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=perturb_generator,
            ) * (1e-5 * target.abs().mean())
            if c_cfg.source_perturbation == "add":
                encoder_input = encoder_input.clone()
                encoder_input[:, :1] = target + perturbation
                target = encoder_input[:, :1]

        progress = epoch / c_cfg.train_epochs
        tau = c_cfg.tau * np.exp(progress * np.log(0.1 / c_cfg.tau))
        reconstructions, bins, soft_codes, priors = model(
            encoder_input, si_batch, tau=tau
        )
        loss = torch.zeros((), device=encoder_input.device)
        sample_indices = torch.arange(encoder_input.shape[0], device=encoder_input.device)
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
        return loss


__all__ = ["EncoderHistoryDedupedWZQuantizer", "EncoderHistoryWZcfgQuant"]
