"""Frozen study: last ran against commit 3df7927; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

from typing import ClassVar

import torch

from FL_code.FL_core.codec import Access, ReconstructionHistory
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.multi_round import _MultiRoundTrainingMixin
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.round_codecs import (
    Oracle_RoundCodec,
    T_RoundCodec,
)
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant

from .encoder_history_quantizer import (
    EncoderHistoryDedupedWZQuantizer,
    EncoderHistoryWZcfgQuant,
)
from .index_aware_model import IndexAwareEncoderDecoderLayeredRNN
from .index_quantizer import IndexAwareDedupedWZQuantizer, IndexAwareWZcfgQuant


class _IndexAwareRoundCodec:
    """Select the experimental quantizer and fit an uncached index-aware entropy prior."""

    config_class: ClassVar[type[IndexAwareWZcfgQuant]] = IndexAwareWZcfgQuant
    quantizer_class: ClassVar[type[IndexAwareDedupedWZQuantizer]] = (
        IndexAwareDedupedWZQuantizer
    )
    c_cfg: IndexAwareWZcfgQuant
    quantizer: IndexAwareDedupedWZQuantizer

    def conditional_code_lengths(
        self,
        quantizer: DedupedWZQuantizerCancer,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: WZcfgQuant,
    ) -> tuple[torch.Tensor, EncoderDecoderLayeredRNN]:
        """Fit the separate prior while retaining the quantizer's frozen index conditioner."""
        assert quantizer.vector_size is not None
        source_model = quantizer.coding_model
        prior_cfg = c_cfg.model_copy(update={"marginal_loss": False, "cache_dir": None})

        def model_factory(indexed_side_info_size: int) -> EncoderDecoderLayeredRNN:
            assert indexed_side_info_size >= 2
            # TI quantization uses worker-local SI, while its separately trained
            # entropy prior legitimately sees the wider server-side SI window.
            model = quantizer._new_coding_model(indexed_side_info_size - 1)
            assert isinstance(model, IndexAwareEncoderDecoderLayeredRNN)
            model.index_conditioner.load_state_dict(
                source_model.index_conditioner.state_dict()
            )
            model.set_vector_size(quantizer.vector_size)
            return model

        prior_model = PriorCalculator.make_trained_prior_model(
            bins.long(),
            soft_codes,
            side_info,
            prior_cfg,
            model_factory=model_factory,
        )
        prior = quantizer.prior_calculator.compute_prior_from_network(
            prior_model, bins, side_info
        )
        return (
            PriorCalculator.compute_code_lengths(prior, bins, prior_cfg.num_planes),
            prior_model,
        )


class OI_RoundCodec(_IndexAwareRoundCodec, Oracle_RoundCodec):
    """Oracle round whose encoder, decoder, and prior share index conditioning."""

    round_name: ClassVar[str] = "OI"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY


class TI_RoundCodec(_IndexAwareRoundCodec, T_RoundCodec):
    """Worker-trained round whose encoder, decoder, and prior share index conditioning."""

    round_name: ClassVar[str] = "TI"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO


class OMI_RoundCodec(_MultiRoundTrainingMixin, OI_RoundCodec):
    """Index-aware multi-round oracle using server-visible history from every client."""

    round_name: ClassVar[str] = "OMI"


class TMI_RoundCodec(_MultiRoundTrainingMixin, TI_RoundCodec):
    """Index-aware multi-round worker codec using only its temporal history."""

    round_name: ClassVar[str] = "TMI"


class _UncachedReferencePrior:
    """Keep comparison caches limited to the expensive quantizer models."""

    def conditional_code_lengths(
        self,
        quantizer: IndexAwareDedupedWZQuantizer,
        bins: torch.Tensor,
        soft_codes: torch.Tensor,
        side_info: torch.Tensor,
        c_cfg: IndexAwareWZcfgQuant,
    ) -> tuple[torch.Tensor, EncoderDecoderLayeredRNN]:
        """Train the ordinary entropy-prior estimate without persisting it."""
        return super().conditional_code_lengths(
            quantizer,
            bins,
            soft_codes,
            side_info,
            c_cfg.model_copy(update={"cache_dir": None}),
        )


class OM_RoundCodec(_UncachedReferencePrior, _MultiRoundTrainingMixin, Oracle_RoundCodec):
    """Non-index oracle reference trained on current and lagged problems."""

    round_name: ClassVar[str] = "OM"


class TM_RoundCodec(_UncachedReferencePrior, _MultiRoundTrainingMixin, T_RoundCodec):
    """Non-index temporal reference trained on current and lagged problems."""

    round_name: ClassVar[str] = "TM"


class _EncoderHistoryRoundMixin:
    """Select the requesting client's raw-source local-history columns as encoder context."""

    config_class: ClassVar[type[EncoderHistoryWZcfgQuant]] = EncoderHistoryWZcfgQuant
    quantizer_class: ClassVar[type[EncoderHistoryDedupedWZQuantizer]] = (
        EncoderHistoryDedupedWZQuantizer
    )
    c_cfg: EncoderHistoryWZcfgQuant

    def get_training_data(
        self,
        given_history: ReconstructionHistory,
        client_id: int,
    ) -> tuple[None, list[torch.Tensor]]:
        """Retain OM/TM problems and mark the local raw-history slice for the encoder."""
        target, side_information = super().get_training_data(given_history, client_id)
        history = given_history.view(self.can_decode_where, client_id)
        client_position = tuple(history).index(client_id)
        start = client_position * self.history_window
        self.c_cfg = self.c_cfg.model_copy(update={
            "encoder_side_information_indices": tuple(
                range(start, start + self.history_window)
            )
        })
        return target, side_information


class OM_FE_RoundCodec(_EncoderHistoryRoundMixin, OM_RoundCodec):
    """OM codec whose encoder also receives the client's raw-source local history."""

    round_name: ClassVar[str] = "OM_FE"


class TM_FE_RoundCodec(_EncoderHistoryRoundMixin, TM_RoundCodec):
    """TM codec whose encoder also receives the client's raw-source local history."""

    round_name: ClassVar[str] = "TM_FE"


__all__ = [
    "OI_RoundCodec",
    "OM_FE_RoundCodec",
    "OM_RoundCodec",
    "OMI_RoundCodec",
    "TI_RoundCodec",
    "TM_FE_RoundCodec",
    "TM_RoundCodec",
    "TMI_RoundCodec",
]
