"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

import torch

from FL_code.FL_core.cache import cache_call, model_training_hash
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer,
    WZcfgQuant,
    WZQuantizerCancer,
)


def _train_prepared_parameters(
    quantizer_class: type[WZQuantizerCancer],
    c_cfg: WZcfgQuant,
    x_prep: torch.Tensor,
    side_info: torch.Tensor,
    vector_size: int,
) -> tuple[dict[str, torch.Tensor], tuple[float, ...]]:
    """Train a quantizer on normalized rows selected by the partition codec."""
    quantizer = quantizer_class(
        c_cfg=c_cfg,
        si_size=side_info.shape[1],
    )
    quantizer.vector_size = vector_size
    quantizer._fit_prepared(x_prep, side_info)
    assert quantizer.training_prior_stage_rates is not None
    return quantizer.coding_model.state_dict(), quantizer.training_prior_stage_rates


class PreparedWZQuantizerCancer(DedupedWZQuantizerCancer):
    """Report quantizer supporting training on a partition of normalized rows."""

    def train_on_prepared(
        self,
        x_prep: torch.Tensor,
        side_info: torch.Tensor,
        vector_size: int,
    ) -> None:
        """Train on already-normalized rows selected from the complete vector.

        The partition codec normalizes the whole vector once, then gives each quantizer only its
        selected rows. ``vector_size`` remains the length of the complete coded vector.
        """
        state_dict, training_prior_stage_rates = cache_call(
            _train_prepared_parameters,
            None if self.cache_dir is None else self.cache_dir / "quantizers",
            model_training_hash,
            self.cache_overwrite,
            quantizer_class=WZQuantizerCancer,
            c_cfg=self.c_cfg,
            x_prep=x_prep,
            side_info=side_info,
            vector_size=vector_size,
        )
        self.vector_size = vector_size
        self.coding_model.load_state_dict(state_dict)
        self.training_prior_stage_rates = training_prior_stage_rates
        self.training_prior_rate = sum(training_prior_stage_rates)
