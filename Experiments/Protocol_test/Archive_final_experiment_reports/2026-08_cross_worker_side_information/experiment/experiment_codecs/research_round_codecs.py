"""Frozen study: last ran against commit bb5d9f1; not maintained. Reads the removed _/data/federated_learning/raw_client_deltas layout and imports removed symbols."""

from __future__ import annotations

import gc
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, ReconstructionHistory
from FL_code.cancer_protocol.multi_round import _MultiRoundTrainingMixin
from FL_code.cancer_protocol.round_codecs import (
    CancerRecord,
    Oracle_RoundCodec,
    _WZRoundCodec,
)
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer


class N_RoundCodec(_WZRoundCodec):
    """Train and price a quantizer with no side information as the non-DSC baseline."""

    round_name: ClassVar[str] = "N"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY
    _send_quantizer_state: bool

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return no SI; the raw target arrives during pre-encode training."""
        return None, []

    def build_quantizer(
        self, x: torch.Tensor | None, si: list[torch.Tensor] | None
    ) -> DedupedWZQuantizerCancer:
        """Create the marginal quantizer without training it before the raw delta arrives."""
        assert x is None and si == []
        assert self._prior_side_information is None, (
            "N cannot use server SI for its entropy prior; set "
            "retrained_prior_uses_server_si=False."
        )
        self.c_cfg = self.c_cfg.model_copy(update={"marginal_loss": True})
        self.quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=0)
        self._send_quantizer_state = False
        return self.quantizer

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Fit the marginal quantizer on the current raw target before encoding."""
        if self.round_name_full == "F":
            assert self.quantizer.side_info_list_used is not None
            assert self.quantizer.vector_size is not None
            assert self.quantizer.training_prior_stage_rates is not None
            assert not self._send_quantizer_state
            return
        assert self.quantizer.side_info_list_used is None
        assert not self._send_quantizer_state
        self.quantizer.train_model(delta_vec, None)
        assert self.quantizer.vector_size is not None
        assert self.quantizer.training_prior_stage_rates is not None
        gc.collect()
        torch.cuda.empty_cache()
        self._send_quantizer_state = True

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Encode with the trained marginal model and transmit newly trained state once."""
        assert self.quantizer.side_info_list_used is not None, (
            f"{self.round_name_full} must run pre_encode_training_steps before encoding.")
        assert self.quantizer.vector_size is not None
        assert self.quantizer.training_prior_stage_rates is not None
        payload = super().get_encod_payload(delta_vec, record)
        if self._send_quantizer_state:
            payload["quantizer_state"] = self.quantizer.coding_model.decoder_state_dict()
            self._send_quantizer_state = False
        return payload


class OD_RoundCodec(Oracle_RoundCodec):
    """Oracle quantizer trained with own-worker SI columns withheld or scrambled on some rows."""

    # TODO: OD is intentionally incomplete. Restore it with experiment-local config fields for
    # dropout/span/scrambling and an experimental quantizer that perturbs normalized training SI.
    # If missingness must be signalled, wrap the model locally to replace a trailing 0/1 flag with
    # a learned embedding before decoder and conditional-prior inputs.

    round_name: ClassVar[str] = "OD"

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return server SI and record the column span belonging to the current worker."""
        assert self.c_cfg.own_si_dropout or self.c_cfg.own_si_scramble, (
            f"{self.round_name_full} requires own_si_dropout or own_si_scramble."
        )
        history = given_history.view(self.can_decode_where, client_id)
        clients = list(history)
        assert client_id in clients, f"Client {client_id} has no server history to perturb."
        start = sum(len(history[earlier]) for earlier in clients[:clients.index(client_id)])
        self.c_cfg = type(self.c_cfg).model_validate(
            self.c_cfg.model_dump()
            | {"own_si_span": (start, start + len(history[client_id]))}
        )
        return None, [
            entry.tensor for client_history in history.values() for entry in client_history
        ]


class OM_RoundCodec(_MultiRoundTrainingMixin, Oracle_RoundCodec):
    """Train an oracle round on current and selected lagged repetitions of its problem."""

    round_name: ClassVar[str] = "OM"
