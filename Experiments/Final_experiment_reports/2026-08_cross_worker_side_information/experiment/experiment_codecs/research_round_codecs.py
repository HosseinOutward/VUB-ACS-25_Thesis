from __future__ import annotations

import gc
from collections.abc import Mapping
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, ReconstructionHistory
from FL_code.cancer_protocol.round_codecs import (
    CancerRecord,
    Oracle_RoundCodec,
    T_RoundCodec,
    _WZRoundCodec,
)
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer


class N_RoundCodec(_WZRoundCodec):
    """Train and price a quantizer with no side information as the non-DSC baseline."""

    round_name: ClassVar[str] = "N"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return an explicitly empty SI set; the raw target arrives during encode."""
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
        return self.quantizer

    def get_encod_payload(
        self, delta_vec: torch.Tensor, record: CancerRecord
    ) -> dict[str, Any]:
        """Fit the marginal quantizer at the worker and transmit its decoder state."""
        self.quantizer.train_model(delta_vec, None)
        gc.collect()
        torch.cuda.empty_cache()
        payload = super().get_encod_payload(delta_vec, record)
        payload["quantizer_state"] = self.quantizer.coding_model.decoder_state_dict()
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


class OM_RoundCodec(T_RoundCodec):
    """Train an oracle round on current and selected lagged repetitions of its problem."""

    round_name: ClassVar[str] = "OM"
    can_decode_where: ClassVar[Access] = Access.SERVER_ONLY

    train_offsets: tuple[int, ...]
    history_window: int

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        train_offsets: tuple[int, ...],
        history_window: int,
    ) -> None:
        super().__init__(cfg_options, round_name_full)
        assert train_offsets and history_window > 0
        self.train_offsets = train_offsets
        self.history_window = history_window

    @classmethod
    def history_depth(cls, train_offsets: tuple[int, ...]) -> int:
        """Retain history through the oldest configured lagged training problem."""
        return max(train_offsets)

    @classmethod
    def protocol_inputs(
        cls,
        history_window: int,
        train_offsets: tuple[int, ...],
        previous_codec: _WZRoundCodec | None,
    ) -> dict[str, Any]:
        """Supply the selected offsets and normal SI-window width."""
        return {"history_window": history_window, "train_offsets": train_offsets}

    def get_training_data(
        self, given_history: ReconstructionHistory, client_id: int
    ) -> tuple[None, list[torch.Tensor]]:
        """Return current SI and retain each selected lagged target/SI training problem."""
        history = given_history.view(self.can_decode_where, client_id)
        depth = max(self.train_offsets)
        assert all(
            len(ledger) == self.history_window + depth for ledger in history.values()
        ), (
            f"{self.round_name_full} needs {self.history_window + depth} entries per client; "
            f"got {[len(ledger) for ledger in history.values()]}."
        )
        self.extra_training_pairs = [
            (
                history[client_id][-offset].tensor.float(),
                [
                    entry.tensor
                    for ledger in history.values()
                    for entry in ledger[-offset - self.history_window:-offset]
                ],
            )
            for offset in self.train_offsets
        ]
        return None, [
            entry.tensor
            for ledger in history.values()
            for entry in ledger[-self.history_window:]
        ]
