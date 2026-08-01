from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, BaseProtocol, BaseRoundCodec, ReconstructionHistory

from .round_codecs import (
    CancerRecord,
    F_RoundCodec,
    Oracle_RoundCodec,
    P_RoundCodec,
    R_RoundCodec,
    T_RoundCodec,
    _WZRoundCodec,
)
from .wz_quantizer import WZcfgQuant


class _WZProtocol(BaseProtocol):
    """Schedule WZ round policies and own the decoder-visible reconstruction ledgers."""

    max_per_client_recons_history = 5
    warmup_round_codecs: ClassVar[tuple[str, ...]]
    routine_round_codecs: ClassVar[tuple[str, ...]]
    protocol_name: ClassVar[str]

    refresh_frozen_si: bool = True
    retrained_prior_uses_server_si: bool = True
    train_offsets: tuple[int, ...] = (1, 2, 3)
    last_trained_codec: dict[int, _WZRoundCodec]
    _lagged_history: ReconstructionHistory | None
    _train_at_round: int | None = None

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
    ) -> None:
        super().__init__(
            options,
            protocol_name_full,
            sd_slices,
            cache_dir,
            cache_overwrite,
        )
        self.last_trained_codec = {}
        lagged_depth = max(
            (
                round_codec.history_depth(self.train_offsets)
                for round_codec in self._round_codec_classes.values()
                if issubclass(round_codec, _WZRoundCodec)
            ),
            default=0,
        )
        self._lagged_history = (
            ReconstructionHistory(
                max_per_client=self.max_per_client_recons_history + lagged_depth,
                routine_accesses=self.routine_accesses(),
            )
            if lagged_depth
            else None
        )
        if self._train_at_round is not None:
            assert self._train_at_round >= self.max_per_client_recons_history + lagged_depth, (
                f"train_at_round={self._train_at_round} leaves fewer than "
                f"{self.max_per_client_recons_history + lagged_depth} preceding history rounds.")

    def histories(self) -> tuple[ReconstructionHistory, ...]:
        """Return the ordinary decoder ledger and any deeper lagged-training ledger."""
        if self._lagged_history is None:
            return super().histories()
        return (*super().histories(), self._lagged_history)

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the scheduled codec with only the history and model state its policy requests."""
        round_codec_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        if not issubclass(round_codec_class, _WZRoundCodec):
            return round_codec_class(parsed.options, round_name_full)

        previous = self.last_trained_codec.get(client_id)
        if round_codec_class is F_RoundCodec:
            assert previous is not None, f"No frozen quantizer exists for client {client_id}."
            side_information = None
            if self.refresh_frozen_si:
                side_information = [
                    entry.tensor
                    for client_history in self._recons_history.view(
                        previous.can_decode_where, client_id
                    ).values()
                    for entry in client_history
                ]
                assert len(side_information) == previous.quantizer.side_info_size, (
                    "Refreshed side information no longer matches the frozen quantizer.")
            frozen = previous.frozen_round_codec(round_name_full, side_information)
            assert isinstance(frozen, _WZRoundCodec)
            if self.retrained_prior_uses_server_si:
                frozen.set_retrained_prior_si(self._recons_history, client_id)
            return frozen

        assert isinstance(parsed.options, dict)
        history_depth = round_codec_class.history_depth(self.train_offsets)
        training_history = self._recons_history
        if history_depth:
            assert self._lagged_history is not None
            training_history = self._lagged_history
        codec = round_codec_class(
            parsed.options | {
                "sd_slices": self.sd_slices,
                "cache_dir": self.cache_dir,
                "cache_overwrite": self.cache_overwrite,
            },
            round_name_full,
            **round_codec_class.protocol_inputs(
                self.max_per_client_recons_history,
                self.train_offsets,
                previous,
            ),
        )
        assert isinstance(codec, _WZRoundCodec)

        target, side_information = codec.get_training_data(training_history, client_id)
        if self.retrained_prior_uses_server_si:
            codec.set_retrained_prior_si(self._recons_history, client_id)
        codec.build_quantizer(target, side_information)
        self.last_trained_codec[client_id] = codec
        return codec

    def config_snapshot(self) -> dict[str, Any]:
        """Return the WZ schedule, quantizer defaults, and SI/history policies."""
        snapshot = super().config_snapshot()
        snapshot["quantizer_defaults"] = WZcfgQuant(
            bins_per_plane=1,
            num_planes=1,
        ).model_dump(
            mode="json",
            exclude={
                "bins_per_plane",
                "num_planes",
                "norm_slices",
                "cache_dir",
                "cache_overwrite",
            },
        )
        snapshot["refresh_frozen_si"] = self.refresh_frozen_si
        snapshot["retrained_prior_uses_server_si"] = self.retrained_prior_uses_server_si
        if self._lagged_history is not None:
            snapshot["train_offsets"] = self.train_offsets
            snapshot["max_per_client_offset_history"] = self._lagged_history.max_per_client
        return snapshot


class WZCancerProtocol(_WZProtocol):
    """Cancer-model WZ schedule with training rounds followed by frozen-model rounds."""

    warmup_round_codecs: tuple[str, ...] = (
        "I",
        "I",
        "O|bins_per_plane=8|num_planes=3",
        "O|bins_per_plane=4|num_planes=3",
    )
    routine_round_codecs: tuple[str, ...] = (
        "O|bins_per_plane=3|num_planes=3",
        "F",
        "F",
        "F",
        "F",
    )
    protocol_name: ClassVar[str] = "wz_cancer"

    def options_to_config(
        self,
        replace_O_with: str | None = None,
        extra_train_tokens: Sequence[str] = (),
        refresh_frozen_si: bool = True,
        retrained_prior_uses_server_si: bool = True,
        train_at_round: int | None = None,
        train_offsets: int | Sequence[int] | None = None,
    ) -> None:
        """Configure SI policies and optionally replace/collapse the scheduled training rounds."""
        assert isinstance(refresh_frozen_si, bool)
        assert isinstance(retrained_prior_uses_server_si, bool)
        self.refresh_frozen_si = refresh_frozen_si
        self.retrained_prior_uses_server_si = retrained_prior_uses_server_si
        if train_offsets is not None:
            offsets = (
                tuple(range(1, train_offsets + 1))
                if isinstance(train_offsets, int)
                else tuple(train_offsets)
            )
            assert offsets and all(isinstance(offset, int) and offset > 0 for offset in offsets), (
                "train_offsets must be a positive count or a sequence of positive offsets.")
            assert len(set(offsets)) == len(offsets), (
                f"train_offsets repeats an offset: {offsets}.")
            self.train_offsets = tuple(sorted(offsets))

        assert not extra_train_tokens or replace_O_with is not None, (
            "extra_train_tokens configures the replacement round, so replace_O_with is required.")
        if replace_O_with is not None:
            replacement = "|".join((replace_O_with, *extra_train_tokens))
            self.warmup_round_codecs = tuple(
                round_name.replace("O|", f"{replacement}|")
                if round_name.startswith("O|")
                else round_name
                for round_name in self.warmup_round_codecs
            )
            self.routine_round_codecs = tuple(
                round_name.replace("O|", f"{replacement}|")
                if round_name.startswith("O|")
                else round_name
                for round_name in self.routine_round_codecs
            )
        if train_at_round is not None:
            assert isinstance(train_at_round, int) and train_at_round >= 0
            self._train_at_round = train_at_round
            self.warmup_round_codecs = (
                ("I",) * train_at_round + (self.routine_round_codecs[0],))
            self.routine_round_codecs = ("F",)


__all__ = [
    "CancerRecord",
    "F_RoundCodec",
    "Oracle_RoundCodec",
    "P_RoundCodec",
    "R_RoundCodec",
    "T_RoundCodec",
    "WZCancerProtocol",
]
