from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, BaseProtocol, BaseRoundCodec, CompressionRecord

from .history import CompressedReconstructionHistory, HistoryVector, ReconstructionHistory
from .round_codecs import (
    CancerRecord,
    F_RoundCodec,
    Oracle_RoundCodec,
    T_RoundCodec,
    _WZRoundCodec,
)
from .wz_quantizer import WZcfgQuant


class _WZProtocol(BaseProtocol):
    """Schedule WZ round policies and own the decoder-visible reconstruction ledger.

    The ledgers are the protocol's own state, not the FL loop's: the loop only hands each decoded
    reconstruction back through `after_decode`. One training problem per round: each scheduled codec
    is trained from the ledger the decoder can see, and frozen rounds replay the model the last
    training round produced. Training on more than the current round is an experiment-side extension
    of `allocate_training_histories` and `training_problem`; nothing here assumes it exists.
    """

    max_per_client_recons_history: ClassVar[int | None] = 5
    _recons_history: ReconstructionHistory
    # Every ledger this protocol commits decoded reconstructions to, and that a replay seeds.
    histories: tuple[ReconstructionHistory, ...]
    warmup_round_codecs: tuple[str, ...]
    routine_round_codecs: tuple[str, ...]
    protocol_name: ClassVar[str]

    refresh_frozen_si: bool = True
    retrained_prior_uses_server_si: bool = True
    last_trained_codec: dict[int, _WZRoundCodec]

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
        self._recons_history = CompressedReconstructionHistory(self.max_per_client_recons_history)
        self.histories = (self._recons_history,)
        self.allocate_training_histories()

    def allocate_training_histories(self) -> None:
        """Append ledgers beyond the decoder-visible one; a single-round schedule needs none."""

    def install_histories(self, histories: tuple[ReconstructionHistory, ...]) -> None:
        """Install an independent cached decoder-history state for sparse replay."""
        assert len(histories) == len(self.histories)
        assert [history.config_snapshot() for history in histories] == [
            history.config_snapshot() for history in self.histories]
        self.histories = histories
        self._recons_history = histories[0]

    def after_decode(
        self, reconst: torch.Tensor, record: CompressionRecord, round_codec: BaseRoundCodec
    ) -> None:
        """Commit the round's reconstruction under the access level that codec decodes with."""
        for history in self.histories:
            history.commit(reconst, record, round_codec.can_decode_where)

    def training_problem(
        self, codec: _WZRoundCodec, round_id: int, client_id: int
    ) -> tuple[torch.Tensor | None, list[HistoryVector] | None]:
        """Return the (target, SI) this codec trains on, read from the decoder-visible ledger."""
        return codec.get_training_data(self._recons_history, client_id)

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the scheduled codec with only the history and model state its policy requests."""
        round_codec_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        if round_codec_class is F_RoundCodec:
            return self._reuse_trained_codec(
                round_name_full, self.last_trained_codec.get(client_id), client_id)

        if not issubclass(round_codec_class, _WZRoundCodec):
            return round_codec_class(parsed.options, round_name_full)

        assert isinstance(parsed.options, dict)
        codec = round_codec_class(
            parsed.options | {
                "sd_slices": self.sd_slices,
                "cache_dir": self.cache_dir,
                "cache_overwrite": self.cache_overwrite,
                # Cases share a cache root, so schedule provenance must separate histories that
                # happen to have the same target and side-information count.
                "training_cache_discriminator": (
                    self.protocol_name_full,
                    *self.warmup_round_codecs,
                    *self.routine_round_codecs,
                ),
            },
            round_name_full,
        )
        assert isinstance(codec, _WZRoundCodec)

        target, side_information = self.training_problem(codec, round_id, client_id)
        if self.retrained_prior_uses_server_si:
            codec.set_retrained_prior_si(self._recons_history, client_id)
        codec.build_quantizer(target, side_information)
        self.last_trained_codec[client_id] = codec
        return codec

    def _reuse_trained_codec(
        self, round_name_full: str, previous: _WZRoundCodec | None, client_id: int
    ) -> _WZRoundCodec:
        """Reuse the client's trained codec, advancing its SI window if configured."""
        assert previous is not None, f"No frozen quantizer exists for client {client_id}."
        assert previous.quantizer.vector_size is not None, (
            f"Client {client_id}'s previous codec has not completed pre-encode training and encoding.")
        assert previous.quantizer.side_info_list_used is not None
        assert previous.quantizer.training_prior_stage_rates is not None
        assert not getattr(previous, "_send_quantizer_state", False), (
            f"Client {client_id}'s trained decoder state has not been transmitted yet.")
        side_information = None
        # A marginal quantizer codes without side information, so there is no SI window to
        # advance; refreshing is silently inapplicable rather than a configuration error.
        if self.refresh_frozen_si and not previous.quantizer.no_side_info:
            side_information = [
                entry.value
                for client_history in \
                     self._recons_history.view(previous.can_decode_where, client_id).values()
                for entry in client_history
            ]
            assert len(side_information) == previous.quantizer.side_info_size, (
                "Refreshed side information no longer matches the frozen quantizer.")
        previous.round_name_full = round_name_full
        previous.dump_element_metrics = False
        previous._original_delta_vec = None
        if side_information is not None:
            previous.refresh_side_information(side_information)
        if self.retrained_prior_uses_server_si:
            previous.set_retrained_prior_si(self._recons_history, client_id)
        return previous

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
        snapshot["max_per_client_recons_history"] = self.max_per_client_recons_history
        snapshot["refresh_frozen_si"] = self.refresh_frozen_si
        snapshot["retrained_prior_uses_server_si"] = self.retrained_prior_uses_server_si
        snapshot["history"] = self._recons_history.config_snapshot()
        # Codec facts and resolved transport settings, so D/E read oracle grants and rate-field
        # availability from here rather than inferring them from a display label like codec_id.
        for round_name, entry in snapshot["round_codecs"].items():
            round_codec_class = self._round_codec_classes[round_name]
            if issubclass(round_codec_class, _WZRoundCodec):
                entry["codec_provenance"] = round_codec_class.codec_provenance(**entry["options"])
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
        refresh_frozen_si: bool = True,
        retrained_prior_uses_server_si: bool = True,
    ) -> None:
        """Configure the SI policies used by frozen and retrained codecs."""
        assert isinstance(refresh_frozen_si, bool)
        assert isinstance(retrained_prior_uses_server_si, bool)
        self.refresh_frozen_si = refresh_frozen_si
        self.retrained_prior_uses_server_si = retrained_prior_uses_server_si


__all__ = [
    "CancerRecord",
    "F_RoundCodec",
    "Oracle_RoundCodec",
    "T_RoundCodec",
    "WZCancerProtocol",
]
