from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import copy
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, BaseProtocol, BaseRoundCodec, CompressionRecord

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


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One reconstructed delta committed by the WZ protocol after a successful decode."""
    tensor: torch.Tensor
    round_id: int
    client_id: int
    access: Access
    round_name: str | None


class ReconstructionHistory:
    """Bounded ledger of reconstructed deltas that enforces decoder-side visibility.

    Owned by the WZ protocol, not by the FL loop: only a schedule whose codecs condition on earlier
    reconstructions needs one. Both ledgers hold the same entry objects, so a reconstruction is
    stored once no matter how many processes may read it.
    """

    def __init__(
        self, max_per_client: int | None,
        # Reconstructions are retained exactly as the decoder produced them. Kept as an asserted
        # field rather than a bare assumption because half precision cannot hold these values:
        # mean |delta| is around 1e-5, already inside float16's subnormals.
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert max_per_client != 0, 'to disable reconstruction history, set max_per_client to None'
        self.max_per_client: int = max_per_client or 0
        self.dtype: torch.dtype = dtype
        self._server: dict[int, list[HistoryEntry]] = {}
        self._temporal: dict[int, list[HistoryEntry]] = {}
        self.frozen: bool = self.max_per_client == 0

    def freeze(self) -> None:
        """Prevent further commits to the reconstruction history."""
        self.frozen = True

    def config_snapshot(self) -> dict[str, Any]:
        """Describe the retained history representation used by a protocol run."""
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "max_per_client": self.max_per_client,
            "dtype": str(self.dtype),
        }

    def copy(self) -> ReconstructionHistory:
        """Return a state copy that reuses the same immutable history entries.

        Shallow-copied and then given its own ledger lists, rather than rebuilt field by field: a
        field added to `__init__` would otherwise be silently missing from every copy.
        """
        copied = copy.copy(self)
        copied._server = {
            client_id: entries.copy() for client_id, entries in self._server.items()
        }
        copied._temporal = {
            client_id: entries.copy() for client_id, entries in self._temporal.items()
        }
        return copied

    def commit(self, reconst: torch.Tensor, record: CompressionRecord, access: Access) -> None:
        """Commit a reconstructed tensor to every ledger allowed to retain it."""
        assert not self.frozen, "ReconstructionHistory is frozen and cannot be modified."
        assert reconst.dtype == self.dtype and reconst.device == torch.device("cpu")
        if access is Access.NONE:
            return

        entry = HistoryEntry(
            tensor=reconst,
            round_id=record.round_id,
            client_id=record.client_id,
            access=access,
            round_name=record.round_name_full,
        )
        self._add_entry(self._server, entry)
        if access is Access.TEMPORAL_TOO:
            self._add_entry(self._temporal, entry)

    def view(self, access: Access, client_id: int) -> dict[int, tuple[HistoryEntry, ...]]:
        """Return the history entries visible to the requested process.

        `client_id` selects the worker asking. A worker sees only its own temporal ledger, while
        the server sees every client's, so under SERVER_ONLY the caller's id narrows nothing and
        the whole ledger set comes back keyed by client.
        """
        if access is Access.NONE:
            return dict()

        if access is Access.TEMPORAL_TOO:
            assert client_id in self._temporal, (
                f"No temporal reconstruction history exists for client {client_id}.")
            return {client_id: tuple(self._temporal[client_id])}

        assert access is Access.SERVER_ONLY, f"Unknown access policy: {access!r}."
        return {
            visible_client: tuple(ledger)
            for visible_client, ledger in self._server.items()
        }

    def _add_entry(self, history: dict[int, list[HistoryEntry]], entry: HistoryEntry) -> None:
        ledger = history.setdefault(entry.client_id, [])
        ledger.append(entry)
        if len(ledger) > self.max_per_client:
            ledger.pop(0)


class _WZProtocol(BaseProtocol):
    """Schedule WZ round policies and own the decoder-visible reconstruction ledger.

    The ledgers are the protocol's own state, not the FL loop's: the loop only hands each decoded
    reconstruction back through `after_decode`. One training problem per round: each scheduled codec
    is trained from the ledger the decoder can see, and frozen rounds replay the model the last
    training round produced. Training on more than the current round is an experiment-side extension
    of `allocate_training_histories`, `codec_protocol_inputs` and `training_problem`; nothing here
    assumes it exists.
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
        self._recons_history = ReconstructionHistory(self.max_per_client_recons_history)
        self.histories = (self._recons_history,)
        self.allocate_training_histories()
        if self._train_at_round is not None:
            required = max(history.max_per_client for history in self.histories)
            assert self._train_at_round >= required, (
                f"train_at_round={self._train_at_round} leaves fewer than {required} preceding "
                "history rounds.")

    def allocate_training_histories(self) -> None:
        """Append ledgers beyond the decoder-visible one; a single-round schedule needs none."""

    def after_decode(
        self, reconst: torch.Tensor, record: CompressionRecord, round_codec: BaseRoundCodec
    ) -> None:
        """Commit the round's reconstruction under the access level that codec decodes with."""
        for history in self.histories:
            history.commit(reconst, record, round_codec.can_decode_where)

    def codec_protocol_inputs(
        self, round_codec_class: type[_WZRoundCodec], previous: _WZRoundCodec | None
    ) -> dict[str, Any]:
        """Return constructor inputs the protocol supplies; ordinary WZ rounds need none."""
        return {}

    def training_problem(
        self, codec: _WZRoundCodec, round_id: int, client_id: int
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
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

        previous = self.last_trained_codec.get(client_id)
        assert isinstance(parsed.options, dict)
        codec = round_codec_class(
            parsed.options | {
                "sd_slices": self.sd_slices,
                "cache_dir": self.cache_dir,
                "cache_overwrite": self.cache_overwrite,
            },
            round_name_full,
            **self.codec_protocol_inputs(round_codec_class, previous),
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
                entry.tensor
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
    ) -> None:
        """Configure SI policies and optionally replace/collapse the scheduled training rounds."""
        assert isinstance(refresh_frozen_si, bool)
        assert isinstance(retrained_prior_uses_server_si, bool)
        self.refresh_frozen_si = refresh_frozen_si
        self.retrained_prior_uses_server_si = retrained_prior_uses_server_si

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
