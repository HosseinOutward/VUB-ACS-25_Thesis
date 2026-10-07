from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, BaseProtocol, BaseRoundCodec, CompressionRecord

from .history import CompressedReconstructionHistory, HistoryVector, ReconstructionHistory
from .round_codecs import (
    CancerRecord,
    F_RoundCodec,
    Oracle_RoundCodec,
    ServerSIPriorTrainingPlan,
    ServerSIPriorMixin,
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

    # Oracle knowledge: frozen rounds send no model bytes, so a prior refit on them is never paid for.
    refit_priors_on_frozen: bool = False
    last_trained_codec: dict[int, _WZRoundCodec]

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
    ) -> None:
        super().__init__(options, protocol_name_full, sd_slices)
        self.last_trained_codec = {}
        self._recons_history = CompressedReconstructionHistory(
            self.max_per_client_recons_history, self.sd_slices
        )
        self.histories = (self._recons_history,)
        self.allocate_training_histories()

    def allocate_training_histories(self) -> None:
        """Append ledgers beyond the decoder-visible one; a single-round schedule needs none."""

    def after_decode(
        self, reconst: torch.Tensor, record: CompressionRecord, round_codec: BaseRoundCodec
    ) -> None:
        """Commit the round's reconstruction under the access level that codec decodes with."""
        for history in self.histories:
            history.commit(reconst, record, round_codec.history_access)

    def training_problem(
        self, codec: _WZRoundCodec, round_id: int, client_id: int
    ) -> tuple[torch.Tensor | None, list[HistoryVector] | None]:
        """Return the (target, SI) this codec trains on, read from the decoder-visible ledger."""
        return codec.get_training_data(self._recons_history, client_id)

    def quantizer_initial_states(
        self, codec: _WZRoundCodec, client_id: int, si_size: int
    ) -> tuple[Mapping[str, torch.Tensor], ...]:
        """Return the protocol-selected initialization states for a new quantizer."""
        return ()

    def quantizer_config_overrides(self, round_id: int, client_id: int) -> Mapping[str, Any]:
        """Return protocol-owned quantizer options for this training call."""
        return {}

    def server_si_prior_training_plan(
        self, codec: ServerSIPriorMixin,
    ) -> ServerSIPriorTrainingPlan:
        """Explicitly request a cold server-SI prior for an ordinary trained T round."""
        return ServerSIPriorTrainingPlan(
            initial_states=(None,), parent_slots=(),
            attempts_per_parent=codec.c_cfg.prior_train_repeats,
        )

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the scheduled codec with only the history and model state its policy requests."""
        round_codec_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        if round_codec_class is F_RoundCodec:
            return self._reuse_trained_codec(
                round_name_full, self.last_trained_codec.get(client_id), client_id)

        if not issubclass(round_codec_class, _WZRoundCodec):
            return round_codec_class(parsed.options, round_name_full)

        assert isinstance(parsed.options, dict)
        protocol_overrides = self.quantizer_config_overrides(round_id, client_id)
        assert not parsed.options.keys() & protocol_overrides.keys(), (
            "Protocol quantizer overrides duplicate scheduled round options."
        )
        effective_round_name = round_name_full + "".join(
            f"|{name}" if value is True else f"|{name}={value!r}"
            for name, value in protocol_overrides.items()
        )
        codec = round_codec_class(
            parsed.options | {
                "sd_slices": self.sd_slices,
            } | protocol_overrides,
            effective_round_name,
        )
        assert isinstance(codec, _WZRoundCodec)

        target, side_information = self.training_problem(codec, round_id, client_id)
        initial_states = self.quantizer_initial_states(codec, client_id, len(side_information))
        codec.refit_priors_on_frozen = self.refit_priors_on_frozen
        if isinstance(codec, ServerSIPriorMixin):
            codec.set_server_si_prior_training(
                self._recons_history, client_id, self.server_si_prior_training_plan(codec))
        codec.build_quantizer(
            target,
            side_information,
            initial_states,
        )
        self.last_trained_codec[client_id] = codec
        return codec

    def _reuse_trained_codec(
        self, round_name_full: str, previous: _WZRoundCodec | None, client_id: int
    ) -> _WZRoundCodec:
        """Reuse the client's trained codec, advancing its SI window to the most recent history."""
        assert previous is not None, f"No frozen quantizer exists for client {client_id}."
        assert previous.quantizer.vector_size is not None, (
            f"Client {client_id}'s previous codec has not completed pre-encode training and encoding.")
        assert previous.quantizer.side_info_list_used is not None
        assert previous.quantizer.quantizer_prior_stage_rates is not None
        assert not getattr(previous, "_send_quantizer_state", False), (
            f"Client {client_id}'s trained decoder state has not been transmitted yet.")
        # A marginal quantizer codes without side information, so there is no SI window to advance.
        if not previous.quantizer.no_side_info:
            side_information = [
                entry.value
                for client_history in self._recons_history.view(previous.history_access, client_id).values()
                for entry in client_history
            ]
            assert len(side_information) == previous.quantizer.side_info_size, (
                "Refreshed side information no longer matches the frozen quantizer.")
            previous.refresh_side_information(side_information)
        previous.round_name_full = round_name_full
        previous.dump_element_metrics = False
        previous.warm_parent_slots = ()
        previous._original_delta_vec = None
        previous.refit_priors_on_frozen = self.refit_priors_on_frozen
        if isinstance(previous, ServerSIPriorMixin):
            previous.set_server_si_prior_training(self._recons_history, client_id, None)
        return previous

    def config_snapshot(self) -> dict[str, Any]:
        """Return the WZ schedule, quantizer defaults, and SI/history policies."""
        snapshot = super().config_snapshot()
        snapshot["candidate_acceptance"] = "finite_usable_and_total_hard_prior_le_same_symbol_empirical_marginal_v1"
        snapshot["quantizer_defaults"] = WZcfgQuant(
            bins_per_plane=1,
            num_planes=1,
        ).model_dump(
            mode="json",
            exclude={
                "bins_per_plane",
                "num_planes",
                "norm_slices",
            },
        )
        snapshot["max_per_client_recons_history"] = self.max_per_client_recons_history
        snapshot["refit_priors_on_frozen"] = self.refit_priors_on_frozen
        snapshot["history"] = self._recons_history.config_snapshot()
        # Codec facts and resolved transport settings, so D/E read oracle grants and rate-field
        # availability from here rather than inferring them from a display label like codec_id.
        for round_name, entry in snapshot["round_codecs"].items():
            round_codec_class = self._round_codec_classes[round_name]
            if issubclass(round_codec_class, _WZRoundCodec):
                entry["codec_provenance"] = round_codec_class.codec_provenance(**entry["options"])
                resolved = round_codec_class.config_class(**{
                    key: value for key, value in entry["options"].items()
                    if key in round_codec_class.config_class.model_fields
                })
                entry["numerical_runtime"] = resolved.numerical_policy.runtime(
                    torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"),
                    mixed_precision=resolved.mixed_precision)
        return snapshot


class WZCancerProtocol(_WZProtocol):
    """Cancer-model WZ schedule with training rounds followed by frozen-model rounds."""

    # Identity rounds commit to both ledgers, so every client's history is full before any training.
    warmup_round_codecs: tuple[str, ...] = ("I",) * 5
    routine_round_codecs: tuple[str, ...] = (
        "O|bins_per_plane=3|num_planes=3",
        "F",
        "F",
        "F",
        "F",
    )
    protocol_name: ClassVar[str] = "wz_cancer"

    def options_to_config(self, refit_priors_on_frozen: bool = False) -> None:
        """Configure optional fixed-symbol F-round prior adaptation."""
        assert isinstance(refit_priors_on_frozen, bool)
        self.refit_priors_on_frozen = refit_priors_on_frozen


class WarmWZCancerProtocol(WZCancerProtocol):
    """Cancer WZ schedule maintaining five compatible shared warm-training lineages."""

    pool_size: ClassVar[int] = 5
    parents_per_refresh: ClassVar[int] = 2
    history_prep_round_codecs: ClassVar[tuple[str, ...]] = WZCancerProtocol.warmup_round_codecs
    warmup_round_codecs: tuple[str, ...] = history_prep_round_codecs
    routine_round_codecs: tuple[str, ...] = WZCancerProtocol.routine_round_codecs
    protocol_name: ClassVar[str] = "wz_cancer_warm"

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
    ) -> None:
        super().__init__(options, protocol_name_full, sd_slices)
        self.quantizer_pool: list[dict[str, torch.Tensor]] = []
        self.quantizer_pool_si_size: int | None = None
        self.server_si_prior_pool: list[dict[str, torch.Tensor]] = []
        self.warm_refresh_count: int = 0

    def quantizer_config_overrides(self, round_id: int, client_id: int) -> Mapping[str, Any]:
        """Switch eligible training calls to warm mode only after the shared pool is full."""
        if round_id < len(self.history_prep_round_codecs) or len(self.quantizer_pool) < self.pool_size:
            return {}
        assert len(self.quantizer_pool) == self.pool_size
        return {"warm_training": True}

    def quantizer_initial_states(
        self, codec: _WZRoundCodec, client_id: int, si_size: int
    ) -> tuple[Mapping[str, torch.Tensor], ...]:
        """Select two parents, asserting that warmup established the expected pool and SI width.

        Oracle knowledge: pool parents are full quantizer states, encoder included, fitted on other
        workers' data; a T worker warm-starting from them receives parameters no payload pays for.
        """
        if not codec.c_cfg.warm_training:
            return super().quantizer_initial_states(codec, client_id, si_size)
        assert len(self.quantizer_pool) == self.pool_size, (
            f"Warm training requires {self.pool_size} shared base quantizers; "
            f"completed training calls produced {len(self.quantizer_pool)}."
        )
        assert self.quantizer_pool_si_size == si_size, (
            f"Warm training expected {self.quantizer_pool_si_size} "
            f"side-information vectors, got {si_size}."
        )
        start = self.warm_refresh_count * self.parents_per_refresh % self.pool_size
        codec.warm_parent_slots = tuple(
            (start + offset) % self.pool_size for offset in range(self.parents_per_refresh)
        )
        return tuple(self.quantizer_pool[slot] for slot in codec.warm_parent_slots)

    def server_si_prior_training_plan(
        self, codec: ServerSIPriorMixin,
    ) -> ServerSIPriorTrainingPlan:
        """Use aligned cold or warm server-SI-prior lineages for a T codec."""
        if not codec.c_cfg.warm_training:
            return super().server_si_prior_training_plan(codec)
        assert len(self.server_si_prior_pool) == self.pool_size, (
            f"Warm server-SI-prior training requires {self.pool_size} base priors; "
            f"completed training calls produced {len(self.server_si_prior_pool)}."
        )
        assert len(codec.warm_parent_slots) == self.parents_per_refresh
        return ServerSIPriorTrainingPlan(
            initial_states=tuple(
                self.server_si_prior_pool[slot] for slot in codec.warm_parent_slots),
            parent_slots=codec.warm_parent_slots,
            attempts_per_parent=codec.c_cfg.warm_retrain_repeats,
        )

    def after_decode(
        self, reconst: torch.Tensor, record: CompressionRecord, round_codec: BaseRoundCodec
    ) -> None:
        """Commit reconstruction first, then advance the completed pool lineages."""
        super().after_decode(reconst, record, round_codec)
        if not isinstance(round_codec, _WZRoundCodec) or round_codec.round_name_full == "F":
            return

        if round_codec.c_cfg.warm_training:
            states = round_codec.quantizer.warm_retrained_states
            assert len(round_codec.warm_parent_slots) == len(states) == self.parents_per_refresh
            for slot, state in zip(round_codec.warm_parent_slots, states, strict=True):
                self.quantizer_pool[slot] = {
                    name: value.detach().cpu().clone() for name, value in state.items()
                }
            if isinstance(round_codec, ServerSIPriorMixin):
                result = round_codec.server_si_prior_training_result
                assert result is not None
                assert len(result.lineage_states) == len(round_codec.warm_parent_slots)
                for slot, state in zip(
                    round_codec.warm_parent_slots, result.lineage_states, strict=True,
                ):
                    self.server_si_prior_pool[slot] = {
                        name: value.detach().cpu().clone() for name, value in state.items()
                    }
            self.warm_refresh_count += 1
            return

        if record.round_id < len(self.history_prep_round_codecs):
            return
        quantizer = round_codec.quantizer
        assert quantizer.quantizer_prior_stage_rates is not None
        assert len(self.quantizer_pool) < self.pool_size, (
            "A cold training call completed after the shared warm pool was already full."
        )
        if not self.quantizer_pool:
            self.quantizer_pool_si_size = quantizer.side_info_size
        else:
            assert self.quantizer_pool_si_size == quantizer.side_info_size, (
                f"Base quantizer {len(self.quantizer_pool)} has incompatible SI width."
            )
        assert round_codec.joint_quantizer_state is not None
        self.quantizer_pool.append({
            name: value.detach().cpu().clone()
            for name, value in round_codec.joint_quantizer_state.items()
        })
        if isinstance(round_codec, ServerSIPriorMixin):
            result = round_codec.server_si_prior_training_result
            assert result is not None
            self.server_si_prior_pool.append({
                name: value.detach().cpu().clone()
                for name, value in result.model.state_dict().items()
            })

    def config_snapshot(self) -> dict[str, Any]:
        """Record the fixed pool construction and balanced parent-selection policy."""
        snapshot = super().config_snapshot()
        snapshot["warm_quantizer_pool"] = {
            "pool_size": self.pool_size,
            "parents_per_refresh": self.parents_per_refresh,
            "selection": "balanced_rotating_window_v1",
            "history_readiness": "warmup_schedule_asserted_at_warm_entry",
            "server_si_prior_pool": "aligned_parent_slots_for_T",
        }
        return snapshot


__all__ = [
    "CancerRecord",
    "F_RoundCodec",
    "Oracle_RoundCodec",
    "ServerSIPriorTrainingPlan",
    "ServerSIPriorMixin",
    "T_RoundCodec",
    "WZCancerProtocol",
    "WarmWZCancerProtocol",
]
