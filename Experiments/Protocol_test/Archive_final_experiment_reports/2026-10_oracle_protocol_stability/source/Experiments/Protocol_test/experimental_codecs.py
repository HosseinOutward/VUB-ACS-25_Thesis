"""Multi-round O/T experiments: raw historical targets paired with capture-time decoder SI."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access
from FL_code.cancer_protocol.bin_coder.rans_mixin import RansRoundMixin
from FL_code.cancer_protocol.cancer_protocol import WarmWZCancerProtocol, WZCancerProtocol
from FL_code.cancer_protocol.history import HistoryVector, ReconstructionHistory
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.prior_code import PriorCalculator, PriorTrainingResult
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec, T_RoundCodec, _WZRoundCodec
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer, PreprocessMetadata, SideInfoMatrix, _SideInfoRows, _stacked_problem_sampler,
)


class _StackedSideInfo(SideInfoMatrix):
    """Row concatenation consumed through the existing lazy SI matrix and row-view interface."""

    def __init__(self, blocks: Sequence[SideInfoMatrix]) -> None:
        assert blocks and len({block.shape for block in blocks}) == 1
        assert len({block.device for block in blocks}) == 1
        self._blocks: tuple[SideInfoMatrix, ...] = tuple(blocks)
        self._block_size: int = blocks[0].shape[0]
        self._shape = (len(blocks) * self._block_size, blocks[0].shape[1])
        self._device, self._dtype = blocks[0].device, blocks[0].dtype

    def to(self, device: torch.device | str) -> _StackedSideInfo:
        """Move compact block payloads while keeping inherited row selection lazy."""
        self._blocks = tuple(block.to(device) for block in self._blocks)
        self._device = self._blocks[0].device
        return self

    def __getitem__(self, index: slice | torch.Tensor | Sequence[int] | int) -> torch.Tensor:
        """Decode requested rows from their own normalized training problems."""
        rows = self.rows(index)._resolved_indices()
        result = torch.empty((rows.numel(), self.shape[1]), device=self.device, dtype=self.dtype)
        for block_id, block in enumerate(self._blocks):
            mask = rows.div(self._block_size, rounding_mode="floor") == block_id
            result[mask] = block[rows[mask].remainder(self._block_size)]
        return result

    def iter_code_columns(self, rows: torch.Tensor | None = None) -> Iterator[torch.Tensor]:
        """Keep compact grouping keys in separate namespaces for independently normalized blocks."""
        rows = torch.arange(self.shape[0], device=self.device) if rows is None else rows
        masks = tuple(rows.div(self._block_size, rounding_mode="floor") == i for i in range(len(self._blocks)))
        generators = tuple(
            block.iter_code_columns(rows[mask].remainder(self._block_size))
            for block, mask in zip(self._blocks, masks, strict=True)
        )
        for columns in zip(*generators, strict=True):
            result = torch.empty(rows.numel(), device=self.device, dtype=torch.int64)
            for block_id, (mask, column) in enumerate(zip(masks, columns, strict=True)):
                assert bool((column < 1 << 48).all()), "SI grouping keys exceed the block namespace."
                result[mask] = column + (block_id << 48)
            yield result

    def code_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Return compact grouping keys for sampled rows without materializing full SI."""
        return torch.stack(tuple(self.iter_code_columns(rows)), dim=1)


class _MultiRoundQuantizer(DedupedWZQuantizerCancer):
    """Use the ordinary quantizer training loop with a lazy stack of historical SI problems."""

    def _stack_training_pairs(
        self, x_prep: torch.Tensor, side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor,
        extra_pairs: Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]],
        problem_weights: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, SideInfoMatrix]:
        assert isinstance(side_info, SideInfoMatrix)
        assert extra_pairs, "Multi-round training requires historical targets."
        assert problem_weights is None or len(problem_weights) == 1 + len(extra_pairs)
        rows = x_prep.shape[0]
        stacked_x = torch.empty((rows * (1 + len(extra_pairs)), 1), device=x_prep.device, dtype=x_prep.dtype)
        stacked_x[:rows] = x_prep
        blocks = [side_info]
        for i, (target, si) in enumerate(extra_pairs, start=1):
            assert target.numel() == rows and len(si) == self.side_info_size
            prepared, metadata = self.preprocess_x(target)
            stacked_x[i * rows:(i + 1) * rows] = prepared
            blocks.append(self.normalize_side_info_tensor(metadata, si))
        if problem_weights is not None:
            self.training_index_sampler = _stacked_problem_sampler(problem_weights, rows, stacked_x.device)
        return stacked_x, _StackedSideInfo(blocks)


class _MultiRoundTrainingMixin:
    """Supply aligned extra training rows while retaining the existing O/T codec behavior."""

    quantizer_class: ClassVar[type[DedupedWZQuantizerCancer]] = _MultiRoundQuantizer
    historical_problems: tuple[tuple[torch.Tensor, Sequence[HistoryVector]], ...]
    historical_server_si: tuple[Sequence[HistoryVector], ...]
    historical_encodings: tuple[tuple[torch.Tensor, torch.Tensor, PreprocessMetadata], ...]
    problem_weights: tuple[float, ...]

    def extra_training_problems(self) -> Sequence[tuple[torch.Tensor, Sequence[HistoryVector]]]:
        """Return the raw historical targets and their decoder-visible SI snapshots."""
        assert hasattr(self, "historical_problems"), "OM/TM require a wz_cancer_multi protocol."
        return self.historical_problems

    def training_problem_weights(self) -> tuple[float, ...]:
        """Sample current and capture-time historical pairs with the protocol's declared shares."""
        return self.problem_weights

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Encode historical raw targets once with the final encoder for all subsequent prior fits."""
        super().pre_encode_training_steps(delta_vec)
        self.historical_encodings = tuple(
            self.quantizer.encoding_process(target) for target, _ in self.historical_problems)
        assert all(codes is not None for _, codes, _ in self.historical_encodings)

    def fit_prior_model(
        self, bins: torch.Tensor, soft_codes: torch.Tensor, side_info: SideInfoMatrix,
        access: Access, target_model: EncoderDecoderLayeredRNN | None,
        initial_states: Sequence[Mapping[str, torch.Tensor] | None], attempts_per_parent: int,
    ) -> PriorTrainingResult:
        """Fit native or full-SI priors on the same weighted, capture-time-paired raw targets."""
        historical_si = (self.historical_server_si if access is Access.SERVER_ONLY
                         else tuple(si for _, si in self.historical_problems))
        blocks = [side_info]
        for (_, _, metadata), si in zip(self.historical_encodings, historical_si, strict=True):
            blocks.append(self.quantizer.normalize_side_info_tensor(metadata, si))
        return PriorCalculator.fit_prior_model(
            torch.cat([bins, *(encoded[0].long() for encoded in self.historical_encodings)], dim=1),
            torch.cat([soft_codes, *(encoded[1] for encoded in self.historical_encodings)], dim=1),
            _StackedSideInfo(blocks), self.c_cfg.model_copy(update={"marginal_loss": False}),
            target_model, initial_states, attempts_per_parent, problem_weights=self.problem_weights,
        )

    def release_round_resources(self) -> None:
        """Drop training-only raw targets when the client-round finishes."""
        super().release_round_resources()
        self.historical_problems = ()
        self.historical_server_si = ()
        self.historical_encodings = ()


class OM_RoundCodec(_MultiRoundTrainingMixin, Oracle_RoundCodec):
    """Train oracle WZ on current and historical raw targets with matching server SI."""

    round_name: ClassVar[str] = "OM"


class TM_RoundCodec(_MultiRoundTrainingMixin, T_RoundCodec):
    """Train temporal WZ on current and historical raw targets with matching temporal SI."""

    round_name: ClassVar[str] = "TM"


class OMR_RoundCodec(RansRoundMixin, OM_RoundCodec):
    """Multi-round oracle WZ with server-SI rANS transport."""

    round_name: ClassVar[str] = "OMR"
    transport_prior_access: ClassVar[Access] = Access.SERVER_ONLY


class TMR_RoundCodec(RansRoundMixin, TM_RoundCodec):
    """Multi-round temporal WZ with temporal rANS and a full-server-SI rANS comparison."""

    round_name: ClassVar[str] = "TMR"
    transport_prior_access: ClassVar[Access] = Access.TEMPORAL_TOO


@dataclass(frozen=True, slots=True)
class _HistoricalProblem:
    """Raw target and immutable history snapshot captured by the runner before compression."""

    target: HistoryVector
    history: ReconstructionHistory


class _MultiRoundProtocolMixin:
    """Own bounded historical problems and select lags through the codec's existing SI policy."""

    train_offsets: tuple[int, ...] = (1, 2)
    train_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    _problems: dict[int, dict[int, _HistoricalProblem]]

    def __init__(
        self, options: Mapping[str, Any] | None = None, protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
    ) -> None:
        super().__init__(options, protocol_name_full, sd_slices)
        self._problems = {}

    def options_to_config(
        self, train_offsets: Sequence[int] = (1, 2), train_weights: Sequence[float] = (0.5, 0.3, 0.2), **options: Any,
    ) -> None:
        """Select historical lags and shares ordered as current, then increasing lag."""
        offsets = tuple(train_offsets)
        assert offsets and all(type(offset) is int and offset > 0 for offset in offsets)
        assert len(set(offsets)) == len(offsets), "train_offsets must be unique."
        self.train_offsets = tuple(sorted(offsets))
        weights = tuple(train_weights)
        assert len(weights) == 1 + len(offsets) and all(weight > 0 and isfinite(weight) for weight in weights), (
            f"train_weights needs one finite positive share for current and each lag; got {weights} for {offsets}.")
        self.train_weights = weights
        super().options_to_config(**options)

    @property
    def history_seed_depth(self) -> int:
        """Include the oldest target's complete SI window in the raw trace prefix."""
        assert self.max_per_client_recons_history is not None
        return self.max_per_client_recons_history + max(self.train_offsets)

    def remember_target(self, target: torch.Tensor, round_id: int, client_id: int) -> None:
        """Capture the exact raw target before the runner commits its decoded reconstruction."""
        if self._schedule_placed and not any(
            issubclass(self._get_curr_round_codec_name(round_id + offset)[0], _MultiRoundTrainingMixin)
            for offset in self.train_offsets
        ):
            return
        history = self._recons_history.view(Access.SERVER_ONLY, client_id)
        depth = self.max_per_client_recons_history
        assert depth is not None
        if client_id not in history or any(len(entries) < depth for entries in history.values()):
            return
        snapshot = self._recons_history.copy()
        snapshot.frozen = True
        problems = self._problems.setdefault(client_id, {})
        problems[round_id] = _HistoricalProblem(HistoryVector(target), snapshot)
        for stale in tuple(r for r in problems if r < round_id - max(self.train_offsets)):
            del problems[stale]

    def training_problem(
        self, codec: _WZRoundCodec, round_id: int, client_id: int,
    ) -> tuple[torch.Tensor | None, list[HistoryVector] | None]:
        """Pair each requested past raw target with the SI visible when that target was encoded."""
        target, si = super().training_problem(codec, round_id, client_id)
        if isinstance(codec, _MultiRoundTrainingMixin):
            extras = []
            server_si = []
            for offset in self.train_offsets:
                historical = self._problems.get(client_id, {}).get(round_id - offset)
                assert historical is not None, (
                    f"{codec.round_name} round {round_id}, client {client_id} lacks lag {offset}; "
                    f"seed {self.history_seed_depth} preceding raw rounds."
                )
                _, historical_si = codec.get_training_data(historical.history, client_id)
                assert historical_si is not None and si is not None and len(historical_si) == len(si)
                extras.append((historical.target.materialize(), historical_si))
                server_si.append(tuple(
                    entry.value for entries in historical.history.view(Access.SERVER_ONLY, client_id).values()
                    for entry in entries))
                if offset == max(self.train_offsets):
                    del self._problems[client_id][round_id - offset]
            codec.historical_problems = tuple(extras)
            codec.historical_server_si = tuple(server_si)
            codec.problem_weights = self.train_weights
        return target, si

    def config_snapshot(self) -> dict[str, Any]:
        """Include raw-target alignment and lag selection in the experiment's semantic identity."""
        snapshot = super().config_snapshot()
        snapshot["multi_round_training"] = {
            "train_offsets": self.train_offsets,
            "train_weights": self.train_weights,
            "target": "raw_client_delta",
            "side_information": "decoder_visible_snapshot_before_target_encoding",
            "sampling": "weighted_current_then_increasing_lag",
            "stack": "lazy_independently_normalized_si_blocks_v1",
            "prior_training": "weighted_final_encoder_symbols_with_capture_time_native_and_server_si_v1",
            "history_seed_depth": self.history_seed_depth,
        }
        return snapshot


class MultiRoundWZCancerProtocol(_MultiRoundProtocolMixin, WZCancerProtocol):
    """Sparse experimental protocol supplying aligned historical problems to OM/TM codecs."""

    warmup_round_codecs: tuple[str, ...] = ("I",) * 7
    routine_round_codecs: tuple[str, ...] = ("OMR|bins_per_plane=2|num_planes=3", "F", "F", "F", "F")
    protocol_name: ClassVar[str] = "wz_cancer_multi"


class MultiRoundWarmWZCancerProtocol(_MultiRoundProtocolMixin, WarmWZCancerProtocol):
    """Sparse experimental protocol combining historical training with the existing warm pool."""

    history_prep_round_codecs: ClassVar[tuple[str, ...]] = ("I",) * 7
    warmup_round_codecs: tuple[str, ...] = history_prep_round_codecs
    routine_round_codecs: tuple[str, ...] = MultiRoundWZCancerProtocol.routine_round_codecs
    protocol_name: ClassVar[str] = "wz_cancer_multi_warm"
