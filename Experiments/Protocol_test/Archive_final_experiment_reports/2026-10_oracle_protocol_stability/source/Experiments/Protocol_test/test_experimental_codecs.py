"""Validate historical target/SI alignment and the experimental codecs' real training path."""

import csv
import json
from pathlib import Path

import pytest
import torch

from Experiments.Protocol_test.experimental_codecs import (
    OM_RoundCodec, OMR_RoundCodec, TM_RoundCodec, _MultiRoundQuantizer, _StackedSideInfo,
)
from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, simulate_compression
from FL_code.FL_core.utils import StateDictManager
from FL_code.cancer_protocol.history import HistoryVector, ReconstructionHistory
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import PreprocessMetadata, SideInfoMatrix, unique_row_groups


@pytest.mark.parametrize("codec_class", [OM_RoundCodec, TM_RoundCodec])
def test_historical_targets_are_raw_and_si_preserves_client_order(
    codec_class: type[OM_RoundCodec | TM_RoundCodec],
) -> None:
    """Raw targets differ from decoded history, whose same-round visibility is preserved."""
    protocol = create_protocol("wz_cancer_multi|train_offsets=(1,3)")
    protocol.set_schedule(8, (codec_class.round_name + "|bins_per_plane=2|num_planes=3", "F"))
    protocol._recons_history = ReconstructionHistory(5)
    for round_id in range(8):
        for client_id in range(2):
            raw = torch.tensor([float(100 * client_id + round_id)])
            protocol.remember_target(raw, round_id, client_id)
            protocol._recons_history.commit(
                raw + 1000, CompressionRecord(round_id, client_id, "I"), Access.TEMPORAL_TOO,
            )
    options = {"bins_per_plane": 2, "num_planes": 3, "sd_slices": [slice(None)]}
    codec = codec_class(options, codec_class.round_name)
    _, current_si = protocol.training_problem(codec, 8, 1)
    extras = codec.extra_training_problems()
    assert codec.training_problem_weights() == (0.5, 0.3, 0.2)
    assert [target.item() for target, _ in extras] == [107, 105]
    expected_temporal = [1102, 1103, 1104, 1105, 1106]
    expected_server = [1003, 1004, 1005, 1006, 1007] + expected_temporal
    expected = expected_server if codec_class is OM_RoundCodec else expected_temporal
    assert [value.materialize().item() for value in extras[0][1]] == expected
    assert [value.materialize().item() for value in codec.historical_server_si[0]] == expected_server
    assert [value.materialize().item() for value in current_si][-1] == 1107
    protocol._recons_history.commit(
        torch.tensor([9999.]), CompressionRecord(8, 0, "I"), Access.TEMPORAL_TOO,
    )
    assert [value.materialize().item() for value in extras[0][1]] == expected


def test_lazy_stack_matches_dense_rows_and_nested_selections() -> None:
    """Repeated mixed-block indices retain their exact order and independent normalization."""
    blocks = tuple(
        SideInfoMatrix(
            [HistoryVector(torch.tensor([1., 1., 3., 4.]))], 4,
            torch.tensor([[float(i + 1), float(i)]]), [slice(0, 4)], torch.device("cpu"),
        ) for i in range(2)
    )
    stack = _StackedSideInfo(blocks)
    dense = torch.cat([block[:] for block in blocks])
    rows = torch.tensor([7, 0, 5, 0, 2, 4])
    torch.testing.assert_close(stack[rows], dense[rows])
    torch.testing.assert_close(stack.rows(slice(2, 8)).rows(torch.tensor([5, 0]))[:], dense[[7, 2]])
    assert stack.code_rows(rows).shape == (6, 1)
    representatives, inverse, collisions = unique_row_groups((stack,), 26)
    assert not collisions.numel()
    torch.testing.assert_close(dense[representatives][inverse], dense)
    assert stack.code_rows(torch.tensor([0, 4]))[0].item() != stack.code_rows(torch.tensor([0, 4]))[1].item()


@pytest.mark.parametrize("token", ["OMR", "TMR"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Quantizer fitting requires CUDA")
def test_real_cold_warm_and_frozen_rans(token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise historical fitting, warm parents, frozen SI refresh, and both TMR rate reports."""
    torch.manual_seed(43)
    # Two-epoch fixtures test protocol/transport plumbing; the statistical gate has separate tests.
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda model, bins, si, weights=None: (1.0, 2.0)))
    original_fit = PriorCalculator.fit_prior_model
    original_encode = _MultiRoundQuantizer.encoding_process
    encoded: list[tuple[torch.Tensor, torch.Tensor]] = []
    fitted_widths: list[int] = []

    def encode(
        quantizer: _MultiRoundQuantizer, raw: torch.Tensor, *args: object, **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, PreprocessMetadata]:
        bins, codes, metadata = original_encode(quantizer, raw, *args, **kwargs)
        assert codes is not None
        encoded.append((bins, codes))
        return bins, codes, metadata

    def fit(bins: torch.Tensor, codes: torch.Tensor, si: SideInfoMatrix, *args: object, **kwargs: object) -> object:
        assert bins.shape == (3, 3 * 128) and codes.shape == (3, 3 * 128, 2)
        assert isinstance(si, _StackedSideInfo) and len(si._blocks) == 3
        assert kwargs["problem_weights"] == (.5, .3, .2)
        # The two historical encodes precede the current encode; both prior fits reuse them.
        order = (encoded[-1], encoded[-3], encoded[-2])
        torch.testing.assert_close(bins, torch.cat([pair[0].long() for pair in order], dim=1))
        torch.testing.assert_close(codes, torch.cat([pair[1] for pair in order], dim=1))
        fitted_widths.append(si.shape[1])
        return original_fit(bins, codes, si, *args, **kwargs)

    monkeypatch.setattr(PriorCalculator, "fit_prior_model", staticmethod(fit))
    monkeypatch.setattr(_MultiRoundQuantizer, "encoding_process", encode)
    protocol = create_protocol("wz_cancer_multi_warm", [slice(0, 128)])
    protocol.pool_size = 1
    protocol.parents_per_refresh = 1
    token_options = (
        "|bins_per_plane=2|num_planes=3|rans_lanes=4|hidden_dim=4|layers=1"
        "|train_epochs=2|train_sample_size=128|train_batch_size=64"
        "|quantizer_train_repeats=1|prior_train_repeats=1|prior_retrain_epochs=1"
    )
    protocol.set_schedule(7, (token + token_options, "F"))
    manager = StateDictManager(torch.nn.Linear(128, 1, bias=False))
    for round_id in range(9):
        for client_id in range(2):
            raw = torch.linspace(-1, 1, 128) * (1 + round_id / 10) + client_id / 10
            protocol.remember_target(raw, round_id, client_id)
            if round_id < 7:
                protocol._recons_history.commit(
                    raw, CompressionRecord(round_id, client_id, "I"), Access.TEMPORAL_TOO,
                )
                continue
            reconstructed = simulate_compression(protocol, raw, client_id, round_id, manager, tmp_path, {}, (), ())
            assert bool(torch.isfinite(reconstructed).all())
    with (tmp_path / "compression_records.csv").open() as file:
        records = list(csv.DictReader(file))
    assert len(records) == 4
    assert fitted_widths == ([10, 10] if token == "OMR" else [5, 10, 5, 10])
    assert records[1]["warm_parent_slots"] == "[0]"
    for record in records:
        assert float(record["symbol_error_rate"]) == 0
        rates = json.loads(record["coder_stage_rates"])
        assert len(rates) == 3 and sum(rates) == pytest.approx(float(record["coder_rate"]))
        access = "temporal_si" if token == "TMR" else "server_si"
        assert float(record["coder_rate"]) == float(record[f"{access}_rans_rate"])
        if token == "TMR":
            assert record["temporal_si_rans_rate"] and record["server_si_rans_rate"]


def test_missing_full_lagged_problem_fails() -> None:
    """A five-round prefix cannot provide a lag-1 target with five earlier SI rounds."""
    protocol = create_protocol("wz_cancer_multi")
    for round_id in range(5):
        raw = torch.tensor([float(round_id)])
        protocol.remember_target(raw, round_id, 0)
        protocol._recons_history.commit(raw, CompressionRecord(round_id, 0, "I"), Access.TEMPORAL_TOO)
    codec = OMR_RoundCodec({"bins_per_plane": 2, "num_planes": 3, "sd_slices": None}, "OMR")
    with pytest.raises(AssertionError, match="lacks lag 1"):
        protocol.training_problem(codec, 5, 0)
