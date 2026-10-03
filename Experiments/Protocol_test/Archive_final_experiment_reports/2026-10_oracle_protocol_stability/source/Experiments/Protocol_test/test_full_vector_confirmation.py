"""Check retained fallback equivalence, chronological selection, and real confirmation plumbing."""

from copy import deepcopy
from collections.abc import Iterator
import json
from pathlib import Path

import pytest
import torch

from Experiments.Protocol_test.full_vector_confirmation import (
    Candidate, embed_temporal, evaluate, fit_embedded_prior, select_candidates, train_oracle,
)
from Experiments.Protocol_test import full_vector_confirmation as confirmation
from Experiments.Protocol_test.sparse_replay import ReplayCase, ReplayPlan, case_protocol, load_plan
from FL_code.FL_core.utils import StateDictManager
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer, new_rnn_model


@pytest.mark.parametrize("columns", [torch.tensor([0, 2, 7]), torch.tensor([4, 5, 6])])
def test_embedding_preserves_every_output(columns: torch.Tensor) -> None:
    """Extra SI may vary arbitrarily without altering the retained temporal fallback."""
    torch.manual_seed(43)
    t = new_rnn_model(3, 2, 3, False, 4, 1).eval()
    o = embed_temporal(t, columns, 9)
    x, si = torch.randn(128, 1), torch.randn(128, 9)
    with torch.no_grad():
        a, b = t(x, si[:, columns]), o(x, si)
    for left, right in zip(a, b, strict=True):
        for old, embedded in zip(left, right, strict=True):
            torch.testing.assert_close(old, embedded, atol=2e-6, rtol=2e-6)


def test_selection_retains_joint_prior_and_exact_t_candidate() -> None:
    """Bad fine-tuning and bad O optimization cannot remove the stable candidates from selection."""
    rows = [{"phase": "validation", "round": 8, "stage": 3, "variant": variant,
             "policy": policy, "prior_rate": rate, "hard_rd": objective}
            for variant, policy, rate, objective in (
                ("T", "joint", .7, 2.), ("T", "fine", .9, 2.2),
                ("T_in_O", "joint", .7, 2.), ("T_in_O", "fine", 2., 3.3),
                ("O_from_T", "joint", .8, 2.4), ("O_from_T", "fine", 1.5, 3.),
                ("O_cold", "joint", .8, 2.5), ("O_cold", "fine", 1., 2.7),
            )]
    selected = select_candidates(rows)
    assert selected["oracle_variant"] == "T_in_O"
    assert selected["oracle_policy"] == selected["temporal_policy"] == "joint"
    with pytest.raises(AssertionError, match="round 8"):
        select_candidates([{**row, "phase": "future", "round": 9} for row in rows])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Candidate setup uses CUDA")
def test_extra_retries_keep_required_accepted_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lineage can recover after attempt4 without accepting bad rates or reducing its quota."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=1, hidden_dim=4, layers=1, warm_training=True)
    quantizer = WZQuantizerCancer(c_cfg=cfg, si_size=1)
    quantizer.candidate_failure_tolerance = confirmation.RETRY_FAILURE_TOLERANCE
    quantizer.vector_size = 4
    losses = iter([.01, 4., .02, .03, 2.])
    rates = iter([(1.2, 1.), (.8, 1.), (1.3, 1.), (1.4, 1.), (.9, 1.)])
    calls: list[object] = []

    def train(loss_fn: object, model: object, *args: object, **kwargs: object) -> float:
        calls.append(model)
        return next(losses)

    monkeypatch.setattr("FL_code.cancer_protocol.wz_quantizer.wz_model_training_loop", train)
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda *args: next(rates)))
    monkeypatch.setattr(quantizer, "encode_prepared", lambda *args: (torch.tensor([[0, 1, 0, 1]]), torch.zeros(1, 4, 2)))
    monkeypatch.setattr(quantizer, "decode_prepared", lambda *args: torch.zeros(4))
    model, loss = quantizer._train_finite_candidate(torch.zeros(4, 1), torch.zeros(4, 1), 1.,
        quantizer.coding_model.state_dict(), 2, "test")
    assert len(calls) == 5 and model is calls[-1] and loss == 2.
    assert WZQuantizerCancer.candidate_failure_tolerance == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Confirmation uses CUDA")
def test_full_vector_confirmation_cold_warm_prior_and_rans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny real fits exercise all full-vector code paths without asserting two-epoch convergence."""
    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda *args: (.8, 1.)))
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3, norm_slices=[slice(0, 128)],
        hidden_dim=4, layers=1, train_epochs=2, prior_retrain_epochs=1,
        train_batch_size=64, train_sample_size=128, quantizer_train_repeats=1, prior_train_repeats=1)
    raw = torch.linspace(-1, 1, 128)
    full_si = [HistoryVector(raw + i / 10) for i in range(9)]
    historical = tuple((raw + i / 10, full_si) for i in (1, 2))
    cold = train_oracle(cfg, raw, full_si, historical, None, 43)
    t = new_rnn_model(3, 2, 3, False, 4, 1)
    columns = torch.tensor([3, 4, 5])
    embedded = embed_temporal(t, columns, 9)
    warm = train_oracle(cfg, raw, full_si, historical, embedded, 44)
    fine = fit_embedded_prior(cfg, embedded, raw, full_si, historical)
    for candidate in (cold, warm, Candidate(embedded, {"joint": deepcopy(embedded), "fine": fine}, False)):
        rows = evaluate(candidate, cfg, raw, full_si, columns)
        assert len(rows) == 6
        assert all(row["rans_rate"] > 0 and row["wmape"] > 0 for row in rows)
    baseline = Candidate(t, {"joint": deepcopy(t)}, True)
    left, right = evaluate(baseline, cfg, raw, full_si, columns), evaluate(
        Candidate(embedded, {"joint": deepcopy(embedded)}, False), cfg, raw, full_si, columns)
    for a, b in zip(left, right, strict=True):
        assert a["rans_rate"] == b["rans_rate"]
        assert a["wmape"] == pytest.approx(b["wmape"], rel=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Confirmation uses CUDA")
def test_confirmation_worker_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a tiny complete decoded trajectory with cold/warm parents, chronological selection, and frozen reports."""
    plan_path = Path(__file__).parent / "plans/full_vector_confirmation_retry8.json"
    plan = load_plan(plan_path).model_copy(update={"output_root": tmp_path / "runs"})
    original_cfg = WZcfgQuant

    def config(**options: object) -> WZcfgQuant:
        return original_cfg(**options, hidden_dim=4, layers=1, train_epochs=2, prior_retrain_epochs=1,
            train_batch_size=64, train_sample_size=128, quantizer_train_repeats=1, prior_train_repeats=1)

    def protocol(plan: ReplayPlan, case: ReplayCase, manager: StateDictManager) -> object:
        token = ("TMR|bins_per_plane=2|num_planes=3|rans_lanes=4|hidden_dim=4|layers=1"
                 "|train_epochs=2|prior_retrain_epochs=1|train_batch_size=64|train_sample_size=128"
                 "|quantizer_train_repeats=1|prior_train_repeats=1")
        return case_protocol(plan, case.model_copy(update={"round_codecs": (token,) + ("F",) * 9}), manager)

    class TinyTrace:
        """Deterministic six-client fixture retains a genuine five-slot warm-pool transition."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def identity(self) -> dict[str, str]:
            return {"scope": "test-only tiny trace"}

        def stream(self, first: int, last: int) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
            for round_id in range(first, last + 1):
                yield round_id, tuple(torch.linspace(-1, 1, 128) * (1 + round_id / 10) + client / 10 for client in range(6))

    monkeypatch.setattr(PriorCalculator, "candidate_rates", staticmethod(lambda *args: (.8, 1.)))
    monkeypatch.setattr(confirmation, "load_plan", lambda path: plan)
    monkeypatch.setattr(confirmation, "initialize_model", lambda *args: torch.nn.Linear(128, 1, bias=False))
    monkeypatch.setattr(confirmation, "case_protocol", protocol)
    monkeypatch.setattr(confirmation, "WZcfgQuant", config)
    monkeypatch.setattr(confirmation, "Trace", TinyTrace)
    monkeypatch.setattr(confirmation, "CLIENTS", (0, 5))
    monkeypatch.setattr(confirmation, "_provenance", lambda *args: {"scope": "test-only"})
    confirmation.run(plan_path)
    confirmation.render(plan.output_root)
    assert (plan.output_root / "complete.json").is_file()
    assert (plan.output_root / "report.md").is_file()
    for client in (0, 5):
        models = torch.load(plan.output_root / "models" / f"client{client}.pt", weights_only=True)
        assert set(models) == {"T", "T_in_O", "O_from_T", "O_cold"}
        selected = json.loads((plan.output_root / "models" / f"selection_client{client}.json").read_text())
        assert selected["oracle_variant"] in {"T_in_O", "O_from_T", "O_cold"}
    assert len(list((plan.output_root / "attempts").glob("*.json"))) == 18
