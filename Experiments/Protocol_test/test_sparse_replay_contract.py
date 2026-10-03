"""Check ordinary replay plans and records without importing archived experimental codecs."""

import json
from pathlib import Path

import pandas as pd
import pytest

from Experiments.Protocol_test.records import validate_case_records
from Experiments.Protocol_test.sparse_replay import case_protocol, load_plan
from FL_code.FL_core.codec import create_protocol
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant, WZQuantizerCancer


@pytest.mark.parametrize("name", [
    "o_svhn_30c_f9_warm", "t_svhn_30c_f9_warm", "or_svhn_30c_f9", "tr_svhn_30c_f9",
    "or_svhn_30c_f9_warm", "tr_svhn_30c_f9_warm", "or_tr_svhn_30c_start12_seed44",
])
def test_retained_plans_load_without_archived_registration(name: str) -> None:
    """Ordinary O/T plans keep their trace device, ten-round schedules and acceptance identity."""
    plan = load_plan(Path(__file__).parent / "plans" / f"{name}.json")
    assert plan.trace_device == "cuda:0"
    for case in plan.cases:
        protocol = case_protocol(plan, case)
        assert len(case.round_codecs) == 10 and case.round_codecs[1:] == ("F",) * 9
        assert "candidate_acceptance" in protocol.config_snapshot()


@pytest.mark.parametrize("name", ["wz_cancer_multi", "wz_cancer_multi_warm", "wz_cancer_confirmation"])
def test_archived_protocols_are_not_registered(name: str) -> None:
    """Importing the retained runner cannot enable historical experimental protocols."""
    with pytest.raises(AssertionError):
        create_protocol(name)


def test_warm_records_accept_only_the_current_suffix(tmp_path: Path) -> None:
    """The current warm suffix validates; archived protocol-specific suffix fields do not."""
    path = tmp_path / "compression_records.csv"
    metadata = {"protocol": "wz_cancer_warm", "start_round": 5,
                "round_codecs": ("OR|bins_per_plane=2", "F"), "client_ids": (0, 1)}
    data = pd.DataFrame([
        (5, 0, "OR|bins_per_plane=2"), (5, 1, "OR|bins_per_plane=2|warm_training"),
        (6, 0, "F"), (6, 1, "F"),
    ], columns=("round_id", "client_id", "round_name_full"))
    data.to_csv(path, index=False)
    assert len(validate_case_records(path, metadata)) == 4
    data.loc[1, "round_name_full"] += "|warm_retrain_repeats=2"
    data.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="Incomplete, duplicated, or reordered"):
        validate_case_records(path, metadata)


def test_stability_defaults_and_retry_budget_remain_unchanged() -> None:
    """Archival does not promote the confirmation-only retry extension into ordinary WZ."""
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3)
    warm = cfg.warm_retraining_config()
    assert (cfg.layers, cfg.lr, cfg.train_epochs) == (4, .0028364098431326932, 140)
    assert (warm.train_epochs, warm.lr, warm.tau) == (35, cfg.lr * .3, .5)
    assert WZQuantizerCancer.candidate_failure_tolerance == 2


@pytest.mark.parametrize("codec", ["OR", "TR"])
def test_retained_plan_has_complete_all_client_windows(codec: str) -> None:
    """Tracked warm plans retain three placements and all-client full-window measurement."""
    path = Path(__file__).parent / "plans" / f"{codec.lower()}_svhn_30c_f9_warm.json"
    plan = load_plan(path)
    assert [case.start_round for case in plan.cases] == [5, 12, 20]
    assert json.loads(plan.fl_config.read_text())["num_clients"] == 30
