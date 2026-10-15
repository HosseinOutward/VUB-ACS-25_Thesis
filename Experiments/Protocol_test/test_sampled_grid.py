"""Verify grid cases and sample-only optimizer overrides before full-size experiments."""

from pathlib import Path
import csv
from itertools import product
from typing import Any

import numpy as np
import pytest

from Experiments.Protocol_test import sampled_local_grid
from Experiments.Protocol_test.starting_model_comparison import ComparisonPlan, RawSampleQuantizer, SampleFitConfig
from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant


def test_grid_expands_only_sample_cases() -> None:
    """Adding sample variants creates distinct directories without duplicating ordinary O."""
    settings = tuple(SampleFitConfig(name=f"e{epochs}", train_epochs=epochs, lr=.001, tau=.25) for epochs in (28, 42))
    plan = ComparisonPlan(name="test", output_root=Path("output"), reference_root=Path("reference"),
                          methods=("O", "S"), seeds=(43, 44), sample_fits=settings)
    assert plan.cases == ("O_seed43", "S_e28_seed43", "S_e42_seed43", "O_seed44", "S_e28_seed44", "S_e42_seed44")


def test_legacy_plan_preserves_case_labels() -> None:
    """Existing measurement plans retain their method/seed directory layout."""
    plan = ComparisonPlan(name="test", output_root=Path("output"), reference_root=Path("reference"), methods=("O", "S_tau025"), seeds=(43,))
    assert plan.cases == ("O_seed43", "S_tau025_seed43")


@pytest.mark.parametrize("epochs,lr,tau,count,batch", [(28, .0006807383623518464, .2, 2_400_000, 5_000), (35, .000850922952939808, .25, 3_000_000, 6_250), (42, .0010211075435277695, .3, 3_600_000, 7_500)])
def test_grid_optimizer_leaves_prior_configuration_ordinary(epochs: int, lr: float, tau: float, count: int, batch: int) -> None:
    """Grid settings are applied after warm conversion without changing the native-prior config."""
    config = WZcfgQuant(bins_per_plane=2, num_planes=3, warm_training=True, hidden_dim=4, layers=1)
    quantizer = RawSampleQuantizer(config, si_size=2)
    quantizer.sample_fit_config = SampleFitConfig(name="test", train_epochs=epochs, lr=lr, tau=tau, sample_size=count, train_batch_size=batch)
    effective = quantizer.training_config()
    assert (effective.train_epochs, effective.lr, effective.tau) == (epochs, lr, tau)
    assert effective.train_batch_size == quantizer.train_batch_size == batch
    assert quantizer.sample_fit_config.sample_size == count
    assert quantizer.c_cfg.train_batch_size == 6250
    assert quantizer.c_cfg is config
    assert quantizer.c_cfg.warm_retraining_config().tau == .5
    assert quantizer.c_cfg.prior_retrain_epochs == 30


def test_corner_axis_report_recovers_known_direction_and_interaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end reporting recovers a known LR effect and epoch–tau interaction with matched seeds."""
    base_lr = .0008509229529398079
    levels = ((28, 35, 42), (.8, 1., 1.2), (.2, .25, .3), (2_400_000, 3_000_000, 3_600_000), (5_000, 6_250, 7_500))
    middle = tuple(values[1] for values in levels)
    coordinates = set(product(*[(values[0], values[-1]) for values in levels])) | {middle}
    coordinates |= {tuple(value if i == changed else middle[i] for i in range(5))
                    for changed, values in enumerate(levels) for value in (values[0], values[-1])}
    grid = tuple(SampleFitConfig(name=f"e{epochs}_l{round(multiplier*100):03}_t{round(tau*100):03}_n{count}_b{batch}",
                                train_epochs=epochs, lr=base_lr*multiplier, tau=tau, sample_size=count, train_batch_size=batch)
                 for epochs, multiplier, tau, count, batch in sorted(coordinates))
    plan = sampled_local_grid.GridPlan(name="toy", output_root=tmp_path / "results", reference_root=tmp_path,
        report_path=tmp_path / "report.md", control_roots=(), worker_plans=(tmp_path / "worker.json",), grid=grid,
        center_name="e35_l100_t025_n3000000_b6250")

    def records(scale: float) -> list[dict[str, Any]]:
        return [dict(fit_seed=seed, record=dict(
            quantizer_prior_stage_rates=[.2*(scale+(seed-45)*.001), .25*(scale+(seed-45)*.001), .3*(scale+(seed-45)*.001)],
            server_si_prior_stage_rates=[.2*(scale+(seed-45)*.001), .25*(scale+(seed-45)*.001), .3*(scale+(seed-45)*.001)],
            coder_stage_rates=[.2*(scale+(seed-45)*.001), .25*(scale+(seed-45)*.001), .3*(scale+(seed-45)*.001)],
            server_si_rans_stage_rates=[.2*(scale+(seed-45)*.001), .25*(scale+(seed-45)*.001), .3*(scale+(seed-45)*.001)],
            stage_wmapes=[70., 55., 40.], marginal_rate=2.)) for seed in range(43, 48)]

    groups = {config.name: records(1 + .01*(config.train_epochs-35)*(config.tau-.25) + .5*(config.lr/base_lr-1)) for config in grid}
    groups |= {"O": records(1.), "O_tau025": records(1.05), "F*": records(1.5)[:1]}
    monkeypatch.setattr(sampled_local_grid, "read_runs", lambda *_args, **_kwargs: groups)
    destination = tmp_path / "analysis"
    sampled_local_grid.publish(plan, destination, complete=True)
    with (destination / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 43
    assert float(next(row for row in summary if row["name"] == plan.center_name)["all_five_gap_vs_o"]) == pytest.approx(0., abs=1e-12)
    high_lr = next(row for row in summary if row["name"] == "e35_l120_t025_n3000000_b6250")
    assert float(high_lr["all_five_gap_vs_o"]) == pytest.approx(10.)
    assert high_lr["retained_seeds"] == "[43, 44, 45]"
    with (destination / "interaction_effects.csv").open() as stream:
        interactions = list(csv.DictReader(stream))
    row = next(row for row in interactions if row["first_parameter"] == "train_epochs" and row["second_parameter"] == "tau")
    assert len(interactions) == 80
    with (destination / "direction_effects.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 85
    db = 20*np.log10(np.array([.4, .55, .7]))
    base_mean_rate = np.trapezoid([.75, .45, .2], db) / (db[-1]-db[0])
    assert float(row["interaction_delta_bpf"]) == pytest.approx(.01*14*.1*base_mean_rate)
    assert plan.report_path.exists() and (destination / "parameter_directions.png").exists()
    with (destination / "comparison_gaps.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2*2*46*46
