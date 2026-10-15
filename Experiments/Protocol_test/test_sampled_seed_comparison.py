"""Check independent trimming, untouched fresh-seed cohorts and paired aggregate RD uncertainty."""
from pathlib import Path
import csv
import json
from typing import Any

import pytest

from Experiments.Protocol_test import sampled_seed_comparison as study
from Experiments.Protocol_test.five_seed_tables import mean_points
from Experiments.Protocol_test.starting_model_comparison import SampleFitConfig


def record(scale: float) -> dict[str, Any]:
    """Provide a monotone three-stage RD measurement with a known multiplicative rate effect."""
    rates = [.2*scale, .25*scale, .3*scale]
    return dict(quantizer_prior_stage_rates=rates, server_si_prior_stage_rates=rates,
                coder_stage_rates=rates, server_si_rans_stage_rates=rates, stage_wmapes=[70., 55., 40.], marginal_rate=2.)


@pytest.fixture
def plan(tmp_path: Path) -> study.SeedComparisonPlan:
    """Describe twenty paired toy fits with the same independent-trimming policies as the study."""
    configs = {label: SampleFitConfig(name=label, train_epochs=28, lr=.00068, tau=tau,
                                     sample_size=3600000, train_batch_size=7500) for label, tau in (('S04', .2), ('S08', .3))}
    return study.SeedComparisonPlan(name='toy', output_root=tmp_path/'output', reference_root=tmp_path,
        source_root=tmp_path/'original', report_path=tmp_path/'report.md', worker_plans=(tmp_path/'worker.json',),
        configurations=configs, seeds=tuple(range(43, 63)), original_seeds=tuple(range(43, 48)),
        ranking_span_db=(-5., -4.), trim_count=12, bootstrap_samples=50)


@pytest.fixture
def groups() -> dict[str, list[dict[str, Any]]]:
    """Make each configuration favor opposite seeds while sharing the same all-twenty mean."""
    return {label: [dict(fit_seed=seed, record=record(1 + .02*(seed-43 if label=='S04' else 62-seed)))
                   for seed in range(43, 63)] for label in ('S04', 'S08')}


def test_fresh_cohort_and_independent_selection(plan: study.SeedComparisonPlan, groups: dict[str, list[dict[str, Any]]]) -> None:
    """Original and fresh seeds retain different effects, while each method keeps its own strongest seeds."""
    result = study.summarize(groups, plan)
    for cohort, count in (('original5_avg', 5), ('new15_avg', 15), ('avg20', 20), ('trimmed12', 12), ('best3', 3)):
        assert all(point['observations']==count for point in result['cohorts'][cohort].values())
    selected = result['selection']
    assert [row['seed'] for row in selected if row['cohort']=='best3' and row['method']=='S04' and row['retained']] == [43, 44, 45]
    assert [row['seed'] for row in selected if row['cohort']=='best3' and row['method']=='S08' and row['retained']] == [60, 61, 62]
    assert study.gap(result['cohorts']['original5_avg']['S08'], result['cohorts']['original5_avg']['S04']) > 0
    assert study.gap(result['cohorts']['new15_avg']['S08'], result['cohorts']['new15_avg']['S04']) < 0
    assert study.gap(result['cohorts']['avg20']['S08'], result['cohorts']['avg20']['S04']) == pytest.approx(0., abs=1e-12)
    assert len(result['paired']) == 20


def test_declared_ranking_span_is_not_silently_changed(plan: study.SeedComparisonPlan, groups: dict[str, list[dict[str, Any]]]) -> None:
    """Out-of-range seed ranking leaves trimmed results unavailable and preserves every untrimmed observation."""
    result = study.summarize(groups, plan.model_copy(update={'ranking_span_db': (-50., -40.)}))
    assert all(point is None for cohort in ('trimmed12', 'best3') for point in result['cohorts'][cohort].values())
    assert all(point['observations']==20 for point in result['cohorts']['avg20'].values())
    assert not any(row['retained'] for row in result['selection'])


def test_common_span_preserves_a_seed_outside_the_original_envelope(plan: study.SeedComparisonPlan, groups: dict[str, list[dict[str, Any]]]) -> None:
    """A shifted valid curve narrows the common span instead of losing that seed from the ranking."""
    groups['S08'][0]['record']['stage_wmapes'] = [63., 52., 40.]
    result = study.summarize(groups, plan)
    assert result['ranking_span_db'][1] < plan.ranking_span_db[1]
    assert all(row['common_mean_bpf'] is not None for row in result['selection'])
    assert all(point['observations']==12 for point in result['cohorts']['trimmed12'].values())


def test_paired_bootstrap_recomputes_aggregate_rate_gap() -> None:
    """A constant paired five-percent rate reduction yields the same five-percent bootstrap interval."""
    result = study.bootstrap_gap([record(.95)]*20, [record(1.)]*20, 100)
    assert result['valid_draws'] == 100
    assert result['low_percent'] == pytest.approx(-5.)
    assert result['high_percent'] == pytest.approx(-5.)


def test_completed_report_contains_all_cohorts_and_raw_stage_metrics(plan: study.SeedComparisonPlan, groups: dict[str, list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch) -> None:
    """The automatic final publisher writes complete matrices, selection and measured stage tables."""
    source=plan.source_root/'final_analysis/comparison_with_t';source.mkdir(parents=True)
    reference=mean_points([record(1.)])
    (source/'aggregate_points.json').write_text(json.dumps({'all_five': {'O': reference, 'F*': reference}, 'best_three': {'O': reference}}))
    monkeypatch.setattr(study, 'read_runs', lambda *_args, **_kwargs: (groups, []))
    output=plan.output_root/'analysis'
    study.publish(plan, output, complete=True)
    with (output/'comparison_gaps.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2*13*13
    with (output/'stage_points.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==39
    assert all(not row['marginal'] for row in rows if row['stage']!='3')
    assert plan.report_path.exists() and (output/'paired_rd.pdf').exists()
    assert json.loads((output/'statistics.json').read_text())['cohorts']['avg20']['S08']['observations']==20
