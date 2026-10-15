"""Detached paired S04/S08 seed extension with automatic raw-metric and uncertainty reporting."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from Experiments.Protocol_test.five_seed_tables import mean_points, mean_rate
from Experiments.Protocol_test.live_plot import watch
from Experiments.Protocol_test.rd_metrics import wmape_to_db
from Experiments.Protocol_test.sampled_local_grid import gap, run_workers, valid_curve
from Experiments.Protocol_test.starting_model_comparison import SampleFitConfig, load_plan as load_worker_plan
from FL_code.FL_core.explicit_cache import atomic_output, content_hash


class SeedComparisonPlan(BaseModel):
    """Paired seed study produced before training and consumed by the supervisor, workers and report."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    output_root: Path
    reference_root: Path
    source_root: Path
    report_path: Path
    worker_plans: tuple[Path, ...] = Field(min_length=1, max_length=3)
    configurations: dict[str, SampleFitConfig]
    seeds: tuple[int, ...]
    original_seeds: tuple[int, ...]
    ranking_span_db: tuple[float, float]
    trim_count: int = Field(ge=3)
    bootstrap_samples: int = Field(default=2_000, ge=1)

    @property
    def cases(self) -> tuple[str, ...]:
        """Expose only newly scheduled fits for detached health coverage."""
        return tuple(case for path in self.worker_plans for case in load_worker_plan(path).cases)


def load_plan(path: Path) -> SeedComparisonPlan:
    """Resolve a fresh paired plan and verify that its workers schedule every new seed exactly once."""
    plan = SeedComparisonPlan.model_validate_json(path.read_text())
    plan = plan.model_copy(update={key: (path.parent/getattr(plan, key)).resolve() for key in
                                  ("output_root", "reference_root", "source_root", "report_path")} |
                           {"worker_plans": tuple((path.parent/value).resolve() for value in plan.worker_plans)})
    assert set(plan.configurations) == {"S04", "S08"}
    first, second = plan.configurations["S04"], plan.configurations["S08"]
    assert first.model_dump(exclude={"name", "tau"}) == second.model_dump(exclude={"name", "tau"})
    assert (first.tau, second.tau) == (.2, .3), "S04/S08 must retain their historical tau settings."
    assert len(plan.seeds) == len(set(plan.seeds)) == 20 and len(plan.original_seeds) == 5
    assert set(plan.original_seeds) < set(plan.seeds) and plan.trim_count == 12
    assert plan.ranking_span_db[0] < plan.ranking_span_db[1] and plan.trim_count < len(plan.seeds)
    source = json.loads((plan.source_root/"grid_plan.json").read_text())
    assert Path(source["reference_root"]) == plan.reference_root
    catalog = {config["name"]: SampleFitConfig.model_validate(config) for config in source["grid"]}
    assert all(config == catalog[config.name] for config in plan.configurations.values())
    workers = [load_worker_plan(path) for path in plan.worker_plans]
    fits = [fit for worker in workers for fit in worker.fits]
    expected = {(config.name, seed) for config in plan.configurations.values() for seed in set(plan.seeds)-set(plan.original_seeds)}
    assert {(fit.sample_fit.name, fit.seed) for fit in fits} == expected and len(fits) == len(expected)
    assert all(worker.methods == ("S",) and worker.fit_device == "cuda:0" and worker.reference_root == plan.reference_root for worker in workers)
    assert all(fit.sample_fit in plan.configurations.values() for fit in fits)
    return plan


def read_runs(plan: SeedComparisonPlan, *, complete: bool) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    """Import the five historical fits and read new measurements without performance-based exclusions."""
    labels = {config.name: label for label, config in plan.configurations.items()}
    groups: dict[str, list[dict[str, Any]]] = {label: [] for label in plan.configurations}
    paths = [plan.source_root/"cases"/f"S_{config.name}_seed{seed}"/"target/complete.json"
             for config in plan.configurations.values() for seed in plan.original_seeds]
    paths += sorted((plan.output_root/"cases").glob("*/*/complete.json"))
    historical = {label: json.loads((plan.source_root/"cases"/f"S_{config.name}_seed{plan.original_seeds[0]}"/"target/complete.json").read_text())
                  for label, config in plan.configurations.items()}
    initial_hash = historical["S04"]["initial_state_hash"]
    for path in paths:
        run = json.loads(path.read_text())
        label = labels[run["sample_fit"]["name"]]
        assert run["method"] == "S" and run["sample_fit"] == plan.configurations[label].model_dump()
        assert run["initial_state_hash"] == initial_hash and run["quantizer_si_columns"] == 150
        assert run["training_config"] == historical[label]["training_config"], f"Training settings changed: {path}"
        assert run["record"]["symbol_error_rate"] == 0, f"Native symbol errors: {path}"
        assert np.isclose(run["record"]["stage_wmapes"][-1], run["record"]["wmape"])
        groups[label].append(run)
    for label, group in groups.items():
        group.sort(key=lambda run: run["fit_seed"])
        seeds = [run["fit_seed"] for run in group]
        assert len(seeds) == len(set(seeds)) and set(plan.original_seeds) <= set(seeds) <= set(plan.seeds)
        if complete:
            assert seeds == list(plan.seeds), f"Incomplete seed coverage: {label} {seeds}"
    reference = json.loads((plan.source_root/"shards/part1/test_config.json").read_text())
    for path in plan.worker_plans:
        manifest_path = load_worker_plan(path).output_root/"test_config.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert all(manifest[key] == reference[key] for key in ("reference_manifest", "baseline_state_hash", "numerical_runtime")), f"Unmatched worker: {manifest_path}"
    return groups, paths


def validate_artifacts(plan: SeedComparisonPlan, paths: list[Path], destination: Path) -> None:
    """Check seeded sample pools, starting weights, prior-only weight changes and archived numerical sources."""
    import torch

    baseline_path, = (plan.reference_root/"baseline").glob("*.pt")
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)["state"]
    runs = [json.loads(path.read_text()) for path in paths]
    vector_size = runs[0]["record"]["model_size"]
    count = plan.configurations["S04"].sample_size
    expected: dict[int, str] = {}
    for seed in plan.seeds:
        coordinate_seed = int(torch.randint(0, 2**31, (), generator=torch.Generator().manual_seed(seed)).item())
        expected[seed] = content_hash(torch.from_numpy(np.sort(np.random.default_rng(coordinate_seed).choice(vector_size, count, replace=False))))
    checks = []
    for path, run in zip(paths, runs, strict=True):
        weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
        assert content_hash({key: baseline[key] for key in weights["joint"]}) == run["initial_state_hash"]
        assert content_hash(weights["sample_indices"]) == expected[run["fit_seed"]], f"Incorrect seed/sample pool: {path}"
        assert all(torch.equal(value, weights["native"][key]) for key, value in weights["joint"].items() if not key.startswith("conditional")), f"Prior fit changed quantizer: {path}"
        checks.append(dict(path=str(path), seed=run["fit_seed"], sample_hash=expected[run["fit_seed"]], joint_hash=content_hash(weights["joint"])))
    assert len(checks) == 2*len(plan.seeds)
    archives = [*(plan.source_root/"shards").glob("*/sources.tar.gz"), *(load_worker_plan(path).output_root/"sources.tar.gz" for path in plan.worker_plans)]
    source_hashes = {}
    for path in archives:
        with tarfile.open(path) as archive:
            sources = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.name.startswith("FL_code/") or member.name == "Experiments/Protocol_test/starting_model_comparison.py"}
        source_hashes[str(path)] = content_hash(sources)
    assert len(set(source_hashes.values())) == 1, "Numerical FL or sample-fit worker sources changed since the five-seed grid."
    (destination/"validation.json").write_text(json.dumps(dict(weights=checks, fl_sources=source_hashes,
        paired_sample_pools_exact=True, quantizers_unchanged_in_prior_fit=True), indent=2))


def bootstrap_gap(first: list[dict[str, Any]], second: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    """Resample paired seeds and recompute aggregate RD gaps, returning a percentile interval."""
    assert len(first) == len(second) and first
    indices = np.random.default_rng(20261009).integers(0, len(first), (draws, len(first)))
    aggregates = []
    for records in (first, second):
        rates = np.array([np.cumsum(record["server_si_rans_stage_rates"]) for record in records])[indices].mean(axis=1)
        raw = np.array([record["stage_wmapes"] for record in records])[indices].mean(axis=1)
        aggregates.append([dict(server_si_rans_stage_rates=rate.tolist(), stage_wmapes=wmape.tolist(), wmape_db=wmape_to_db(wmape).tolist()) for rate, wmape in zip(rates, raw, strict=True)])
    values = np.array([gap(row, column) for row, column in zip(*aggregates, strict=True)])
    valid = np.isfinite(values)
    bounds = np.quantile(values, [.025, .975]).tolist() if valid.all() else [None, None]
    return dict(draws=draws, valid_draws=int(valid.sum()), low_percent=bounds[0], high_percent=bounds[1])


def summarize(groups: dict[str, list[dict[str, Any]]], plan: SeedComparisonPlan) -> dict[str, Any]:
    """Compare the original, fresh and pooled seeds with independent best-three and best-60% selection."""
    cohorts: dict[str, dict[str, Any]] = {}
    selection = []
    for cohort, seeds in (("original5_avg", plan.original_seeds), ("new15_avg", tuple(seed for seed in plan.seeds if seed not in plan.original_seeds)), ("avg20", plan.seeds)):
        cohorts[cohort] = {label: mean_points([run["record"] for run in group if run["fit_seed"] in seeds]) for label, group in groups.items()}
    valid_points = [mean_points([run["record"]]) for group in groups.values() for run in group
                    if valid_curve(mean_points([run["record"]]))]
    low = max([plan.ranking_span_db[0], *(min(point["wmape_db"]) for point in valid_points)])
    high = min([plan.ranking_span_db[1], *(max(point["wmape_db"]) for point in valid_points)])
    ranking_span = [low, high] if valid_points and low < high else None
    scores: dict[str, list[tuple[dict[str, Any], float | None]]] = {}
    for label, group in groups.items():
        scores[label] = []
        for run in group:
            point = mean_points([run["record"]])
            score = mean_rate(point, low, high) if ranking_span is not None and valid_curve(point) and min(point["wmape_db"]) <= low < high <= max(point["wmape_db"]) else None
            scores[label].append((run, score))
    for cohort, count in (("trimmed12", plan.trim_count), ("best3", 3)):
        cohorts[cohort] = {}
        for label, scored in scores.items():
            kept = sorted(((run, score) for run, score in scored if score is not None), key=lambda pair: (pair[1], pair[0]["fit_seed"]))[:count]
            retained = {run["fit_seed"] for run, _score in kept} if len(kept) == count else set()
            cohorts[cohort][label] = mean_points([run["record"] for run, _score in kept]) if retained else None
            selection.extend(dict(cohort=cohort, method=label, seed=run["fit_seed"], common_mean_bpf=score,
                                  retained=run["fit_seed"] in retained, reason="ranked RD" if score is not None else "unrankable on declared common span") for run, score in scored)
    paired = []
    for first, second in zip(groups["S04"], groups["S08"], strict=True):
        assert first["fit_seed"] == second["fit_seed"]
        rate_first = next(score for run, score in scores["S04"] if run["fit_seed"] == first["fit_seed"])
        rate_second = next(score for run, score in scores["S08"] if run["fit_seed"] == first["fit_seed"])
        paired.append(dict(seed=first["fit_seed"], original=first["fit_seed"] in plan.original_seeds,
            s08_vs_s04_percent=gap(mean_points([second["record"]]), mean_points([first["record"]])),
            s04_common_bpf=rate_first, s08_common_bpf=rate_second,
            common_delta_s08_minus_s04_bpf=rate_second-rate_first if rate_first is not None and rate_second is not None else None))
    intervals = {}
    for cohort, seeds in (("avg20", plan.seeds), ("new15_avg", tuple(seed for seed in plan.seeds if seed not in plan.original_seeds))):
        records = {label: [run["record"] for run in group if run["fit_seed"] in seeds] for label, group in groups.items()}
        intervals[cohort] = bootstrap_gap(records["S08"], records["S04"], plan.bootstrap_samples)
    return dict(cohorts=cohorts, selection=selection, paired=paired, bootstrap=intervals, ranking_span_db=ranking_span)


def publish(plan: SeedComparisonPlan, destination: Path, *, complete: bool) -> None:
    """Refresh paired plots or publish all raw metrics, independently trimmed curves and uncertainty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    groups, _paths = read_runs(plan, complete=complete)
    shared = sorted(set.intersection(*(set(run["fit_seed"] for run in group) for group in groups.values())))
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, group in groups.items():
        point = mean_points([run["record"] for run in group if run["fit_seed"] in shared])
        axis.plot(point["wmape_db"], point["server_si_rans_stage_rates"], "o-", label=f"{label}: {len(shared)}/{len(plan.seeds)} paired seeds")
    axis.set(xlabel="WMAPE (dB)", ylabel="Cumulative full-SI rANS (bits/float)", title="S04 vs S08: paired average" + ("" if complete else " — PARTIAL"))
    axis.legend(); axis.grid(alpha=.2); figure.tight_layout()
    figure.savefig(destination/"paired_rd.png", dpi=180); figure.savefig(destination/"paired_rd.pdf")
    plt.close(figure)
    (destination/"coverage.json").write_text(json.dumps(dict(complete=complete, paired_seeds=shared,
        available={label: [run["fit_seed"] for run in group] for label, group in groups.items()}), indent=2))
    if not complete:
        return
    result = summarize(groups, plan)
    (destination/"raw_records.json").write_text(json.dumps(groups, indent=2))
    (destination/"statistics.json").write_text(json.dumps(result, indent=2))
    references = json.loads((plan.source_root/"final_analysis/comparison_with_t/aggregate_points.json").read_text())
    points = {f"{label} {cohort}": point for cohort, methods in result["cohorts"].items() for label, point in methods.items()}
    points |= {"O avg5": references["all_five"]["O"], "O trimmed3": references["best_three"]["O"], "F": references["all_five"]["F*"]}
    rates = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates", "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    stage_rows = [dict(method=label, stage=stage+1, marginal=point["marginal_rate"] if point is not None and stage==2 else None,
        **{field: point[field][stage] if point is not None and point[field] is not None else None for field in rates},
        wmape_db=point["wmape_db"][stage] if point is not None else None, raw_wmape_percent=point["stage_wmapes"][stage] if point is not None else None)
        for label, point in points.items() for stage in range(3)]
    matrices = [dict(rate_source=field, row=label, column=other, rate_change_percent=gap(point, reference, field))
        for field in ("server_si_rans_stage_rates", "coder_stage_rates") for label, point in points.items() for other, reference in points.items()]
    comparisons = [dict(cohort=cohort, s08_vs_s04_percent=gap(methods["S08"], methods["S04"]),
                        s04_vs_s08_percent=gap(methods["S04"], methods["S08"]), **result["bootstrap"].get(cohort, {}))
                   for cohort, methods in result["cohorts"].items()]
    for filename, rows in (("selection.csv", result["selection"]), ("paired_seed_metrics.csv", result["paired"]),
                           ("stage_points.csv", stage_rows), ("comparison_gaps.csv", matrices), ("cohort_comparisons.csv", comparisons)):
        with (destination/filename).open("w") as stream:
            writer=csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader(); writer.writerows(rows)
    diagnostics = []
    for label, group in groups.items():
        for run in group:
            point = mean_points([run["record"]]); flags = []
            if not valid_curve(point): flags.append("nonmonotone stage RD")
            if point["quantizer_prior_stage_rates"][-1] > point["marginal_rate"]: flags.append("joint prior exceeds marginal")
            flags.extend(f"stage {stage+1} native growth {rate:.6f}" for stage, rate in enumerate(run["record"]["coder_stage_rates"]) if rate < .01 or rate > 1.05)
            diagnostics.append(dict(method=label, seed=run["fit_seed"], flags=flags))
    (destination/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    paired_counts = {cohort: dict(ranked_pairs=sum(row["common_delta_s08_minus_s04_bpf"] is not None for row in result["paired"] if row["seed"] in seeds),
        s08_lower_common_rate=sum(row["common_delta_s08_minus_s04_bpf"] < 0 for row in result["paired"] if row["seed"] in seeds and row["common_delta_s08_minus_s04_bpf"] is not None))
        for cohort, seeds in (("avg20", plan.seeds), ("new15_avg", tuple(seed for seed in plan.seeds if seed not in plan.original_seeds)))}

    def cell(value: float | None) -> str:
        """Render unmeasured or unrankable statistics explicitly."""
        return f"{value:+.6f}" if value is not None and np.isfinite(value) else "—"

    lines = ["# S04 versus S08: expanded seed comparison", "", "**Complete: 20 matched fit seeds per configuration (43–62), including 15 new seeds each.**", "",
        "S04: tau .20; S08: tau .30. Both use 3.6M raw samples, batch 7500, 28 epochs, LR 0.0006807383623518464.",
        "Starting weights, target, 150 SI columns and numerical policy are unchanged; RTX 3090/cuda:0. Samples are identical for each paired seed. Quantizer and entropy-prior fitting retain the original schedules.",
        "This continues the raw-S oracle diagnostic on one client/one trained round. It does not measure acquisition cost or the trained-plus-nine-frozen population protocol.",
        "All rates are averaged cumulatively and WMAPE as raw percentages before converting to dB. Individual gap percentages are never averaged.",
        "Avg20 is primary. New15_avg excludes the five seeds used during configuration selection. Trimmed12 preserves the prior 60% retention fraction; best3 preserves the earlier absolute best-three rule.",
        f"Ranking intersects the declared original grid envelope {plan.ranking_span_db} with all valid individual curves: effective WMAPE dB span {result['ranking_span_db']}. Each configuration selects its own seeds; unrankable trimmed curves are missing, without extrapolation.",
        "O avg5 and O trimmed3 are fixed historical five-seed controls, not expanded twenty-seed O measurements. F is unchanged without fitting.",
        "No client exclusions: 1/1 retained, with/without exclusions identical. Stage anomalies are flagged without silently removing them.", "",
        f"![Paired average RD]({os.path.relpath(destination / 'paired_rd.png', plan.report_path.parent)})", "",
        "| Cohort | S08 versus S04 % | S04 versus S08 % | Paired bootstrap 95% interval for S08 versus S04 % |", "|---|---:|---:|---|"]
    for row in comparisons:
        interval = result["bootstrap"].get(row["cohort"])
        bounds = f"[{cell(interval['low_percent'])}, {cell(interval['high_percent'])}]" if interval else "—"
        lines.append(f"| {row['cohort']} | {cell(row['s08_vs_s04_percent'])} | {cell(row['s04_vs_s08_percent'])} | {bounds} |")
    lines.append(f"\nSame-span paired seed counts: {paired_counts}.")
    lines.extend(["", "Negative favors the row. Bootstrap resamples matched fit seeds and recomputes mean curves; it measures fit-seed uncertainty on this fixed target, not uncertainty across clients or starting models.", "",
        "| Seed | S08 versus S04 % | S04 common bpf | S08 common bpf | S08 minus S04 common bpf |", "|---|---:|---:|---:|---:|"])
    for row in result["paired"]:
        lines.append(f"| {row['seed']} | " + " | ".join(cell(row[key]) for key in ("s08_vs_s04_percent", "s04_common_bpf", "s08_common_bpf", "common_delta_s08_minus_s04_bpf")) + " |")
    for cohort in ("trimmed12", "best3"):
        for label in groups:
            kept = [row["seed"] for row in result["selection"] if row["cohort"]==cohort and row["method"]==label and row["retained"]]
            lines.append(f"\n{label} {cohort} retained seeds: {kept}.")
    for field in ("server_si_rans_stage_rates", "coder_stage_rates"):
        lines.extend(["", f"## Full square: {field}", "", "Row versus column (%) with reverse entries recomputed. Before/after client exclusions identical.", "", "```text", " "*23 + " ".join(f"{label:>23}" for label in points)])
        for label, point in points.items():
            lines.append(f"{label:>23} " + " ".join(f"{cell(gap(point, reference, field)):>23}" for reference in points.values()))
        lines.extend(["```", ""])
    lines.extend(["## Cumulative stage rates (bits/float)", "", "Marginal totals are measured only at the final stage; — is unmeasured. F priors are frozen, with no refit.", "",
        "| Method | Stage | Marginal | Joint/pre-prior fit | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB | Raw WMAPE % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in stage_rows:
        lines.append(f"| {row['method']} | {row['stage']} | " + " | ".join(cell(row[key]) for key in ("marginal", *rates, "wmape_db", "raw_wmape_percent")) + " |")
    flags = [row for row in diagnostics if row["flags"]]
    lines.extend(["", "## Diagnostics", "", "All native symbol errors are zero. Per-stage marginal entropy was not measured; stage death is not inferred.",
        *[f"- {row['method']} seed {row['seed']}: {', '.join(row['flags'])}." for row in flags],
        "No stage monotonicity, rate-growth or prior/marginal flags." if not flags else "",
        f"\nRaw artifacts and reproducible analysis: `{destination}`."])
    report = "\n".join(lines)+"\n"
    (destination/"report.md").write_text(report)
    (destination/"analysis_source.py").write_text(Path(__file__).read_text())
    with atomic_output(plan.report_path) as temporary: temporary.write_text(report)
    print(f"Published expanded S04/S08 comparison: {plan.report_path}", flush=True)


def run(plan: SeedComparisonPlan) -> None:
    """Launch bounded detached children, validate their artifacts and automatically publish the completed report."""
    plan.output_root.mkdir(parents=True, exist_ok=True)
    (plan.output_root/"cases").mkdir(exist_ok=True)
    (plan.output_root/"seed_plan.json").write_text(plan.model_dump_json(indent=2))
    (plan.output_root/"test_config.json").write_text(json.dumps(dict(study=plan.model_dump(mode="json"), scheduled_cases=len(plan.cases)), indent=2))
    for path in plan.worker_plans:
        worker = load_worker_plan(path)
        for label in worker.cases:
            link = plan.output_root/"cases"/label
            if not link.is_symlink(): link.symlink_to(worker.output_root/"cases"/label, target_is_directory=True)
    run_workers(plan.worker_plans, plan.output_root/"workers")
    destination = plan.output_root/"final_analysis"
    destination.mkdir(exist_ok=True)
    _groups, paths = read_runs(plan, complete=True)
    validate_artifacts(plan, paths, destination)
    publish(plan, destination, complete=True)


def main() -> None:
    """Provide detached execution, disposable live plots and completed-report regeneration."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path); parser.add_argument("--out", type=Path)
    parser.add_argument("--runner-pid", type=int); parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--report", action="store_true")
    args=parser.parse_args()
    if args.runner_pid is not None:
        assert args.control_dir is not None
        watch(args.path, args.runner_pid, args.control_dir, plot_module=__spec__.name,
              source_paths=(Path(__file__), Path(__file__).with_name("five_seed_tables.py")))
    elif args.out is not None:
        publish(SeedComparisonPlan.model_validate_json((args.path/"seed_plan.json").read_text()), args.out, complete=False)
    else:
        plan=load_plan(args.path)
        if args.report:
            destination=plan.output_root/"final_analysis"; destination.mkdir(exist_ok=True)
            _groups, paths=read_runs(plan, complete=True)
            validate_artifacts(plan, paths, destination); publish(plan, destination, complete=True)
        else:
            run(plan)


if __name__ == "__main__":
    main()
