"""Detached raw-S corner/axis grid with matched seeds and automatic final reporting."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from itertools import combinations, product
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from Experiments.Protocol_test.five_seed_tables import mean_points, mean_rate
from Experiments.Protocol_test.live_plot import watch
from Experiments.Protocol_test.starting_model_comparison import SampleFitConfig, load_plan as load_worker_plan
from Experiments.Protocol_test.starting_model_tables import curve_gap
from FL_code.FL_core.explicit_cache import atomic_output, content_hash


class GridPlan(BaseModel):
    """Corner/axis catalog, immutable controls and bounded worker plans for one study."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    output_root: Path
    reference_root: Path
    report_path: Path
    control_roots: tuple[Path, ...]
    worker_plans: tuple[Path, ...] = Field(min_length=1, max_length=3)
    grid: tuple[SampleFitConfig, ...] = Field(min_length=1)
    center_name: str
    seeds: tuple[int, ...] = (43, 44, 45, 46, 47)
    parameters: tuple[str, ...] = ("train_epochs", "lr", "tau", "sample_size", "train_batch_size")

    @property
    def cases(self) -> tuple[str, ...]:
        """Expose only newly scheduled measurements to the detached health supervisor."""
        return tuple(case for path in self.worker_plans for case in load_worker_plan(path).cases)


def load_plan(path: Path) -> GridPlan:
    """Resolve a fresh grid plan's saved references, workers and report destination."""
    plan = GridPlan.model_validate_json(path.read_text())
    updates = {key: (path.parent / getattr(plan, key)).resolve() for key in ("output_root", "reference_root", "report_path")}
    updates |= {key: tuple((path.parent / value).resolve() for value in getattr(plan, key))
                for key in ("control_roots", "worker_plans")}
    plan = plan.model_copy(update=updates)
    assert len(set(plan.cases)) == len(plan.cases), "Grid worker plans schedule duplicate cases."
    assert len({config.name for config in plan.grid}) == len(plan.grid), "Grid names are not unique."
    assert plan.center_name in {config.name for config in plan.grid}
    levels = [sorted({getattr(config, field) for config in plan.grid}) for field in plan.parameters]
    assert all(len(values) == 3 for values in levels), "This study needs three levels per parameter."
    middle = tuple(values[1] for values in levels)
    expected_coordinates = set(product(*[(values[0], values[-1]) for values in levels])) | {middle}
    expected_coordinates |= {tuple(value if i == changed else middle[i] for i in range(len(levels)))
                             for changed, values in enumerate(levels) for value in (values[0], values[-1])}
    assert {tuple(getattr(config, field) for field in plan.parameters) for config in plan.grid} == expected_coordinates, "Incomplete corner/axis grid."
    center = next(config for config in plan.grid if config.name == plan.center_name)
    assert tuple(getattr(center, field) for field in plan.parameters) == middle, "Declared center is not the middle setting."
    expected = {(config.name, seed) for config in plan.grid if config.name != plan.center_name for seed in plan.seeds}
    actual = {(fit.sample_fit.name, fit.seed) for path in plan.worker_plans for fit in load_worker_plan(path).fits}
    assert actual == expected, "Worker plans do not cover the declared non-center grid."
    assert all(load_worker_plan(path).reference_root == plan.reference_root for path in plan.worker_plans)
    assert all(load_worker_plan(path).methods == ("S",) and load_worker_plan(path).seeds == plan.seeds
               for path in plan.worker_plans), "Grid workers must measure raw S with the matched seeds."
    catalog = {config.name: config for config in plan.grid}
    assert all(fit.sample_fit == catalog[fit.sample_fit.name] for path in plan.worker_plans for fit in load_worker_plan(path).fits)
    return plan


def read_runs(plan: GridPlan, *, complete: bool) -> dict[str, list[dict[str, Any]]]:
    """Read controls and grid results, preserving every completed seed without performance filtering."""
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in ("O", "O_tau025", *(config.name for config in plan.grid))}
    reference_manifest = json.loads((plan.reference_root / "test_config.json").read_text())
    config_by_name = {config.name: config for config in plan.grid}
    center = config_by_name[plan.center_name]
    center_training = None
    runtime = None
    initial_hash = None
    for root in plan.control_roots:
        manifest = json.loads((root / "test_config.json").read_text())
        assert manifest["reference_manifest"] == reference_manifest
        if runtime is None:
            runtime = manifest["numerical_runtime"]
        assert manifest["numerical_runtime"] == runtime
        for path in sorted((root / "cases").glob("*/*/complete.json")):
            run = json.loads(path.read_text())
            if run["method"] not in {"O", "O_tau025", "S_tau025"}:
                continue
            key = plan.center_name if run["method"] == "S_tau025" else run["method"]
            if key == plan.center_name:
                assert all(run["training_config"][key] == value for key, value in center.model_dump(exclude={"name", "sample_size"}).items()), "Imported center optimizer differs."
                assert manifest["plan"]["sample_size"] == 3_000_000, "Imported center sample size differs."
                center_training = run["training_config"]
            groups[key].append(run)
            if initial_hash is None:
                initial_hash = run["initial_state_hash"]
            assert run["initial_state_hash"] == initial_hash
    assert center_training is not None
    for path in sorted((plan.output_root / "cases").glob("*/*/complete.json")):
        run = json.loads(path.read_text())
        assert run["method"] == "S"
        assert run["initial_state_hash"] == initial_hash and run["quantizer_si_columns"] == 150
        config = config_by_name[run["sample_fit"]["name"]]
        assert run["sample_fit"] == config.model_dump()
        assert all(run["training_config"][key] == value for key, value in config.model_dump(exclude={"name", "sample_size"}).items())
        varied = set(plan.parameters) - {"sample_size"}
        assert {key: value for key, value in run["training_config"].items() if key not in varied} == {
            key: value for key, value in center_training.items() if key not in varied}, "Non-grid quantizer settings changed."
        groups[config.name].append(run)
    for path in plan.worker_plans:
        worker = load_worker_plan(path)
        manifest_path = worker.output_root / "test_config.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            assert manifest["reference_manifest"] == reference_manifest and manifest["numerical_runtime"] == runtime
    for label, group in groups.items():
        group.sort(key=lambda run: run["fit_seed"])
        seeds = [run["fit_seed"] for run in group]
        assert len(set(seeds)) == len(seeds), f"Duplicate fit seeds: {label}"
        if complete or label in {"O", "O_tau025", plan.center_name}:
            assert seeds == list(plan.seeds), f"Incomplete matched seed coverage: {label} {seeds}"
        for run in group:
            assert run["record"]["symbol_error_rate"] == 0
            assert np.isclose(run["record"]["stage_wmapes"][-1], run["record"]["wmape"], rtol=1e-10)
    frozen = json.loads((plan.reference_root / "cases/F/target/complete.json").read_text())
    assert frozen["weights_unchanged"] and frozen["quantizer_fits"] == frozen["prior_fits"] == 0
    groups["F*"] = [dict(fit_seed=43, record=frozen["record"])]
    return groups


def valid_curve(point: dict[str, Any]) -> bool:
    """Identify stage curves that support unambiguous piecewise-linear RD ranking."""
    return bool(np.all(np.diff(point["wmape_db"]) < 0) and np.all(np.diff(point["server_si_rans_stage_rates"]) > 0))


def gap(row: dict[str, Any] | None, column: dict[str, Any] | None, field: str = "server_si_rans_stage_rates") -> float:
    """Preserve unavailable trimmed curves as missing comparison cells."""
    return curve_gap(row, column, field) if row is not None and column is not None else float("nan")


def validate_artifacts(plan: GridPlan, destination: Path) -> None:
    """Verify seed/count sampling, unchanged quantizers during prior fit and archived FL sources."""
    destination.mkdir(parents=True, exist_ok=True)
    import torch

    center_pools: dict[int, str] = {}
    for root in plan.control_roots:
        for path in (root / "cases").glob("S_tau025*/*/complete.json"):
            result = json.loads(path.read_text())
            weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
            center_pools[result["fit_seed"]] = content_hash(weights["sample_indices"])
    assert set(center_pools) == set(plan.seeds), "Missing imported center sample pools."
    expected_pools: dict[tuple[int, int], str] = {}
    for seed, count in product(plan.seeds, sorted({config.sample_size for config in plan.grid})):
        coordinate_seed = int(torch.randint(0, 2**31, (), generator=torch.Generator().manual_seed(seed)).item())
        expected = torch.from_numpy(np.sort(np.random.default_rng(coordinate_seed).choice(11_181_642, count, replace=False)))
        expected_pools[seed, count] = content_hash(expected)
        if count == 3_000_000:
            assert expected_pools[seed, count] == center_pools[seed], "Center sample coordinates differ from the seeded draw."
    weight_checks: list[dict[str, Any]] = []
    for path in sorted((plan.output_root / "cases").glob("*/*/complete.json")):
        result = json.loads(path.read_text())
        weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
        count = result["sample_fit"]["sample_size"]
        assert content_hash(weights["sample_indices"]) == expected_pools[result["fit_seed"], count], f"Seed/count sample coordinates differ: {path}"
        assert all(torch.equal(value, weights["native"][key]) for key, value in weights["joint"].items()
                   if not key.startswith("conditional")), f"Prior fitting changed quantizer: {path}"
        weight_checks.append(dict(config=result["sample_fit"]["name"], seed=result["fit_seed"], sample_size=count,
                                 joint_hash=content_hash(weights["joint"]), native_hash=content_hash(weights["native"])))
    assert len(weight_checks) == len(plan.cases), "Incomplete saved grid weights."
    source_hashes = {}
    for root in (*plan.control_roots, *(load_worker_plan(path).output_root for path in plan.worker_plans)):
        with tarfile.open(root / "sources.tar.gz") as archive:
            sources = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.name.startswith("FL_code/")}
        source_hashes[str(root)] = content_hash(sources)
    assert len(set(source_hashes.values())) == 1, "Numerical FL sources changed across grid workers/controls."
    (destination / "validation.json").write_text(json.dumps(dict(weights=weight_checks, fl_sources=source_hashes,
        sample_pools_match_seed_and_count=True, quantizers_unchanged_in_prior_fit=True), indent=2))


def publish(plan: GridPlan, destination: Path, *, complete: bool) -> None:
    """Publish matched partial heatmaps or all completed raw measurements, directions and interactions."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    groups = read_runs(plan, complete=complete)
    if complete:
        (destination / "raw_records.json").write_text(json.dumps(groups, indent=2))
    labels = {config.name: f"S{index:02}" for index, config in enumerate(plan.grid, 1)}
    controls = {label: mean_points([run["record"] for run in groups[label]]) for label in ("O", "O_tau025", "F*")}
    all_points = {labels[name]: mean_points([run["record"] for run in group]) for name, group in groups.items() if name in labels and group}
    levels = {field: sorted({getattr(config, field) for config in plan.grid}) for field in plan.parameters}
    center = next(config for config in plan.grid if config.name == plan.center_name)
    values: dict[str, float] = {}
    for config in plan.grid:
        group = groups[config.name]
        if group:
            matched = {run["fit_seed"] for run in group}
            control = mean_points([run["record"] for run in groups["O"] if run["fit_seed"] in matched])
            values[config.name] = gap(all_points[labels[config.name]], control)
    limit = max([abs(value) for value in values.values() if np.isfinite(value)] or [1.])
    count_levels = [levels["sample_size"][0], levels["sample_size"][-1]]
    batch_levels = [levels["train_batch_size"][0], levels["train_batch_size"][-1]]
    epoch_levels = [levels["train_epochs"][0], levels["train_epochs"][-1]]
    columns = list(product((levels["lr"][0], levels["lr"][-1]), (levels["tau"][0], levels["tau"][-1])))
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, (count, batch) in zip(axes.flat, product(count_levels, batch_levels), strict=True):
        data = np.full((2, 4), np.nan)
        for config in plan.grid:
            if config.sample_size != count or config.train_batch_size != batch or config.name not in values:
                continue
            row, column = epoch_levels.index(config.train_epochs), columns.index((config.lr, config.tau))
            data[row, column] = values[config.name]
            axis.text(column, row, f"{values[config.name]:+.2f}%\n{len(groups[config.name])}/5 seeds", ha="center", va="center", fontsize=9)
        artist = axis.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set(xticks=range(4), xticklabels=[f"LR {lr/center.lr:.1f}×\nτ {tau}" for lr, tau in columns],
                 yticks=range(2), yticklabels=epoch_levels, xlabel="LR / initial tau", ylabel="Epochs",
                 title=f"Sample count {count/1e6:.1f}M; batch {batch}")
    figure.colorbar(artist, ax=list(axes.flat), label="Full-SI RD gap vs matched O (%)")
    figure.suptitle("S combined directions: " + ("all-five means" if complete else "PARTIAL; paired available seeds"))
    figure.savefig(destination / "grid_all_seed_rd.png", dpi=180, bbox_inches="tight")
    figure.savefig(destination / "grid_all_seed_rd.pdf", bbox_inches="tight")
    plt.close(figure)
    coverage = {name: [run["fit_seed"] for run in group] for name, group in groups.items()}
    (destination / "coverage.json").write_text(json.dumps(dict(complete=complete, seeds=coverage), indent=2))
    if not complete:
        return
    individual = [(name, run["fit_seed"], mean_points([run["record"]])) for name, group in groups.items() for run in group]
    valid = [point for _name, _seed, point in individual if valid_curve(point)]
    low = max(min(point["wmape_db"]) for point in valid)
    high = min(max(point["wmape_db"]) for point in valid)
    selections = []
    best_points: dict[str, dict[str, Any] | None] = {}
    for name, group in groups.items():
        if name == "F*":
            best_points[name] = controls[name]
            continue
        scored = []
        for run in group:
            point = mean_points([run["record"]])
            score = mean_rate(point, low, high) if low < high and valid_curve(point) else None
            scored.append((run, score))
        kept = sorted([(run, score) for run, score in scored if score is not None], key=lambda pair: (pair[1], pair[0]["fit_seed"]))[:3]
        kept_seeds = [run["fit_seed"] for run, _score in kept] if len(kept) == 3 else []
        key = labels.get(name, name)
        best_points[key] = mean_points([run["record"] for run, _score in kept]) if kept_seeds else None
        selections.extend(dict(label=key, config=name, seed=run["fit_seed"], common_mean_bpf=score, retained=run["fit_seed"] in kept_seeds,
                               reason="ranked RD performance" if score is not None else "unavailable monotone/common-span ranking") for run, score in scored)
    all_points = controls | all_points
    best_points = {key: best_points[key] for key in all_points}
    cohorts = {"all_five": all_points, "best_three": best_points}
    common_rates = {name: (mean_rate(all_points[labels[name]], low, high)
                          if low < high and valid_curve(all_points[labels[name]])
                          and min(all_points[labels[name]]["wmape_db"]) <= low < high <= max(all_points[labels[name]]["wmape_db"])
                          else None) for name in labels}
    (destination / "aggregate_points.json").write_text(json.dumps(cohorts, indent=2))
    (destination / "selection.json").write_text(json.dumps(dict(common_distortion_db=[low, high] if low < high else None, runs=selections), indent=2))
    summary = []
    for config in plan.grid:
        label = labels[config.name]
        full, best = all_points[label], best_points[label]
        summary.append(dict(label=label, **config.model_dump(), lr_multiplier=config.lr / center.lr,
                            all_five_gap_vs_o=gap(full, all_points["O"]), best_three_gap_vs_o=gap(best, best_points["O"]),
                            paired_wins_vs_o=sum(gap(mean_points([run["record"]]),
                                mean_points([next(control["record"] for control in groups["O"] if control["fit_seed"] == run["fit_seed"])])) < 0
                                for run in groups[config.name]),
                            all_five_gap_vs_center=gap(full, all_points[labels[plan.center_name]]),
                            common_mean_full_si_bpf=common_rates[config.name], final_full_si_bpf=full["server_si_rans_stage_rates"][-1],
                            final_wmape_percent=full["stage_wmapes"][-1], retained_seeds=[item["seed"] for item in selections if item["label"] == label and item["retained"]]))
    directions, interactions = [], []
    fields = plan.parameters
    coordinates = {tuple(getattr(config, field) for field in fields): config.name for config in plan.grid}
    for changed in fields:
        fixed = [field for field in fields if field != changed]
        for values_fixed in product(*(levels[field] for field in fixed)):
            context = dict(zip(fixed, values_fixed, strict=True))
            names = [coordinates.get(tuple(value if field == changed else context[field] for field in fields)) for value in levels[changed]]
            scores = [common_rates[name] if name is not None else None for name in names]
            if sum(score is not None for score in scores) < 2:
                continue
            directions.append(dict(parameter=changed, **context, low_mean_bpf=scores[0], center_mean_bpf=scores[1], high_mean_bpf=scores[2],
                low_to_high_delta_bpf=scores[2]-scores[0] if scores[0] is not None and scores[2] is not None else None,
                low_to_center_delta_bpf=scores[1]-scores[0] if None not in scores[:2] else None,
                center_to_high_delta_bpf=scores[2]-scores[1] if None not in scores[1:] else None))
    for first, second in combinations(fields, 2):
        fixed_fields = [field for field in fields if field not in {first, second}]
        for fixed_values in product(*[(levels[field][0], levels[field][-1]) for field in fixed_fields]):
            context = dict(zip(fixed_fields, fixed_values, strict=True))
            corners = [common_rates[coordinates[tuple(({first: a, second: b} | context)[field] for field in fields)]]
                       for a, b in product((levels[first][0], levels[first][-1]), (levels[second][0], levels[second][-1]))]
            interactions.append(dict(first_parameter=first, second_parameter=second, **context,
                                     interaction_delta_bpf=corners[3]-corners[2]-corners[1]+corners[0] if None not in corners else None))
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, parameter in zip(list(axes.flat)[:len(fields)], fields, strict=True):
        for row in directions:
            if row["parameter"] != parameter:
                continue
            context = ", ".join(f"{field}={row[field]:.5g}" for field in fields if field != parameter)
            x = np.array(levels[parameter]) / center.lr if parameter == "lr" else levels[parameter]
            y = [row[key] if row[key] is not None else np.nan for key in ("low_mean_bpf", "center_mean_bpf", "high_mean_bpf")]
            available = np.isfinite(y)
            axis.plot(np.array(x)[available], np.array(y)[available], "o-", label=context)
        axis.set(xlabel="LR multiplier" if parameter == "lr" else parameter, ylabel="Common-span full-SI mean bits/float")
        axis.grid(alpha=.2)
        axis.legend(fontsize=6)
    axes.flat[-1].set_visible(False)
    figure.suptitle("Parameter directions at fixed other settings; all-five means")
    figure.tight_layout()
    figure.savefig(destination / "parameter_directions.png", dpi=180)
    figure.savefig(destination / "parameter_directions.pdf")
    plt.close(figure)
    diagnostics = []
    for name, seed, point in individual:
        flags = []
        if not valid_curve(point):
            flags.append("nonmonotone stage RD")
        if point["quantizer_prior_stage_rates"][-1] > point["marginal_rate"]:
            flags.append("joint prior exceeds final marginal")
        for stage, rate in enumerate(np.diff([0., *point["coder_stage_rates"]]), 1):
            if rate < .01 or rate > 1.05:
                flags.append(f"stage {stage} native rate growth {rate:.6f}")
        diagnostics.append(dict(config=name, seed=seed, flags=flags))
    (destination / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    matrices = [dict(cohort=cohort, rate_source=field, row=row, column=column, rate_change_percent=gap(point, reference, field))
                for cohort, points in cohorts.items() for field in ("server_si_rans_stage_rates", "coder_stage_rates")
                for row, point in points.items() for column, reference in points.items()]
    paired = [dict(config=config.name, seed=run["fit_seed"], rate_gap_vs_o=gap(mean_points([run["record"]]),
                   mean_points([next(control["record"] for control in groups["O"] if control["fit_seed"] == run["fit_seed"])])))
              for config in plan.grid for run in groups[config.name]]
    for filename, rows in (("summary.csv", summary), ("selection.csv", selections), ("direction_effects.csv", directions),
                           ("interaction_effects.csv", interactions), ("comparison_gaps.csv", matrices), ("paired_seed_gaps.csv", paired)):
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    stage_rows = []
    rate_fields = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates", "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    for cohort, points in cohorts.items():
        for label, point in points.items():
            for stage in range(3):
                stage_rows.append(dict(cohort=cohort, label=label, stage=stage + 1,
                    marginal=point["marginal_rate"] if point is not None and stage == 2 else None,
                    **{field: point[field][stage] if point is not None and point[field] is not None else None for field in rate_fields},
                    wmape_db=point["wmape_db"][stage] if point is not None else None,
                    wmape_percent=point["stage_wmapes"][stage] if point is not None else None))
    with (destination / "stage_points.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    lines = ["# Local S optimizer grid", "", "**Complete: 43 configurations × five fit seeds; 210 new fits and five retained center fits.**", "",
             "Raw-S diagnostic on the fixed SVHN client-0/round-5 gradient, seed-43 starting model and 150 decoder SI columns. RTX 3090/cuda:0; unchanged numerical policy.",
             "Sample pools vary with count and are identical across optimizer settings within each seed/count. Separate entropy-prior fitting stays at the ordinary 30 epochs/three accepted candidates.",
             "Quantizer epochs, LR, initial tau, sample count and batch size vary in a 32-corner + 10-axis + center design. Their interactions include the optimizer trajectory and the subsequent prior's continuing RNG stream.",
             "This is one static trained-round diagnostic, not a trained-plus-nine-frozen population result.",
             "All-five means are used for parameter-direction and interaction measurements; best-three means are reported alongside them.",
             "Rates are averaged cumulatively and WMAPE in raw percent before dB conversion and RD comparison. Percent gaps are never averaged.",
             "Best-three selection ranks each method independently on the one common span across valid individual curves; unrankable/nonmonotone curves remain explicitly missing.",
             f"Selection range in WMAPE dB: {[low, high] if low < high else '— (no common span; best-three results unavailable)'}. Invalid individual curves: {len(individual) - len(valid)}.",
             "No client exclusions: 1/1 retained; with/without symmetric whole-window client exclusions is identical. Performance-based seed trimming is separate.", "",
             f"![All-five RD grid]({os.path.relpath(destination / 'grid_all_seed_rd.png', plan.report_path.parent)})", "",
             f"![Parameter directions]({os.path.relpath(destination / 'parameter_directions.png', plan.report_path.parent)})", "",
             "| ID | Samples | Batch | Epochs | LR multiplier | Initial tau | All-five RD gap vs O % | Best-three RD gap vs O % | All-five gap vs center % | All-five final full-SI bpf | All-five final raw WMAPE % | Retained seeds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    def cell(value: float | None) -> str:
        """Render unavailable numerical cells without inferred values."""
        return f"{value:.6f}" if value is not None and np.isfinite(value) else "—"

    for row in summary:
        lines.append(f"| {row['label']} | {row['sample_size']} | {row['train_batch_size']} | {row['train_epochs']} | {row['lr_multiplier']:.1f} | {row['tau']} | " +
                     " | ".join(cell(row[field]) for field in ("all_five_gap_vs_o", "best_three_gap_vs_o", "all_five_gap_vs_center", "final_full_si_bpf", "final_wmape_percent")) + f" | {row['retained_seeds']} |")
    ranking = sorted((row for row in summary if np.isfinite(row["all_five_gap_vs_o"])), key=lambda row: row["all_five_gap_vs_o"])
    lines.extend(["", "Lowest measured all-five full-SI gaps versus O (exploratory selection from the grid):", ""])
    for row in ranking[:5]:
        lines.append(f"- {row['label']}: {row['sample_size']/1e6:.1f}M samples, batch {row['train_batch_size']}, {row['train_epochs']} epochs, LR {row['lr_multiplier']:.1f}×, tau {row['tau']}; gap {row['all_five_gap_vs_o']:+.6f}%; same-seed wins {row['paired_wins_vs_o']}/5.")
    lines.extend(["", "Directional effects and interactions are in `direction_effects.csv` and `interaction_effects.csv`.",
                  "They use absolute mean interpolated full-SI rates over the common span (bits/float), preserving the other parameter values.",
                  "Center-axis changes measure all three levels; combined corner contexts measure the two endpoints only. Intermediate corner settings are unmeasured.",
                  "Interaction delta = high/high − high/low − low/high + low/low. A negative delta improves more than the additive prediction; positive worsens more.", ""])
    for cohort, points in cohorts.items():
        lines.extend([f"## {cohort}: full square RD changes", "", "Row versus column (%); negative favors row; reverse entries recomputed. Before/after client exclusions (none) are identical.", ""])
        for field in ("server_si_rans_stage_rates", "coder_stage_rates"):
            lines.extend([field, "", "```text", "          " + " ".join(f"{key:>9}" for key in points)])
            for row, point in points.items():
                lines.append(f"{row:>9} " + " ".join(f"{cell(gap(point, reference, field)):>9}" for reference in points.values()))
            lines.extend(["```", ""])
        lines.extend(["Stage rates: cumulative bits/float, marginal total at final stage only. F* has no prior refit. — is unmeasured.", "",
                      "| Method | Stage | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB | Raw WMAPE % |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in stage_rows:
            if row["cohort"] == cohort:
                lines.append(f"| {row['label']} | {row['stage']} | " + " | ".join(cell(row[field]) for field in ("marginal", *rate_fields, "wmape_db", "wmape_percent")) + " |")
        lines.append("")
    lines.extend(["## Diagnostics", "", "All native symbol errors are zero. Per-stage marginal entropy was not measured; symbol-stage death is not inferred.",
                  *[f"- {item['config']} seed {item['seed']}: {', '.join(item['flags'])}." for item in diagnostics if item["flags"]],
                  "" if any(item["flags"] for item in diagnostics) else "No stage monotonicity, native rate growth or prior/marginal failures flagged.", "",
                  f"Raw case artifacts: `{plan.output_root}`. Full analysis CSV/JSON/PNG/PDF: `{destination}`.", ""])
    text = "\n".join(lines)
    (destination / "report.md").write_text(text)
    with atomic_output(plan.report_path) as temporary:
        temporary.write_text(text)
    (destination / "analysis_source.py").write_text(Path(__file__).read_text())
    print(f"Published completed S grid: {plan.report_path}", flush=True)


def run_workers(worker_plans: tuple[Path, ...], logs: Path) -> None:
    """Run declared comparison workers concurrently and fail if any child exits unsuccessfully."""
    logs.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        children = []
        for index, path in enumerate(worker_plans, 1):
            log = stack.enter_context((logs / f"part{index}.log").open("ab", buffering=0))
            child = subprocess.Popen((sys.executable, "-u", "-m", "Experiments.Protocol_test.starting_model_comparison", str(path)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append((path, child))
            print(f"STARTED {path.name}: pid={child.pid}", flush=True)
        pending = {os.pidfd_open(child.pid): (path, child) for path, child in children}
        for descriptor in pending:
            stack.callback(os.close, descriptor)
        while pending:
            ready, _writes, _errors = select.select(list(pending), [], [])
            for descriptor in ready:
                path, child = pending.pop(descriptor)
                result = child.wait()
                assert result == 0, f"Experiment worker failed: {path}, exit={result}; inspect {logs}"
                print(f"FINISHED {path.name}", flush=True)


def run(plan: GridPlan) -> None:
    """Run three memory-bounded child workers and generate the final report without chat supervision."""
    plan.output_root.mkdir(parents=True, exist_ok=True)
    cases_root = plan.output_root / "cases"
    cases_root.mkdir(exist_ok=True)
    with atomic_output(plan.output_root / "grid_plan.json") as temporary:
        temporary.write_text(plan.model_dump_json(indent=2))
    with atomic_output(plan.output_root / "test_config.json") as temporary:
        temporary.write_text(json.dumps(dict(study=plan.model_dump(mode="json"), scheduled_cases=len(plan.cases), reused_center_seeds=plan.seeds), indent=2))
    workers = [load_worker_plan(path) for path in plan.worker_plans]
    for worker in workers:
        for label in worker.cases:
            link = cases_root / label
            if not link.is_symlink():
                link.symlink_to(worker.output_root / "cases" / label, target_is_directory=True)
    run_workers(plan.worker_plans, plan.output_root / "workers")
    validate_artifacts(plan, plan.output_root / "final_analysis")
    publish(plan, plan.output_root / "final_analysis", complete=True)


def main() -> None:
    """Provide experiment, disposable live-plot and explicit completed-report entry points."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.runner_pid is not None:
        assert args.control_dir is not None
        watch(args.path, args.runner_pid, args.control_dir, plot_module=__spec__.name,
              source_paths=(Path(__file__), Path(__file__).with_name("five_seed_tables.py")))
    elif args.out is not None:
        publish(GridPlan.model_validate_json((args.path / "grid_plan.json").read_text()), args.out, complete=False)
    else:
        plan = load_plan(args.path)
        if args.report:
            validate_artifacts(plan, plan.output_root / "final_analysis")
            publish(plan, plan.output_root / "final_analysis", complete=True)
        else:
            run(plan)


if __name__ == "__main__":
    main()
