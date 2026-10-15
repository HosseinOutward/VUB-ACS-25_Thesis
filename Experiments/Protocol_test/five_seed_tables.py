"""Publish five-seed means and explicitly ranked best-three RD measurements."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Experiments.Protocol_test.rd_metrics import wmape_to_db
from Experiments.Protocol_test.starting_model_tables import curve_gap, matrix_text
from FL_code.FL_core.explicit_cache import content_hash


def mean_points(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average cumulative rates and raw WMAPE before dB conversion or RD comparison."""
    points = {key: (np.mean([np.cumsum(record[key]) for record in records], axis=0).tolist()
                    if all(record.get(key) is not None for record in records) else None)
              for key in ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
                          "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")}
    points["stage_wmapes"] = np.mean([record["stage_wmapes"] for record in records], axis=0).tolist()
    points["wmape_db"] = wmape_to_db(np.array(points["stage_wmapes"])).tolist()
    points["marginal_rate"] = float(np.mean([record["marginal_rate"] for record in records]))
    points["observations"] = len(records)
    return points


def mean_rate(point: dict[str, Any], low: float, high: float) -> float:
    """Integrate piecewise-linear full-SI rate over the fixed common distortion interval."""
    x, y = np.array(point["wmape_db"])[::-1], np.array(point["server_si_rans_stage_rates"])[::-1]
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) < 0), "Invalid stage curve for seed ranking."
    assert x[0] <= low < high <= x[-1], "Seed ranking would extrapolate."
    grid = np.unique(np.r_[low, x[(x > low) & (x < high)], high])
    return float(np.trapezoid(np.interp(grid, x, y), grid) / (high - low))


def publish(reference_root: Path, roots: tuple[Path, ...], destination: Path) -> None:
    """Validate all matched fits and publish per-codec best-three and all-five raw metrics."""
    manifests = [json.loads((root / "test_config.json").read_text()) for root in roots]
    for key in ("baseline_state_hash", "reference_manifest", "numerical_runtime"):
        assert all(manifest[key] == manifests[0][key] for manifest in manifests), f"Unmatched {key}."
    assert all(Path(manifest["plan"]["reference_root"]).resolve() == reference_root.resolve() for manifest in manifests)
    paths = [path for root in roots for path in sorted((root / "cases").glob("*/*/complete.json"))]
    runs = [json.loads(path.read_text()) for path in paths]
    grouped = {method: sorted((run for run in runs if run["method"] == method), key=lambda run: run["fit_seed"])
               for method in ("O", "O_tau025", "S_tau025", "T")}
    assert all([run["fit_seed"] for run in group] == [43, 44, 45, 46, 47] for group in grouped.values()), "Incomplete five-seed coverage."
    assert all(len({json.dumps(run["training_config"], sort_keys=True) for run in group}) == 1
               for group in grouped.values()), "Optimizer changed across fit seeds."
    assert grouped["S_tau025"][0]["training_config"] == grouped["O_tau025"][0]["training_config"], "Unmatched S/O_tau025 optimizers."
    expected_identities = {manifest["identity"] for manifest in manifests}
    assert all(run["identity"] in expected_identities for run in runs)
    baseline = torch.load(manifests[0]["baseline_artifact"], map_location="cpu", weights_only=True)["state"]
    assert content_hash(baseline) == manifests[0]["baseline_state_hash"], "Starting weights changed."
    diagnostics = []
    for path, run in zip(paths, runs, strict=True):
        weights = torch.load(path.with_name("weights.pt"), map_location="cpu", weights_only=True)
        assert all(torch.equal(value, weights["native"][key]) for key, value in weights["joint"].items()
                   if not key.startswith("conditional")), f"Prior fitting changed the quantizer: {path}"
        state = {key: baseline[key] for key in weights["joint"]}
        if run["method"] == "T":
            state["decoder.input_layers.0.weight"] = torch.cat((state["decoder.input_layers.0.weight"][:, :2], state["decoder.input_layers.0.weight"][:, 2:7]), dim=1)
            state["conditionalRNN.input_layers.0.weight"] = torch.cat((state["conditionalRNN.input_layers.0.weight"][:, :5], state["conditionalRNN.input_layers.0.weight"][:, -2:]), dim=1)
        assert run["initial_state_hash"] == content_hash(state), f"Starting weights differ: {path}"
        assert run["quantizer_si_columns"] == (5 if run["method"] == "T" else 150)
        record = run["record"]
        assert record["symbol_error_rate"] == 0.0, f"Symbol errors: {path}"
        flags = []
        if np.any(np.diff(record["stage_wmapes"]) >= 0):
            flags.append("nonmonotone distortion")
        if record["quantizer_prior_rate"] > record["marginal_rate"]:
            flags.append("joint prior exceeds marginal")
        for stage, rate in enumerate(record["coder_stage_rates"], 1):
            if rate < .01 or rate > 1.05:
                flags.append(f"stage {stage} native stream growth {rate:.6f} bpf")
        for field in ("temporal_si_prior", "server_si_prior"):
            if record.get(f"{field}_stage_rates") is not None:
                assert np.isclose(sum(record[f"{field}_stage_rates"]), record[f"{field}_rate"], rtol=2e-6)
        diagnostics.append(dict(method=run["method"], seed=run["fit_seed"], flags=flags))
    assert len({run["initial_state_hash"] for run in grouped["T"]}) == 1, "T starting projection changed."
    frozen = json.loads((reference_root / "cases/F/target/complete.json").read_text())
    assert frozen["weights_unchanged"] and frozen["quantizer_fits"] == frozen["prior_fits"] == 0
    ordinary_path = next(path for path, run in zip(paths, runs, strict=True) if run["method"] == "O")
    deployed = torch.load(ordinary_path.with_name("weights.pt"), map_location="cpu", weights_only=True)["joint"]
    assert content_hash(OrderedDict((key, baseline[key]) for key in deployed)) == frozen["frozen_state_hash"]
    individual = {(run["method"], run["fit_seed"]): mean_points([run["record"]]) for run in runs}
    frozen_point = mean_points([frozen["record"]])
    all_curves = [*individual.values(), frozen_point]
    low = max(min(point["wmape_db"]) for point in all_curves)
    high = min(max(point["wmape_db"]) for point in all_curves)
    assert low < high, "No common distortion range for seed ranking."
    baseline_rate = mean_rate(frozen_point, low, high)
    scores = {key: mean_rate(point, low, high) for key, point in individual.items()}
    retained = {method: sorted(group, key=lambda run: (scores[(method, run["fit_seed"])], run["fit_seed"]))[:3]
                for method, group in grouped.items()}
    selections = [dict(method=method, seed=run["fit_seed"], common_mean_full_si_bpf=scores[(method, run["fit_seed"])],
                       common_gap_vs_f_percent=100 * (scores[(method, run["fit_seed"])] / baseline_rate - 1),
                       retained=run in retained[method]) for method, group in grouped.items() for run in group]
    cohorts = {label: {method: mean_points([run["record"] for run in group]) for method, group in population.items()}
               for label, population in (("best_three", retained), ("all_five", grouped))}
    cohorts = {label: {"O": points["O"], "O_tau025": points["O_tau025"], "F*": frozen_point,
                       "S_tau025": points["S_tau025"], "T": points["T"]} for label, points in cohorts.items()}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "mean_stage_points.json").write_text(json.dumps(cohorts, indent=2))
    (destination / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (destination / "selection.json").write_text(json.dumps(dict(common_distortion_db=[low, high],
        common_distortion_raw_percent=(100 * 10 ** (np.array([low, high]) / 20)).tolist(),
        ranking="lower mean full-SI rANS over common interval; ties by seed; separate selection per codec",
        scores=selections), indent=2))
    with (destination / "selection.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selections[0]))
        writer.writeheader()
        writer.writerows(selections)
    lines = ["## Five-seed extension: best-three and all-five means (8 October 2026)", "",
             "Fit seeds **43–47**; original completed seeds 43–45 retained, seeds 46–47 newly fitted.",
             "Starting weights, SVHN client-0 round-5 target, all 150 SI columns and numerical policy stay fixed at baseline seed 43.",
             "T uses its five temporal columns for its quantizer/native coder and all 150 for the separately refitted comparison prior.",
             "O_tau025 retains the matched S-optimizer control. S_tau025 is the privileged raw-label diagnostic; practical S still acquires reconstructed labels.",
             "This measures fit/sample randomness on one fixed gradient, not FL-trace or starting-weight seed robustness.", "",
             f"Rank each codec's runs by mean full-SI rANS at equal distortion over the **same** interval [{low:.6f}, {high:.6f}] WMAPE dB.",
             "Lower is better. Equivalently rank the RD gap against fixed F* on that interval. Discard the two largest scores independently per codec; ties by seed.",
             "These are intentionally selected best-three means; the untrimmed five-seed means and all paired seeds remain visible.",
             "Average cumulative stage rates and raw WMAPE first, then convert WMAPE to dB and compare aggregate curves.",
             "Do not average per-seed gap percentages. Each matrix uses the pair's complete shared distortion span; the fixed common span applies only to selection.", "",
             "| Codec | Seed | Common-span full-SI mean bpf | Common-span gap vs F* % | Included in best three |",
             "|---|---:|---:|---:|---|"]
    for item in selections:
        lines.append(f"| {item['method']} | {item['seed']} | {item['common_mean_full_si_bpf']:.9f} | {item['common_gap_vs_f_percent']:+.6f} | {'yes' if item['retained'] else 'discarded'} |")
    stage_rows, gap_rows = [], []
    fields = ("quantizer_prior_stage_rates", "temporal_si_prior_stage_rates", "server_si_prior_stage_rates",
              "coder_stage_rates", "temporal_si_rans_stage_rates", "server_si_rans_stage_rates")
    for cohort, points in cohorts.items():
        lines.extend(["", f"### {'Best three of five' if cohort == 'best_three' else 'All five, untrimmed'}: mean stage RD", "",
                      "Cumulative bits/float. Marginal is a final-stage scalar total; — is unmeasured. F* has no prior refit.", "",
                      "| Method | Stage | Marginal | Joint/pre-fit prior | Temporal prior | Full-SI fitted prior | Native rANS | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
        for method, point in points.items():
            for stage in range(3):
                values = [point["marginal_rate"] if stage == 2 else None,
                          *(point[field][stage] if point[field] is not None else None for field in fields)]
                cells = ["—" if value is None else f"{value:.6f}" for value in values]
                lines.append(f"| {method} | {stage+1} | " + " | ".join(cells) +
                             f" | {point['wmape_db'][stage]:.6f} ({point['stage_wmapes'][stage]:.6f}%) |")
                stage_rows.append([cohort, method, stage+1, *values, point["wmape_db"][stage], point["stage_wmapes"][stage]])
        lines.extend(["", "Equal-distortion rate changes (%), row versus column; negative favors row; reverse entries recomputed.",
                      "Client exclusions: none; with/without symmetric whole-window client exclusions is identical, 1/1 client retained.",
                      f"Fit coverage: {'3/5 separately selected' if cohort == 'best_three' else '5/5'} per codec; fixed F* (one model). One static round, no frozen evaluation window.", ""])
        for field, title in (("server_si_rans_stage_rates", "Full-SI rANS (primary)"), ("coder_stage_rates", "Native transport (T uses temporal SI)")):
            lines.extend([title + ":", "", "```text", matrix_text(points, field), "```", ""])
            gap_rows.extend(dict(cohort=cohort, rate_source=field, row=row, column=column,
                                 rate_change_percent=curve_gap(point, reference, field))
                            for row, point in points.items() for column, reference in points.items())
    lines.extend(["### Same-seed comparisons, all five", "",
                  "Pairwise full-SI curve gaps remain diagnostic; the aggregate result comes from averaged curves.", "",
                  "| Candidate | Reference | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Seed 47 | Wins | Best-three mean-curve gap % | All-five mean-curve gap % |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    paired_rows = []
    for candidate, reference in (("O_tau025", "O"), ("S_tau025", "O"), ("S_tau025", "O_tau025"), ("T", "O")):
        gaps = [curve_gap(individual[(candidate, seed)], individual[(reference, seed)], "server_si_rans_stage_rates") for seed in range(43,48)]
        paired_rows.extend(dict(candidate=candidate, reference=reference, seed=seed, rate_change_percent=gap)
                           for seed, gap in zip(range(43,48), gaps, strict=True))
        aggregates = [curve_gap(points[candidate], points[reference], "server_si_rans_stage_rates") for points in cohorts.values()]
        lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.6f}" for value in gaps) +
                     f" | {sum(value < 0 for value in gaps)}/5 | " + " | ".join(f"{value:+.6f}" for value in aggregates) + " |")
    seed_cohorts = {label: {method: mean_points([run["record"] for run in group if run["fit_seed"] in seeds])
                           for method, group in grouped.items()}
                    for label, seeds in (("original_three", (43, 44, 45)), ("added_two", (46, 47)))}
    (destination / "seed_cohort_means.json").write_text(json.dumps(seed_cohorts, indent=2))
    lines.extend(["", "### Added-seed diagnostic with the same averaging rule", "",
                  "This separates the additional seeds from the switch from historical medians to arithmetic means.",
                  "Each cell compares curves built from mean cumulative rates and mean raw WMAPE; it is not a mean of individual gaps.", "",
                  "| Candidate | Reference | Original seeds 43–45 mean-curve gap % | Added seeds 46–47 mean-curve gap % |",
                  "|---|---|---:|---:|"])
    for candidate, reference in (("S_tau025", "O"), ("S_tau025", "O_tau025"), ("T", "O")):
        gaps = [curve_gap(points[candidate], points[reference], "server_si_rans_stage_rates") for points in seed_cohorts.values()]
        lines.append(f"| {candidate} | {reference} | " + " | ".join(f"{value:+.6f}" for value in gaps) + " |")
    lines.extend(["", "### Individual run stage measurements", "",
                  "Every retained and discarded seed is included. Cumulative bpf; marginal final-stage only.", "",
                  "| Codec | Seed | Stage | Marginal | Joint prior | Temporal prior | Full-SI prior | Native rANS | Temporal rANS | Full-SI rANS | Raw WMAPE % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for (method, seed), point in sorted(individual.items()):
        for stage in range(3):
            values = [point["marginal_rate"] if stage == 2 else None,
                      *(point[field][stage] if point[field] is not None else None for field in fields), point["stage_wmapes"][stage]]
            lines.append(f"| {method} | {seed} | {stage+1} | " + " | ".join("—" if value is None else f"{value:.9f}" for value in values) + " |")
    lines.extend(["", "Stage diagnostics:", "", *[f"- {item['method']} seed {item['seed']}: {', '.join(item['flags']) or 'monotone stages; no flagged rate growth/prior failure'}." for item in diagnostics],
                  "", "All native symbol errors are zero. Per-stage marginals were not measured; symbol-stage death is not inferred.", ""])
    (destination / "tables.md").write_text("\n".join(lines))
    with (destination / "mean_stage_points.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cohort", "method", "stage", "marginal", "joint_prior", "temporal_prior", "full_si_prior", "native_rans", "temporal_rans", "full_si_rans", "wmape_db", "wmape_percent"])
        writer.writerows(stage_rows)
    with (destination / "comparison_gaps.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gap_rows[0]))
        writer.writeheader()
        writer.writerows(gap_rows)
    with (destination / "paired_seed_gaps.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (cohort, points) in zip(axes, cohorts.items(), strict=True):
        for method, point in points.items():
            axis.plot(point["server_si_rans_stage_rates"], point["wmape_db"], "o-", label=method)
        axis.set(xlabel="Cumulative full-SI rANS bits/float", title=cohort.replace("_", " "))
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel("WMAPE (dB)")
    figure.tight_layout()
    figure.savefig(destination / "five_seed_rd.png", dpi=180)
    figure.savefig(destination / "five_seed_rd.pdf")
    plt.close(figure)
    (destination / "analysis_source.py").write_text(Path(__file__).read_text())
    print(f"Published five-seed tables: {destination}")


def main() -> None:
    """Build all five-seed tables from retained F and matched completed fit outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    publish(args.reference_root, tuple(args.roots), args.out)


if __name__ == "__main__":
    main()
