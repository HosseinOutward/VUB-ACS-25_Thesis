"""Audit received-sample distortion against the held-out gradient and baseline fit target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from FL_code.FL_core.explicit_cache import content_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "_" / "experiments" / "s_received_mae_five_seed_20261012"
DEFAULT_VECTORS = ROOT / "_" / "experiments" / "s_distortion_audit_20261014" / "vector" / "source_vectors.pt"
DEFAULT_OUTPUT = ROOT / "_" / "experiments" / "s_distortion_audit_20261014" / "sample"
DEFAULT_SLICE_MAP = ROOT / "_" / "experiments" / "s_distortion_audit_20261014" / "vector" / "slice_parameter_map.csv"
SEEDS = (43, 44, 45, 46, 47)
TAIL_FRACTIONS = (0.0001, 0.001, 0.01, 0.05, 0.10)
RAW_MAE_EXPERIMENT_43 = ROOT / "_" / "experiments" / "s_received_raw_mae_acq_ld05_seed43_20261013"
RAW_MAE_EXPERIMENT_44_47 = ROOT / "_" / "experiments" / "s_received_raw_mae_acq_ld05_confirm_seeds44_47_20261013"


def read_vectors(path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[slice, ...], dict[str, Any]]:
    """Load the CPU-only held-out and baseline vectors saved by the shared audit extraction."""
    vectors: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    target, baseline = vectors["target"].flatten(), vectors["baseline"].flatten()
    slices = tuple(vectors["slices"])
    assert target.shape == baseline.shape and target.dtype == baseline.dtype == torch.float32
    source_manifest = path.with_name("source_manifest.json")
    provenance = json.loads(source_manifest.read_text())
    assert content_hash(target) == provenance["target_content_hash"]
    assert content_hash(baseline) == provenance["baseline_content_hash"]
    assert hashlib.sha256(target.numpy().tobytes()).hexdigest() == provenance["target_sha256"]
    assert hashlib.sha256(baseline.numpy().tobytes()).hexdigest() == provenance["baseline_sha256"]
    return target, baseline, slices, provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a sequence of flat diagnostic rows with stable columns."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def audit_seed(seed: int, experiment: Path, target: torch.Tensor, baseline: torch.Tensor,
               slices: tuple[slice, ...]
               ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
                          list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure received reconstruction errors and their raw-L1 numerator contributions for one seed."""
    case = experiment / "cases" / f"current_e1_mae_received_prefix_bins_seed{seed}" / "target"
    complete = json.loads((case / "complete.json").read_text())
    weights: dict[str, torch.Tensor] = torch.load(case / "weights.pt", map_location="cpu", weights_only=True)
    indices = weights["sample_indices"].long()
    decoded = weights["sample_reconstruction"].float()
    assert indices.numel() == 3_600_000 and decoded.numel() == indices.numel()
    assert indices.min() >= 0 and indices.max() < target.numel() and torch.all(indices[1:] > indices[:-1])
    assert complete["fit_seed"] == seed and complete["sample_indices_hash"] == content_hash(indices)
    assert complete["sample_acquisition_distortion"] == "mae"
    assert complete["record"]["sample_fit_mode"] == "received_prefix_bins"
    assert not complete["record"]["sample_bootstrap"]
    raw_experiment = RAW_MAE_EXPERIMENT_43 if seed == 43 else RAW_MAE_EXPERIMENT_44_47
    raw_case = raw_experiment / "cases" / f"current_e1_raw_mae_received_prefix_bins_seed{seed}" / "target"
    raw_complete = json.loads((raw_case / "complete.json").read_text())
    raw_weights: dict[str, torch.Tensor] = torch.load(raw_case / "weights.pt", map_location="cpu", weights_only=True)
    raw_indices = raw_weights["sample_indices"].long()
    raw_decoded = raw_weights["sample_reconstruction"].float()
    assert raw_complete["sample_acquisition_distortion"] == "raw_mae"
    assert raw_complete["sample_indices_hash"] == content_hash(raw_indices) == complete["sample_indices_hash"]
    assert torch.equal(raw_indices, indices), f"Sample coordinates differ in seed{seed} objective comparison."
    y = target[indices].numpy().astype(np.float64)
    x = decoded.numpy().astype(np.float64)
    raw_x = raw_decoded.numpy().astype(np.float64)
    b = baseline[indices].numpy().astype(np.float64)
    coordinate_indices = indices.numpy()
    error = x - y
    abs_error, abs_target = np.abs(error), np.abs(y)
    raw_error, raw_abs_error = raw_x - y, np.abs(raw_x - y)
    denom, numerator = float(abs_target.sum()), float(abs_error.sum())
    wmape = 100 * numerator / denom
    recorded_wmape = float(complete["record"]["sample_wmape"])
    assert np.isclose(wmape, recorded_wmape, rtol=1e-5, atol=1e-5), (
        f"seed{seed} reconstructed sample WMAPE {wmape} differs from saved {recorded_wmape}")
    raw_wmape = 100 * float(raw_abs_error.sum()) / float(abs_target.sum())
    assert np.isclose(raw_wmape, raw_complete["record"]["sample_wmape"], rtol=1e-5, atol=1e-5)

    zero_target = y == 0
    nonzero_target = ~zero_target
    zero_count = int(zero_target.sum())
    nonzero_abs = abs_target[nonzero_target]
    near_zero_cut = float(np.quantile(nonzero_abs, .20))
    near_zero_nonzero = nonzero_target & (abs_target <= near_zero_cut)
    true_sign_flip = (x * y) < 0
    zero_error = float(abs_error[zero_target].sum())
    summary = dict(seed=seed, sample_n=len(indices), vector_n=len(target), sample_fraction=len(indices) / len(target),
                   raw_l1_numerator=numerator, heldout_raw_l1_denominator=denom, sample_wmape_pct=wmape,
                   raw_mae_lambda05_l1_numerator=float(raw_abs_error.sum()),
                   raw_mae_lambda05_sample_wmape_pct=raw_wmape,
                   raw_mae_lambda05_minus_normalized_mae_wmape_pp=raw_wmape - wmape,
                   sample_coder_rate=float(complete["record"]["sample_coder_rate"]),
                   raw_mae_lambda05_sample_coder_rate=float(raw_complete["record"]["sample_coder_rate"]),
                   sample_coder_rate_delta_raw_minus_normalized=float(
                       raw_complete["record"]["sample_coder_rate"] - complete["record"]["sample_coder_rate"]),
                   baseline_to_heldout_raw_l1=float(np.abs(b - y).sum()),
                   baseline_to_heldout_wmape_pct=100 * float(np.abs(b - y).sum()) / denom,
                   mean_abs_target=float(abs_target.mean()), mean_abs_baseline_shift=float(np.abs(b - y).mean()),
                   signed_error_sum=float(error.sum()), mean_signed_error=float(error.mean()),
                   signed_error_over_heldout_l1=float(error.sum()) / denom,
                   exact_zero_target_count=zero_count, exact_zero_target_pct=100 * zero_count / len(y),
                   exact_zero_decoded_nonzero_pct=100 * float(np.mean(x[zero_target] != 0)),
                   exact_zero_decoded_unique_levels=int(np.unique(x[zero_target]).size),
                   exact_zero_error_l1=zero_error, exact_zero_error_share_pct=100 * zero_error / numerator,
                   exact_zero_mean_decoded=float(x[zero_target].mean()),
                   nonzero_to_zero_pct=100 * float(np.mean(x[nonzero_target] == 0)),
                   nonzero_true_sign_flip_count=int(true_sign_flip.sum()),
                   nonzero_true_sign_flip_pct=100 * float(true_sign_flip[nonzero_target].mean()),
                   near_zero_nonzero_threshold=near_zero_cut,
                   near_zero_nonzero_count=int(near_zero_nonzero.sum()),
                   near_zero_nonzero_error_share_pct=100 * float(abs_error[near_zero_nonzero].sum()) / numerator,
                   near_zero_nonzero_sign_flip_pct=100 * float(true_sign_flip[near_zero_nonzero].mean()),
                   abs_error_p50=float(np.quantile(abs_error, .50)), abs_error_p90=float(np.quantile(abs_error, .90)),
                   abs_error_p99=float(np.quantile(abs_error, .99)), abs_error_p999=float(np.quantile(abs_error, .999)),
                   abs_error_max=float(abs_error.max()), abs_target_error_corr=float(np.corrcoef(abs_target, abs_error)[0, 1]),
                   abs_baseline_shift_error_corr=float(np.corrcoef(np.abs(y - b), abs_error)[0, 1]),
                   positive_target_mean_error=float(error[y > 0].mean()),
                   negative_target_mean_error=float(error[y < 0].mean()),
                   positive_target_through_origin_slope=float(np.dot(y[y > 0], x[y > 0]) /
                                                              np.dot(y[y > 0], y[y > 0])),
                   negative_target_through_origin_slope=float(np.dot(y[y < 0], x[y < 0]) /
                                                              np.dot(y[y < 0], y[y < 0])),
                   saved_sample_wmape_pct=recorded_wmape)

    signed_rows = []
    for sign, selected in (("positive", y > 0), ("negative", y < 0)):
        target_l2 = float(np.dot(y[selected], y[selected]))
        signed_rows.append(dict(seed=seed, target_sign=sign, coordinate_count=int(selected.sum()),
                                target_l1=float(abs_target[selected].sum()), error_l1=float(abs_error[selected].sum()),
                                signed_error_sum=float(error[selected].sum()), mean_signed_error=float(error[selected].mean()),
                                target_l2=target_l2,
                                through_origin_decoded_slope=float(np.dot(y[selected], x[selected]) / target_l2),
                                true_sign_flips=int(((x * y) < 0)[selected].sum()),
                                raw_mae_lambda05_error_l1=float(raw_abs_error[selected].sum()),
                                raw_mae_lambda05_mean_signed_error=float(raw_error[selected].mean()),
                                raw_mae_lambda05_through_origin_slope=float(np.dot(y[selected], raw_x[selected]) /
                                                                            target_l2)))
    zero_selected = y == 0
    signed_rows.append(dict(seed=seed, target_sign="exact_zero", coordinate_count=int(zero_selected.sum()),
                            target_l1=0.0, error_l1=float(abs_error[zero_selected].sum()),
                            signed_error_sum=float(error[zero_selected].sum()),
                            mean_signed_error=float(error[zero_selected].mean()),
                            target_l2=0.0, through_origin_decoded_slope=float("nan"), true_sign_flips=0,
                            raw_mae_lambda05_error_l1=float(raw_abs_error[zero_selected].sum()),
                            raw_mae_lambda05_mean_signed_error=float(raw_error[zero_selected].mean()),
                            raw_mae_lambda05_through_origin_slope=float("nan")))

    tail_rows: list[dict[str, Any]] = []
    order = np.argsort(abs_error)
    for fraction in TAIL_FRACTIONS:
        count = max(1, int(np.ceil(len(order) * fraction)))
        selected = order[-count:]
        tail_rows.append(dict(seed=seed, tail_fraction=fraction, coordinate_count=count,
                              error_l1=float(abs_error[selected].sum()),
                              share_of_sample_error_pct=100 * float(abs_error[selected].sum()) / numerator,
                              heldout_l1_denominator=float(abs_target[selected].sum()),
                              share_of_sample_denominator_pct=100 * float(abs_target[selected].sum()) / denom,
                              mean_abs_error=float(abs_error[selected].mean()),
                              mean_abs_target=float(abs_target[selected].mean()),
                              mean_abs_baseline_shift=float(np.abs(y[selected] - b[selected]).mean())))

    bins = np.unique(np.quantile(nonzero_abs, np.linspace(0, 1, 6)))
    target_rows: list[dict[str, Any]] = []
    if len(bins) > 1:
        groups = np.minimum(np.searchsorted(bins[1:-1], nonzero_abs, side="right"), len(bins) - 2)
        for group in range(len(bins) - 1):
            group_selected = groups == group
            selected = np.zeros(len(y), dtype=bool)
            selected[np.flatnonzero(nonzero_target)[group_selected]] = True
            target_rows.append(dict(seed=seed, heldout_abs_magnitude_bin=group + 1,
                                    abs_target_low=float(bins[group]), abs_target_high=float(bins[group + 1]),
                                    coordinate_count=int(selected.sum()),
                                    share_of_sample_error_pct=100 * float(abs_error[selected].sum()) / numerator,
                                    share_of_sample_denominator_pct=100 * float(abs_target[selected].sum()) / denom,
                                    mean_abs_error=float(abs_error[selected].mean()),
                                    true_sign_flip_pct=100 * float(true_sign_flip[selected].mean()),
                                    abs_shift_error_corr=float(np.corrcoef(np.abs(y[selected] - b[selected]),
                                                                          abs_error[selected])[0, 1]),
                                    mean_target=float(y[selected].mean()), mean_decoded=float(x[selected].mean()),
                                    signed_error_over_target_l2=float(np.dot(y[selected], error[selected]) /
                                                                      np.dot(y[selected], y[selected])),
                                    wmape_pct=100 * float(abs_error[selected].sum()) / float(abs_target[selected].sum()),
                                    raw_mae_lambda05_error_l1=float(raw_abs_error[selected].sum()),
                                    raw_mae_lambda05_wmape_pct=100 * float(raw_abs_error[selected].sum()) /
                                                               float(abs_target[selected].sum()),
                                    raw_minus_normalized_error_l1=float(raw_abs_error[selected].sum() -
                                                                       abs_error[selected].sum())))
    slice_rows = []
    for slice_index, vector_slice in enumerate(slices):
        start, stop = vector_slice.start or 0, vector_slice.stop or len(target)
        left, right = np.searchsorted(coordinate_indices, (start, stop))
        if left == right:
            continue
        slice_error, slice_target = abs_error[left:right], abs_target[left:right]
        slice_num, slice_den = float(slice_error.sum()), float(slice_target.sum())
        slice_y = y[left:right]
        slice_x = x[left:right]
        slice_raw_error = raw_error[left:right]
        slice_rows.append(dict(seed=seed, norm_slice=slice_index, vector_start=start, vector_stop=stop,
                               vector_coordinate_count=stop - start, sample_coordinate_count=right - left,
                               sample_fraction=(right - left) / (stop - start), raw_l1_numerator=slice_num,
                               target_l1=slice_den, zero_target_count=int(np.count_nonzero(slice_y == 0)),
                               nonzero_true_sign_flips=int(np.count_nonzero((slice_y * slice_x) < 0)),
                               positive_target_l1=float(np.abs(slice_y[slice_y > 0]).sum()),
                               negative_target_l1=float(np.abs(slice_y[slice_y < 0]).sum()),
                               positive_signed_error_sum=float(error[left:right][slice_y > 0].sum()),
                               negative_signed_error_sum=float(error[left:right][slice_y < 0].sum()),
                               signed_error_sum=float(error[left:right].sum()),
                               target_l2=float(np.square(slice_y).sum()),
                               squared_error_sum=float(np.square(error[left:right]).sum()),
                               heldout_raw_l1_denominator=slice_den,
                               share_of_sample_error_pct=100 * slice_num / numerator,
                               share_of_sample_denominator_pct=100 * slice_den / denom,
                               wmape_pct=100 * slice_num / slice_den if slice_den else float("nan"),
                               mse=float(np.square(error[left:right]).mean()),
                               raw_mae_lambda05_error_l1=float(raw_abs_error[left:right].sum()),
                               raw_mae_lambda05_squared_error_sum=float(np.square(raw_error[left:right]).sum()),
                               raw_mae_lambda05_wmape_pct=100 * float(raw_abs_error[left:right].sum()) / slice_den
                               if slice_den else float("nan"),
                               raw_minus_normalized_error_l1=float(raw_abs_error[left:right].sum() - slice_num),
                               mean_signed_error=float(error[left:right].mean()),
                               raw_mae_lambda05_mean_signed_error=float(slice_raw_error.mean())))
    return summary, tail_rows, target_rows, slice_rows, signed_rows


def main() -> None:
    """Write per-seed sample distortion, tail concentration and magnitude-bin audits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--slice-map", type=Path, default=DEFAULT_SLICE_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    target, baseline, slices, provenance = read_vectors(args.vectors)
    reference = json.loads((args.experiment / "test_config.json").read_text())["reference_manifest"]
    assert content_hash(target) == reference["target_hash"]
    summaries, tails, magnitude_bins, norm_slices, signed = [], [], [], [], []
    for seed in SEEDS:
        summary, seed_tails, seed_bins, seed_slices, seed_signed = audit_seed(
            seed, args.experiment, target, baseline, slices)
        summaries.append(summary)
        tails.extend(seed_tails)
        magnitude_bins.extend(seed_bins)
        norm_slices.extend(seed_slices)
        signed.extend(seed_signed)
        print(f"seed{seed}: sample WMAPE={summary['sample_wmape_pct']:.4f}% "
              f"L1 tail top 0.1%={next(row['share_of_sample_error_pct'] for row in seed_tails if row['tail_fraction'] == .001):.2f}%")
    args.output.mkdir(parents=True, exist_ok=True)
    total_num = sum(row["raw_l1_numerator"] for row in summaries)
    total_den = sum(row["heldout_raw_l1_denominator"] for row in summaries)
    raw_total_num = sum(row["raw_mae_lambda05_l1_numerator"] for row in summaries)
    slice_map = {int(row["slice_index"]): row["parameter"] for row in csv.DictReader(args.slice_map.open())}
    assert len(slice_map) == len(slices)
    for row in norm_slices:
        row["parameter"] = slice_map[int(row["norm_slice"])]
        row["layer_group"] = row["parameter"].rsplit(".", 1)[0]
    layer_groups: dict[str, dict[str, Any]] = {}
    for row in norm_slices:
        group = layer_groups.setdefault(row["parameter"], dict(parameter=row["parameter"],
            vector_coordinate_count=0, sample_coordinate_count=0, target_l1=0.0, target_l2=0.0,
            error_l1=0.0, squared_error_sum=0.0, zero_target_count=0, nonzero_true_sign_flips=0,
            raw_mae_lambda05_error_l1=0.0, raw_mae_lambda05_squared_error_sum=0.0,
            positive_signed_error_sum=0.0, negative_signed_error_sum=0.0, signed_error_sum=0.0,
            seed_deltas={}))
        for key in ("vector_coordinate_count", "sample_coordinate_count", "target_l1", "target_l2",
                    "raw_l1_numerator", "squared_error_sum", "zero_target_count", "nonzero_true_sign_flips",
                    "raw_mae_lambda05_error_l1", "raw_mae_lambda05_squared_error_sum"):
            output_key = "error_l1" if key == "raw_l1_numerator" else key
            group[output_key] += row[key]
        for key in ("positive_signed_error_sum", "negative_signed_error_sum", "signed_error_sum"):
            group[key] += row[key]
        group["seed_deltas"][int(row["seed"])] = row["raw_minus_normalized_error_l1"]
    layer_rows = []
    for row in layer_groups.values():
        row["sample_fraction"] = row["sample_coordinate_count"] / row["vector_coordinate_count"]
        row["error_l1_share_pct"] = 100 * row["error_l1"] / total_num
        row["target_l1_share_pct"] = 100 * row["target_l1"] / total_den
        row["sample_wmape_pct"] = 100 * row["error_l1"] / row["target_l1"] if row["target_l1"] else float("nan")
        row["sample_mse"] = row["squared_error_sum"] / row["sample_coordinate_count"]
        row["raw_mae_lambda05_wmape_pct"] = (100 * row["raw_mae_lambda05_error_l1"] / row["target_l1"]
                                                if row["target_l1"] else float("nan"))
        row["raw_mae_lambda05_sample_mse"] = (row["raw_mae_lambda05_squared_error_sum"] /
                                               row["sample_coordinate_count"])
        row["raw_minus_normalized_error_l1"] = row["raw_mae_lambda05_error_l1"] - row["error_l1"]
        row["raw_error_l1_share_pct"] = 100 * row["raw_mae_lambda05_error_l1"] / raw_total_num
        row["true_sign_flip_pct_nonzero"] = 100 * row["nonzero_true_sign_flips"] / (
            row["sample_coordinate_count"] - row["zero_target_count"])
        row["raw_lower_error_seed_count"] = sum(delta < 0 for delta in row["seed_deltas"].values())
        row["mean_raw_minus_normalized_error_l1_per_seed"] = float(np.mean(list(row["seed_deltas"].values())))
        del row["seed_deltas"]
        layer_rows.append(row)
    write_csv(args.output / "sample_seed_summary.csv", summaries)
    write_csv(args.output / "sample_error_tails.csv", tails)
    write_csv(args.output / "sample_target_magnitude_bins.csv", magnitude_bins)
    write_csv(args.output / "sample_norm_slice_distortion.csv", norm_slices)
    write_csv(args.output / "sample_layer_distortion.csv", layer_rows)
    write_csv(args.output / "sample_signed_calibration.csv", signed)
    all_coords = sum(row["sample_n"] for row in summaries)
    zero_coords = sum(row["exact_zero_target_count"] for row in summaries)
    zero_error = sum(row["exact_zero_error_l1"] for row in summaries)
    nonzero_coords = all_coords - zero_coords
    true_flips = sum(row["nonzero_true_sign_flip_count"] for row in summaries)
    pooled_shift = sum(row["baseline_to_heldout_raw_l1"] for row in summaries) / total_den * 100
    mean_primary_sample_rate = float(np.mean([row["sample_coder_rate"] for row in summaries]))
    mean_raw_sample_rate = float(np.mean([row["raw_mae_lambda05_sample_coder_rate"] for row in summaries]))
    tail_mean = {fraction: float(np.mean([row["share_of_sample_error_pct"] for row in tails
                                          if row["tail_fraction"] == fraction]))
                 for fraction in TAIL_FRACTIONS}
    low_nonzero = [row for row in magnitude_bins if row["heldout_abs_magnitude_bin"] == 1]
    low_nonzero_error_share = float(np.mean([row["share_of_sample_error_pct"] for row in low_nonzero]))
    low_nonzero_den_share = float(np.mean([row["share_of_sample_denominator_pct"] for row in low_nonzero]))
    low_nonzero_flips = float(np.mean([row["true_sign_flip_pct"] for row in low_nonzero]))
    low_nonzero_wmape = float(np.mean([row["wmape_pct"] for row in low_nonzero]))
    sign_calibration = {sign: [row for row in signed if row["target_sign"] == sign]
                        for sign in ("positive", "negative")}
    sign_slopes = {sign: sum(row["target_l2"] * row["through_origin_decoded_slope"] for row in rows) /
                   sum(row["target_l2"] for row in rows) for sign, rows in sign_calibration.items()}
    sign_bias = {sign: sum(row["signed_error_sum"] for row in rows) /
                 sum(row["coordinate_count"] for row in rows) for sign, rows in sign_calibration.items()}
    shift_corr_by_bin = {index: float(np.mean([row["abs_shift_error_corr"] for row in magnitude_bins
                                               if row["heldout_abs_magnitude_bin"] == index]))
                         for index in sorted({row["heldout_abs_magnitude_bin"] for row in magnitude_bins})}
    top_layers = sorted(layer_rows, key=lambda row: row["error_l1"], reverse=True)[:3]
    top_layer_report = "; ".join(
        f"`{row['parameter']}` ({row['error_l1_share_pct']:.1f}% sample-error share; "
        f"raw λ=.5 changes sample L1 by {row['raw_minus_normalized_error_l1']:+.2f} in "
        f"{row['raw_lower_error_seed_count']}/5 seeds)" for row in top_layers)
    key_findings = [
        dict(metric="pooled_sample_wmape_pct", value=100 * total_num / total_den,
             definition="100 * pooled sum|decoded-heldout| / pooled sum|heldout|"),
        dict(metric="pooled_raw_mae_lambda05_sample_wmape_pct", value=100 * raw_total_num / total_den,
             definition="Matched raw-unit-MAE lambda=.5 contrast; same sample coordinates and heldout denominator"),
        dict(metric="raw_mae_lambda05_minus_normalized_mae_sample_wmape_pp",
             value=100 * (raw_total_num - total_num) / total_den,
             definition="Negative means lower sample distortion for the raw-unit-MAE lambda=.5 contrast"),
        dict(metric="normalized_mae_mean_sample_coder_rate_bits_per_float", value=mean_primary_sample_rate,
             definition="Mean of sample coder rate across seeds; metadata excluded"),
        dict(metric="raw_mae_lambda05_mean_sample_coder_rate_bits_per_float", value=mean_raw_sample_rate,
             definition="Mean of sample coder rate across seeds; metadata excluded"),
        dict(metric="pooled_baseline_heldout_wmape_pct", value=pooled_shift,
             definition="100 * pooled sum|baseline-heldout| / pooled sum|heldout|; domain shift diagnostic"),
        dict(metric="exact_zero_target_coordinate_pct", value=100 * zero_coords / all_coords,
             definition="Share of sampled coordinates with heldout target exactly zero"),
        dict(metric="exact_zero_decoded_nonzero_pct", value=float(np.mean(
            [row["exact_zero_decoded_nonzero_pct"] for row in summaries])),
             definition="Mean across seeds; exact-zero targets decoded to nonzero values"),
        dict(metric="exact_zero_error_numerator_share_pct", value=100 * zero_error / total_num,
             definition="Pooled raw-L1 error share from exact-zero heldout targets"),
        dict(metric="nonzero_true_sign_flip_pct", value=100 * true_flips / nonzero_coords,
             definition="100 * count(decoded*heldout<0) / count(heldout!=0)"),
        dict(metric="positive_target_through_origin_slope", value=sign_slopes["positive"],
             definition="Pooled sum(target*decoded)/sum(target^2), target>0"),
        dict(metric="negative_target_through_origin_slope", value=sign_slopes["negative"],
             definition="Pooled sum(target*decoded)/sum(target^2), target<0"),
        dict(metric="positive_target_mean_signed_error", value=sign_bias["positive"],
             definition="Pooled mean(decoded-target), target>0"),
        dict(metric="negative_target_mean_signed_error", value=sign_bias["negative"],
             definition="Pooled mean(decoded-target), target<0"),
        dict(metric="lowest_nonzero_magnitude_quintile_error_share_pct", value=low_nonzero_error_share,
             definition="Mean across seeds; raw-L1 numerator share"),
        dict(metric="lowest_nonzero_magnitude_quintile_denominator_share_pct", value=low_nonzero_den_share,
             definition="Mean across seeds; heldout sum|target| denominator share"),
        dict(metric="lowest_nonzero_magnitude_quintile_wmape_pct", value=low_nonzero_wmape,
             definition="Mean across seeds; raw-L1 WMAPE within lowest nonzero-|target| quintile"),
        dict(metric="lowest_nonzero_magnitude_quintile_true_sign_flip_pct", value=low_nonzero_flips,
             definition="Mean across seeds; true sign flips among nonzero values"),
        *[dict(metric=f"magnitude_bin_{index}_abs_shift_error_correlation", value=value,
               definition="Within nonzero heldout-|target| band, correlation between |target-baseline| and |decoded-target|")
          for index, value in shift_corr_by_bin.items()],
        *[dict(metric=f"top_error_{100*fraction:g}pct_raw_l1_share_pct", value=tail_mean[fraction],
               definition="Mean across seeds; top coordinates ranked by absolute reconstruction error")
          for fraction in TAIL_FRACTIONS],
    ]
    write_csv(args.output / "sample_key_findings.csv", key_findings)
    report = ["# Received-sample distortion audit", "",
              "Primary case: normalized-MAE acquisition, final MSE, received-prefix bins; seeds 43–47. "
              "Each case's transmitted coordinates are current held-out target values `target[indices]`; "
              "the baseline vector is the acquisition-fit target. `sample_reconstruction` is decoded back to raw units. "
              "The acquisition objective is explicit in each completion record; final distortion uses the config default MSE, "
              "which these older completion records do not repeat. "
              f"Held-out target hash `{reference['target_hash']}` matches source snapshot `{provenance['snapshot']}`; "
              f"baseline hash `{provenance['baseline_content_hash']}`. Per-seed index hashes are checked against completion records.", "",
              f"Pooled sample WMAPE from raw sums: {100 * total_num / total_den:.5f}% "
              f"({total_num:.9g} / {total_den:.9g}; {all_coords:,} sample observations).", "",
              f"On the same coordinates, the raw-unit-MAE λ=.5 contrast has pooled sample WMAPE "
              f"{100 * raw_total_num / total_den:.5f}% (difference "
              f"{100 * (raw_total_num-total_num)/total_den:+.3f} points) and mean sample coder rate "
              f"{mean_raw_sample_rate:.5f} versus {mean_primary_sample_rate:.5f} bits/float (metadata excluded). "
              "It is lower on seeds 44–47, higher on seed43; "
              "per-slice and per-magnitude error changes are in `sample_norm_slice_distortion.csv` and "
              "`sample_target_magnitude_bins.csv`. This distortion contrast does not establish better RD; sample rate and full-vector cost differ.", "",
              f"Heldout-vs-baseline mismatch on the same coordinates has pooled WMAPE {pooled_shift:.2f}%; "
              "this is a domain-shift diagnostic, not distortion of transmitted values. Absolute error correlates "
              f"{np.mean([row['abs_target_error_corr'] for row in summaries]):.3f} with |heldout target| and "
              f"{np.mean([row['abs_baseline_shift_error_corr'] for row in summaries]):.3f} with |heldout−baseline|.", "",
              f"Exact-zero heldout targets are {100 * zero_coords / all_coords:.2f}% of sample coordinates. They all decode nonzero, "
              f"yet contribute only {100 * zero_error / total_num:.3f}% of raw-L1 error. Among nonzero heldout coordinates, "
              f"the true sign-flip rate is {100 * true_flips / nonzero_coords:.2f}%. The lowest nonzero-|target| quintile "
              f"accounts for {low_nonzero_error_share:.2f}% of error numerator and {low_nonzero_den_share:.2f}% of denominator "
              f"(WMAPE {low_nonzero_wmape:.1f}%, true flips {low_nonzero_flips:.1f}%); higher magnitude bands have sharply fewer flips.", "",
              f"Through-origin decoded/target slopes are {sign_slopes['positive']:.3f} for positive targets and "
              f"{sign_slopes['negative']:.3f} for negative targets; pooled mean signed errors are "
              f"{sign_bias['positive']:.3g} and {sign_bias['negative']:.3g}, respectively, consistent with modest shrinkage toward zero. "
              "Within heldout-magnitude quintiles, the correlation between |heldout−baseline| and absolute received error is "
              + ", ".join(f"{value:.2f}" for value in shift_corr_by_bin.values())
              + "; this controls magnitude coarsely but not parameter/layer, so it is not a causal domain-shift effect.", "",
              f"Largest sample-error parameter groups: {top_layer_report}.", "",
              "Error-tail numerator shares, averaged across seeds: "
              + ", ".join(f"top {100*fraction:g}% = {tail_mean[fraction]:.2f}%" for fraction in TAIL_FRACTIONS) + ".", "",
              "Seed-level raw numerator, held-out denominator, bias, sign flips and error quantiles are in `sample_seed_summary.csv`. "
              "Tail rows rank coordinates by absolute reconstruction error and report both their raw-L1 error share and held-out `sum|target|` share. "
              "Per-normalization-slice sample numerator/denominator coverage is in `sample_norm_slice_distortion.csv` for joining to model-layer names. "
              "The same measurements aggregated by named parameter, including zero targets and true nonzero sign flips, are in `sample_layer_distortion.csv`; signed calibration by target sign is in `sample_signed_calibration.csv`. "
              "Retained sample symbols are an rANS byte stream and per-coordinate prior probabilities are not saved; no coordinate-level rate is attributed here. "
              "Exact-zero held-out targets are analyzed separately from nonzero-target magnitude bands; sign flips mean `target*decoded<0`. "
              "Magnitude bands use nonzero held-out `|target|` quantiles; tied cutpoints are collapsed, so bands can contain unequal numbers of coordinates. "
              "These are descriptive sample audits, not causal attributions to the full-vector RD gap.", ""]
    (args.output / "README.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
