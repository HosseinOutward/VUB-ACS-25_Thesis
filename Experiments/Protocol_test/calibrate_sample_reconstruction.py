"""Measure held-out benefit from cheap positive gains on received sample reconstructions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Experiments.Protocol_test.analyze_sample_distortion import DEFAULT_EXPERIMENT, DEFAULT_OUTPUT, DEFAULT_SLICE_MAP, DEFAULT_VECTORS, SEEDS, read_vectors
from FL_code.FL_core.explicit_cache import content_hash
from FL_code.FL_core.utils import decompress_data_list


SPLIT_SEED = 20261014
OUTPUT = DEFAULT_OUTPUT / "gain_calibration"


def positive_l1_gain(target_centered: np.ndarray, decoded_centered: np.ndarray) -> float:
    """Fit the nonnegative L1-optimal gain as a weighted median of target/decoded ratios."""
    active = decoded_centered != 0
    if not active.any():
        return 1.0
    ratios = target_centered[active] / decoded_centered[active]
    weights = np.abs(decoded_centered[active])
    order = np.argsort(ratios)
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    return max(0.0, float(ratios[order[min(index, len(order) - 1)]]))


def fit_gains(target: np.ndarray, decoded: np.ndarray, centers: np.ndarray, groups: np.ndarray,
              loss: str, per_parameter: bool, group_count: int) -> np.ndarray:
    """Fit one or one-per-parameter positive gains using training-half raw targets."""
    centered_target, centered_decoded = target - centers, decoded - centers
    gain_count = group_count if per_parameter else 1
    gains = np.ones(gain_count, dtype=np.float64)
    for group in range(gain_count):
        selected = (groups == group) if per_parameter else np.ones(len(target), dtype=bool)
        t, r = centered_target[selected], centered_decoded[selected]
        if loss == "l1":
            gains[group] = positive_l1_gain(t, r)
        else:
            denominator = float(np.dot(r, r))
            gains[group] = max(0.0, float(np.dot(r, t)) / denominator) if denominator else 1.0
    return gains


def reconstruction(decoded: np.ndarray, centers: np.ndarray, groups: np.ndarray, gains: np.ndarray,
                   per_parameter: bool) -> np.ndarray:
    """Apply a gain around the decoder-known normalization center for each coordinate."""
    gain = gains[groups] if per_parameter else gains[0]
    return centers + gain * (decoded - centers)


def wmape(target: np.ndarray, decoded: np.ndarray) -> tuple[float, float, float]:
    """Return raw L1 numerator, denominator and their WMAPE percentage."""
    numerator, denominator = float(np.abs(decoded - target).sum()), float(np.abs(target).sum())
    return numerator, denominator, 100 * numerator / denominator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat calibration results with stable columns."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run() -> None:
    """Fit and evaluate global/per-parameter gains on deterministic disjoint coordinate halves."""
    target, _, slices, provenance = read_vectors(DEFAULT_VECTORS)
    parameter_names = {int(row["slice_index"]): row["parameter"]
                       for row in csv.DictReader(DEFAULT_SLICE_MAP.open())}
    assert len(parameter_names) == len(slices)
    summary_rows, layer_rows, gain_rows = [], [], []
    for seed in SEEDS:
        case = DEFAULT_EXPERIMENT / "cases" / f"current_e1_mae_received_prefix_bins_seed{seed}" / "target"
        complete = json.loads((case / "complete.json").read_text())
        weights: dict[str, torch.Tensor] = torch.load(case / "weights.pt", map_location="cpu", weights_only=True)
        indices = weights["sample_indices"].long()
        decoded = weights["sample_reconstruction"].numpy().astype(np.float64)
        assert complete["sample_indices_hash"] == content_hash(indices)
        payload = decompress_data_list((case / "payload.bin").read_bytes())
        packet = decompress_data_list(payload["sample_payload"])
        assert torch.equal(packet.indices(), indices)
        factors = packet.metadata.norm_factors.cpu().numpy()
        assert factors.shape == (len(slices), 2) and not packet.metadata.outliers.positions
        coordinate_ids = indices.numpy()
        boundaries = np.array([part.start for part in slices] + [slices[-1].stop])
        groups = np.searchsorted(boundaries[1:], coordinate_ids, side="right")
        assert groups.max() < len(slices)
        y = target[indices].numpy().astype(np.float64)
        centers = factors[groups, 1].astype(np.float64)
        split = np.random.default_rng(SPLIT_SEED + seed).permutation(len(indices))
        train_ids, eval_ids = split[:len(split) // 2], split[len(split) // 2:]
        methods = (("global_l1", "l1", False), ("parameter_l1", "l1", True),
                   ("global_l2", "l2", False), ("parameter_l2", "l2", True))
        for method, loss, per_parameter in methods:
            gains = fit_gains(y[train_ids], decoded[train_ids], centers[train_ids], groups[train_ids],
                              loss, per_parameter, len(slices))
            calibrated = reconstruction(decoded[eval_ids], centers[eval_ids], groups[eval_ids], gains,
                                        per_parameter)
            before_num, denominator, before_wmape = wmape(y[eval_ids], decoded[eval_ids])
            after_num, _, after_wmape = wmape(y[eval_ids], calibrated)
            train_reconstruction = reconstruction(decoded[train_ids], centers[train_ids], groups[train_ids],
                                                  gains, per_parameter)
            _, _, train_after_wmape = wmape(y[train_ids], train_reconstruction)
            summary_rows.append(dict(seed=seed, method=method, loss=loss,
                                     gain_scope="per named parameter" if per_parameter else "global",
                                     gain_values=len(gains), gain_float32_bytes=4 * len(gains),
                                     split_seed=SPLIT_SEED + seed, train_coordinates=len(train_ids),
                                     eval_coordinates=len(eval_ids), eval_raw_l1_before=before_num,
                                     eval_raw_l1_after=after_num, eval_target_l1=denominator,
                                     eval_wmape_before_pct=before_wmape, eval_wmape_after_pct=after_wmape,
                                     eval_wmape_change_points=after_wmape - before_wmape,
                                     train_wmape_after_pct=train_after_wmape,
                                     gain_min=float(gains.min()), gain_median=float(np.median(gains)),
                                     gain_max=float(gains.max()), gain_mean=float(gains.mean())))
            for group in range(len(slices)):
                selected = groups[eval_ids] == group
                if not selected.any():
                    continue
                group_y = y[eval_ids][selected]
                group_before = decoded[eval_ids][selected]
                group_after = calibrated[selected]
                before_group_num, group_den, before_group_wmape = wmape(group_y, group_before)
                after_group_num, _, after_group_wmape = wmape(group_y, group_after)
                gain_value = gains[group] if per_parameter else gains[0]
                layer_rows.append(dict(seed=seed, method=method, parameter=parameter_names[group],
                                       norm_slice=group, eval_coordinate_count=int(selected.sum()),
                                       target_l1=group_den, error_l1_before=before_group_num,
                                       error_l1_after=after_group_num,
                                       numerator_change=after_group_num - before_group_num,
                                       wmape_before_pct=before_group_wmape, wmape_after_pct=after_group_wmape,
                                       gain=float(gain_value)))
                if per_parameter:
                    gain_rows.append(dict(seed=seed, method=method, parameter=parameter_names[group],
                                          norm_slice=group, gain=float(gain_value),
                                          train_coordinate_count=int(np.count_nonzero(groups[train_ids] == group)),
                                          eval_coordinate_count=int(selected.sum())))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "gain_calibration_by_seed.csv", summary_rows)
    write_csv(OUTPUT / "gain_calibration_by_parameter.csv", layer_rows)
    write_csv(OUTPUT / "fitted_parameter_gains.csv", gain_rows)
    methods = ("global_l1", "parameter_l1", "global_l2", "parameter_l2")
    aggregate = []
    for method in methods:
        rows = [row for row in summary_rows if row["method"] == method]
        numerator = sum(row["eval_raw_l1_before"] for row in rows)
        denominator = sum(row["eval_target_l1"] for row in rows)
        calibrated_numerator = sum(row["eval_raw_l1_after"] for row in rows)
        aggregate.append(dict(method=method, eval_coordinates=sum(row["eval_coordinates"] for row in rows),
                              eval_wmape_before_pct=100 * numerator / denominator,
                              eval_wmape_after_pct=100 * calibrated_numerator / denominator,
                              eval_wmape_change_points=100 * (calibrated_numerator - numerator) / denominator,
                              mean_gain_float32_bytes=float(np.mean([row["gain_float32_bytes"] for row in rows]))))
    write_csv(OUTPUT / "gain_calibration_pooled.csv", aggregate)
    (OUTPUT / "README.md").write_text(
        "# Held-out sample gain calibration\n\n"
        "Diagnostic only: no quantizer or decoder was retrained and no production codec was changed. "
        "Each seed's sorted transmitted coordinates are split by a deterministic random permutation into disjoint halves. "
        "Gains are fit on raw held-out sample values from the train half and evaluated only on the other half. "
        "The calibrated value is `center + gain * (decoded - center)`, where each exact center comes from the received sample packet's decoder-visible normalization metadata. "
        "The global gain uses one positive scalar; the parameter gain uses one scalar per each of the 62 named parameters. "
        "L1 gains use the exact weighted-median solution; L2 gains use the analytic nonnegative least-squares solution. "
        "The listed float32 cost is 4 bytes global or 248 bytes per-parameter, before protocol metadata. "
        "Per-seed WMAPE is based on raw L1 sums. These held-out sample results do not imply a whole-vector RD gain or include downstream retraining.\n\n"
        f"Source snapshot: `{provenance['snapshot']}`. See `gain_calibration_pooled.csv`, `gain_calibration_by_seed.csv`, "
        "`gain_calibration_by_parameter.csv`, and `fitted_parameter_gains.csv`.\n")


if __name__ == "__main__":
    run()
