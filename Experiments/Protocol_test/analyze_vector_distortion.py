"""Read-only coordinate audit of matched O and sampled-S target reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from FL_code.FL_core.explicit_cache import content_hash
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.utils import StateDictManager, decompress_data_list
from FL_code.run_fl import FLConfig
from FL_code.cancer_protocol.bin_coder.rans_coder import HEADER, _digest
from FL_code.cancer_protocol.wz_quantizer import (
    DedupedWZQuantizerCancer, SideInfoColumnsCache, WZcfgQuant,
)
from FL_code.cancer_protocol.prior_code import PriorCalculator
from Experiments.Protocol_test.sampled_codec import _SampleQuantizer

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "_" / "experiments"
SOURCE = EXPERIMENTS / "frozen_starting_svhn_seed43_rtx3090_20261007/problem/c13435385fb67dcc38578bab6ea6080e119612b356dff5517102221110e5c5a0.pkl"
S_ROOT = EXPERIMENTS / "s_received_mae_five_seed_20261012"
O_ROOT = EXPERIMENTS / "s_dedicated_refiner_coldfit_five_seed_20261011"
OUTPUT = EXPERIMENTS / "s_distortion_audit_20261014/vector"


def _quantizer(config: dict[str, Any], slices: tuple[slice, ...], si: tuple[Any, ...],
               weights: dict[str, Any], method: str, indices: torch.Tensor | None = None,
               sample_reconstruction: torch.Tensor | None = None) -> DedupedWZQuantizerCancer:
    """Build only the saved decoder architecture and attach its decoder-visible side information."""
    config = config | {"num_planes": 3, "norm_slices": slices}
    if method == "O":
        quantizer = DedupedWZQuantizerCancer(WZcfgQuant.model_validate(config), len(si), SideInfoColumnsCache())
    else:
        full_config = WZcfgQuant.model_validate(config | {"num_planes": 4})
        quantizer = _SampleQuantizer(
            c_cfg=full_config, si_size=len(si), side_info_columns_cache=SideInfoColumnsCache(),
            refinement_planes=1,
        )
        quantizer.sample_indices = indices
        quantizer.sample_reconstruction = sample_reconstruction
        quantizer.c_cfg = quantizer.c_cfg.model_copy(update={"num_planes": 3})
        quantizer.coding_model.num_planes = 3
    quantizer.coding_model.load_state_dict(weights["native"], strict=True)
    quantizer.side_info_list_used = list(si)
    quantizer.vector_size = 11_181_642
    return quantizer


def _decode(path: Path, quantizer: DedupedWZQuantizerCancer, target: torch.Tensor, method: str,
            indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, str, Any]:
    """Regenerate encoder symbols, verify their wire digest, then decode the saved quantized stream."""
    packet = decompress_data_list(path.read_bytes())
    transmitted, metadata = packet["payload_content"]
    normalized, _ = quantizer.preprocess_x(target, normalization_factors=metadata.norm_factors)
    bins, _ = quantizer.encode_prepared(normalized, batch_size=25_000)
    if method == "S":
        ordinary = torch.ones(quantizer.vector_size, dtype=torch.bool)
        assert indices is not None
        ordinary[indices] = False
        expected_positions = int(ordinary.sum())
        wire_bins = bins[:, ordinary]
    else:
        expected_positions = quantizer.vector_size
        wire_bins = bins
    positions, stages_count, alphabet, _, _, expected_digest = HEADER.unpack_from(transmitted)
    assert (positions, stages_count, alphabet) == (expected_positions, 3, 2), (
        f"{method} saved payload header disagrees with its expected symbol shape at {path}.")
    assert _digest(wire_bins) == expected_digest, (
        f"Regenerated {method} symbols do not match the saved rANS symbol digest at {path}.")
    if method == "S":
        restored = torch.zeros((3, quantizer.vector_size), dtype=bins.dtype)
        restored[:, ordinary] = wire_bins
        bins = restored
    stages = quantizer.decoding_stages_process((bins, metadata), batch_size=25_000)
    if method == "S":
        # Decode replaces the sample after ordinary reconstruction, as the deployed protocol does.
        stages[:, indices] = quantizer.sample_reconstruction
    return stages, bins, f"header={expected_digest.hex()} regenerated={_digest(wire_bins).hex()}", metadata


def _bucket_rows(target: torch.Tensor, baseline: torch.Tensor, reconstruction: torch.Tensor,
                 sample_mask: torch.Tensor,
                 seed: int, method: str, stage: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize support, magnitude, SI mismatch, layer and sample effects without raw arrays."""
    error = reconstruction - target
    absolute_error = error.abs()
    denominator = target.abs().mean().item()
    target_abs = target.abs()
    mismatch = (target - baseline).abs()
    sample_rows: list[dict[str, Any]] = []
    for sampled, mask in ((True, sample_mask), (False, ~sample_mask)):
        source = target[mask]
        predicted = reconstruction[mask]
        absolute = (predicted - source).abs()
        support = source != 0
        sample_rows.append({
            "seed": seed, "method": method, "stage": stage, "sampled": sampled,
            "coordinates": int(mask.sum()), "wmape_l1_share_percent": 100 * absolute.sum().item() / target.abs().sum().item(),
            "target_zero_fraction": (~support).float().mean().item(),
            "false_positive_fraction_of_zeros": ((~support) & (predicted != 0)).sum().item() / max((~support).sum().item(), 1),
            "false_negative_fraction_of_nonzeros": (support & (predicted == 0)).sum().item() / max(support.sum().item(), 1),
            "true_sign_flip_fraction_nonzero": ((source * predicted < 0) & support).sum().item() / max(support.sum().item(), 1),
            "signed_bias_over_target_mean_abs": (predicted - source).mean().item() / denominator,
            "l1_per_coordinate": absolute.mean().item(),
        })
    feature_rows: list[dict[str, Any]] = []
    for feature_name, feature in (("target_abs", target_abs), ("baseline_mismatch_abs", mismatch)):
        zero_mask = feature == 0
        masks = [("exact_zero", zero_mask)]
        nonzero = (feature > 0).nonzero().flatten()
        for decile, group_indices in enumerate(torch.tensor_split(nonzero[torch.argsort(feature[nonzero], stable=True)], 10), start=1):
            mask = torch.zeros_like(zero_mask)
            mask[group_indices] = True
            masks.append((f"nonzero_rank_decile_{decile}", mask))
        for bucket_name, mask in masks:
            if not bool(mask.any()):
                continue
            true_nonzero = target[mask] != 0
            pred = reconstruction[mask]
            err = error[mask]
            feature_rows.append({
                "seed": seed, "method": method, "stage": stage, "feature": feature_name, "bucket": bucket_name,
                "coordinates": int(mask.sum()), "coordinate_l1_share_percent": 100 * absolute_error[mask].sum().item() / target.abs().sum().item(),
                "mean_feature_value": feature[mask].mean().item(), "p50_feature_value": feature[mask].median().item(),
                "target_zero_fraction": (~true_nonzero).float().mean().item(),
                "false_positive_fraction_of_zeros": ((target[mask] == 0) & (pred != 0)).sum().item() / max((target[mask] == 0).sum().item(), 1),
                "nonzero_magnitude_p50": target[mask][true_nonzero].abs().median().item() if true_nonzero.any() else 0.0,
                "mean_abs_error": err.abs().mean().item(), "mean_signed_error": err.mean().item(),
            })
    return sample_rows, feature_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one compact table, keeping empty artifacts explicit."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(seeds: tuple[int, ...], output: Path = OUTPUT) -> None:
    """Decode the saved O/S artifacts only and report coordinate-level distortion patterns."""
    output.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("rb") as stream:
        problem = pickle.load(stream)
    target = problem["target"].float().cpu()
    baseline = problem["baseline"].float().cpu()
    si = tuple(problem["si"])
    slices = tuple(problem["slices"])
    expected_si_hash = json.loads((S_ROOT / "test_config.json").read_text())["reference_manifest"]["si_hash"]
    actual_si_hash = content_hash(tuple(content_hash(value.materialize()) for value in si))
    assert actual_si_hash == expected_si_hash, f"Cached source SI hash mismatch: {actual_si_hash} != {expected_si_hash}."
    del problem
    fl_config = FLConfig.model_validate_json((ROOT / "Experiments/Protocol_test/plans/svhn_30c_fl_config.json").read_text())
    manager = StateDictManager(initialize_model(fl_config, torch.device("cpu")))
    assert tuple((part.start, part.stop) for part in slices) == tuple((part.start, part.stop) for part in manager.get_slices())
    config = json.loads((EXPERIMENTS / "frozen_starting_svhn_seed43_rtx3090_20261007/test_config.json").read_text())["plan"]["quantizer"]
    config["norm_slices"] = slices
    layer_ids = torch.empty(target.numel(), dtype=torch.int32)
    for layer, part in enumerate(slices):
        layer_ids[part] = layer
    sample_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    zero_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    for seed in seeds:
        s_case = S_ROOT / f"cases/current_e1_mae_received_prefix_bins_seed{seed}/target"
        o_case = O_ROOT / f"cases/O_seed{seed}/target"
        s_weights = torch.load(s_case / "weights.pt", map_location="cpu", weights_only=True)
        o_weights = torch.load(o_case / "weights.pt", map_location="cpu", weights_only=True)
        indices = s_weights["sample_indices"].long().cpu()
        sample_mask = torch.zeros(target.numel(), dtype=torch.bool)
        sample_mask[indices] = True
        for method, case, weights in (("O", o_case, o_weights), ("S", s_case, s_weights)):
            print(f"decoding seed{seed} {method}", flush=True)
            quantizer = _quantizer(config, slices, si, weights, method,
                                   indices if method == "S" else None,
                                   s_weights["sample_reconstruction"] if method == "S" else None)
            stages, bins, symbol_check, metadata = _decode(
                case / "payload.bin", quantizer, target, method, indices if method == "S" else None)
            record = next(csv.DictReader((case / "compression_records.csv").open()))
            recorded_wmapes = json.loads(record["stage_wmapes"])
            ordinary = torch.ones(target.numel(), dtype=torch.bool)
            if method == "S":
                ordinary[indices] = False
            wire_bins = bins[:, ordinary]
            side_info = quantizer.side_info_tensor(metadata)[ordinary]
            conditional_prior = PriorCalculator.compute_prior_from_network(
                quantizer.coding_model, wire_bins, side_info)
            conditional_lengths = PriorCalculator.compute_code_lengths(conditional_prior, wire_bins, 3)
            marginal_prior = PriorCalculator.compute_marginal_prior(
                wire_bins, quantizer.coding_model.bins_per_plane, 3)
            marginal_lengths = PriorCalculator.compute_code_lengths(marginal_prior, wire_bins, 3)
            stage_rates = json.loads(record["server_si_prior_stage_rates"])
            top_error = torch.topk((stages[2, ordinary] - target[ordinary]).abs(),
                                   k=max(1, wire_bins.shape[1] // 100)).indices
            total_conditional_nll = conditional_lengths.sum(dim=0)
            total_marginal_nll = marginal_lengths.sum(dim=0)
            for stage in range(3):
                rate_rows.append({
                    "seed": seed, "method": method, "stage": stage + 1,
                    "wire_coordinates": wire_bins.shape[1],
                    "native_model_nll_proxy_bits_per_symbol": conditional_lengths[stage].mean().item(),
                    "recorded_server_si_prior_bits_per_symbol": stage_rates[stage],
                    "native_minus_recorded_bits_per_symbol": conditional_lengths[stage].mean().item() - stage_rates[stage],
                    "marginal_symbol_surprisal_bits_per_symbol": marginal_lengths[stage].mean().item(),
                    "native_model_nll_proxy_top_stage3_error_1pct_share_percent": (
                        100 * total_conditional_nll[top_error].sum().item() / total_conditional_nll.sum().item()),
                    "marginal_surprisal_top_stage3_error_1pct_share_percent": (
                        100 * total_marginal_nll[top_error].sum().item() / total_marginal_nll.sum().item()),
                    "interpretation": "model NLL proxy; native prior CDF does not reproduce physical rANS stream",
                })
            denominator = target.abs().mean().item()
            for stage, reconstruction in enumerate(stages, start=1):
                wmape = (reconstruction - target).abs().mean().item() / denominator * 100
                assert abs(wmape - recorded_wmapes[stage - 1]) < 0.02, (
                    f"{method} seed{seed} stage{stage} WMAPE {wmape} differs from record {recorded_wmapes[stage-1]}.")
                validation_rows.append({
                    "seed": seed, "method": method, "stage": stage,
                    "regenerated_wmape_percent": wmape, "recorded_wmape_percent": recorded_wmapes[stage - 1],
                    "wmape_delta_points": wmape - recorded_wmapes[stage - 1],
                    "recorded_numerical_runtime": record["numerical_runtime"],
                    "symbol_digest_check": symbol_check,
                })
                sample_part, feature_part = _bucket_rows(
                    target, baseline, reconstruction, sample_mask, seed, method, stage)
                sample_rows.extend(sample_part)
                feature_rows.extend(feature_part)
                error_abs = (reconstruction - target).abs()
                order = torch.argsort(error_abs, descending=True)
                for fraction in (.0001, .001, .01, .05, .10):
                    count = max(1, int(target.numel() * fraction))
                    squared_error = error_abs.square()
                    for ranking_metric, ranked in (("absolute_error", error_abs), ("squared_error", squared_error)):
                        top = torch.argsort(ranked, descending=True)[:count]
                        outlier_rows.append({
                            "seed": seed, "method": method, "stage": stage, "tail_fraction": fraction,
                            "ranking_metric": ranking_metric, "coordinate_count": count,
                            "l1_error_share_percent": 100 * error_abs[top].sum().item() / error_abs.sum().item(),
                            "squared_error_share_percent": 100 * squared_error[top].sum().item() / squared_error.sum().item(),
                            "target_zero_fraction": (target[top] == 0).float().mean().item(),
                            "sampled_fraction": sample_mask[top].float().mean().item(),
                            "mean_abs_target": target[top].abs().mean().item(),
                            "mean_abs_baseline_mismatch": (target[top] - baseline[top]).abs().mean().item(),
                        })
                if stage == 3:
                    zero = target == 0
                    nonzero = ~zero
                    error = reconstruction - target
                    zero_rows.append({
                        "seed": seed, "method": method, "stage": stage,
                        "target_zero_count": int(zero.sum()), "target_zero_fraction": zero.float().mean().item(),
                        "zero_false_positive_count": int((zero & (reconstruction != 0)).sum()),
                        "zero_false_positive_l1_share_percent": 100 * error[zero].abs().sum().item() / target.abs().sum().item(),
                        "nonzero_false_negative_count": int((nonzero & (reconstruction == 0)).sum()),
                        "true_sign_flips_nonzero": int((target[nonzero] * reconstruction[nonzero] < 0).sum()),
                        "nonzero_count": int(nonzero.sum()),
                    })
                    for layer, (name, part) in enumerate(zip(manager.keys, slices, strict=True)):
                        err = error[part].abs()
                        target_l1 = target[part].abs().sum().item()
                        layer_rows.append({
                            "seed": seed, "method": method, "layer_index": layer, "parameter": name,
                            "coordinates": part.stop - part.start,
                            "sample_fraction": sample_mask[part].float().mean().item(),
                            "target_abs_mean": target[part].abs().mean().item(),
                            "baseline_mismatch_abs_mean": (target[part] - baseline[part]).abs().mean().item(),
                            "target_zero_fraction": (target[part] == 0).float().mean().item(),
                            "l1_error_contribution_percent": 100 * err.sum().item() / target.abs().sum().item(),
                            "target_l1_share_percent": 100 * target_l1 / target.abs().sum().item(),
                            "local_wmape_percent": 100 * err.sum().item() / max(target_l1, 1e-30),
                        "false_positive_fraction_of_zeros": (
                                ((target[part] == 0) & (reconstruction[part] != 0)).sum().item()
                                / max((target[part] == 0).sum().item(), 1)),
                            "false_positive_reconstruction_abs_p50": (
                                reconstruction[part][target[part] == 0].abs().median().item()
                                if bool((target[part] == 0).any()) else 0.0),
                            "false_positive_reconstruction_abs_p99": (
                                torch.quantile(reconstruction[part][target[part] == 0].abs(), .99).item()
                                if bool((target[part] == 0).any()) else 0.0),
                            "signed_error_over_abs_target_mean": error[part].mean().item() / denominator,
                        })
            del stages, bins, quantizer
            print(f"decoded seed{seed} {method}: {symbol_check}", flush=True)
            torch.cuda.empty_cache()
    _write_csv(output / "sampled_vs_unsampled.csv", sample_rows)
    _write_csv(output / "feature_deciles.csv", feature_rows)
    _write_csv(output / "layer_stage3.csv", layer_rows)
    _write_csv(output / "zero_support_stage3.csv", zero_rows)
    _write_csv(output / "error_outlier_tails.csv", outlier_rows)
    _write_csv(output / "validation.csv", validation_rows)
    _write_csv(output / "rate_proxy_seed43.csv", rate_rows)
    manifest = {
        "source": SOURCE.as_posix(), "source_target_hash": content_hash(target),
        "source_baseline_hash": content_hash(baseline), "coordinate_count": target.numel(),
        "side_info_vectors": len(si), "layer_count": len(slices), "seeds": list(seeds),
        "decoding": "saved encoder weights regenerated symbols from exact target using payload metadata; expected and regenerated rANS-header BLAKE2 digests are persisted in validation.csv; saved decoder weights and SI decoded stages; no fit",
        "sample_reconstruction_scale": "raw (verified against recorded sample WMAPE)",
        "physical_rans_decode": "prior CDF did not reproduce stream; diagnostics use regenerated symbols only after wire symbol-digest match",
        "rate_attribution": "per-coordinate actual rANS byte allocation unavailable; stage records retain full-stream measured rates",
        "side_info_content_hash": actual_si_hash,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    """Select matched seeds for a bounded read-only diagnostic pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(tuple(args.seeds), args.output)


if __name__ == "__main__":
    main()
