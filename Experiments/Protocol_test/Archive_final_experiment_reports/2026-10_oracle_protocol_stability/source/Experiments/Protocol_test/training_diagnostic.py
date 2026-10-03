"""Controlled sampled-trace O/T training and frozen-prior diagnostics; not a protocol replay."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep

import numpy as np
import pandas as pd
import torch

from Experiments.Protocol_test.sparse_replay import load_fl_config, load_plan
from Experiments.Protocol_test.rd_metrics import rd_curve_gap, wmape_to_db
from FL_code.FL_core.codec import Access, CompressionRecord
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.bin_coder.rans_coder import HEADER, RansConfig, decode, encode
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.history import CompressedReconstructionHistory
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant, new_rnn_model, wz_model_training_loop


CLIENTS = (0, 5, 11)
ROUNDS = (5, 6, 7, 8, 12, 13)
SEEDS = (43, 44, 45)
TRAIN_ROWS = 16_384
VALID_ROWS = 4_096


def embed_temporal(model: EncoderDecoderLayeredRNN, columns: torch.Tensor, width: int) -> EncoderDecoderLayeredRNN:
    """Embed a temporal model exactly into an oracle model, initially ignoring extra SI."""
    oracle = new_rnn_model(3, 2, width, False, 128, 4)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for key, offset in (("decoder.input_layers.0.weight", 2), ("conditionalRNN.input_layers.0.weight", 0)):
        old = state[key]
        expanded = torch.zeros((old.shape[0], width + 2))
        expanded[:, columns + offset] = old[:, offset:offset + columns.numel()]
        if offset:
            expanded[:, :2] = old[:, :2]
        else:
            expanded[:, width:] = old[:, columns.numel():]
        state[key] = expanded
    oracle.load_state_dict(state)
    return oracle.eval()


def collect_problems(plan_path: Path, output: Path) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    """Capture full-vector-normalized samples with common raw-history SI for both methods."""
    plan = load_plan(plan_path)
    cfg = load_fl_config(plan.fl_config)
    slices = StateDictManager(initialize_model(cfg, torch.device("cpu"))).get_slices()
    history = CompressedReconstructionHistory(5, slices)
    trace = Trace(cfg, dataset_revision=plan.dataset_revision, device=torch.device(plan.trace_device))
    indices = torch.randperm(slices[-1].stop, generator=torch.Generator().manual_seed(1043))[:TRAIN_ROWS + VALID_ROWS]
    owners = torch.bucketize(indices, torch.tensor([part.stop for part in slices]), right=True)
    problems = {}
    for round_id, deltas in trace.stream(0, max(ROUNDS)):
        print(f"Collect trace round {round_id}", flush=True)
        for client_id, target in enumerate(deltas):
            if round_id in ROUNDS and client_id in CLIENTS:
                si = [entry.value for entries in history.view(Access.SERVER_ONLY, client_id).values() for entry in entries]
                temporal_values = [entry.value for entries in history.view(Access.TEMPORAL_TOO, client_id).values() for entry in entries]
                columns = torch.tensor([next(i for i, value in enumerate(si) if value is own) for own in temporal_values])
                quantizer = DedupedWZQuantizerCancer(c_cfg=WZcfgQuant(bins_per_plane=2, num_planes=3, norm_slices=slices), si_size=len(si))
                normalized, metadata = quantizer.preprocess_x(target)
                matrix = quantizer.normalize_side_info_tensor(metadata, si)
                problems[client_id, round_id] = {
                    "x": normalized[indices.to(normalized.device)].cpu(),
                    "si": matrix[indices.to(matrix.device)].cpu(),
                    "raw": target[indices].clone(),
                    "factors": metadata.norm_factors[owners].clone(),
                    "columns": columns,
                }
                del quantizer, normalized, matrix
                torch.cuda.empty_cache()
            history.commit(target, CompressionRecord(round_id, client_id, "common_raw_history"), Access.TEMPORAL_TOO)
    torch.save(problems, output / "problems.pt")
    return problems


def fit_quantizer(cfg: WZcfgQuant, problems: list[dict[str, torch.Tensor]], temporal: bool, seed: int,
                  initial: EncoderDecoderLayeredRNN | None = None) -> tuple[EncoderDecoderLayeredRNN, float]:
    """Run one retained candidate using the unchanged joint training loss and configured schedule."""
    set_global_seed(seed)
    x = torch.cat([problem["x"][:TRAIN_ROWS] for problem in problems]).cuda()
    si = torch.cat([problem["si"][:TRAIN_ROWS, problem["columns"]] if temporal else problem["si"][:TRAIN_ROWS]
                    for problem in problems]).cuda()
    model = new_rnn_model(3, 2, si.shape[1], False, cfg.hidden_dim, cfg.layers) if initial is None else deepcopy(initial)
    loss = wz_model_training_loop(
        DedupedWZQuantizerCancer.compute_loss, model, model.parameters(), (x,), si, cfg,
        batch_size=cfg.train_batch_size, label=f"Diagnostic seed {seed}",
        wmspe_denom=x.square().mean().item() / 2 + 1e-8,
        perturb_generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    assert np.isfinite(loss), f"Non-finite candidate loss at seed {seed}"
    return model.cpu().eval(), loss


def fixed_bins(model: EncoderDecoderLayeredRNN, problems: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every historical target with the final encoder, never with an older encoder."""
    with torch.no_grad():
        model.cuda().eval()
        pairs = [model.encode(problem["x"][:TRAIN_ROWS].cuda()) for problem in problems]
        bins = torch.cat([torch.stack(pair[0]) for pair in pairs], dim=1).cpu()
        codes = torch.cat([torch.stack(pair[1]) for pair in pairs], dim=1).cpu()
        model.cpu()
    return bins, codes


def fit_prior(cfg: WZcfgQuant, joint: EncoderDecoderLayeredRNN, problems: list[dict[str, torch.Tensor]],
              temporal: bool, native_temporal: bool, seed: int) -> EncoderDecoderLayeredRNN:
    """Fit fixed-symbol priors with native initialization or an independent alternate-SI model."""
    set_global_seed(seed)
    # Full-vector fits hit the same sample cap; repeat current rows to match that update budget here.
    if len(problems) == 1:
        problems = problems * 2
    bins, codes = fixed_bins(joint, problems)
    si = torch.cat([problem["si"][:TRAIN_ROWS, problem["columns"]] if temporal else problem["si"][:TRAIN_ROWS]
                    for problem in problems])
    target = deepcopy(joint) if temporal == native_temporal else None
    return PriorCalculator.fit_prior_model(bins, codes, si, cfg, target, (None,), cfg.prior_train_repeats,
                                          batch_size=cfg.train_batch_size).model.cpu().eval()


def evaluate(joint: EncoderDecoderLayeredRNN, problem: dict[str, torch.Tensor], temporal: bool,
             priors: dict[str, EncoderDecoderLayeredRNN], heldout: bool) -> list[dict[str, float | int | None]]:
    """Measure stage RD and actual lossless rANS, keeping the sample's flush overhead explicit."""
    rows = slice(TRAIN_ROWS, None) if heldout else slice(None)
    x, raw, factors = (problem[name][rows] for name in ("x", "raw", "factors"))
    full_si = problem["si"][rows]
    native_si = full_si[:, problem["columns"]] if temporal else full_si
    with torch.no_grad():
        joint.cuda().eval()
        bins_list, codes = joint.encode(x.cuda())
        recons = joint.decode(codes, native_si.cuda())
        bins = torch.stack(bins_list).cpu()
        wmapes = [(recons[stage].cpu().flatten() * factors[:, 0] + factors[:, 1] - raw).abs().mean().item()
                  / raw.abs().mean().item() * 100 for stage in range(3)]
        hard_mse = [(value.cpu() - x).square().mean().item() for value in recons]
        joint.cpu()
        pre = np.cumsum(PriorCalculator.compute_plane_rates(PriorCalculator.compute_prior_from_network(joint, bins, native_si), bins, 3))
        measured = {}
        for source, model in priors.items():
            si = full_si[:, problem["columns"]] if source == "temporal" else full_si
            rates = PriorCalculator.compute_plane_rates(PriorCalculator.compute_prior_from_network(model, bins, si), bins, 3)

            def prior(context: torch.Tensor, stage: int) -> torch.Tensor:
                padded = torch.zeros_like(bins)
                padded[:context.shape[0]] = context.cpu()
                return PriorCalculator.compute_prior_from_network(model, padded, si)[stage].cuda()

            payloads = [encode(bins[:stage].cuda(), prior, RansConfig(lanes=2048)) for stage in (1, 2, 3)]
            assert torch.equal(decode(payloads[-1], prior).cpu(), bins), "rANS decoded symbols differ"
            measured[source] = (np.cumsum(rates), [8 * len(payload) / bins.shape[1] for payload in payloads], rates)
        overhead = 8 * (HEADER.size + 4 * 2048) / bins.shape[1]
    return [{
        "stage": stage + 1, "pre_rate": float(pre[stage]), "wmape": wmapes[stage],
        "wmape_db": float(wmape_to_db(np.array([wmapes[stage]]))[0]), "rans_overhead": overhead,
        "hard_mse": hard_mse[stage],
        **{f"{source}_{kind}": float(values[index][stage])
           for source, values in measured.items() for index, kind in enumerate(("prior", "rans", "stage_rate"))},
    } for stage in range(3)]


def render(output: Path) -> None:
    """Publish all candidates, equal-distortion gaps, selection audit, and a live RD plot."""
    files = sorted((output / "attempts").glob("*.json"))
    if not files:
        return
    records = pd.DataFrame([row for path in files for row in json.loads(path.read_text())["rows"]])
    records["pre_stage_rate"] = records.groupby(["client", "seed", "variant", "prior_mode", "round", "heldout"]).pre_rate.diff().fillna(records.pre_rate)
    records.to_csv(output / "stage_points.csv", index=False)
    issues = records.melt(id_vars=["client", "seed", "variant", "prior_mode", "round", "heldout", "stage"],
                          value_vars=["pre_stage_rate", "temporal_stage_rate", "full_stage_rate"],
                          var_name="source", value_name="rate")
    issues = issues[(issues.rate < .01) | (issues.rate > 1.05)].copy()
    issues["status"] = np.where(issues.rate < .01, "dead", "blown")
    issues.to_csv(output / "suspect_stages.csv", index=False)
    matched_jobs = records.groupby(["client", "seed"]).variant.nunique()
    matched_jobs = matched_jobs[matched_jobs == 3].index
    matched = records[records.set_index(["client", "seed"]).index.isin(matched_jobs)].copy()
    if not matched.empty:
        matched["phase"] = np.where(matched.heldout, "historical_validation", "future")
        matched["native_prior"] = np.where(matched.variant == "T", matched.temporal_prior, matched.full_prior)
        matched["native_rans"] = np.where(matched.variant == "T", matched.temporal_rans, matched.full_rans)
        fields = ["pre_rate", "temporal_prior", "full_prior", "temporal_rans", "full_rans", "native_prior", "native_rans", "wmape"]
        summary = matched.groupby(["phase", "variant", "prior_mode", "stage"])[fields].mean().reset_index()
        summary["wmape_db"] = wmape_to_db(summary.wmape.to_numpy())
        summary.to_csv(output / "summary_stage_points.csv", index=False)
        summary_gaps = []
        for (phase, mode), group in summary.groupby(["phase", "prior_mode"]):
            reference = group[group.variant == "T"].sort_values("stage")
            for variant, candidate in group.groupby("variant"):
                if variant == "T":
                    continue
                candidate = candidate.sort_values("stage")
                for source in fields[:-1]:
                    if candidate[source].isna().any() or reference[source].isna().any():
                        continue
                    summary_gaps.append({"phase": phase, "prior_mode": mode, "variant": variant, "source": source} |
                                        rd_curve_gap(candidate[source].to_numpy(), candidate.wmape_db.to_numpy(),
                                                     reference[source].to_numpy(), reference.wmape_db.to_numpy()))
        pd.DataFrame(summary_gaps).to_csv(output / "summary_comparison_gaps.csv", index=False)
        report = ["# Controlled sampled-trace diagnostic (incomplete until complete.json)", "",
                  f"Matched completed client/seed jobs: {list(matched_jobs)}.", "",
                  "Same compressed raw SI history; smaller optimizer-update budget than full-vector runs.",
                  "2048-lane rANS on short samples is flush-dominated: use priors to diagnose training, not sample rANS savings to conclude protocol superiority.",
                  "Preserve rows report the original joint prior, not a fine-tuned prior. Alternate-SI rates are unavailable there.", "",
                  "| Phase | Model | Prior fit | Stage | Pre-fit | Temporal prior | Full prior | Temporal rANS | Full rANS | WMAPE dB (raw %) |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for row in summary.to_dict("records"):
            rates = " | ".join("—" if pd.isna(row[field]) else f"{row[field]:.4f}" for field in fields[:5])
            report.append(f"| {row['phase']} | {row['variant']} | {row['prior_mode']} | {row['stage']} | {rates} | {row['wmape_db']:.3f} ({row['wmape']:.3f}%) |")
        report += ["", "Equal-distortion gaps vs T: positive means worse. All available rate-source gaps are in summary_comparison_gaps.csv.", "",
                   "| Phase | Model | Prior fit | Source | Gap bits/float | Gap % |", "|---|---|---|---|---:|---:|"]
        for row in summary_gaps:
            if row["source"] not in ("native_prior", "full_prior", "native_rans", "full_rans"):
                continue
            report.append(f"| {row['phase']} | {row['variant']} | {row['prior_mode']} | {row['source']} | {row['rate_gap_bits']:+.5f} | {row['rate_gap_percent']:+.3f}% |")
        (output / "report.md").write_text("\n".join(report) + "\n")
    gaps = []
    keys = ["client", "seed", "round", "heldout", "prior_mode"]
    for key, group in records.groupby(keys):
        ref = group[group.variant == "T"].sort_values("stage")
        if len(ref) != 3:
            continue
        for variant, candidate in group.groupby("variant"):
            if variant == "T":
                continue
            candidate = candidate.sort_values("stage")
            for source in ("pre_rate", "temporal_prior", "full_prior", "temporal_rans", "full_rans"):
                if source not in candidate or candidate[source].isna().any() or ref[source].isna().any():
                    continue
                gaps.append(dict(zip(keys, key, strict=True)) | {"variant": variant, "source": source} | rd_curve_gap(
                    candidate[source].to_numpy(), candidate.wmape_db.to_numpy(), ref[source].to_numpy(), ref.wmape_db.to_numpy()))
    pd.DataFrame(gaps).to_csv(output / "comparison_gaps.csv", index=False)
    # Selection uses only held-out coordinates at rounds 5/6, never future rounds.
    selection = records[(records.heldout) & (records["round"].isin((5, 6))) & (records.prior_mode == "preserve")].copy()
    selection["native_prior"] = np.where(selection.variant == "T", selection.temporal_prior, selection.full_prior)
    selection["validation_rd"] = selection.hard_rd
    scores = selection.groupby(["client", "variant", "seed"])[["training_loss", "validation_rd"]].mean().reset_index()
    scores.to_csv(output / "selection_scores.csv", index=False)
    selected = []
    for (client, variant), group in scores.groupby(["client", "variant"]):
        if len(group) != len(SEEDS):
            continue
        for selector in ("training_loss", "validation_rd"):
            seed = group.loc[group[selector].idxmin(), "seed"]
            chosen = records[(records.client == client) & (records.variant == variant) & (records.seed == seed)].copy()
            chosen["selector"] = selector
            selected.append(chosen)
    if selected:
        deployed = pd.concat(selected, ignore_index=True)
        deployed.to_csv(output / "selected_stage_points.csv", index=False)
        selected_gaps = []
        for key, group in deployed.groupby(["client", "selector", "round", "heldout", "prior_mode"]):
            ref = group[group.variant == "T"].sort_values("stage")
            if len(ref) != 3:
                continue
            for variant, candidate in group.groupby("variant"):
                if variant == "T":
                    continue
                candidate = candidate.sort_values("stage")
                for source in ("pre_rate", "temporal_prior", "full_prior", "temporal_rans", "full_rans"):
                    if candidate[source].isna().any() or ref[source].isna().any():
                        continue
                    selected_gaps.append(dict(zip(("client", "selector", "round", "heldout", "prior_mode"), key, strict=True)) |
                                         {"variant": variant, "source": source} | rd_curve_gap(
                                             candidate[source].to_numpy(), candidate.wmape_db.to_numpy(), ref[source].to_numpy(), ref.wmape_db.to_numpy()))
        pd.DataFrame(selected_gaps).to_csv(output / "selected_comparison_gaps.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for (variant, mode), group in records[(~records.heldout) & (records["round"] > 6)].groupby(["variant", "prior_mode"]):
        curve = group.groupby("stage")[["wmape", "full_rans"]].mean()
        if curve.full_rans.isna().all():
            continue
        axis.plot(wmape_to_db(curve.wmape.to_numpy()), curve.full_rans, "o-", label=f"{variant}/{mode}")
    axis.set(xlabel="WMAPE (dB)", ylabel="Full-SI rANS bits/float (sample flush included)")
    axis.legend(fontsize=7)
    figure.tight_layout()
    temporary = output / f"rd.{os.getpid()}.tmp.png"
    figure.savefig(temporary)
    temporary.replace(output / "rd.png")
    plt.close(figure)


def run(plan_path: Path) -> None:
    """Retain three seeds for each model, compare prior fits, and evaluate unseen rounds."""
    plan = load_plan(plan_path)
    torch.cuda.set_device(plan.gpu_ids[0])
    output = plan.output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "attempts").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    manifest = {
        "scope": "sampled diagnostic, common compressed RAW history; not exact warm-run reproduction",
        "train_rows_per_round": TRAIN_ROWS, "heldout_rows_per_round": VALID_ROWS,
        "clients": CLIENTS, "rounds": ROUNDS, "seeds": SEEDS, "train_rounds": [6, 5],
        "trace_device": plan.trace_device, "sample_seed": 1043,
        "selection": "historical heldout coordinates only; future rounds evaluation only",
        "cfg": WZcfgQuant(bins_per_plane=2, num_planes=3).model_dump(mode="json"),
        "embedding": "zero extra SI weights; O_from_T uses default warm schedule (35 epochs)",
        "prior_update_control": "repeat current-only rows twice; equal draws and optimizer steps to two-round fit",
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == json.loads(json.dumps(manifest)), f"Diagnostic identity mismatch: {manifest_path}"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    problems = torch.load(output / "problems.pt", weights_only=True) if (output / "problems.pt").exists() else collect_problems(plan_path, output)
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3)
    for client in CLIENTS:
        training = [problems[client, round_id] for round_id in (6, 5)]
        for seed in SEEDS:
            temporal_model = None
            for variant in ("T", "O", "O_from_T"):
                stem = f"client{client}_{variant}_seed{seed}"
                model_file = output / "models" / f"{stem}.pt"
                if (output / "attempts" / f"{stem}.json").exists():
                    if variant == "T":
                        temporal_model = new_rnn_model(3, 2, 5, False, 128, 4)
                        temporal_model.load_state_dict(torch.load(model_file, weights_only=True)["state"])
                    continue
                temporal = variant == "T"
                initial = None
                if variant == "O_from_T":
                    assert temporal_model is not None
                    initial = embed_temporal(temporal_model, training[0]["columns"], training[0]["si"].shape[1])
                    with torch.no_grad():
                        probe = training[0]
                        t_outputs = temporal_model.cpu().eval()(probe["x"][:64], probe["si"][:64, probe["columns"]])
                        o_outputs = initial(probe["x"][:64], probe["si"][:64])
                        assert all(torch.allclose(a, b, atol=2e-6, rtol=2e-6)
                                   for left, right in zip(t_outputs, o_outputs, strict=True)
                                   for a, b in zip(left, right, strict=True)), "T-in-O embedding is not equivalent"
                training_cfg = cfg.warm_retraining_config() if initial is not None else cfg
                print(f"BEGIN {stem}, epochs={training_cfg.train_epochs}", flush=True)
                if model_file.exists():
                    saved = torch.load(model_file, weights_only=True)
                    joint = new_rnn_model(3, 2, 5 if temporal else 150, False, 128, 4)
                    joint.load_state_dict(saved["state"])
                    loss = saved["loss"]
                else:
                    joint, loss = fit_quantizer(training_cfg, training, temporal, seed, initial)
                    torch.save({"state": joint.state_dict(), "loss": loss}, model_file)
                if temporal:
                    temporal_model = deepcopy(joint)
                rows = []
                for mode in ("preserve", "current", "multi"):
                    priors = {"temporal" if temporal else "full": deepcopy(joint)} if mode == "preserve" else {
                        source: fit_prior(cfg, joint, training[:1] if mode == "current" else training, source == "temporal", temporal, seed + 1000)
                        for source in ("temporal", "full")
                    }
                    for source, prior in priors.items():
                        torch.save(prior.state_dict(), output / "models" / f"{stem}_{mode}_{source}.pt")
                    for round_id in ROUNDS:
                        for heldout in ((True,) if round_id in (5, 6) else (False,)):
                            stage_rows = evaluate(joint, problems[client, round_id], temporal, priors, heldout)
                            denominator = torch.cat([p["x"][:TRAIN_ROWS] for p in training]).square().mean().item() / 2 + 1e-8
                            previous_rate = 0.0
                            for row in stage_rows:
                                hard_rd = cfg.reconst_ld * row["hard_mse"] / denominator + (row["pre_rate"] - previous_rate) * np.log(2)
                                previous_rate = row["pre_rate"]
                                rows.append({"client": client, "seed": seed, "variant": variant, "prior_mode": mode,
                                             "round": round_id, "heldout": heldout, "training_loss": loss,
                                             "hard_rd": hard_rd, **row})
                temporary = output / "attempts" / f"{stem}.tmp"
                temporary.write_text(json.dumps({"rows": rows}))
                temporary.replace(output / "attempts" / f"{stem}.json")
                print(f"DONE {stem}: {len(rows)} stage rows", flush=True)
                torch.cuda.empty_cache()
    (output / "complete.json").write_text(json.dumps({"attempts": len(CLIENTS) * len(SEEDS) * 3}))


def main() -> None:
    """Run under detached_replay, or act as its change-triggered live diagnostic plotter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render(args.path)
        return
    if args.runner_pid is None:
        run(args.path)
        return
    previous = None
    while True:
        signature = tuple((p.name, p.stat().st_mtime_ns) for p in sorted((args.path / "attempts").glob("*.json")))
        if signature != previous:
            try:
                result = subprocess.run((sys.executable, "-m", "Experiments.Protocol_test.training_diagnostic", str(args.path), "--render"), timeout=600)
                if result.returncode == 0:
                    previous = signature
            except subprocess.TimeoutExpired:
                print("Diagnostic plot refresh timed out; retaining last good figure", flush=True)
        try:
            os.kill(args.runner_pid, 0)
        except ProcessLookupError:
            return
        sleep(30)


if __name__ == "__main__":
    main()
