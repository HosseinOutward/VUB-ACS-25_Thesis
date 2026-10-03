"""Full-budget O/T confirmation on a shared decoded TMR warm-pool trajectory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch

from Experiments.Protocol_test.experimental_codecs import (
    MultiRoundWarmWZCancerProtocol, OM_RoundCodec, TMR_RoundCodec, _MultiRoundQuantizer,
)
from Experiments.Protocol_test.rd_metrics import rd_curve_gap
from Experiments.Protocol_test.sparse_replay import (
    _codec_seed, _provenance, case_protocol, load_fl_config, load_plan,
)
from FL_code.FL_core.codec import Access, BaseRoundCodec, CompressionRecord, record_reconstruction_metrics, simulate_compression
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.bin_coder.rans_coder import RansConfig, decode, encode
from FL_code.cancer_protocol.brent_wz_models import EncoderDecoderLayeredRNN
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.prior_code import INFERENCE_BATCH, PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant, new_rnn_model


CLIENTS = (0, 5, 11, 21)
TRAIN_ROUND = 7
VALIDATION_ROUND = 8
LAST_ROUND = 16
RETRY_FAILURE_TOLERANCE = 6


class RetryConfirmationProtocol(MultiRoundWarmWZCancerProtocol):
    """Experiment-only extra quantizer retries without changing accepted quotas or production defaults."""

    protocol_name: ClassVar[str] = "wz_cancer_confirmation"

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Give both cold and warm baseline quantizers six rejected-attempt slots instead of two."""
        codec = super().create_round_codec(round_id, client_id)
        if isinstance(codec, TMR_RoundCodec):
            codec.quantizer.candidate_failure_tolerance = RETRY_FAILURE_TOLERANCE
        return codec

    def config_snapshot(self) -> dict[str, Any]:
        """Identify the changed retry budget independently from the acceptance rule and optimizer."""
        snapshot = super().config_snapshot()
        snapshot["quantizer_retry_failure_tolerance"] = RETRY_FAILURE_TOLERANCE
        return snapshot


@dataclass
class Candidate:
    """CPU joint model and fixed-symbol priors produced at round 7 and frozen thereafter."""

    joint: EncoderDecoderLayeredRNN
    priors: dict[str, EncoderDecoderLayeredRNN]
    temporal: bool


def save_candidates(candidates: dict[str, Candidate], path: Path) -> None:
    """Atomically retain tensor-only checkpoints that remain loadable outside the worker module."""
    payload = {name: {"temporal": candidate.temporal, "joint": candidate.joint.state_dict(),
                      "priors": {policy: model.state_dict() for policy, model in candidate.priors.items()}}
               for name, candidate in candidates.items()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def embed_temporal(model: EncoderDecoderLayeredRNN, columns: torch.Tensor, width: int) -> EncoderDecoderLayeredRNN:
    """Embed every temporal encoder, decoder, and prior weight exactly, zeroing extra SI inputs."""
    oracle = new_rnn_model(model.num_planes, model.bins_per_plane, width, False,
                          model.decoder.hidden_dim, model.decoder.layers, model.shared_encoder)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    bins = model.bins_per_plane
    for key, offset in (("decoder.input_layers.0.weight", bins), ("conditionalRNN.input_layers.0.weight", 0)):
        old = state[key]
        expanded = old.new_zeros((old.shape[0], width + bins))
        expanded[:, columns + offset] = old[:, offset:offset + columns.numel()]
        if offset:
            expanded[:, :bins] = old[:, :bins]
        else:
            expanded[:, width:] = old[:, columns.numel():]
        state[key] = expanded
    oracle.load_state_dict(state)
    return oracle.eval()


def train_oracle(
    cfg: WZcfgQuant, target: torch.Tensor, full_si: Sequence[HistoryVector],
    historical: tuple[tuple[torch.Tensor, Sequence[HistoryVector]], ...],
    initial: EncoderDecoderLayeredRNN | None, seed: int,
) -> Candidate:
    """Train production-repeat cold or embedded-T warm O, retaining its original joint prior."""
    set_global_seed(seed)
    codec = OM_RoundCodec({"bins_per_plane": cfg.bins_per_plane, "num_planes": cfg.num_planes,
                          "sd_slices": cfg.norm_slices}, "OM")
    codec.c_cfg = cfg.model_copy(update={"warm_training": initial is not None})
    codec.historical_problems = historical
    codec.historical_server_si = tuple(si for _, si in historical)
    codec.problem_weights = (.5, .3, .2)
    codec.si = list(full_si)
    codec.quantizer = _MultiRoundQuantizer(c_cfg=codec.c_cfg, si_size=len(full_si))
    codec.quantizer.candidate_failure_tolerance = RETRY_FAILURE_TOLERANCE
    if initial is not None:
        codec.quantizer.load_warm_start_states((initial.state_dict(),))
    codec._send_quantizer_state = False
    codec.pre_encode_training_steps(target)
    joint = deepcopy(codec.quantizer.coding_model).cpu().eval()
    bins, codes, metadata = codec.quantizer.encoding_process(target)
    fine = codec.fit_prior_model(bins.long(), codes, codec.quantizer.side_info_tensor(metadata),
        Access.SERVER_ONLY, deepcopy(joint), (None,), cfg.prior_train_repeats).model.cpu().eval()
    codec.release_round_resources()
    return Candidate(joint, {"joint": deepcopy(joint), "fine": fine}, False)


def fit_embedded_prior(
    cfg: WZcfgQuant, embedded: EncoderDecoderLayeredRNN, target: torch.Tensor,
    full_si: Sequence[HistoryVector], historical: tuple[tuple[torch.Tensor, Sequence[HistoryVector]], ...],
) -> EncoderDecoderLayeredRNN:
    """Fine-tune the embedded prior on three correctly paired final-encoder symbol blocks."""
    codec = OM_RoundCodec({"bins_per_plane": cfg.bins_per_plane, "num_planes": cfg.num_planes,
                          "sd_slices": cfg.norm_slices}, "OM")
    codec.c_cfg = cfg.model_copy(update={"warm_training": False})
    codec.historical_problems = historical
    codec.historical_server_si = tuple(si for _, si in historical)
    codec.problem_weights = (.5, .3, .2)
    codec.quantizer = _MultiRoundQuantizer(c_cfg=codec.c_cfg, si_size=len(full_si))
    codec.quantizer.coding_model = deepcopy(embedded)
    codec.quantizer.set_side_information(full_si)
    codec.historical_encodings = tuple(codec.quantizer.encoding_process(raw) for raw, _ in historical)
    bins, codes, metadata = codec.quantizer.encoding_process(target)
    result = codec.fit_prior_model(bins.long(), codes, codec.quantizer.side_info_tensor(metadata),
        Access.SERVER_ONLY, deepcopy(embedded), (None,), cfg.prior_train_repeats)
    codec.release_round_resources()
    return result.model.cpu().eval()


def evaluate(
    candidate: Candidate, cfg: WZcfgQuant, raw: torch.Tensor,
    full_si: Sequence[HistoryVector], temporal_columns: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    """Evaluate full-vector hard RD and both legitimate native/full-SI rANS sources."""
    native_si = [full_si[int(i)] for i in temporal_columns] if candidate.temporal else list(full_si)
    quantizer = DedupedWZQuantizerCancer(c_cfg=cfg, si_size=len(native_si))
    quantizer.coding_model = candidate.joint
    quantizer.set_side_information(native_si)
    bins, _, metadata = quantizer.encoding_process(raw)
    si = quantizer.side_info_tensor(metadata)
    stages = quantizer.decoding_stages_process((bins, metadata))
    distortions = [record_reconstruction_metrics(raw, stage)["wmape"] for stage in stages]
    normalized, _ = quantizer.preprocess_x(raw)
    normalized_stages = quantizer.decode_stages_prepared(bins, si, INFERENCE_BATCH)
    mse = (normalized_stages - normalized.cpu().flatten()).square().mean(dim=1).numpy()
    denominator = normalized.square().mean().item() / 2 + 1e-8
    pre = np.cumsum(PriorCalculator.compute_plane_rates(
        PriorCalculator.compute_prior_from_network(candidate.joint, bins, si), bins, cfg.num_planes))
    rows = []
    for policy, model in candidate.priors.items():
        access = "full" if policy == "full_check" or not candidate.temporal else "temporal"
        prior_si = quantizer.normalize_side_info_tensor(metadata, full_si) if access == "full" else si
        probabilities = PriorCalculator.compute_prior_from_network(model, bins, prior_si)
        plane_rates = PriorCalculator.compute_plane_rates(probabilities, bins, cfg.num_planes)
        rates = np.cumsum(plane_rates)

        def prior(context: torch.Tensor, stage: int) -> torch.Tensor:
            # Derive probabilities from decoded context, never the unavailable future symbols.
            padded = torch.zeros_like(bins)
            padded[:context.shape[0]] = context.cpu()
            return PriorCalculator.compute_prior_from_network(model, padded, prior_si)[stage].cuda()

        payloads = [encode(bins[:stage].cuda(), prior, RansConfig(lanes=2048)) for stage in (1, 2, 3)]
        assert torch.equal(decode(payloads[-1], prior).cpu(), bins), "Confirmation rANS symbol mismatch"
        rans_rates = [8 * len(payload) / bins.shape[1] for payload in payloads]
        for stage in range(3):
            rows.append({"policy": policy, "access": access, "stage": stage + 1,
                "pre_rate": float(pre[stage]), "prior_rate": float(rates[stage]),
                "prior_stage_rate": float(plane_rates[stage]),
                "rans_rate": rans_rates[stage],
                "rans_stage_rate": rans_rates[stage] - (rans_rates[stage - 1] if stage else 0),
                "wmape": distortions[stage],
                "hard_rd": float((cfg.reconst_ld * mse / denominator + np.array(plane_rates) * np.log(2)).mean())})
    return rows


def select_candidates(rows: list[dict[str, object]]) -> dict[str, object]:
    """Select prior per fixed encoder, then quantizer by chronological-validation hard RD."""
    assert rows and all(row["phase"] == "validation" and row["round"] == VALIDATION_ROUND for row in rows), (
        "Candidate selection may only consume chronological-validation round 8.")
    eligible = [row for row in rows if row["policy"] != "full_check" and row["stage"] == 3]
    best_prior = {variant: min((row for row in eligible if row["variant"] == variant), key=lambda row: row["prior_rate"])
                  for variant in {str(row["variant"]) for row in eligible}}
    oracle = min((row for variant, row in best_prior.items() if variant != "T"), key=lambda row: row["hard_rd"])
    return {"oracle_variant": oracle["variant"], "oracle_policy": oracle["policy"],
            "temporal_policy": best_prior["T"]["policy"],
            "prior_choices": {variant: row["policy"] for variant, row in best_prior.items()}}


def run(plan_path: Path) -> None:
    """Replay all 30 TMR clients serially and freeze confirmation choices before future evaluation."""
    plan = load_plan(plan_path)
    assert len(plan.cases) == 1 and plan.cases[0].start_round == TRAIN_ROUND
    torch.cuda.set_device(plan.gpu_ids[0])
    output = plan.output_root
    assert not output.exists(), f"Confirmation is not resumable; preserve partial output and use a fresh root: {output}"
    output.mkdir(parents=True)
    for directory in ("attempts", "models"):
        (output / directory).mkdir()
    fl_cfg = load_fl_config(plan.fl_config)
    manager = StateDictManager(initialize_model(fl_cfg, torch.device("cpu")))
    protocol = case_protocol(plan, plan.cases[0], manager)
    assert isinstance(protocol, RetryConfirmationProtocol), "Use full_vector_confirmation_retry8.json for the revised retry budget."
    assert not protocol.refit_priors_on_frozen
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3, norm_slices=manager.get_slices())
    manifest = {"clients": CLIENTS, "train_rounds": [7, 6, 5], "weights": [.5, .3, .2],
        "validation_round": VALIDATION_ROUND, "future_rounds": list(range(9, 17)),
        "history": "shared decoded TMR trajectory, all 30 clients, real five-slot/two-parent warm pool",
        "scope": "full-vector mechanism confirmation, not an independent O protocol trajectory",
        "selection": "prior NLL per encoder, then hard normalized MSE plus native NLL; validation round8 only",
        "candidate_seeds": "baseline _codec_seed; embedded prior +500; O-from-T +1000; cold O +2000",
        "quantizer_retry_failure_tolerance": RETRY_FAILURE_TOLERANCE,
        "cfg": cfg.model_dump(mode="json", exclude={"norm_slices"}),
        "norm_slices": [(part.start, part.stop, part.step) for part in manager.get_slices()],
        "protocol": protocol.config_snapshot(),
        "provenance": _provenance(plan, plan_path)}
    trace = Trace(fl_cfg, dataset_revision=plan.dataset_revision, device=torch.device(plan.trace_device))
    manifest["trace_identity"] = trace.identity()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    candidates: dict[int, dict[str, Candidate]] = {}
    choices: dict[int, dict[str, object]] = {}
    baseline = output / "cases/TMR/start7"
    baseline.mkdir(parents=True)
    for round_id, deltas in trace.stream(0, LAST_ROUND):
        print(f"TRACE round {round_id}", flush=True)
        for client, raw in enumerate(deltas):
            protocol.remember_target(raw, round_id, client)
            if round_id < TRAIN_ROUND:
                for history in protocol.histories:
                    history.commit(raw, CompressionRecord(round_id, client, "raw_prefix"), Access.TEMPORAL_TOO)
                continue
            full_si = [entry.value for entries in protocol._recons_history.view(Access.SERVER_ONLY, client).values() for entry in entries]
            temporal_si = [entry.value for entries in protocol._recons_history.view(Access.TEMPORAL_TOO, client).values() for entry in entries]
            columns = torch.tensor([next(i for i, value in enumerate(full_si) if value is own) for own in temporal_si])
            historical = ()
            if round_id == TRAIN_ROUND and client in CLIENTS:
                historical = tuple((problem.target.materialize(), tuple(entry.value
                    for entries in problem.history.view(Access.SERVER_ONLY, client).values() for entry in entries))
                    for problem in (protocol._problems[client][6], protocol._problems[client][5]))
            seed = _codec_seed(plan.seed, TRAIN_ROUND, round_id, client)
            set_global_seed(seed)
            simulate_compression(protocol, raw, client, round_id, manager, baseline, {}, (), ())
            if client not in CLIENTS:
                continue
            codec = protocol.last_trained_codec[client]
            assert isinstance(codec, TMR_RoundCodec)
            if round_id == TRAIN_ROUND:
                joint = new_rnn_model(3, 2, 5, False, cfg.hidden_dim, cfg.layers, cfg.shared_heads)
                joint.load_state_dict(codec.joint_quantizer_state)
                embedded = embed_temporal(joint, columns, len(full_si))
                probe = _MultiRoundQuantizer(c_cfg=cfg, si_size=5)
                normalized, metadata = probe.preprocess_x(raw)
                full_matrix = probe.normalize_side_info_tensor(metadata, full_si)
                with torch.no_grad():
                    x, y = normalized[:64].cpu(), full_matrix[:64].cpu()
                    t, o = joint.eval()(x, y[:, columns]), embedded(x, y)
                    assert all(torch.allclose(a, b, atol=2e-6, rtol=2e-6)
                               for left, right in zip(t, o, strict=True) for a, b in zip(left, right, strict=True)), "T-in-O is not exact"
                del probe, normalized, full_matrix
                fine_temporal = deepcopy(codec.quantizer.coding_model).cpu().eval()
                candidates[client] = {
                    "T": Candidate(joint, {"joint": deepcopy(joint), "fine": fine_temporal,
                        "full_check": deepcopy(codec._server_si_prior_model).cpu().eval()}, True),
                    "T_in_O": Candidate(embedded, {"joint": deepcopy(embedded),
                        "temporal_fine": embed_temporal(fine_temporal, columns, len(full_si))}, False),
                }
                save_candidates(candidates[client], output / "models" / f"client{client}.pt")
                set_global_seed(seed + 500)
                candidates[client]["T_in_O"].priors["fine"] = fit_embedded_prior(cfg, embedded, raw, full_si, historical)
                save_candidates(candidates[client], output / "models" / f"client{client}.pt")
                for variant, initial, offset in (("O_from_T", embedded, 1000), ("O_cold", None, 2000)):
                    candidates[client][variant] = train_oracle(cfg, raw, full_si, historical, initial, seed + offset)
                    save_candidates(candidates[client], output / "models" / f"client{client}.pt")
                print(f"CANDIDATES complete client={client}", flush=True)
                continue
            rows = []
            for variant, candidate in candidates[client].items():
                measured = evaluate(candidate, cfg, raw, full_si, columns)
                rows.extend({"client": client, "round": round_id, "variant": variant,
                             "phase": "validation" if round_id == VALIDATION_ROUND else "future", **row} for row in measured)
            if round_id == VALIDATION_ROUND:
                choices[client] = select_candidates(rows)
                (output / "models" / f"selection_client{client}.json").write_text(json.dumps(choices[client], indent=2))
            for row in rows:
                choice = choices[client]
                row["selected"] = ((row["variant"] == "T" and row["policy"] == choice["temporal_policy"])
                    or (row["variant"] == choice["oracle_variant"] and row["policy"] == choice["oracle_policy"]))
            temporary = output / "attempts" / f"round{round_id}_client{client}.tmp"
            temporary.write_text(json.dumps(rows))
            temporary.replace(temporary.with_suffix(".json"))
            print(f"EVALUATED round={round_id} client={client} rows={len(rows)}", flush=True)
            torch.cuda.empty_cache()
    (output / "complete.json").write_text(json.dumps({"validation_clients": len(choices), "future_rounds": 8}))


def render(output: Path) -> None:
    """Publish every frozen candidate, selected curves, stage failures, and equal-distortion gaps."""
    paths = sorted((output / "attempts").glob("*.json"))
    if not paths:
        return
    data = pd.DataFrame([row for path in paths for row in json.loads(path.read_text())])
    data["wmape_db"] = 20 * np.log10(data.wmape / 100)
    data.to_csv(output / "stage_points.csv", index=False)
    data[(data.prior_stage_rate < .01) | (data.prior_stage_rate > 1.05)
         | (data.rans_stage_rate < .01) | (data.rans_stage_rate > 1.05)].to_csv(output / "suspect_stages.csv", index=False)
    summary = data.groupby(["phase", "variant", "policy", "access", "stage"])[["pre_rate", "prior_rate", "rans_rate", "wmape"]].mean().reset_index()
    summary["wmape_db"] = 20 * np.log10(summary.wmape / 100)
    summary.to_csv(output / "summary_stage_points.csv", index=False)
    gaps = []
    for phase, group in summary.groupby("phase"):
        ref = group[group.variant.eq("T") & group.policy.eq("fine")].sort_values("stage")
        for (variant, policy), curve in group.groupby(["variant", "policy"]):
            curve = curve.sort_values("stage")
            for source in ("pre_rate", "prior_rate", "rans_rate"):
                gaps.append({"phase": phase, "variant": variant, "policy": policy, "source": source,
                    **rd_curve_gap(curve[source].to_numpy(), curve.wmape_db.to_numpy(), ref[source].to_numpy(), ref.wmape_db.to_numpy())})
    selected = data[data.selected].copy()
    selected["method"] = np.where(selected.variant.eq("T"), "selected_T", "selected_O")
    selected.to_csv(output / "selected_stage_points.csv", index=False)
    for phase, group in selected.groupby("phase"):
        curves = group.groupby(["method", "stage"])[["pre_rate", "prior_rate", "rans_rate", "wmape"]].mean().reset_index()
        a, b = (curves[curves.method.eq(label)].sort_values("stage") for label in ("selected_O", "selected_T"))
        for source in ("pre_rate", "prior_rate", "rans_rate"):
            gaps.append({"phase": phase, "variant": "selected_O_vs_selected_T", "policy": "chronological_validation", "source": source,
                **rd_curve_gap(a[source].to_numpy(), 20 * np.log10(a.wmape.to_numpy() / 100),
                              b[source].to_numpy(), 20 * np.log10(b.wmape.to_numpy() / 100))})
    pd.DataFrame(gaps).to_csv(output / "comparison_gaps.csv", index=False)
    lines = ["# Full-vector confirmation (incomplete until complete.json)", "",
        "Production full-vector training and update budget; common decoded TMR warm-pool history.",
        "Round8 is chronological validation; rounds9–16 are untouched frozen evaluation.",
        "Joint priors and exact T-in-O candidates remain explicit alternatives. No every-round fine-tuning.",
        "Rates are cumulative bits/float; WMAPE is averaged in linear space before conversion to dB.", "",
        "| Phase | Model | Prior policy | Stage | Pre-fit | Temporal prior | Full-SI prior | Temporal rANS | Full-SI rANS | WMAPE dB (raw %) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary[~summary.policy.eq("full_check")].to_dict("records"):
        full = row
        if row["variant"] == "T":
            check = summary[summary.phase.eq(row["phase"]) & summary.variant.eq("T")
                            & summary.policy.eq("full_check") & summary.stage.eq(row["stage"])]
            full = check.iloc[0].to_dict()
        temporal = row if row["access"] == "temporal" else None
        values = [row["pre_rate"], temporal["prior_rate"] if temporal else np.nan,
                  full["prior_rate"], temporal["rans_rate"] if temporal else np.nan, full["rans_rate"]]
        rates = " | ".join("—" if np.isnan(value) else f"{value:.4f}" for value in values)
        lines.append(f"| {row['phase']} | {row['variant']} | {row['policy']} | {row['stage']} | {rates} | {row['wmape_db']:.3f} ({row['wmape']:.3f}%) |")
    lines += ["", "Equal-distortion gaps relative to T's fine-tuned temporal transport; negative favors the candidate.",
        "Selected_O_vs_selected_T instead compares both chronological selections. Full-SI T checks are listed separately above.", "",
        "| Phase | Model | Prior policy | Source | Gap bits/float | Gap % |", "|---|---|---|---|---:|---:|"]
    for row in gaps:
        lines.append(f"| {row['phase']} | {row['variant']} | {row['policy']} | {row['source']} | {row['rate_gap_bits']:+.5f} | {row['rate_gap_percent']:+.2f}% |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for (variant, policy), curve in summary[summary.phase.eq("future")].groupby(["variant", "policy"]):
        axis.plot(curve.wmape_db, curve.rans_rate, "o-", label=f"{variant}/{policy}")
    axis.set(xlabel="WMAPE dB", ylabel="Full-vector rANS bits/float")
    axis.legend(fontsize=6)
    figure.tight_layout()
    temporary = output / f"rd.{os.getpid()}.tmp.png"
    figure.savefig(temporary)
    temporary.replace(output / "rd.png")
    plt.close(figure)


def main() -> None:
    """Run with the detached supervisor or refresh figures in disposable watcher processes."""
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
                result = subprocess.run((sys.executable, "-m", "Experiments.Protocol_test.full_vector_confirmation", str(args.path), "--render"), timeout=600)
                if result.returncode == 0:
                    previous = signature
            except subprocess.TimeoutExpired:
                print("Confirmation plot refresh timed out; retaining last good figure", flush=True)
        try:
            os.kill(args.runner_pid, 0)
        except ProcessLookupError:
            return
        sleep(30)


if __name__ == "__main__":
    main()
