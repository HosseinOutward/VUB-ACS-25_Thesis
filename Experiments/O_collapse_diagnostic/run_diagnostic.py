"""Train single cold O-style quantizer attempts under different side-information sets and log stage collapse."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch

from Experiments.Protocol_test.sparse_replay import load_fl_config, load_plan
from FL_code.FL_core.codec import Access, CompressionRecord, record_reconstruction_metrics
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.trace import Trace
from FL_code.FL_core.utils import StateDictManager, set_global_seed
from FL_code.cancer_protocol.history import CompressedReconstructionHistory, HistoryVector
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, WZcfgQuant

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAN = _REPO_ROOT / "Experiments/Protocol_test/plans/or_svhn_30c_f9_warm.json"
_EPOCH_LINE = re.compile(
    r"^(.*) epoch (\d+)/\d+: loss=([\d.eE+-]+).*hard_mse=\[([^\]]*)\], cumulative_rate=\[([^\]]*)\]",
    re.MULTILINE)


def side_information_sets(
    before_round: CompressedReconstructionHistory, at_turn: CompressedReconstructionHistory, client_id: int,
) -> dict[str, list[HistoryVector]]:
    """Return the compared SI sets for one client at its turn in the trained round.

    `own5` is T's view; `server_prev` is 150 vectors from earlier rounds only; `server_actual` is
    O's real view, where earlier clients' same-round updates replace their oldest round;
    `own5_same_round` adds only those same-round updates to T's view.
    """
    def values(history: CompressedReconstructionHistory, access: Access) -> list[HistoryVector]:
        return [entry.value for entries in history.view(access, client_id).values() for entry in entries]

    same_round = [entries[-1].value for other, entries in at_turn.view(Access.SERVER_ONLY, client_id).items()
                  if other < client_id]
    return {
        "own5": values(at_turn, Access.TEMPORAL_TOO),
        "own5_same_round": values(at_turn, Access.TEMPORAL_TOO) + same_round,
        "server_prev": values(before_round, Access.SERVER_ONLY),
        "server_actual": values(at_turn, Access.SERVER_ONLY),
    }


def round_problems(
    start_round: int, clients: Sequence[int],
) -> tuple[dict[int, tuple[torch.Tensor, dict[str, list[HistoryVector]]]], list[slice]]:
    """Regenerate the trace up to `start_round`; return each client's target and SI sets, plus norm slices."""
    plan = load_plan(_PLAN)
    trace = Trace(load_fl_config(plan.fl_config), dataset_revision=plan.dataset_revision,
                  device=torch.device(plan.trace_device))
    slices = StateDictManager(initialize_model(trace.cfg, torch.device("cpu"))).get_slices()
    history = CompressedReconstructionHistory(5, slices)
    problems = {}
    for round_id, deltas in trace.stream(start_round - 5, start_round):
        if round_id == start_round:
            before_round = history.copy()
        for client_id, delta in enumerate(deltas):
            if round_id == start_round and client_id in clients:
                problems[client_id] = (
                    delta.clone(), side_information_sets(before_round, history.copy(), client_id))
            # Earlier clients' raw same-round deltas stand in for the decoded ones the protocol would hold.
            history.commit(delta, CompressionRecord(round_id, client_id, "raw_history"), Access.TEMPORAL_TOO)
    return problems, slices


def train_attempt(
    target: torch.Tensor, side_information: list[HistoryVector], cfg: WZcfgQuant, seed: int,
    initial_states: Sequence[Mapping[str, torch.Tensor]] = (),
) -> tuple[dict[str, object], list[dict[str, float]], dict[str, torch.Tensor]]:
    """Train one attempt, returning its full-vector stage metrics, logged training curve, and joint state.

    `initial_states` warm-starts one lineage per parent, as the warm protocol does, and requires
    `cfg.warm_training`. `fine_tuned_stage_rates` repeats the codec's native-prior fine-tune on
    the encoded symbols (`_WZRoundCodec.get_encod_payload`), so it is the rate the transport
    actually codes against. The joint state is taken before that fine-tune, as the warm pool is.
    """
    set_global_seed(seed)
    quantizer = DedupedWZQuantizerCancer(c_cfg=cfg, si_size=len(side_information))
    if initial_states:
        quantizer.load_warm_start_states(initial_states)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        quantizer.train_model(target, side_information)
        joint_state = {name: value.detach().cpu().clone() for name, value in quantizer.coding_model.state_dict().items()}
        bins, soft_codes, metadata = quantizer.encoding_process(target)
        stages = quantizer.decoding_stages_process((bins, metadata))
        rates = list(quantizer.quantizer_prior_stage_rates)
        side_info = quantizer.side_info_tensor(metadata)
        PriorCalculator.fit_prior_model(
            bins.long(), soft_codes, side_info, cfg.model_copy(update={"marginal_loss": False}),
            quantizer.coding_model, (None,), cfg.prior_train_repeats)
        prior = quantizer.prior_calculator.compute_prior_from_network(quantizer.coding_model, bins, side_info)
    fine_tuned_rates = PriorCalculator.compute_code_lengths(prior, bins, cfg.num_planes).mean(dim=1).tolist()
    wmapes = [record_reconstruction_metrics(target, stage)["wmape"] for stage in stages]
    curve = [
        {"fit": fit, "epoch": int(epoch), "loss": float(loss),
         **{f"hard_mse_{i}": float(v) for i, v in enumerate(mse.split(","))},
         **{f"cumulative_rate_{i}": float(v) for i, v in enumerate(rate.split(","))}}
        for fit, epoch, loss, mse, rate in _EPOCH_LINE.findall(log.getvalue())
    ]
    symbols_per_stage = [int(plane.unique().numel()) for plane in bins]
    final_loss = min(quantizer.warm_retrained_losses) if initial_states else curve[-1]["loss"]
    return {"stage_wmapes": wmapes, "stage_rates": rates, "fine_tuned_stage_rates": fine_tuned_rates,
            "final_training_loss": final_loss, "symbols_per_stage": symbols_per_stage}, curve, joint_state


def main() -> None:
    """Regenerate the trace up to the trained round and run this worker's share of the attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 10, 29])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--overrides", type=json.loads, default={},
                        help='JSON WZcfgQuant overrides, e.g. \'{"layers": 4}\'')
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    problems, slices = round_problems(args.start_round, args.clients)
    # The protocol's configuration for these runs; one attempt so each collapse is observed, not selected away.
    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3, norm_slices=slices,
                     quantizer_train_repeats=1, training_log_every_epochs=5, **args.overrides)
    jobs = [(client_id, target, name, si)
            for client_id, (target, si_sets) in problems.items() for name, si in si_sets.items()]
    runs = [(job, seed) for job in jobs for seed in args.seeds][args.worker::args.workers]
    (args.out / "curves").mkdir(parents=True, exist_ok=True)
    for (client_id, target, name, si), seed in runs:
        stem = f"client{client_id}_{name}_seed{seed}"
        if (args.out / f"{stem}.json").exists():
            continue
        result, curve, _ = train_attempt(target, si, cfg, seed)
        pd.DataFrame(curve).to_csv(args.out / "curves" / f"{stem}.csv", index=False)
        row = {"client_id": client_id, "si_set": name, "si_vectors": len(si), "seed": seed, **result}
        (args.out / f"{stem}.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
