"""Instrumented TR check on synthetic deltas: which data the quantizer and fine-tuned prior see."""

import math
from pathlib import Path
from types import SimpleNamespace

import torch

from FL_code.FL_core.codec import Access, CompressionRecord, create_protocol, simulate_compression
from FL_code.FL_core.utils import set_global_seed
from FL_code.cancer_protocol import prior_code

N, START, CLIENTS = 200_000, 5, 3
TOKEN = ("TR|bins_per_plane=2|num_planes=3|rans_lanes=64|train_epochs=40|train_sample_size=200000"
         "|quantizer_train_repeats=2|prior_train_repeats=2")
set_global_seed(0)
gen = torch.Generator().manual_seed(0)


def laplace() -> torch.Tensor:
    return torch.distributions.Laplace(0.0, 1e-3).sample((N,))


# client 0: AR(1) on its own past (own temporal SI is informative)
# client 1: fresh noise each round (no SI can help)
# client 2: copy of client 0's same-round delta (only server SI, which sees client 0 first, can help)
deltas: dict[int, list[torch.Tensor]] = {c: [] for c in range(CLIENTS)}
for r in range(START + 2):
    deltas[0].append(laplace() if r == 0 else 0.9 * deltas[0][-1] + 0.45 * laplace())
    deltas[1].append(laplace())
    deltas[2].append(deltas[0][-1] + 0.05 * laplace())

protocol = create_protocol("wz_cancer", [slice(0, N)])
protocol.set_schedule(START, (TOKEN, "F"))
history = protocol.histories[0]
for r in range(START):
    for c in range(CLIENTS):
        history.commit(deltas[c][r], CompressionRecord(r, c, "raw_history"), Access.TEMPORAL_TOO)

fit_calls: list[dict] = []
original_fit = prior_code.PriorCalculator.fit_prior_model


def logged_fit(bins_vec, soft_codes, side_info, c_cfg, target_model, *args, **kwargs):
    fit_calls.append({"bins": bins_vec.clone(), "si_cols": side_info.shape[1], "rows": side_info.shape[0],
                      "fine_tunes_quantizer": target_model is not None})
    return original_fit(bins_vec, soft_codes, side_info, c_cfg, target_model, *args, **kwargs)


prior_code.PriorCalculator.fit_prior_model = staticmethod(logged_fit)
entry_label = {id(e.value): (e.round_id, e.client_id) for es in history._server.values() for e in es}
manager = SimpleNamespace(param_count=N)
out = Path("/tmp/tr_prior_provenance_check")
for r in (START, START + 1):
    for c in range(CLIENTS):
        fit_calls.clear()
        simulate_compression(protocol, deltas[c][r], c, r, manager, out, {}, (), ())
        codec = protocol.last_trained_codec[c]
        entry_label |= {id(e.value): (e.round_id, e.client_id) for es in history._server.values() for e in es}
        si = [entry_label.get(id(v), "??") for v in codec.quantizer.side_info_list_used]
        bins, _, _ = codec.quantizer.encoding_process(deltas[c][r])
        joint = (bins.long().cpu() * torch.tensor([4, 2, 1])[:, None]).sum(0)
        freq = torch.bincount(joint, minlength=8).double() / joint.numel()
        h_joint = -(freq[freq > 0] * freq[freq > 0].log2()).sum().item()
        print(f"round {r} client {c} {codec.round_name_full.split('|')[0]}: SI (round, client) = {si}; "
              f"empirical H(3-stage symbol) = {h_joint:.3f} bit")
        for call in fit_calls:
            print(f"   prior fit: fine_tunes_quantizer={call['fine_tunes_quantizer']} rows={call['rows']} "
                  f"si_cols={call['si_cols']} bins==this target's bins: {torch.equal(call['bins'].cpu(), bins.cpu())}")

import pandas as pd

d = pd.read_csv(out / "compression_records.csv")
print(d[["round_id", "client_id", "quantizer_prior_rate", "temporal_si_prior_rate", "server_si_prior_rate",
         "coder_rate", "marginal_rate", "wmape"]].round(3).to_string())
