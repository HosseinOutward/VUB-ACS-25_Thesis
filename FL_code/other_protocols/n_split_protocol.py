from __future__ import annotations

from typing import Any

import torch

from FL_code.cancer_protocol import CancerConfig, BinsCodecRecord
from FL_code.codec import IdentityCodec
from FL_code.codec_registry import require_int_codec_option
from FL_code.protocol_records import BinsCodecRecord
from FL_code.prior_calculator import PriorCalculator


class NSplitCodec(IdentityCodec):
    """Baseline codec that quantizes values by percentile split points."""

    def __init__(self, fl_cfg: Any, split_points: int) -> None:
        super().__init__(fl_cfg)
        self.split_points = split_points
        self.num_clients = fl_cfg.num_clients
        self.srvr_past_reconst: list[list[torch.Tensor]] = [[] for _ in range(self.num_clients)]
        self.si_vec_size: int | None = None

    def create_record(self, round_id: int, client_id: int) -> BinsCodecRecord:
        return BinsCodecRecord(round_id, client_id, self.split_points, method=f"{self.split_points}-split")

    def get_si_data(self) -> torch.Tensor:
        si_raw = [tensor.float() for history in self.srvr_past_reconst for tensor in history]
        si_raw = [tensor / tensor.abs().quantile(0.99) for tensor in si_raw]

        if not si_raw:
            assert self.si_vec_size is not None, "Side information size must be set before fallback prior creation."
            si_raw = [torch.zeros(self.si_vec_size)]

        si_raw: torch.Tensor = torch.stack(si_raw).cuda().T.to(torch.float32).contiguous()
        return si_raw

    def bin_f(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        perc_v = [torch.quantile(x, i / self.split_points) for i in range(1, self.split_points)]

        bins_vec = torch.zeros(x.shape, dtype=torch.uint8)
        for i, pv in enumerate(perc_v):
            bins_vec[x > pv] = i + 1

        mean_v = torch.zeros(self.split_points, dtype=torch.float16)
        for i in range(self.split_points):
            mean_v[i] = x[bins_vec == i].mean().to(torch.float16)
        return bins_vec.unsqueeze(0), mean_v

    def un_bin_f(self, bins_vec: torch.Tensor, mean_v: torch.Tensor) -> torch.Tensor:
        reconst = torch.zeros(bins_vec.shape[1], dtype=torch.float32)
        for i in range(self.split_points):
            reconst[bins_vec[0]==i] = mean_v[i]
        return reconst

    def _compress(self, delta_vec: torch.Tensor, record: BinsCodecRecord) -> tuple[torch.Tensor, torch.Tensor]:
        self.si_vec_size = len(delta_vec)
        bins_vec, mean_v = self.bin_f(delta_vec)
        payload = (bins_vec, mean_v)

        si_trans = self.get_si_data()
        q_model = PriorCalculator.train_prior_model(
            bins_vec, si_trans, 1, record.bins_per_plane, CancerConfig())
        prior = PriorCalculator._compute_prior_from_network(q_model, bins_vec, si_trans)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins_vec, 1)

        m_prior = PriorCalculator.compute_marginal_prior(bins_vec, record.bins_per_plane, 1)
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(m_prior, bins_vec, 1)

        return payload

    def _decompress(self, payload: tuple[torch.Tensor, torch.Tensor], record: BinsCodecRecord) -> torch.Tensor:
        reconst = self.un_bin_f(*payload)

        history = self.srvr_past_reconst[record.client_id]
        history.append(reconst.to(torch.float16))
        # history.append(payload[0].squeeze())
        if len(history) > CancerConfig().max_side_info_count:
            history.pop(0)

        return reconst

if __name__ == '__main__':
    split_points = 3
    num_clients = 3
    num_rounds = 10
    vector_size = 1_000_000
    base_vector = torch.normal(0, 1, size=(vector_size,))
    codec = NSplitCodec(num_clients, split_points=split_points)

    for round_id in range(num_rounds):
        base_vector = base_vector + torch.normal(0.0, 0.01, size=(vector_size,))
        client_deltas = [base_vector + torch.normal(0.0, 0.1, size=(vector_size,)) for _ in range(num_clients)]

        for ci, d_v in enumerate(client_deltas):
            record = codec.create_record(round_id, ci)
            record.model_size = d_v.shape[0]
            payload = codec.encode(d_v, record)
            reconst = codec.decode(payload, record)
            print(record.to_dict())

    # from matplotlib import pyplot as plt
    # bins_vec, mean_v = codec._compress(base_vector, record)
    # plt.scatter(base_vector.cpu().numpy(), bins_vec.cpu().numpy()+0.2, alpha=0.5, s=0.1, cmap='red')
    # plt.vlines(mean_v.cpu().numpy(), 0, split_points-0.9, alpha=0.3)
    # plt.twinx().hist(base_vector.cpu().numpy(), 200, alpha=0.3)
    # plt.show()
