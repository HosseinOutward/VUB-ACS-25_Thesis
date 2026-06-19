from __future__ import annotations

from typing import Any

import torch


try:
    from FL_code.cancer_protocol import CancerConfig, CancerCodec
except ModuleNotFoundError:
    import sys
    sys.path.append('..')
    from FL_code.cancer_protocol import CancerConfig
from FL_code.run_fl import FLConfig
from FL_code.cancer_protocol import CancerCodec


class SingleTypeCodec(CancerCodec):
    """Cancer protocol variant that forces routine rounds to one training type."""

    def __init__(
        self,
        single_letter: str,
        fl_cfg: FLConfig,
        binary_prot: bool = False,
        quantizer_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__(fl_cfg, binary_prot, quantizer_kwargs)
        self.c_cfg.warmup_phase = tuple(
            (single_letter if phase_type != 'P' else 'P', bins_per_plane, num_planes)
            for phase_type, bins_per_plane, num_planes in self.c_cfg.warmup_phase
        )
        self.c_cfg.routine_phase = tuple(
            (single_letter if phase_type != 'F' else 'F', bins_per_plane, num_planes)
            for phase_type, bins_per_plane, num_planes in self.c_cfg.routine_phase
        )

        assert all(phase[0] in ('P', 'F', single_letter) for phase in self.c_cfg.warmup_phase), (
            f"Warmup phase contains an invalid phase for {single_letter} codec."
        )
        assert all(phase[0] in ('F', single_letter) for phase in self.c_cfg.routine_phase), (
            f"Routine phase contains an invalid phase for {single_letter} codec."
        )


class TemporalCodec(SingleTypeCodec):
    """Cancer protocol variant using temporal training rounds."""

    def __init__(
        self,
        fl_cfg: FLConfig,
        binary_prot: bool = False,
        quantizer_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__('T', fl_cfg, binary_prot, quantizer_kwargs)


class RetrainCodec(SingleTypeCodec):
    """Cancer protocol variant using retraining rounds."""

    def __init__(
        self,
        fl_cfg: FLConfig,
        binary_prot: bool = False,
        quantizer_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__('R', fl_cfg, binary_prot, quantizer_kwargs)


# class CancerSemiMarginal(CancerCodec):
#     def __init__(self, fl_cfg: FLConfig, binary_prot=False, quantizer_kwargs=None):
#         super().__init__(fl_cfg, binary_prot, quantizer_kwargs)
#         replace_marg = lambda x: x+'M' if x in ['R', 'T'] else x
#         self.c_cfg.warmup_phase =  tuple((replace_marg(a[0]),a[1],a[2]) for a in self.c_cfg.warmup_phase)
#         self.c_cfg.routine_phase = tuple((replace_marg(a[0]),a[1],a[2]) for a in self.c_cfg.routine_phase)
#
#         assert [c[0] in ['P', 'F', 'RM', 'TM'] for c in self.c_cfg.warmup_phase]
#         assert [c[0] in ['F', 'RM', 'TM'] for c in self.c_cfg.routine_phase]


if __name__ == '__main__':
    num_clients = 3
    num_rounds = 10
    vector_size = 1_000_000
    base_vector = torch.normal(0, 1, size=(vector_size,))
    codec = LearnedMarginalCodec(FLConfig())
    codec.c_cfg.pretrain_pth_dir = r'../data/pre_trained_pth/'

    # base_vector = base_vector + torch.normal(0.0, 0.01, size=(vector_size,))
    # client_deltas = [base_vector + torch.normal(0.0, 0.1, size=(vector_size,)) for _ in range(num_clients)]
    # codec.srvr_past_reconst = [[c] for c in client_deltas]

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
