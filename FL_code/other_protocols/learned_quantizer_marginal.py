import gc
import torch


try:
    from FL_code.cancer_protocol import CancerConfig
except ModuleNotFoundError:
    import sys
    sys.path.append('..')
    from FL_code.cancer_protocol import CancerConfig
from FL_code.run_fl import FLConfig
from FL_code.cancer_protocol import CancerConfig, CancerCodec, CancerRecord
from FL_code.cancer_quantizer import WZQuantizerCancer


class LearnedSimpleCodec(CancerCodec):
    def __init__(self, fl_cfg: FLConfig, binary_prot=False, quantizer_kwargs=None):
        super().__init__(fl_cfg, binary_prot, quantizer_kwargs)
        self.c_cfg.warmup_phase = tuple(('M' if a[0]!='P' else 'P',a[1],a[2]) for a in self.c_cfg.warmup_phase)
        self.c_cfg.routine_phase = tuple(('M' if a[0]!='F' else 'F',a[1],a[2]) for a in self.c_cfg.routine_phase)

        assert [c[0] in ['P', 'F', 'M'] for c in self.c_cfg.warmup_phase]
        assert [c[0] in ['F', 'M'] for c in self.c_cfg.routine_phase]


if __name__ == '__main__':
    num_clients = 3
    num_rounds = 10
    vector_size = 1_000_000
    base_vector = torch.normal(0, 1, size=(vector_size,))
    codec = LearnedSimpleCodec(FLConfig())
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
