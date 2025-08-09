from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol

class BalancedHybridProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer):
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=4)
        self.hybrid_round_num = 4
        self.is_hybrid_round_f = lambda round_id: (round_id+1)%self.hybrid_round_num in [2,3] and not self.warmup
        self.past_workerside_grads = [[] for _ in range(agent_count)]


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main
    from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
    import torch

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=100_000,
                                          count_side_info_data=0, enable_progress_bar=True, vec_slices=None)
    broadcast_prot = BalancedHybridProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=5)
