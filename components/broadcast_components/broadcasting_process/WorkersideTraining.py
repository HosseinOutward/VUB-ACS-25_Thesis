from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol


class WorkersideTrainingProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep):
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=1)

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        super()._prep_for_next_agent(curr_agent_id, worker_count)
        self.prev_d_flat=[]

if __name__ == "__main__":
    import torch
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=200_000,
                            count_side_info_data=0, enable_progress_bar=True, vec_slices=None)

    broadcast_prot = WorkersideTrainingProtocol(k, base_quantizer)

    _test_main(broadcast_prot, worker_count=k, rounds=3)
