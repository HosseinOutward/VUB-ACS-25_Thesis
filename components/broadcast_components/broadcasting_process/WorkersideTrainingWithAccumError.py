from typing import List, Dict
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.WorkersideTraining import WorkersideTrainingProtocol


class WorkersideTrainingWithAccumErrorProtocol(WorkersideTrainingProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep):
        super().__init__(agent_count, wz_base_quantizer)
        self.accum_error:List[None]|List[Dict] = [None for _ in range(agent_count)]

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server,
                                         force_use_diff_model=None):
        if self.warmup and force_use_diff_model is None:
            assert self.curr_round_id == 0
            assert self.accum_error[agent_id] is None
            self.accum_error[agent_id] = grad_dict
        return super().to_server_prep_data_for_transfer(
            agent_id, grad_dict, encoder_data_sent_by_server, force_use_diff_model)

    def _post_reconstruction_processing(self, agent_id, worker_count, res_vector, result_dict):
        super()._post_reconstruction_processing(agent_id, worker_count, res_vector, result_dict)
        # below if runs only once. have the first error right after warmup
        if self.accum_error[agent_id] is not None:
            assert self.curr_round_id == 0
            self.accum_error[agent_id] = {
                k: self.accum_error[agent_id][k] - result_dict[k] for k in result_dict.keys()}
        return result_dict

    def _build_worker_side_quantizer(self, old_quantizer, training_target, side_info):
        new_quantizer, recons_vect = super()._build_worker_side_quantizer(old_quantizer, training_target, side_info)
        assert self.accum_error[self.curr_agent_id] is None
        self.accum_error[self.curr_agent_id] = training_target-recons_vect
        return new_quantizer, recons_vect


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

    broadcast_prot = WorkersideTrainingWithAccumErrorProtocol(k, base_quantizer)
    broadcast_prot.epoch_count=1

    _test_main(broadcast_prot, worker_count=k, rounds=3)
