import numpy as np
import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep, _get_vec_slices
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, _train_model


class SingleTimeTrainingProtocol(WZServerTrainingPerRoundProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep, epoch_count=45):
        super().__init__(agent_count, wz_base_quantizer, epoch_count, True)

    def _get_side_info_for_grad_recons(self, agent_id):
        if self.warmup:
            return []

        assert all([len(self.past_worker_grad_recons_vec[agent_id]) == 1])
        side_info = [a[0] for i,a in enumerate(self.past_worker_grad_recons_vec) if i != agent_id]
        return side_info

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        assert agent_id == self.curr_agent_id

        if self.warmup:
            self.past_worker_grad_recons_vec[agent_id].append(change_dtype_recursive(curr_recons_vector, torch.float16))
        else:
            self.past_worker_grad_recons_vec[agent_id] = change_dtype_recursive(curr_recons_vector, torch.float16)

        # **************
        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count and self.warmup:
            assert all([len(self.past_worker_grad_recons_vec[agent_id]) == 1])

            self.warmup = False

            for i in range(worker_count):
                target_vec = self.past_worker_grad_recons_vec[i][0]
                side_info = self._get_side_info_for_grad_recons(i)
                quantizer = _train_model(
                    target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                    bins_per_plane=max(16 // (self.curr_round_id + 1), 3),
                    vec_slices=_get_vec_slices(dict_shape),
                    user_logger=self.wz_basic_quantizer.user_logger, reconst_ld=1000)
                self.wz_quantizer_list[i] = quantizer


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=200_000,
                                          count_side_info_data=0, enable_progress_bar=True, vec_slices=None)
    broadcast_prot = SingleTimeTrainingProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=3)
