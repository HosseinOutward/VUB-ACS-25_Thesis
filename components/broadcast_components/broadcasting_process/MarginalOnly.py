import numpy as np
import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, dict_to_array, _get_vec_slices, _train_model, \
    compress_data_list


class MarginalOnly(WZServerTrainingPerRoundProtocol):
    side_info_vec_size = None
    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)
        self.side_info_vec_size = curr_recons_vector.shape[-1]
        self.past_worker_grad_recons_vec[agent_id][-1]=None

    def _get_side_info_for_grad_recons(self, agent_id):
        ans = super()._get_side_info_for_grad_recons(agent_id)
        ans = [np.zeros(self.side_info_vec_size, dtype=np.float32)]*len(ans)
        return ans


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        MarginalOnly(worker_count, base_quantizer, epoch_count=1,))
    _test_main(bp_f, worker_count=2, rounds=50)
