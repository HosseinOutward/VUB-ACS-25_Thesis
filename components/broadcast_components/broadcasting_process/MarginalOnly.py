import numpy as np
import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, dict_to_array, _get_vec_slices, _train_model, \
    compress_data_list


class MarginalOnly(WZServerTrainingPerRoundProtocol):
    side_info_vec_size = None
    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        if self.side_info_vec_size is None:
            self.side_info_vec_size = curr_recons_vector.shape[-1]

        # **************
        self.past_worker_grad_recons_vec[agent_id].append(None)

        # **************
        assert agent_id == self.curr_agent_id
        if len(self.past_worker_grad_recons_vec[agent_id]) > self.si_window_size:
            self.past_worker_grad_recons_vec[agent_id].pop(0)
        if agent_id + 1 >= worker_count and self.warmup:
            assert self.curr_round_id == 0
            self.warmup = False

    def _get_side_info_for_grad_recons(self, agent_id):
        return []


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        MarginalOnly(worker_count, base_quantizer, epoch_count=1,))
    _test_main(bp_f, worker_count=5, rounds=50, no_global_quant=True)
