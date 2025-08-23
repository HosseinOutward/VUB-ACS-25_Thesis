import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, dict_to_array, _get_vec_slices, _train_model, \
    compress_data_list


class MarginalOnly(WZServerTrainingPerRoundProtocol):
    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        assert agent_id == self.curr_agent_id

        # **************
        self.past_worker_grad_recons_vec[agent_id].append([0])

        if len(self.past_worker_grad_recons_vec[agent_id]) > self.si_window_size:
            self.past_worker_grad_recons_vec[agent_id].pop(0)

        # **************
        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count and self.warmup:
            assert self.curr_round_id == 0
            self.warmup = False

        # **************
        # we have at least one complete round, so we train the next WZ_models
        if not self.warmup:
            next_agent = (agent_id + 1) % worker_count
            target_vec = self.past_worker_grad_recons_vec[next_agent][-1]
            side_info = []
            quantizer = _train_model(
                target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                bins_per_plane=int(max(16 // (self.curr_round_id/2 + 1), 4)),
                vec_slices=_get_vec_slices(dict_shape),
                user_logger=self.wz_basic_quantizer.user_logger,
                marginal=True)
            self.wz_quantizer_list[next_agent] = quantizer


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        MarginalOnly(worker_count, base_quantizer, epoch_count=1,))
    _test_main(bp_f, worker_count=2, rounds=50)
