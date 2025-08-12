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
            assert len(self.past_worker_grad_recons_vec[agent_id])==1
            self.past_worker_grad_recons_vec[agent_id][0] = change_dtype_recursive(curr_recons_vector, torch.float16)

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
                    bins_per_plane=int(max(16 // (self.curr_round_id/2 + 1), 4)),
                    vec_slices=_get_vec_slices(dict_shape),
                    user_logger=self.wz_basic_quantizer.user_logger)
                self.wz_quantizer_list[i] = quantizer


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        SingleTimeTrainingProtocol(worker_count, base_quantizer,))
    _test_main(bp_f, worker_count=2, rounds=4)

