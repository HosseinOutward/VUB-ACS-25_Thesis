import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, dict_to_array, _get_vec_slices, _train_model, \
    compress_data_list


class HybridWZBroadcastProtocol(WZServerTrainingPerRoundProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep, hybrid_round_num=5, **kwargs):
        super().__init__(agent_count, wz_base_quantizer, **kwargs)
        self.hybrid_round_num = hybrid_round_num
        self.is_hybrid_round_f = lambda round_id: round_id % self.hybrid_round_num == 0 and not self.warmup
        self.past_workerside_grads = [[] for _ in range(agent_count)]

    def to_worker_prep_data_for_transfer(self, agent_id):
        res = super().to_worker_prep_data_for_transfer(agent_id)
        # to simulate the cost of sending the decoder too. assuming similar size to encoder
        if self.is_hybrid_round_f(self.curr_round_id):
            return res, [res]*1
        return res

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        # to compensate for changes to to_worker_prep_data_for_transfer
        if self.is_hybrid_round_f(self.curr_round_id):
            encoder_data_sent_by_server=encoder_data_sent_by_server[0]

        # ***********
        if self.is_hybrid_round_f(self.curr_round_id):
            target_vec, dict_shape = dict_to_array(grad_dict)
            side_info = self._get_side_info_for_grad_recons(agent_id, force_is_hybrid_round=True)

            if self.wz_basic_quantizer.user_logger:
                self.wz_basic_quantizer.user_logger.round_id -= 1

            quantizer:QuantizerWithDataPrep = _train_model(
                target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                bins_per_plane=int(max(self.hybrid_round_num * 16 // (self.curr_round_id/2 + 1), 4)),
                vec_slices=_get_vec_slices(dict_shape),
                user_logger=self.wz_basic_quantizer.user_logger)
            self.wz_quantizer_list[agent_id] = quantizer

            encoder_data_sent_by_server = quantizer.wz_pl_model.coding_model.encoder.state_dict()
            encoder_data_sent_by_server = compress_data_list(encoder_data_sent_by_server)

            if self.wz_basic_quantizer.user_logger:
                self.wz_basic_quantizer.user_logger.round_id += 1

        return super().to_server_prep_data_for_transfer(agent_id, grad_dict, encoder_data_sent_by_server)

    def _get_side_info_for_grad_recons(self, agent_id, force_is_hybrid_round=None):
        if force_is_hybrid_round is None:
            force_is_hybrid_round = self.is_hybrid_round_f(self.curr_round_id)
        if not force_is_hybrid_round:
            return super()._get_side_info_for_grad_recons(agent_id)

        # hybrid round
        assert not self.warmup
        return self.past_workerside_grads[agent_id]

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        assert agent_id == self.curr_agent_id

        # **************
        new_side_info_to_add = change_dtype_recursive(curr_recons_vector, torch.float16)
        self.past_worker_grad_recons_vec[agent_id].append(new_side_info_to_add)

        if len(self.past_worker_grad_recons_vec[agent_id]) > self.si_window_size:
            self.past_worker_grad_recons_vec[agent_id].pop(0)

        # **************
        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count and self.warmup:
            assert self.curr_round_id == 0
            self.warmup = False

            assert all([len(a)==1 for a in self.past_worker_grad_recons_vec])
            self.past_workerside_grads = [[a[0]] for a in self.past_worker_grad_recons_vec]
        elif self.is_hybrid_round_f(self.curr_round_id):
            assert not self.warmup
            self.past_workerside_grads[agent_id].append(new_side_info_to_add)
            if len(self.past_workerside_grads[agent_id]) > self.si_window_size:
                self.past_workerside_grads[agent_id].pop(0)

        # **************
        # if the next round is hybrid, we don't need to train a new quantizer now
        next_agent = (agent_id + 1) % worker_count
        coming_round = self.curr_round_id + int(next_agent==0)
        coming_is_hybrid = self.is_hybrid_round_f(coming_round)
        if not self.warmup and not coming_is_hybrid:
            # cheat to make the cancer protocol simpler to implement
            if hasattr(self, 'cancer_warmup_done') and self.cancer_warmup_done:
                assert self.curr_round_id>=self.si_window_size
                return

            # train as usual here
            target_vec = self.past_worker_grad_recons_vec[next_agent][-1]
            side_info = self._get_side_info_for_grad_recons(next_agent, force_is_hybrid_round=coming_is_hybrid)
            assert len(side_info) == min(self.curr_round_id*worker_count+agent_id, self.si_window_size*worker_count-1)
            quantizer = _train_model(
                target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                bins_per_plane=int(max(16 // (self.curr_round_id/2 + 1), 4)),
                vec_slices=_get_vec_slices(dict_shape),
                user_logger=self.wz_basic_quantizer.user_logger)
            self.wz_quantizer_list[next_agent] = quantizer



if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        HybridWZBroadcastProtocol(worker_count, base_quantizer, epoch_count=1, hybrid_round_num=5))
    _test_main(bp_f, worker_count=5, rounds=50)
