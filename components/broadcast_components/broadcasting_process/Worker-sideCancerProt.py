import numpy as np
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep, _get_vec_slices
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _train_model


class CancerProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep,
                 small_update=False, update_interval=10, **kwargs):
        self.small_update = small_update
        assert self.small_update == False

        self.update_interval = update_interval
        assert update_interval > 3
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=self.update_interval, **kwargs)
        self.si_window_size = self.update_interval+1

        self.cancer_warmup_done = True

        self.frozen_si = []
        single_is_hybrid_round_f = lambda round_id: round_id % self.hybrid_round_num == 0 and not self.warmup
        self.is_hybrid_round_f = lambda round_id: single_is_hybrid_round_f(round_id)   or\
                                                  single_is_hybrid_round_f(round_id+1) or round_id<self.update_interval
        self.is_freezing_time = lambda round_id: round_id!=1 and (round_id-1) % self.update_interval == 0

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        force_is_hybrid_round = self.is_hybrid_round_f(self.curr_round_id)
        self.frozen_si = self._get_side_info_for_grad_recons(
            agent_id, force_is_hybrid_round=force_is_hybrid_round)
        super().to_server_prep_data_for_transfer(agent_id, grad_dict, encoder_data_sent_by_server)

    def _get_side_info_for_grad_recons(self, agent_id, **kwargs):
        if self.is_freezing_time(self.curr_round_id):
            assert not self.is_hybrid_round_f(self.curr_round_id)
            return self.frozen_si[agent_id]

        return super()._get_side_info_for_grad_recons(agent_id, **kwargs)

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)
        self.past_worker_grad_recons_vec[agent_id][-1]=None


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        CancerProtocol(worker_count, base_quantizer, epoch_count=10,
                       update_interval=4, small_update=False))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)
