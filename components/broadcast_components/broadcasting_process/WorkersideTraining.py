from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol


class WorkersideTrainingProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep, *args, **kwargs):
        super().__init__(agent_count, wz_base_quantizer, *args, hybrid_round_num=1, **kwargs)

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)
        assert len(self.past_workerside_grads[agent_id])==min(self.curr_round_id+1, self.si_window_size) or self.warmup

    def _get_side_info_for_grad_recons(self, agent_id, *args, **kwargs):
        side_info = super()._get_side_info_for_grad_recons(agent_id, *args, **kwargs)
        assert len(self.past_workerside_grads[agent_id]) == min(self.curr_round_id, self.si_window_size) or self.warmup
        return side_info

if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        WorkersideTrainingProtocol(worker_count, base_quantizer))
    _test_main(bp_f, worker_count=5, rounds=50)

