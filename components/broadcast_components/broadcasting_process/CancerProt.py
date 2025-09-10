import numpy as np

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol


def _update_wz_quant_model(qz_model, grad_vector, side_info_vs):
    assert len(side_info_vs) != 0
    temp = np.random.normal(0, np.sqrt(1e-8), len(grad_vector), ).astype(np.float32)
    qz_model.train_model(grad_vector + temp, side_info_vs, epoch=20, batch_size=10_000)
    qz_model.get_set_training_posterior_cdf(grad_vector, side_info_vs)
    return qz_model


class CancerProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep, update_interval=10, **kwargs):
        self.update_interval = update_interval
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=self.update_interval, **kwargs)
        self.si_window_size = self.update_interval
        self.cancer_warmup_done = False

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        temp = lambda: all([len(a)==self.si_window_size for a in self.past_worker_grad_recons_vec])
        if not self.cancer_warmup_done and temp():
            self.cancer_warmup_done=True

        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)

        # retrain the qz when we have fresh workerside grads (i.e. 10 rounds have passed)
        if self.cancer_warmup_done and self.is_hybrid_round_f(self.curr_round_id):
            assert temp()
            self.wz_quantizer_list[agent_id] = _update_wz_quant_model(
                self.wz_quantizer_list[agent_id], curr_recons_vector, self.past_worker_grad_recons_vec)



if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        CancerProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)
