import numpy as np

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol


def _update_wz_quant_model(qz_model, grad_vector, side_info_vs, epochs):
    assert len(side_info_vs) != 0
    if epochs > 0:
        temp = np.random.normal(0, np.sqrt(1e-8), len(grad_vector), ).astype(np.float32)
        qz_model.train_model(grad_vector + temp, side_info_vs, epoch=epochs, batch_size=10_000)
    else:
        print('--- only updating the p_cdf ---')
    qz_model.get_set_training_posterior_cdf(grad_vector, side_info_vs)
    return qz_model


class CancerProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep,
                 small_update=False, update_interval=10, **kwargs):
        self.small_update = small_update
        self.update_interval = update_interval
        assert update_interval > 3
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=self.update_interval, **kwargs)
        self.si_window_size = self.update_interval-1
        self.cancer_warmup_done = False
        self.frozen_quantizers = None

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)

        if not self.cancer_warmup_done and (self.curr_round_id==self.update_interval+1 and agent_id+1==worker_count-1):
            self.cancer_warmup_done=True
            print('--- switching to frozen quantizers for cancer protocol ---')
            self.frozen_quantizers = [a for a in self.wz_quantizer_list]

        # retrain the qz when we have fresh workerside grads (i.e. all-out rounds have passed)
        if self.cancer_warmup_done:
            target_vec = self.past_worker_grad_recons_vec[agent_id][-1]
            side_info = self._get_side_info_for_grad_recons(agent_id, force_is_hybrid_round=False)

            epoch_count = 0
            if self.is_hybrid_round_f(self.curr_round_id):
                print('--- updating frozen quantizers after worker-side ---')

                # since its hybrid round, our quantizer is worker-side, so switch back to frozen versions
                self.wz_quantizer_list[agent_id] = self.frozen_quantizers[agent_id]

                epoch_count = 30
            elif self.small_update:
                print('--- small updating of frozen quantizers ---')
                epoch_count = 3

            self.wz_quantizer_list[agent_id] = _update_wz_quant_model(
                self.wz_quantizer_list[agent_id], target_vec, side_info, epoch_count)

if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        CancerProtocol(worker_count, base_quantizer, epoch_count=10, update_interval=5, small_update=True))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)
