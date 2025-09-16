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

        self.cancer_warmup_done = False

        self.frozen_quantizers = None
        self.frozen_si = None
        single_is_hybrid_round_f = lambda round_id: round_id % self.hybrid_round_num == 0 and not self.warmup
        self.is_hybrid_round_f = lambda round_id: single_is_hybrid_round_f(round_id)   or\
                                                  single_is_hybrid_round_f(round_id+1)
        self.is_freezing_time = lambda round_id: round_id!=1 and (round_id-1) % self.update_interval == 0

    def _get_side_info_for_grad_recons(self, agent_id, **kwargs):
        if self.is_freezing_time(self.curr_round_id):
            assert self.frozen_quantizers is not None
            assert not self.is_hybrid_round_f(self.curr_round_id)

            return self.frozen_si[agent_id]

        return super()._get_side_info_for_grad_recons(agent_id, **kwargs)

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        super()._post_reconstruction_processing(agent_id, worker_count, dict_shape, curr_recons_vector)

        next_agent = (agent_id + 1) % worker_count
        coming_round = self.curr_round_id + int(next_agent==0)
        coming_is_freezing = self.is_freezing_time(coming_round)

        # first time we are switching to frozen quantizers
        if coming_is_freezing and not self.cancer_warmup_done:
            assert next_agent==0
            assert self.is_hybrid_round_f(self.curr_round_id)
            assert self.frozen_quantizers is None

            print('--- switching to frozen quantizers for cancer protocol ---')
            self.cancer_warmup_done=True
            self.frozen_quantizers = []
            self.frozen_si = []

            # the super has trained the first quantizer before setting the cancer warmup flag
            side_info = self._get_side_info_for_grad_recons(0, force_is_hybrid_round=False)
            self.frozen_quantizers.append(self.wz_quantizer_list[0])
            self.frozen_si.append(side_info)

            for i in range(1, worker_count):
                target_vec = self.past_worker_grad_recons_vec[i][-1]
                side_info = self._get_side_info_for_grad_recons(i, force_is_hybrid_round=False)
                qz_model = _train_model(
                    target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                    bins_per_plane=int(max(16 // (self.curr_round_id/2 + 1), 2)),
                    binary_quant=self.binary_quantizer if self.curr_round_id >= 9 else False,
                    vec_slices=_get_vec_slices(dict_shape),
                    user_logger=self.wz_basic_quantizer.user_logger
                )

                self.frozen_quantizers.append(qz_model)
                self.frozen_si.append(side_info)

        elif coming_is_freezing and next_agent==0:
            assert self.frozen_quantizers is not None
            assert len(self.frozen_quantizers) == worker_count
            assert len(self.frozen_si) == worker_count
            assert self.is_hybrid_round_f(self.curr_round_id)

            print('--- updating frozen quantizers after worker-side ---')
            self.frozen_quantizers = []
            self.frozen_si = []
            for i in range(worker_count):
                target_vec = self.past_worker_grad_recons_vec[i][-1]
                side_info = self._get_side_info_for_grad_recons(i, force_is_hybrid_round=False)
                qz_model = _train_model(
                    target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                    bins_per_plane=int(max(16 // (self.curr_round_id + 1), 2)),
                    binary_quant=self.binary_quantizer if self.curr_round_id >= 9 else False,
                    vec_slices=_get_vec_slices(dict_shape),
                    user_logger=self.wz_basic_quantizer.user_logger
                )

                self.frozen_quantizers.append(qz_model)
                self.frozen_si.append(side_info)

        if coming_is_freezing:
            self.wz_quantizer_list=[a for a in self.frozen_quantizers]


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        CancerProtocol(worker_count, base_quantizer, epoch_count=10,
                       update_interval=4, small_update=False))
    _test_main(bp_f, worker_count=2, rounds=25, no_global_quant=True)
