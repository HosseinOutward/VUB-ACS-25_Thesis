import numpy as np
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol


class NoSameWorkerSide(WZServerTrainingPerRoundProtocol):
    def _get_side_info_for_grad_recons(self, agent_id):
        if self.warmup:
            return []

        side_info = []
        for i, past_grads_agent in enumerate(self.past_worker_grad_recons_vec):
            temp = past_grads_agent
            if i == agent_id:
                temp = [a*0 for a in temp[:-1]]
            side_info.extend(temp)

        temp = self.curr_round_id*len(self.past_worker_grad_recons_vec)+self.curr_agent_id
        temp = min(temp, self.si_window_size*len(self.past_worker_grad_recons_vec)-1)
        assert len(side_info) in [temp, temp-1]
        return side_info

if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        NoSameWorkerSide(worker_count, base_quantizer))
    _test_main(bp_f, worker_count=2, rounds=6)
