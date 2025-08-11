import numpy as np
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol


class OnlyGlobalModel(WZServerTrainingPerRoundProtocol):
    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        return (grad_dict, )

    def reconstruct_worker_grads(self, agent_id, worker_broadcast_data, worker_count, global_model_dims):
        return worker_broadcast_data[0]

if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        OnlyGlobalModel(worker_count, base_quantizer))
    _test_main(bp_f, worker_count=2, rounds=4)
