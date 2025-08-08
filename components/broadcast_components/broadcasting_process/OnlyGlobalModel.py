import numpy as np
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol


class OnlyGlobalModel(WZServerTrainingPerRoundProtocol):
    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server,
                                         force_use_diff_model=None):
        if force_use_diff_model is None:
            return (grad_dict, )
        return super().to_server_prep_data_for_transfer(agent_id, grad_dict, encoder_data_sent_by_server,
                                                       force_use_diff_model)

    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count,
                               global_model_dims, force_use_diff_model=None):
        if force_use_diff_model is None:
            return worker_broadcast_data[0]
        if self.curr_round_id!=0: self.warmup=False
        return super().reconstruction_process(agent_id, worker_broadcast_data,
                                              worker_count, global_model_dims, force_use_diff_model)

    # def model_transfer_to_worker_from_server(self, agent_id, server_model_state_dict):
    #     return server_model_state_dict, None