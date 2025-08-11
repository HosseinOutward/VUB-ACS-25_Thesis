from typing import List, OrderedDict
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.WorkersideTraining import WorkersideTrainingProtocol


class WorkersideTrainingWithAccumErrorProtocol(WorkersideTrainingProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep):
        super().__init__(agent_count, wz_base_quantizer)
        self.accum_error:List[dict|None] = [None for _ in range(agent_count)]

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        # add the past accumulated error to the grad_dict
        temp = {k: grad_dict[k].cpu() for k in grad_dict.keys()}
        if self.accum_error[agent_id] is not None:
            grad_dict = OrderedDict({k: grad_dict[k] + self.accum_error[agent_id][k].cuda() for k in grad_dict.keys()})
        self.accum_error[agent_id]=temp

        return super().to_server_prep_data_for_transfer(agent_id, grad_dict, encoder_data_sent_by_server)

    def reconstruct_worker_grads(self, agent_id, worker_broadcast_data, worker_count, global_dims):
        result_dict = super().reconstruct_worker_grads(agent_id, worker_broadcast_data, worker_count, global_dims)
        self.accum_error[agent_id] = OrderedDict(
            {k: self.accum_error[agent_id][k] - v for k, v in result_dict.items()})
        return result_dict


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    bp_f = lambda worker_count, base_quantizer: (
        WorkersideTrainingWithAccumErrorProtocol(worker_count, base_quantizer))
    _test_main(bp_f, worker_count=2, rounds=4)
