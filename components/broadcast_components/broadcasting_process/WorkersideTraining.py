from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol

class WorkersideTrainingProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer):
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=1)

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        super()._prep_for_next_agent(curr_agent_id, worker_count)
        self.prev_d_flat=[]