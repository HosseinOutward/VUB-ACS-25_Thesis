from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
from components.broadcast_components.broadcasting_process.HybridWZBroadcastProtocol import HybridWZBroadcastProtocol

class BalancedHybridProtocol(HybridWZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer):
        super().__init__(agent_count, wz_base_quantizer, hybrid_round_num=4)
        self.hybrid_round_num = 4
        self.is_hybrid_round_f = lambda round_id: (round_id+1)%self.hybrid_round_num in [2,3] and not self.warmup
        self.past_workerside_grads = [[] for _ in range(agent_count)]