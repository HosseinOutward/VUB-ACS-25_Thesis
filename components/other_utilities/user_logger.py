import os
import torch
from pytorch_lightning.loggers import CSVLogger


class UnifiedLoggingClass:
    def __init__(self, name:str = "_dev_debug_test",):
        temp = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\experiments\exp_data\reports of runs'
        self.path_folder = os.path.join(temp, name)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        elif name != "_dev_debug_test":
            raise FileExistsError('Prevented from overwriting existing test folder')

        self.current_folder = None

    def set_aid_rid(self, ag_id, round_s):
        self.current_folder = os.path.join(self.path_folder, f"agent_{ag_id}_round_{round_s}")
        if not os.path.exists(self.path_folder):
            os.makedirs(self.current_folder)

    def broadcast_reporting(self, stats):
        pass

    def fl_sim_log(self, round_s, agent_id, train_metrics_dict, test_metrics_dict):
        pass

    def get_agent_csv_logger(self) -> CSVLogger:
        pass

    def get_wz_csv_logger(self) -> CSVLogger:
        pass


if __name__ == "__main__":
    from components.FL_sim import _main_test, FLSimulator
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
    from components.broadcast_components.reporting_utilities import BroadcastMetricGatheringUtilities

    model, dataset, dataset_test = _main_test()

    # *****************
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_3plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                 count_side_info_data=0, enable_progress_bar=False)
    broadcast_prot_base = WZBroadcastProtocol(5, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base)

    # *****************
    user_logger = UnifiedLoggingClass()

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=5, communication_rounds=3, client_epochs_per_round=10,
        batch_size=10000, dataset_train=dataset, dataset_test=dataset_test,
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
    sim.run_simulation()