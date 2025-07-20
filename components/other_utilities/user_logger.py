import os
import pandas as pd
import numpy as np
from copy import deepcopy

import torch
from pytorch_lightning.loggers import CSVLogger


class UnifiedLoggingClass:
    def __init__(self, worker_count, name:str = "_dev_debug_test",):
        self.worker_count = worker_count
        temp = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\experiments\exp_data\reports of runs'
        self.path_folder = os.path.join(temp, name)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        elif name != "_dev_debug_test":
            raise FileExistsError('Prevented from overwriting existing test folder')

        self.agent_id, self.round_id = None, None

    def set_aid_rid(self, ag_id, round_s):
        self.agent_id = ag_id
        self.round_id = round_s

    def broadcast_reporting(self, stats):
        unified_stats = deepcopy(stats)
        unified_stats.update({'agent_id': self.agent_id, 'round_id': self.round_id})
        folder = '_broadcast_protocol_stats'
        for method in stats.keys():
            file = f'{method}.csv'
            file_path = os.path.join(self.path_folder, folder, file)
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(','.join(unified_stats.keys()) + '\n')
            with open(file_path, 'a') as f:
                f.write(','.join([str(v) for v in unified_stats.values()]) + '\n')

    def fl_sim_log(self, round_s, agent_id, train_metrics_dict, test_metrics_dict):
        temp=lambda a,b: all([k in a.keys() for k in b.keys()])
        assert temp(train_metrics_dict, test_metrics_dict) and temp(test_metrics_dict, train_metrics_dict)

        unified_dict = {'train_' + k: v for k, v in train_metrics_dict.items()}
        unified_dict.update({'test_' + k: v for k, v in test_metrics_dict.items()})
        unified_dict.update({'agent_id': agent_id, 'round_id': round_s})

        current_folder = self.path_folder
        if agent_id == 'global':
            file = '_global_metrics_before_round_start.csv'
        else:
            file = '_agent_metrics_after_training.csv'

        # add line to csv file without overwriting
        file_path = os.path.join(current_folder, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(','.join(unified_dict.keys()) + '\n')
        with open(file_path, 'a') as f:
            f.write(','.join([str(v) for v in unified_dict.values()]) + '\n')

    def get_agent_csv_logger(self) -> CSVLogger:
        folder = 'agent_model_training_logs'
        return CSVLogger(save_dir=self.path_folder, name=folder,
                         version=f"round_{self.round_id}_agent_{self.agent_id}")

    def get_wz_csv_logger(self) -> CSVLogger:
        # NOTE: the wz training is happening to prep for the next agent
        folder = 'wz_training_logs'
        agent_trained_for = (self.agent_id+1)%self.worker_count
        round_trained_for = self.round_id+(agent_trained_for!=self.agent_id)
        return CSVLogger(save_dir=self.path_folder, name=folder,
            version=f"round_{round_trained_for}_agent_{agent_trained_for}")

    # def plotting(self, experiment_name=None, save_plots=True):
    #
    # def _plot_global_metrics(self, ):
    #
    # def _plot_wz_training_metrics(self, ):
    #
    # def _plot_broadcast_protocol_stats(self):
    #
    # def get_unified_data_tables(self, test_train='train'):
    #
    #
    #     return per_worker_step_training,
        




if __name__ == "__main__":
    from components.FL_sim import _main_test, FLSimulator
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
    from components.broadcast_components.reporting_utilities import BroadcastMetricGatheringUtilities

    model, dataset, dataset_test = _main_test()

    # *****************
    user_logger = UnifiedLoggingClass()

    # *****************
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_3plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
            count_side_info_data=0, enable_progress_bar=True, user_logger=user_logger)
    broadcast_prot_base = WZBroadcastProtocol(3, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base)

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=3, communication_rounds=3, client_epochs_per_round=10,
        batch_size=10000, dataset_train=dataset, dataset_test=dataset_test,
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
    sim.run_simulation(broadcast_prot)