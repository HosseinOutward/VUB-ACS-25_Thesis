import os
import warnings

import pandas as pd
import numpy as np
from copy import deepcopy

import torch
from pytorch_lightning.loggers import CSVLogger

_REPORTING_FOLDR_ = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\experiments\exp_data\reports of runs'


class UnifiedLoggingClass:
    def __init__(self, worker_count, name:str = "_dev_debug_test",):
        self.worker_count = worker_count
        self.path_folder = os.path.join(_REPORTING_FOLDR_, name)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        elif name != "_dev_debug_test":
            warnings.warn('Prevented from overwriting existing test folder')
            self.set_aid_rid = None
            self.broadcast_reporting = None
            self.fl_sim_log = None
            self.get_agent_csv_logger = None
            self.get_wz_csv_logger = None
        # else:
        #     import shutil
        #     shutil.rmtree(self.path_folder)
        #     os.makedirs(self.path_folder)

        self.agent_id, self.round_id = None, None

    def set_aid_rid(self, ag_id, round_s):
        self.agent_id = ag_id
        self.round_id = round_s

    def broadcast_reporting(self, stats):
        unified_stats = deepcopy(stats)
        folder = '_broadcast_protocol_stats'
        for method in stats.keys():
            file = f'{method}.csv'
            file_path = os.path.join(self.path_folder, folder, file)
            if not os.path.exists(os.path.join(self.path_folder, folder)):
                os.makedirs(os.path.join(self.path_folder, folder))

            unified_stats[method].update({'agent_id': self.agent_id, 'round_id': self.round_id})
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(','.join(unified_stats[method].keys()) + '\n')
            with open(file_path, 'a') as f:
                f.write(','.join([str(v) for v in unified_stats[method].values()]) + '\n')

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
        round_trained_for = self.round_id+(agent_trained_for==0)
        return CSVLogger(save_dir=self.path_folder, name=folder,
            version=f"round_{round_trained_for}_agent_{agent_trained_for}")

    def get_unified_data_tables(self):
        global_metric_before_round =\
            pd.read_csv(os.path.join(self.path_folder, '_global_metrics_before_round_start.csv'))
        agent_metrics_after_training =\
            pd.read_csv(os.path.join(self.path_folder, '_agent_metrics_after_training.csv'))

        round_count = agent_metrics_after_training['round_id'].max() + 1

        per_worker_training_logs = self._get_trainer_logs('agent_model_training_logs', round_count)
        per_wz_training_logs = self._get_trainer_logs('wz_training_logs', round_count)

        broadcast_entire_stats = {}
        for file in os.listdir(os.path.join(self.path_folder, '_broadcast_protocol_stats')):
            method = file.split('.')[0]
            file_path = os.path.join(self.path_folder, '_broadcast_protocol_stats', file)
            temp={k: list(v.values()) for k, v in pd.read_csv(file_path).to_dict().items()}
            broadcast_entire_stats[method] = {
                k: [v[i:i+round_count] for i in range(0,round_count*self.worker_count, round_count)]
                    for k, v in temp.items()}

        return (per_worker_training_logs, per_wz_training_logs,
                global_metric_before_round, agent_metrics_after_training,
                broadcast_entire_stats)

    def _get_trainer_logs(self, folder_p, round_count):
        temp = os.listdir(os.path.join(self.path_folder, folder_p))
        temp = max([int(f.split('_')[3]) for f in temp])+1
        assert self.worker_count == temp

        res = []
        r_start = 0 if folder_p == '_global_metrics_before_round_start' else 1
        for agent_id in range(self.worker_count):
            last_step_num = 0
            worker_table = pd.DataFrame()
            for round_id in range(r_start, round_count):
                version_folder = f"round_{round_id}_agent_{agent_id}"
                file_path = os.path.join(self.path_folder, folder_p, version_folder, 'metrics.csv')
                assert os.path.exists(file_path)

                table = pd.read_csv(file_path)
                table = table.groupby('step').agg(lambda x: x.ffill().bfill().iloc[0]).reset_index()

                table['step'] += last_step_num
                last_step_num = table['step'].max()+1

                table['agent_id'] = agent_id
                table['round_id'] = round_id

                worker_table = pd.concat([worker_table, table], ignore_index=True)
            temp = worker_table['step']
            worker_table['step'] = temp/temp.max()*round_count
            res.append(worker_table)
        return res

def plot_all_metrics(per_worker_training_logs, per_wz_training_logs,
                     global_metric_before_round, agent_metrics_after_training, broadcast_entire_stats):
    from components.broadcast_components.reporting_utilities import plot_stats

    # broadcast stats plots
    plot_stats(broadcast_entire_stats)

    import matplotlib.pyplot as plt
    import seaborn as sns

    # per_worker_training_logs is a list of DataFrames, one for each agent
    # columns: epoch,step,train_loss,train_m1,m2,...,val_loss,test_m1,...
    # m1 and ... are the metrics that could be different, like mse, acc, auc

    # per_wz_training_logs is a list of DataFrames, one for each agent
    # columns:epoch,step,train_gumble_loss,train_gumble_mape%,train_gumble_mse,train_gumble_rate_bits,
    # train_gumble_real_bit_r,train_loss,train_mape%,train_mse,train_rate_bits,train_real_bit_r,
    # val_loss,val_mape%,val_mse,val_rate_bits,val_real_bit_r

    # each of the dataframes have a steps column which is between 0 and max_rounds

    # global_metric_before_round and agent_metrics_after_training have the following columns:
    # train_loss,train_auc,test_loss,test_auc,agent_id,round_id
    # for global_metric_before_round, agent_id is 'global', but for the other, it is the agent id
    # round id is an integer starting from 0 (before any training) to max_rounds (after all training)
    # note that all the rows might not have the test_x metrics, so treat them separately so you can drop the nan rows

    # I want a set of plots, but some of the columns change,
    # we need the following plots:
    # 1. plot of loss of global model (global_metric_before_round['?_loss']), with
    #       the worker loss in the background (only slightly visible)
    #       grids on, x tick on integers. legend for the global model, and also for a line showing the mean of workers
    #       pay attention to coloring, symbols and others beautification's
    # 2. similar plot to above but for the other metrics, with the same metric from the global_metric_before_round
    # 3. finally a plot for per_wz_training_logs. It's similar to above, but no more global_metric_before_round, instead
    #       every plot has a second y axis on the right with different color that has the data from mean of
    #       broadcast_entire_stats['wz']['mape%'] which is a 2d list of size (round_count, agent_count). take a mean
    #       over the second axis (agent_id).



if __name__ == "__main__":
    from components.FL_sim import _main_test, FLSimulator
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
    from components.broadcast_components.reporting_utilities import BroadcastMetricGatheringUtilities, plot_stats

    model, dataset, dataset_test = _main_test()

    k=3

    # *****************
    user_logger = UnifiedLoggingClass(k)

    # *****************
    temp=user_logger.get_unified_data_tables()
    plot_all_metrics(*temp)
    exit()

    # *****************
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_3plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
            count_side_info_data=0, enable_progress_bar=False, user_logger=user_logger)
    broadcast_prot_base = WZBroadcastProtocol(k, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base, user_logger=user_logger)

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=k, communication_rounds=3, client_epochs_per_round=3,
        batch_size=10000, dataset_train=dataset, dataset_test=dataset_test,
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
    sim.run_simulation(broadcast_prot)