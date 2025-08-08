import os
import pandas as pd
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import torch
from pytorch_lightning.loggers import CSVLogger



class UnifiedLoggingClass:
    def __init__(self, worker_count, name:str = "_dev_debug_test", runs_reporting_folder=None):
        if runs_reporting_folder is None:
            runs_reporting_folder =\
                r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\experiments\run_sim_script\reports of runs'
        self.worker_count = worker_count
        self.path_folder = os.path.join(runs_reporting_folder, name)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        elif name != "_dev_debug_test":
            raise Exception('    ***>> Prevented from overwriting existing test folder<<***    ')
        else:
            input('WARNING: You are about to delete the dev debug folder.')
            import shutil
            shutil.rmtree(self.path_folder)
            os.makedirs(self.path_folder)

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
        unified_dict.update({'val_' + k: v for k, v in test_metrics_dict.items()})
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


def _process_single_csv(file_info, folder_path):
    """Process a single CSV file - designed for parallel execution"""
    round_id, agent_id, folder = file_info
    file_path = os.path.join(folder_path, folder, 'metrics.csv')

    if not os.path.exists(file_path):
        return None, f'WARNING: File not found: {file_path}'

    try:
        table = pd.read_csv(file_path)
        if table.empty:
            return None, f'WARNING: Empty file: {file_path}'

        # Group by step and take the last value for each step
        table = table.groupby('step').agg(lambda x: x.ffill().bfill().iloc[-1]).reset_index()

        # Add metadata
        table['agent_id'] = agent_id
        table['round_id'] = round_id
        table['original_step'] = table['step'].copy()

        return table, None
    except Exception as e:
        return None, f'WARNING: Error processing {file_path}: {e}'


def _get_trainer_logs(path_folder, folder_p, round_count=None, change_step=True, n_workers=None):
    folder_path = os.path.join(path_folder, folder_p)
    if not os.path.exists(folder_path):
        # print(f'WARNING: Folder not found: {folder_path}')
        return []

    # Get all version folders and extract agent/round info
    version_folders = [f for f in os.listdir(folder_path) if f.startswith('round_')]
    if not version_folders:
        # print(f'WARNING: No version folders found in {folder_path}')
        return []

    # Extract round and agent info from folder names
    folder_info = []
    for folder in version_folders:
        try:
            parts = folder.split('_')
            round_id = int(parts[1])
            agent_id = int(parts[3])
            folder_info.append((round_id, agent_id, folder))
        except (IndexError, ValueError):
            # print(f'WARNING: Skipping malformed folder name: {folder}')
            continue

    if not folder_info:
        # print(f'WARNING: No valid version folders found in {folder_path}')
        return []

    # Determine worker count and round range automatically
    worker_count = max([info[1] for info in folder_info]) + 1
    available_rounds = sorted(set([info[0] for info in folder_info]))

    if round_count is not None:
        # Filter to requested round count
        available_rounds = [r for r in available_rounds if r < round_count]

    r_start = 0 if folder_p == 'agent_model_training_logs' else 1
    available_rounds = [r for r in available_rounds if r >= r_start]

    # Filter folder_info to only include valid rounds
    folder_info = [(r, a, f) for r, a, f in folder_info if r in available_rounds]

    # print(f'Processing {len(available_rounds)} rounds for {worker_count} workers in {folder_p} using parallel processing')

    # Determine number of workers for parallel processing
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(folder_info), 8)  # Cap at 8 to avoid overwhelming system

    # Process files in parallel
    all_data = []
    process_func = partial(_process_single_csv, folder_path=folder_path)

    if n_workers > 1 and len(folder_info) > 1:
        # print(f'Using {n_workers} parallel workers to process {len(folder_info)} files')
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_info = {executor.submit(process_func, info): info for info in folder_info}

            # Collect results as they complete
            for future in as_completed(future_to_info):
                table, error = future.result()
                if error:
                    print(error)
                elif table is not None:
                    all_data.append(table)
    else:
        # Fall back to sequential processing for small datasets
        # print('Using sequential processing')
        for info in folder_info:
            table, error = process_func(info)
            if error:
                print(error)
            elif table is not None:
                all_data.append(table)

    if not all_data:
        # print(f'WARNING: No data found for {folder_p}')
        return [pd.DataFrame() for _ in range(worker_count)]

    # Combine all data and sort
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['agent_id', 'round_id', 'original_step']).reset_index(drop=True)

    # Split by worker and adjust steps if needed
    res = []
    for agent_id in range(worker_count):
        worker_data = combined_df[combined_df['agent_id'] == agent_id].copy()

        if worker_data.empty:
            # print(f'WARNING: No data found for agent {agent_id}')
            res.append(pd.DataFrame())
            continue

        if change_step:
            # Recalculate steps to be continuous across rounds
            worker_data = worker_data.sort_values(['round_id', 'original_step']).reset_index(drop=True)
            step_offset = 0

            for round_id in sorted(worker_data['round_id'].unique()):
                round_mask = worker_data['round_id'] == round_id
                round_steps = worker_data.loc[round_mask, 'original_step'].values
                worker_data.loc[round_mask, 'step'] = round_steps + step_offset
                step_offset = worker_data.loc[round_mask, 'step'].max() + 1

        # Remove the temporary column
        worker_data = worker_data.drop('original_step', axis=1)
        res.append(worker_data.reset_index(drop=True))

    return res


def get_unified_data_tables(name, worker_count):
    path_folder = os.path.join(
        r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\experiments\run_sim_script\reports of runs', name)
    global_metric_before_round =\
        pd.read_csv(os.path.join(path_folder, '_global_metrics_before_round_start.csv'))
    agent_metrics_after_training =\
        pd.read_csv(os.path.join(path_folder, '_agent_metrics_after_training.csv'))

    fix_tensor = lambda x: eval(x[6:]) #if type(x) is str else x
    for l in ['val_auc', 'train_auc']:
        for ll in [global_metric_before_round, agent_metrics_after_training]:
            ll[l]=ll[l].apply(fix_tensor)

    round_count = agent_metrics_after_training['round_id'].max() + 1

    per_worker_training_logs = _get_trainer_logs(path_folder, 'agent_model_training_logs', round_count)
    for i in range(len(per_worker_training_logs)):
        temp = per_worker_training_logs[i]['step']
        per_worker_training_logs[i]['step'] = temp/temp.max()*round_count

    per_wz_training_logs = _get_trainer_logs(path_folder, 'wz_training_logs', round_count, False)
    for i in range(len(per_wz_training_logs)):
        temp=per_wz_training_logs[i]
        per_wz_training_logs[i]['real_step'] =\
            temp['round_id'] + temp['agent_id']/worker_count + temp['step']/temp['step'].max()/worker_count

    broadcast_entire_stats = {}
    if os.path.exists(os.path.join(path_folder, '_broadcast_protocol_stats')):
        for file in os.listdir(os.path.join(path_folder, '_broadcast_protocol_stats')):
            method = file.split('.')[0]
            file_path = os.path.join(path_folder, '_broadcast_protocol_stats', file)
            broadcast_entire_stats[method] = pd.read_csv(file_path)

            temp = broadcast_entire_stats[method].agent_id!=0
            broadcast_entire_stats[method].loc[temp, 'mbytes_sent_for_aggre'] = 0.0
            temp=lambda i: (broadcast_entire_stats[method]['round_id']==i).values
            broadcast_entire_stats[method] = {
                k: [broadcast_entire_stats[method][k][temp(i)].values for i in range(round_count)]
                for k in broadcast_entire_stats[method].columns
            }

    return (per_worker_training_logs, per_wz_training_logs,
            global_metric_before_round, agent_metrics_after_training,
            broadcast_entire_stats)


def plot_all_metrics(per_worker_training_logs, per_wz_training_logs,
                     global_metric_before_round, agent_metrics_after_training, broadcast_entire_stats):
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import plot_stats

    # broadcast stats plots
    if len(broadcast_entire_stats)!=0:
        plot_stats(broadcast_entire_stats, no_raw=True)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Helper function to get metric columns
    def get_metric_columns(df, l_col):
        return [col for col in df.columns if l_col in col.lower()]
    loss_cols = get_metric_columns(global_metric_before_round, 'loss')
    auc_cols = get_metric_columns(global_metric_before_round, 'auc')
    
    # 1. Combined Loss and AUC Plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Define colors and markers for different metrics
    colors = {'train_loss': 'darkred', 'val_loss': 'pink', 'train_auc': 'darkblue', 'val_auc': 'lightblue'}
    markers = {'train_loss': 'o', 'val_loss': '^', 'train_auc': 's', 'val_auc': 'D'}

    # Create secondary y-axis for AUC
    ax2 = ax1.twinx()
    for l_col in loss_cols+auc_cols:
        ax = ax1 if 'loss' in l_col else ax2
        # train/val worker backgrounds
        worker_values = []
        for i, worker_df in enumerate(per_worker_training_logs):
            y_vals = worker_df[l_col+'_step']
            x_vals = worker_df['step']
            ax.plot(x_vals, y_vals, alpha=0.15, linewidth=1, color=colors[l_col])
            worker_values.append(y_vals.values)
    
        # Plot worker mean
        mean_values = np.mean(worker_values, axis=0)
        label = f'Workers Mean ({l_col.replace("_", " ").title()})'
        x_vals = per_worker_training_logs[len(per_worker_training_logs)//2]['step']
        ax.plot(x_vals, mean_values, '--', linewidth=1.5,
                 label=label, color=colors[l_col], alpha=0.7)
    
        # Plot global model loss (main lines)
        x_vals = global_metric_before_round['round_id']
        y_vals = global_metric_before_round[l_col]
        label = f"Global Model ({l_col.replace('_', ' ').title()})"
        ax.plot(x_vals, y_vals, marker=markers[l_col], linestyle='-',
                linewidth=3, markersize=6, label=label, color=colors[l_col])

    # Set up axis labels and styling
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='darkred')
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_title('Global Model Loss and AUC vs Training Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('AUC', fontsize=12, fontweight='bold', color='darkblue')
    ax2.tick_params(axis='y', labelcolor='darkblue')

    # Set integer ticks on x-axis
    max_round = global_metric_before_round['round_id'].max()
    ax1.set_xticks(range(0, max_round + 1))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')

    plt.tight_layout()
    plt.show()

    #%%
    # 2. Plot WZ Training Logs (specific metrics, combined across workers)
    # Combine all worker DataFrames into one
    metrics = ['loss', 'mape%', 'mse', 'rate_bits', 'real_bit_r']

    combined_wz_df = pd.concat(per_wz_training_logs, ignore_index=True)
    combined_wz_df = combined_wz_df.sort_values('agent_id').reset_index(drop=True)
    combined_wz_df = combined_wz_df.sort_values('real_step').reset_index(drop=True)

    # find the step where the workers change for each round
    transition_steps_idx = []
    for i in range(1, len(combined_wz_df)):
        if combined_wz_df['agent_id'][i] != combined_wz_df['agent_id'][i - 1]:
            # check if it has non nan val otherwise, add a +1
            if not pd.isna(combined_wz_df['val_mse'][i]):
                transition_steps_idx.append(i)
            elif not pd.isna(combined_wz_df['val_mse'][i+1]):
                transition_steps_idx.append(i+1)
            else:
                assert not pd.isna(combined_wz_df['val_mse'][i-1]), f'{i-1}'
                transition_steps_idx.append(i-1)
    transition_steps_idx = np.array(transition_steps_idx)

    # only consider rows at transition_steps
    combined_wz_df = combined_wz_df[combined_wz_df.index.isin(transition_steps_idx)].reset_index(drop=True)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(20, 6 * len(metrics)))
    axes = axes.flatten()

    # Plot each metric in the group
    for j, metric in enumerate(metrics):
        ax = axes[j]

        train_col = f'train_{metric}'
        val_col = f'val_{metric}'

        # Use different colors for different metrics in the same group
        color = ['blue', 'red']

        # vlines for transition steps
        for step in combined_wz_df['real_step']:
            ax.axvline(x=step, color='gray', linestyle='--', linewidth=0.5, alpha=0.1)

        # Plot metric
        for i in range(2):
            train_val = ['Val', 'Train'][i]
            x_vals = combined_wz_df['real_step']
            y_vals = combined_wz_df[[val_col, train_col][i]]
            ax.plot(x_vals-0.1, y_vals, ['-s', '--o'][i], linewidth=1, color=color[i], alpha=0.7,
                label=f'{train_val} {metric.replace("_", " ").title()}')

        # Create secondary x-axis for worker labels
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.tick_params(axis='x', which='major', pad=0, length=3)

        # Beautify the plot
        ax.set_xlabel('Real Step', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(f'WZ Training: {metric}', fontsize=12, fontweight='bold')
        temp = np.mean(combined_wz_df[train_col].values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('WZ Model (final) Training Metrics (each worker is between the vlines)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    #%%
    # 3. Scatter plots: Bit rates vs MSE/MAPE% colored by real_step
    combined_wz_df = pd.concat(per_wz_training_logs, ignore_index=True)
    combined_wz_df = combined_wz_df.sort_values('agent_id').reset_index(drop=True)
    combined_wz_df = combined_wz_df.sort_values('real_step').reset_index(drop=True)

    # only consider rows at transition_steps
    combined_wz_df = combined_wz_df[combined_wz_df.index.isin(transition_steps_idx)].reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(17, 15))

    mse_db = 10 * np.log10(combined_wz_df['train_mse'] + 1e-10)
    mape_db = 10 * np.log10(combined_wz_df['train_mape%'] + 1e-10)

    # Get bit rate values
    rate_bits = combined_wz_df['train_rate_bits'] if 'train_rate_bits' in combined_wz_df.columns else None
    real_bit_r = combined_wz_df['train_real_bit_r'] if 'train_real_bit_r' in combined_wz_df.columns else None

    # Create color map - using plasma for better directional visualization
    for i, metric in enumerate(['MSE', 'MAPE%']):
        for j, data_name in enumerate(['Rate Bits', 'Real Bit R']):
            metric_data = [mse_db, mape_db][i]
            data = [rate_bits, real_bit_r, ][j]
            scatter = axes[i, j].scatter(data, metric_data, c=combined_wz_df['real_step'],
                                         alpha=0.8, cmap='plasma', s=25)
            axes[i, j].set_xlabel(f'{data_name}', fontsize=10, fontweight='bold')
            axes[i, j].set_ylabel(f'{metric} (dB)', fontsize=10, fontweight='bold')
            axes[i, j].set_title(f'{data_name} vs {metric} (dB)', fontsize=12, fontweight='bold')
            axes[i, j].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[i, j], label='Real Step')

    plt.suptitle('WZ Training: Bit Rates vs Performance Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from components.FL_sim import _main_test, FLSimulator
    from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import WZServerTrainingPerRoundProtocol
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import BroadcastMetricGatheringUtilities, plot_stats

    model, dataset, dataset_test = _main_test()

    k=5

    # *****************
    user_logger = UnifiedLoggingClass(k)

    # *****************
    # temp=get_unified_data_tables("_dev_debug_test", k)
    # plot_all_metrics(*temp)

    # *****************
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'/data/basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
            count_side_info_data=0, enable_progress_bar=False, user_logger=user_logger)
    broadcast_prot_base = WZServerTrainingPerRoundProtocol(k, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base, user_logger=user_logger)

    # *****************
    sim = FLSimulator(
        pl_model=model, num_agents=k, communication_rounds=10, client_epochs_per_round=5,
        batch_size=10000, dataset_train=dataset, dataset_test=dataset_test,
        aggregation_method='fedavg', non_iid_sampling=False, user_logger=user_logger)
    sim.run_simulation(broadcast_prot)
