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


def _get_trainer_logs(path_folder, folder_p, round_count, change_step=True):
    temp = os.listdir(os.path.join(path_folder, folder_p))
    worker_count = max([int(f.split('_')[3]) for f in temp])+1

    res = []
    r_start = 0 if folder_p == 'agent_model_training_logs' else 1
    for agent_id in range(worker_count):
        last_step_num = 0
        worker_table = pd.DataFrame()
        for round_id in range(r_start, round_count):
            version_folder = f"round_{round_id}_agent_{agent_id}"
            file_path = os.path.join(path_folder, folder_p, version_folder, 'metrics.csv')
            assert os.path.exists(file_path)

            table = pd.read_csv(file_path)
            table = table.groupby('step').agg(lambda x: x.ffill().bfill().iloc[0]).reset_index()

            if change_step:
                table['step'] += last_step_num
                last_step_num = table['step'].max()+1

            table['agent_id'] = agent_id
            table['round_id'] = round_id

            worker_table = pd.concat([worker_table, table], ignore_index=True)
        res.append(worker_table)
    return res


def get_unified_data_tables(name, worker_count):
    path_folder = os.path.join(_REPORTING_FOLDR_, name)
    global_metric_before_round =\
        pd.read_csv(os.path.join(path_folder, '_global_metrics_before_round_start.csv'))
    agent_metrics_after_training =\
        pd.read_csv(os.path.join(path_folder, '_agent_metrics_after_training.csv'))
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
    for file in os.listdir(os.path.join(path_folder, '_broadcast_protocol_stats')):
        method = file.split('.')[0]
        file_path = os.path.join(path_folder, '_broadcast_protocol_stats', file)
        temp={k: list(v.values()) for k, v in pd.read_csv(file_path).to_dict().items()}
        broadcast_entire_stats[method] = {
            k: [v[i:i+round_count] for i in range(0,round_count*worker_count, round_count)]
                for k, v in temp.items()}

    return (per_worker_training_logs, per_wz_training_logs,
            global_metric_before_round, agent_metrics_after_training,
            broadcast_entire_stats)

def plot_all_metrics(per_worker_training_logs, per_wz_training_logs,
                     global_metric_before_round, agent_metrics_after_training, broadcast_entire_stats):
    from components.broadcast_components.reporting_utilities import plot_stats

    # broadcast stats plots
    plot_stats(broadcast_entire_stats)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Helper function to get metric columns
    def get_metric_columns(df, metric_type):
        return [col for col in df.columns if metric_type in col.lower()]
    
    # Helper function to plot worker backgrounds and mean
    def plot_worker_backgrounds(ax, worker_logs, metric_col, alpha=0.2, show_mean=True):
        worker_values = []
        for i, worker_df in enumerate(worker_logs):
            if metric_col in worker_df.columns:
                x_vals = worker_df['step'] if 'step' in worker_df.columns else worker_df.index
                y_vals = worker_df[metric_col]
                ax.plot(x_vals, y_vals, alpha=alpha, linewidth=1, color='gray')
                worker_values.append(y_vals.values)
        
        if show_mean and worker_values:
            # Calculate mean across workers at each step
            min_length = min(len(vals) for vals in worker_values)
            truncated_values = [vals[:min_length] for vals in worker_values]
            mean_values = np.mean(truncated_values, axis=0)
            
            # Use same x values as first worker (assuming they're aligned)
            if worker_logs:
                x_vals = worker_logs[0]['step'][:min_length] if 'step' in worker_logs[0].columns else range(min_length)
                ax.plot(x_vals, mean_values, '--', linewidth=2, label='Workers Mean', color='orange', alpha=0.8)
    
    # Find loss and auc columns
    loss_cols = get_metric_columns(global_metric_before_round, 'loss')
    auc_cols = get_metric_columns(global_metric_before_round, 'auc')
    
    # 1. Combined Loss and AUC Plot with dual y-axes
    if loss_cols or auc_cols:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Define colors and markers for different metrics
        colors = {'train_loss': 'darkred', 'val_loss': 'pink', 'train_auc': 'darkblue', 'val_auc': 'lightblue'}
        markers = {'train_loss': 'o', 'val_loss': '^', 'train_auc': 's', 'val_auc': 'D'}
        
        # Plot Loss on primary y-axis
        if loss_cols:
            # Plot worker backgrounds first for loss
            for loss_col in loss_cols:
                worker_loss_cols = get_metric_columns(per_worker_training_logs[0] if per_worker_training_logs else pd.DataFrame(), 'loss')
                matching_worker_col = None
                for wcol in worker_loss_cols:
                    if loss_col.split('_')[-1] in wcol:  # Match train/val prefix
                        matching_worker_col = wcol
                        break
                
                if matching_worker_col:
                    # Use different colors for train/val worker backgrounds
                    metric_type = 'train_loss' if 'train' in loss_col else 'val_loss'
                    worker_values = []
                    for i, worker_df in enumerate(per_worker_training_logs):
                        if matching_worker_col in worker_df.columns:
                            x_vals = worker_df['step'] if 'step' in worker_df.columns else worker_df.index
                            y_vals = worker_df[matching_worker_col]
                            ax1.plot(x_vals, y_vals, alpha=0.15, linewidth=1, color=colors[metric_type])
                            worker_values.append(y_vals.values)
                    
                    # Plot worker mean
                    if worker_values:
                        min_length = min(len(vals) for vals in worker_values)
                        truncated_values = [vals[:min_length] for vals in worker_values]
                        mean_values = np.mean(truncated_values, axis=0)
                        if per_worker_training_logs:
                            x_vals = per_worker_training_logs[0]['step'][:min_length] if 'step' in per_worker_training_logs[0].columns else range(min_length)
                            ax1.plot(x_vals, mean_values, '--', linewidth=1.5, 
                                   label=f'Workers Mean ({loss_col.replace("_", " ").title()})', 
                                   color=colors[metric_type], alpha=0.7)
            
            # Plot global model loss (main lines)
            for loss_col in loss_cols:
                x_vals = global_metric_before_round['round_id']
                y_vals = global_metric_before_round[loss_col]
                metric_type = 'train_loss' if 'train' in loss_col else 'val_loss'
                label = f"Global Model ({loss_col.replace('_', ' ').title()})"
                ax1.plot(x_vals, y_vals, marker=markers[metric_type], linestyle='-', 
                        linewidth=3, markersize=6, label=label, color=colors[metric_type])
            
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='darkred')
            ax1.tick_params(axis='y', labelcolor='darkred')
        
        # Create secondary y-axis for AUC
        if auc_cols:
            ax2 = ax1.twinx()
            
            # Plot worker backgrounds first for AUC
            for auc_col in auc_cols:
                worker_auc_cols = get_metric_columns(per_worker_training_logs[0] if per_worker_training_logs else pd.DataFrame(), 'auc')
                matching_worker_col = None
                for wcol in worker_auc_cols:
                    if auc_col.split('_')[-1] in wcol:  # Match train/val prefix
                        matching_worker_col = wcol
                        break
                
                if matching_worker_col:
                    # Use different colors for train/val worker backgrounds
                    metric_type = 'train_auc' if 'train' in auc_col else 'val_auc'
                    worker_values = []
                    for i, worker_df in enumerate(per_worker_training_logs):
                        if matching_worker_col in worker_df.columns:
                            x_vals = worker_df['step'] if 'step' in worker_df.columns else worker_df.index
                            y_vals = worker_df[matching_worker_col]
                            ax2.plot(x_vals, y_vals, alpha=0.15, linewidth=1, color=colors[metric_type])
                            worker_values.append(y_vals.values)
                    
                    # Plot worker mean
                    if worker_values:
                        min_length = min(len(vals) for vals in worker_values)
                        truncated_values = [vals[:min_length] for vals in worker_values]
                        mean_values = np.mean(truncated_values, axis=0)
                        if per_worker_training_logs:
                            x_vals = per_worker_training_logs[0]['step'][:min_length] if 'step' in per_worker_training_logs[0].columns else range(min_length)
                            ax2.plot(x_vals, mean_values, '--', linewidth=1.5, 
                                   label=f'Workers Mean ({auc_col.replace("_", " ").title()})', 
                                   color=colors[metric_type], alpha=0.7)
            
            # Plot global model AUC (main lines)
            for auc_col in auc_cols:
                x_vals = global_metric_before_round['round_id']
                y_vals = global_metric_before_round[auc_col]
                metric_type = 'train_auc' if 'train' in auc_col else 'val_auc'
                label = f"Global Model ({auc_col.replace('_', ' ').title()})"
                ax2.plot(x_vals, y_vals, marker=markers[metric_type], linestyle='-', 
                        linewidth=3, markersize=6, label=label, color=colors[metric_type])
            
            ax2.set_ylabel('AUC', fontsize=12, fontweight='bold', color='darkblue')
            ax2.tick_params(axis='y', labelcolor='darkblue')
        
        # Common settings
        ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax1.set_title('Global Model Loss and AUC vs Training Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Set integer ticks on x-axis
        max_round = global_metric_before_round['round_id'].max()
        ax1.set_xticks(range(0, max_round + 1))
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = (ax2.get_legend_handles_labels() if auc_cols else ([], []))
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
        
        plt.tight_layout()
        plt.show()
    
    # 2. Plot WZ Training Logs (specific metrics, combined across workers)
    if per_wz_training_logs and len(per_wz_training_logs) > 0:
        # Define the specific metrics we want to plot
        desired_metrics = ['loss', 'mape%', 'mse', 'rate_bits', 'real_bit_r']
        
        # Combine all worker DataFrames into one
        combined_wz_df = pd.concat(per_wz_training_logs, ignore_index=True)
        combined_wz_df = combined_wz_df.sort_values('real_step').reset_index(drop=True)
        
        # Find available metrics with train/val prefixes
        available_metrics = []
        for metric in desired_metrics:
            train_col = f'train_{metric}'
            val_col = f'val_{metric}'
            if train_col in combined_wz_df.columns or val_col in combined_wz_df.columns:
                available_metrics.append(metric)
        
        if available_metrics:
            # Create subplots for each metric
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Calculate discontinuity points (round + agent_id/worker_count)
            worker_count = len(per_wz_training_logs)
            max_round = combined_wz_df['round_id'].max()
            discontinuity_points = []
            for round_id in range(1,max_round + 1):
                for agent_id in range(worker_count):
                    discontinuity_points.append(round_id + agent_id / worker_count)
            
            for i, metric in enumerate(available_metrics):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    continue
                
                # Plot train and val metrics separately
                train_col = f'train_{metric}'
                val_col = f'val_{metric}'
                
                # Plot train metric if available
                if train_col in combined_wz_df.columns:
                    x_vals = combined_wz_df['real_step']
                    y_vals = combined_wz_df[train_col]
                    ax.plot(x_vals, y_vals, '-', linewidth=1,
                            label=f'Train {metric.replace("_", " ").title()}', color='blue', alpha=0.8)
                
                # Plot val metric if available
                if val_col in combined_wz_df.columns:
                    x_vals = combined_wz_df['real_step']
                    y_vals = combined_wz_df[val_col]
                    ax.plot(x_vals, y_vals, '--', linewidth=1,
                            label=f'Val {metric.replace("_", " ").title()}', color='red', alpha=0.8)
                
                # Add vertical lines at discontinuity points
                y_min, y_max = ax.get_ylim()
                for disc_point in discontinuity_points:
                    if disc_point <= combined_wz_df['real_step'].max():
                        ax.axvline(x=disc_point, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
                
                # Add worker labels at discontinuity points
                # Calculate positions for worker labels (at round boundaries)
                worker_label_positions = []
                worker_labels = []
                for round_id in range(1, max_round + 1):
                    for agent_id in range(worker_count):
                        pos = round_id + agent_id / worker_count
                        if pos <= combined_wz_df['real_step'].max():
                            worker_label_positions.append(pos)
                            worker_labels.append(f'w{agent_id + 1}')
                
                # Set custom x-ticks to include worker labels
                existing_ticks = ax.get_xticks()
                all_tick_positions = list(existing_ticks) + worker_label_positions
                all_tick_labels = [f'{int(tick)}' if tick in existing_ticks else '' for tick in existing_ticks]
                
                # Create secondary x-axis for worker labels
                ax_top = ax.twiny()
                ax_top.set_xlim(ax.get_xlim())
                ax_top.set_xticks(worker_label_positions)
                ax_top.set_xticklabels(worker_labels, fontsize=8, rotation=45, alpha=0.7)
                ax_top.tick_params(axis='x', which='major', pad=0, length=3)
                
                # Beautify the plot
                metric_name = metric.replace('_', ' ').title()
                ax.set_xlabel('Real Step', fontsize=10, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
                ax.set_title(f'WZ Training: {metric_name}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
            
            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('WZ Model Training Metrics (Combined Workers)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    # 3. Scatter plots: Bit rates vs MSE/MAPE% colored by real_step
    if per_wz_training_logs and len(per_wz_training_logs) > 0:
        # Combine all worker DataFrames into one
        combined_wz_df = pd.concat(per_wz_training_logs, ignore_index=True)
        combined_wz_df = combined_wz_df.sort_values('real_step').reset_index(drop=True)
        
        # Check if required columns exist
        required_cols = ['train_rate_bits', 'train_real_bit_r', 'train_mse', 'train_mape%']
        available_cols = [col for col in required_cols if col in combined_wz_df.columns]
        
        if len(available_cols) >= 3:  # Need at least one bit rate and one metric
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Convert MSE to decibels (10 * log10(MSE))
            if 'train_mse' in combined_wz_df.columns:
                # Avoid log of zero by adding small epsilon
                mse_db = 10 * np.log10(combined_wz_df['train_mse'] + 1e-10)
            else:
                mse_db = None
            
            # Convert MAPE% to decibels (10 * log10(MAPE%))
            if 'train_mape%' in combined_wz_df.columns:
                # Avoid log of zero by adding small epsilon
                mape_db = 10 * np.log10(combined_wz_df['train_mape%'] + 1e-10)
            else:
                mape_db = None
            
            # Get bit rate values
            rate_bits = combined_wz_df['train_rate_bits'] if 'train_rate_bits' in combined_wz_df.columns else None
            real_bit_r = combined_wz_df['train_real_bit_r'] if 'train_real_bit_r' in combined_wz_df.columns else None
            
            # Get real_step for coloring
            real_step_values = combined_wz_df['real_step']
            
            # Create color map - using plasma for better directional visualization
            scatter_alpha = 0.7
            colormap = 'plasma'  # Better for showing progression (purple to yellow)
            
            # Plot 1: rate_bits vs MSE (dB)
            if rate_bits is not None and mse_db is not None:
                scatter = axes[0, 0].scatter(rate_bits, mse_db, c=real_step_values, 
                                           alpha=scatter_alpha, cmap=colormap, s=20)
                axes[0, 0].set_xlabel('Rate Bits', fontsize=10, fontweight='bold')
                axes[0, 0].set_ylabel('MSE (dB)', fontsize=10, fontweight='bold')
                axes[0, 0].set_title('Rate Bits vs MSE (dB)', fontsize=12, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 0], label='Real Step')
            else:
                axes[0, 0].text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Rate Bits vs MSE (dB) - No Data', fontsize=12)
            
            # Plot 2: real_bit_r vs MSE (dB)
            if real_bit_r is not None and mse_db is not None:
                scatter = axes[0, 1].scatter(real_bit_r, mse_db, c=real_step_values, 
                                           alpha=scatter_alpha, cmap=colormap, s=20)
                axes[0, 1].set_xlabel('Real Bit R', fontsize=10, fontweight='bold')
                axes[0, 1].set_ylabel('MSE (dB)', fontsize=10, fontweight='bold')
                axes[0, 1].set_title('Real Bit R vs MSE (dB)', fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 1], label='Real Step')
            else:
                axes[0, 1].text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Real Bit R vs MSE (dB) - No Data', fontsize=12)
            
            # Plot 3: rate_bits vs MAPE% (dB)
            if rate_bits is not None and mape_db is not None:
                scatter = axes[1, 0].scatter(rate_bits, mape_db, c=real_step_values, 
                                           alpha=scatter_alpha, cmap=colormap, s=20)
                axes[1, 0].set_xlabel('Rate Bits', fontsize=10, fontweight='bold')
                axes[1, 0].set_ylabel('MAPE% (dB)', fontsize=10, fontweight='bold')
                axes[1, 0].set_title('Rate Bits vs MAPE% (dB)', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 0], label='Real Step')
            else:
                axes[1, 0].text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Rate Bits vs MAPE% (dB) - No Data', fontsize=12)
            
            # Plot 4: real_bit_r vs MAPE% (dB)
            if real_bit_r is not None and mape_db is not None:
                scatter = axes[1, 1].scatter(real_bit_r, mape_db, c=real_step_values, 
                                           alpha=scatter_alpha, cmap=colormap, s=20)
                axes[1, 1].set_xlabel('Real Bit R', fontsize=10, fontweight='bold')
                axes[1, 1].set_ylabel('MAPE% (dB)', fontsize=10, fontweight='bold')
                axes[1, 1].set_title('Real Bit R vs MAPE% (dB)', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Real Step')
            else:
                axes[1, 1].text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Real Bit R vs MAPE% (dB) - No Data', fontsize=12)
            
            plt.suptitle('WZ Training: Bit Rates vs Performance Metrics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()



if __name__ == "__main__":
    from components.FL_sim import _main_test, FLSimulator
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
    from components.broadcast_components.reporting_utilities import BroadcastMetricGatheringUtilities, plot_stats

    model, dataset, dataset_test = _main_test()

    k=3

    # *****************
    # user_logger = UnifiedLoggingClass(k)

    # *****************
    temp=get_unified_data_tables("_dev_debug_test", k)
    plot_all_metrics(*temp)

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