import copy
from copy import deepcopy
import numpy as np
import torch
from lightning import seed_everything
from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol, \
    change_dtype_recursive, compress_data_list
from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding
from components.other_utilities.user_logger import UnifiedLoggingClass


def get_obj_size(obj):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(get_obj_size(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(get_obj_size(v) for k, v in obj.items())
    elif hasattr(obj, '_dtype') and hasattr(obj, '__len__'):
        return len(obj) * (obj._dtype.bitwidth // 8)
    elif isinstance(obj, bytes):
        return len(obj)
    else:
        raise


class BroadcastMetricGatheringUtilities:
    def __init__(self, broadcast_prot:WZBroadcastProtocol, user_logger:UnifiedLoggingClass=None):
        self.broadcast_protocol = broadcast_prot
        self.base_stat_dict = {
            'wz': {'mbytes_recived': [], 'mbytes_sent_to_worker': [], 'mse': [],
                   'mape%': [], 'mae': [], 'mbytes_sent_for_aggre': []},
            'raw16': {'mbytes_recived': [], 'mse': [], 'mape%': [], 'mae': [], 'mbytes_sent_for_aggre': []},
            'entropy': {'mbytes_recived': [], 'mse': [], 'mape%': [], 'mae': [], 'mbytes_sent_for_aggre': []},
        }
        self.stats = copy.deepcopy(self.base_stat_dict)
        self.running_stats = copy.deepcopy(self.base_stat_dict)
        self.original_grads = None
        self.current_agent_id = None
        self.user_logger = user_logger

    def reset_running_stats_round_end(self):
        for method_used in self.running_stats.keys():
            for k, v in self.running_stats[method_used].items():
                self.stats[method_used][k].append(v)

        self.running_stats = copy.deepcopy(self.base_stat_dict)

        # log at round start
        if self.user_logger:
            self.user_logger.broadcast_reporting(self.stats)

    def to_worker_prep_data_for_transfer(self, agent_id):
        b_p_res = self.broadcast_protocol.to_worker_prep_data_for_transfer(agent_id)

        # accounting for received data size
        wz_received_size = get_obj_size(b_p_res)
        self.running_stats['wz']['mbytes_sent_to_worker'].append(wz_received_size / (1024 * 1024))

        return b_p_res

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        # deep copy to avoid modification of original gradients
        self.original_grads = copy.deepcopy(grad_dict)
        self.current_agent_id = agent_id

        b_p_res = self.broadcast_protocol.to_server_prep_data_for_transfer(
            agent_id, grad_dict, encoder_data_sent_by_server)

        # accounting for sent data size
        self.running_stats['wz']['mbytes_recived'].append(get_obj_size(b_p_res) / (1024 * 1024))
        assert len(self.running_stats['wz']['mbytes_recived']) ==\
                len(self.running_stats['wz']['mbytes_sent_to_worker'])

        return b_p_res

    def encoding_process(self, agent_id, worker_grad_dict, prob_per_bin):
        # deep copy to avoid modification of original gradients
        self.original_grads = copy.deepcopy(worker_grad_dict)

        b_p_res = self.broadcast_protocol.encoding_process(agent_id, worker_grad_dict, prob_per_bin)
        return b_p_res

    def reconstruction_process(self, *args, **kwargs):
        agent_id = kwargs.get('agent_id') if 'agent_id' in kwargs else args[0]
        assert self.current_agent_id == agent_id, "Current agent ID does not match the provided agent ID."

        reconstructed_grads = self.broadcast_protocol.reconstruction_process(*args, **kwargs)

        original_flat = np.concatenate([v.flatten().cpu().to(torch.float16)
                                        for v in self.original_grads.values()])
        reconstructed_flat_wz = np.concatenate([v.flatten().cpu()
                                                for v in reconstructed_grads.values()])

        # WZ comparison
        mse_f = lambda x,y: np.mean((x-y) ** 2)
        mae_f = lambda x,y: np.mean(np.abs(x - y))
        mape_f = lambda x,y: mae_f(x, y) / np.mean(np.abs(x) + 1e-8) * 100
        self.running_stats['wz']['mse'].append(mse_f(original_flat, reconstructed_flat_wz))
        self.running_stats['wz']['mape%'].append(mape_f(original_flat, reconstructed_flat_wz))
        self.running_stats['wz']['mae'].append(mae_f(original_flat, reconstructed_flat_wz))

        # Raw comparison
        raw_size = get_obj_size(original_flat)
        self.running_stats['raw16']['mbytes_recived'].append(raw_size / (1024 * 1024))
        self.running_stats['raw16']['mse'].append(0)
        self.running_stats['raw16']['mape%'].append(0)
        self.running_stats['raw16']['mae'].append(0)

        # Entropy comparison
        entropy_encoded_data = entropy_coding(original_flat)
        recons_entropy = entropy_decoding(entropy_encoded_data, original_flat.dtype)
        entropy_size = get_obj_size(entropy_encoded_data)
        self.running_stats['entropy']['mbytes_recived'].append(entropy_size / (1024 * 1024))
        self.running_stats['entropy']['mse'].append(mse_f(original_flat, recons_entropy))
        self.running_stats['entropy']['mape%'].append(mape_f(original_flat, recons_entropy))
        self.running_stats['entropy']['mae'].append(mae_f(original_flat, recons_entropy))

        # detect end of round
        if len(self.running_stats['entropy']['mape%'])==worker_count:
            self.reset_running_stats_round_end()

        return reconstructed_grads

    def model_transfer_to_worker_from_server(self, server_model_state_dict):
        recons, compr = self.broadcast_protocol.model_transfer_to_worker_from_server(server_model_state_dict)

        wz_size = get_obj_size(compr)
        self.running_stats['wz']['mbytes_sent_for_aggre'].append(wz_size / (1024 * 1024))

        raw_size = get_obj_size(server_model_state_dict)
        self.running_stats['raw16']['mbytes_sent_for_aggre'].append(raw_size / (1024 * 1024))

        res = change_dtype_recursive(server_model_state_dict, torch.float16)
        res = compress_data_list(res)
        entropy_size = get_obj_size(res)
        self.running_stats['entropy']['mbytes_sent_for_aggre'].append(entropy_size / (1024 * 1024))

        return recons, compr


def plot_stats(stat_dict):
    import matplotlib.pyplot as plt

    # sort stat_dict by these keys ['wz', 'entropy', 'raw16']
    temp = ['wz', 'entropy', 'raw16']
    assert all(k in stat_dict for k in temp), f"Some Key not found in stat_dict: {stat_dict.keys()}"
    stat_dict = {k: deepcopy(stat_dict[k]) for k in temp}

    num_subplots = 3
    fig, ax = plt.subplots(num_subplots, 1, figsize=(15, 4 * num_subplots), sharex=True)

    colors_per_method = {
        'wz': 'tab:blue',
        'raw16': 'tab:purple',
        'entropy': 'tab:green',
    }
    symbol_per_metric = {
        'mbytes_recived': 'o',
        'mbytes_sent_to_worker': 's',
        'mbytes_sent_for_aggre': 'D',
        'mse': 'x',
        'mape%': '^',
    }
    lines_per_metric = {
        'mbytes_recived': '-',
        'mbytes_sent_to_worker': '--',
        'mbytes_sent_for_aggre': '-.',
        'mse': ':',
        'mape%': '-',
    }

    #%%
    # Plotting data transfer sizes on the first subplot (ax[0])
    temp = np.sum(stat_dict['wz']['mbytes_recived'])*0
    for method, metrics in stat_dict.items():
        for k_transfer in ['mbytes_recived', 'mbytes_sent_for_aggre', 'mbytes_sent_to_worker']:
            if k_transfer in metrics:
                temp += np.sum(metrics[k_transfer], axis=1)
        ax[0].plot(temp, label=f'Total transfer - {method}', marker='o', color=colors_per_method[method], alpha=0.9)

    ax[0].set_ylabel('MB')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    ax[0].set_title('Total Data Transfer Size')

    #%%
    # Plotting breakdown of data transfer sizes on the second subplot (ax[1])
    for k_transfer in ['mbytes_recived', 'mbytes_sent_for_aggre']:
        for z_order, (method, metrics) in enumerate(stat_dict.items()):
            plt_name = {
                'mbytes_recived': 'Worker to Server',
                'mbytes_sent_for_aggre': 'Server to Worker (global model)',
            }[k_transfer]
            temp = np.sum(metrics[k_transfer], axis=1)
            offset = z_order*temp*0.01
            ax[1].plot(temp+offset, label=f'{plt_name} - {method}',
                       linestyle=lines_per_metric[k_transfer], marker=symbol_per_metric[k_transfer],
                       color=colors_per_method[method], alpha=0.9, zorder=len(stat_dict) - z_order)
    temp = stat_dict['wz']['mbytes_sent_to_worker']
    ax[1].plot(np.sum(temp, axis=1), linestyle=lines_per_metric[k_transfer],
               label=f'{'Server to Worker (wz encoder)'} - {'wz'}', alpha=0.9, zorder=len(stat_dict))

    ax[1].set_ylabel('MB')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    ax[1].set_title('Breakdown of Data Transfer Size')

    #%%
    # Plotting MSE and MAPE on the second subplot (ax[2]) with a shared x-axis
    ax2 = ax[2].twinx()
    for k_transfer in ['mse', 'mape%']:
        for z_order, (method, metrics) in enumerate(stat_dict.items()):
            ax_to_plot_on = ax[2] if k_transfer == 'mse' else ax2
            ax_to_plot_on.plot(np.mean(metrics[k_transfer], axis=1), label=f'{k_transfer.upper()} - {method}',
                     linestyle=lines_per_metric[k_transfer], marker=symbol_per_metric[k_transfer],
                     color=colors_per_method[method], alpha=0.9, zorder=len(stat_dict) - z_order)

    ax[2].set_xlabel('Rounds')
    ax[2].set_ylabel('MSE', color='tab:blue')
    ax[2].tick_params(axis='y', labelcolor='tab:blue')
    ax[2].grid(True, which='both', axis='y', linestyle='-.', linewidth=0.5)

    ax2.set_ylabel('MAPE %', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax[2].set_title('MSE and MAPE% Comparison')

    # Combine legends from both y-axes
    lines, labels = ax[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    #%%
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.WZ_broadcast import _test_main

    worker_count = 2

    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_3plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                    count_side_info_data=0, enable_progress_bar=True)
    broadcast_prot_base = WZBroadcastProtocol(worker_count, base_quantizer)
    broadcast_prot = BroadcastMetricGatheringUtilities(broadcast_prot_base)

    _test_main(broadcast_prot, worker_count)

    # report --------------------------------
    plot_stats(broadcast_prot.stats)
