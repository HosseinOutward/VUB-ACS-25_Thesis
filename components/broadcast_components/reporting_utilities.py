import copy
import sys
import numpy as np
import torch
from lightning import seed_everything

from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding


def get_obj_size(obj):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(get_obj_size(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(get_obj_size(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj)


class BroadcastReportingUtilities:
    def __init__(self, broadcast_prot:WZBroadcastProtocol):
        self.broadcast_protocol = broadcast_prot
        self.base_stat_dict = {
            'wz': {'mbytes_moved_total': [], 'mbytes_sent_to_worker': [], 'mse': [], 'mape%': []},
            'raw16': {'mbytes_moved_total': [], 'mse': [], 'mape%': []},
            'entropy': {'mbytes_moved_total': [], 'mse': [], 'mape%': []}
        }
        self.stats = copy.deepcopy(self.base_stat_dict)
        self.running_stats = copy.deepcopy(self.base_stat_dict)
        self.original_grads = None
        self.current_agent_id = None

    def reset_running_stats_round_end(self):
        for method_used in self.running_stats.keys():
            for k, v in self.running_stats[method_used].items():
                self.stats[method_used][k].append(v)

        self.running_stats = copy.deepcopy(self.base_stat_dict)

    def to_worker_from_server_data_transfer(self, agent_id):
        b_p_res = self.broadcast_protocol.to_worker_from_server_data_transfer(agent_id)

        # accounting for received data size
        wz_received_size = get_obj_size(b_p_res)
        self.running_stats['wz']['mbytes_sent_to_worker'].append(wz_received_size / (1024 * 1024))

        return b_p_res

    def to_server_from_worker_data_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        # deep copy to avoid modification of original gradients
        self.original_grads = copy.deepcopy(grad_dict)
        self.current_agent_id = agent_id

        b_p_res = self.broadcast_protocol.to_server_from_worker_data_transfer(
            agent_id, grad_dict, encoder_data_sent_by_server)

        # accounting for sent data size
        self.running_stats['wz']['mbytes_moved_total'].append(get_obj_size(b_p_res) / (1024 * 1024) +
                                                              self.running_stats['wz']['mbytes_sent_to_worker'][-1])
        assert len(self.running_stats['wz']['mbytes_moved_total']) == len(self.running_stats['wz']['mbytes_sent_to_worker'])

        return b_p_res

    def encoding_process(self, agent_id, worker_grad_dict, prob_per_bin):
        # deep copy to avoid modification of original gradients
        self.original_grads = copy.deepcopy(worker_grad_dict)

        b_p_res = self.broadcast_protocol.encoding_process(agent_id, worker_grad_dict, prob_per_bin)
        return b_p_res

    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, *args, **kwargs):
        assert self.current_agent_id == agent_id, "Current agent ID does not match the provided agent ID."

        reconstructed_grads = self.broadcast_protocol.reconstruction_process(
            agent_id, worker_broadcast_data, worker_count, *args, **kwargs)

        original_flat = np.concatenate([v.flatten().cpu()
                                        for v in self.original_grads.values()])
        reconstructed_flat_wz = np.concatenate([v.flatten().cpu()
                                                for v in reconstructed_grads.values()])
        assert original_flat.dtype == torch.float16 and reconstructed_flat_wz.dtype == torch.float16

        # WZ comparison
        mse_f = lambda x,y: np.mean((x-y) ** 2)
        mape_f = lambda x,y: np.mean(np.abs(x - y)) / np.mean(np.abs(x) + 1e-8) * 100
        self.running_stats['wz']['mse'].append(mse_f(original_flat, reconstructed_flat_wz))
        self.running_stats['wz']['mape%'].append(mape_f(original_flat, reconstructed_flat_wz))

        # Raw comparison
        raw_size = get_obj_size(original_flat)
        self.running_stats['raw16']['mbytes_moved_total'].append(raw_size / (1024 * 1024))
        self.running_stats['raw16']['mse'].append(0)
        self.running_stats['raw16']['mape%'].append(0)

        # Entropy comparison
        entropy_encoded_data = entropy_coding(original_flat)
        recons_entropy = entropy_decoding(entropy_encoded_data, original_flat.dtype)
        entropy_size = get_obj_size(entropy_encoded_data)
        self.running_stats['entropy']['mbytes_moved_total'].append(entropy_size / (1024 * 1024))
        self.running_stats['entropy']['mse'].append(mse_f(original_flat, recons_entropy))
        self.running_stats['entropy']['mape%'].append(mape_f(original_flat, recons_entropy))

        if len(self.running_stats['entropy']['mape%'])==worker_count:
            self.reset_running_stats_round_end()

        return reconstructed_grads


if __name__ == '__main__':
    import pprint
    from components.other_utilities.models_to_train import ResNetPLModel
    from experiments.resnet_parameter_corr_between_worker import load_grad_files

    # --------------------------------
    torch.set_float32_matmul_precision('medium')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    # --------------------------------
    worker_count = 2
    rounds = 2
    seed_everything(42)

    # load testing data --------------------------------
    model_shape_dict = {
        f'aaa_{i}': (*np.random.randint(1, 5, size=np.random.randint(3)),
            (np.random.randint(10_000, 100_000)*1000)//1000)
        for i in range(10)
    }

    grad_test_data = [
            [{k: torch.normal(0,1,size=v).to('cuda') * 2 - 1 for k, v in model_shape_dict.items()}
            for _ in range(worker_count)]
        for _ in range(rounds)]

    broadcast_prot_base = WZBroadcastProtocol(worker_count,'RNN',
            train_sample_size=100_000, metric_report_flag=True, lr=1e-5, num_planes=3, bins_per_plane=2)
    broadcast_prot = BroadcastReportingUtilities(broadcast_prot_base)

    # simulate the WZ encoding and reconstruction process --------------------------------
    prev = []
    for round, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            print(f'>> Round {round}, Agent {ag_id}')
            server_data_sent_to_worker = broadcast_prot.to_worker_from_server_data_transfer(ag_id)
            encoded_ag_broadcast = broadcast_prot.to_server_from_worker_data_transfer(
                            ag_id, grad, server_data_sent_to_worker)

            decoded_agent_broadcast = broadcast_prot.reconstruction_process(
                ag_id, encoded_ag_broadcast, worker_count, model_shape_dict, prev, )

            prev.append(decoded_agent_broadcast)

    # report --------------------------------
    print("Compression Reporting:")
    pprint.pprint(broadcast_prot.running_stats)
