import sys
from typing import Dict

import numpy as np
import torch

from components.broadcast_components.compressor.entropy_coding import entropy_coding
from components.broadcast_components.quantizer.simple import simple_quantize
from components.other_utilities.models_to_train import ResNetPLModel


class ReportingUtilities:
    def __init__(self, ):
        model_instance = ResNetPLModel(num_classes=10, resnet_version='resnet18')

        self.param_keys = [k for k, v in model_instance.named_parameters() if v.requires_grad]

        self.org_grad_dict_list = {k: [] for k in self.param_keys}
        self.recons_err_perc_dict_list = {k: [] for k in self.param_keys}
        self.byte_mb_size_per_method = {tt: [] for tt in ['raw', 'wz', 'entropy']}
        self.total_error = []

        self.state_report_per_round = []

    def record_compr_stats(self, func):
        from components.broadcast_components.broadcasting_process.WZ_broadcast import dict_to_array_and_normalize

        def wrapper(worker_grad_dict, agent_id):
            assert agent_id == len(self.org_grad_dict_list[self.param_keys[0]]), \
                "something wrong with the order of execution, agent_id given too soon"

            # record org grads sent ---------------
            for k in self.param_keys:
                self.org_grad_dict_list[k].append(worker_grad_dict[k])

            org_data_byte_size = sum(v.nbytes for v in worker_grad_dict.values()) / 1024 ** 2

            # Call the original function to get wz encoded data ---------------
            encoded_data, min_v, max_v, dtype = func(worker_grad_dict, agent_id)

            min_max_vec_size=2 * min_v[0].dtype.itemsize * len(min_v) / 1024 ** 2
            enc_data_byte_size = (len(encoded_data) / 1024 ** 2 + min_max_vec_size)

            # simulate entropy only coding ------------------------------
            entr_grad_flat_normal = dict_to_array_and_normalize(worker_grad_dict, min_v, max_v)
            entr_quantized_data = simple_quantize(entr_grad_flat_normal)
            entr_encoded_data = entropy_coding(entr_quantized_data)

            entr_enc_data_byte_size = len(entr_encoded_data) / 1024 ** 2

            # recoding ------------------------------
            self.byte_mb_size_per_method['raw'].append(org_data_byte_size)
            self.byte_mb_size_per_method['wz'].append(enc_data_byte_size)
            self.byte_mb_size_per_method['entropy'].append(entr_enc_data_byte_size)

            return encoded_data, min_v, max_v, dtype

        return wrapper

    def report_reconst_wrapper(self, func):
        def wrapper(worker_broadcast_data, agent_id, worker_count, global_model_dims, previous_data, ):
            assert agent_id == len(self.recons_err_perc_dict_list[self.param_keys[0]]), \
                "something wrong with the order of execution, agent_id given too soon"

            result_dict = func(worker_broadcast_data, agent_id, worker_count, global_model_dims, previous_data, )

            error_dict = {}
            total_error = 0
            total_element_count = sum(v[0].numel() for v in self.org_grad_dict_list.values())
            for k, v in result_dict.items():
                v = torch.Tensor(v).cuda()
                temp = self.org_grad_dict_list[k][agent_id]
                error_dict[k] = (v - temp).abs().sum()
                error_dict[k] = (error_dict[k] / temp.abs().sum()).item()
                total_error += error_dict[k] * (temp.numel() / total_element_count)

            self.recons_err_perc_dict_list = {k: v + [error_dict[k]]
                                              for k, v in self.recons_err_perc_dict_list.items()}
            self.total_error.append(total_error)

            # Collecting decompression error statistics and reset ---------------
            temp = agent_id == worker_count - 1
            if self.total_error is None or temp:
                if temp:
                    self.state_report_per_round.append({
                        'MB size info per agent': self.byte_mb_size_per_method,
                        '% error per layer per agent': self.recons_err_perc_dict_list,
                        '% total error per agent': self.total_error,
                    })
                self.org_grad_dict_list = {k: [] for k in self.param_keys}
                self.byte_mb_size_per_method = {tt: [] for tt in ['raw', 'wz', 'entropy']}
                self.recons_err_perc_dict_list = {k: [] for k in self.param_keys}
                self.total_error = []

            return result_dict

        return wrapper


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol
from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol, \
    dict_to_array_and_normalize, recover_shape_and_denormal_to_dict
from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding


def get_obj_size(obj):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple)):
        return sum(get_obj_size(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(get_obj_size(k) + get_obj_size(v) for k, v in obj.items())
    else:
        return sys.getsizeof(obj)


class BroadcastReporter:
    def __init__(self, broadcast_prot:WZBroadcastProtocol):
        self.broadcast_protocol = broadcast_prot
        self.stats = {
            'wz': {'bytes_sent': [], 'bytes_received': [], 'mse': [], 'mape': []},
            'raw': {'bytes_sent': [], 'mse': [], 'mape': []},
            'entropy': {'bytes_sent': [], 'mse': [], 'mape': []}
        }
        self.original_grads = None


if __name__ == '__main__':
    import pprint
    from components.other_utilities.models_to_train import ResNetPLModel
    from experiments.resnet_parameter_corr_between_worker import load_grad_files

    # --------------------------------
    torch.set_float32_matmul_precision('high')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    # --------------------------------
    num_agents = 4
    broadcast_prot = WZBroadcastProtocol(agent_count=num_agents, quantizer_type='RNN',
        lr=1e-3, bins_per_plane=2, num_planes=3, metric_report_flag=True, train_sample_size=100_000)
    broadcast_reporter = BroadcastReporter(broadcast_prot)

    # --------------------------------
    model_shape_dict = {
        k: v.shape
        for k, v in ResNetPLModel(num_classes=10, resnet_version='resnet18').named_parameters() if v.requires_grad
    }

    # load testing data --------------------------------
    # This loads gradients from a previous experiment to simulate workers
    temp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    grad_test_data = load_grad_files(
        temp, list(model_shape_dict.keys()),
        [f"../../experiments/exp_data/gradients_resnet/adam/gradients_resnet_t{2}/" for i in range(2)],
        curr_round=0, current_epoch=0
    )
    grad_test_data = [
        {k: torch.tensor(v[i][j]).reshape(model_shape_dict[k]).to('cuda')
         for k, v in grad_test_data.items()}
        for j in range(2) for i in range(len(list(grad_test_data.values())[0]))
    ]
    # grad_test_data = [grad_test_data[:len(temp)], grad_test_data[len(temp):]]
    grad_test_data = grad_test_data[:num_agents]


    # simulate the WZ encoding and reconstruction process --------------------------------
    previous_data = []
    for agent_id in range(num_agents):
        # Server to worker communication
        server_data = broadcast_reporter.to_worker_from_server_data_transfer(agent_id)

        # Worker to server communication
        worker_grad_dict = grad_test_data[agent_id]
        worker_data = broadcast_reporter.to_server_from_worker_data_transfer(agent_id, worker_grad_dict, server_data)

        # Server reconstructs the gradients
        reconstructed_grads = broadcast_reporter.reconstruction_process(
            agent_id, worker_data, num_agents, model_shape_dict, previous_data
        )
        previous_data.append(reconstructed_grads)

    # report --------------------------------
    print("Compression Reporting:")
    stats = broadcast_reporter.get_stats()
    pprint.pprint(stats)

