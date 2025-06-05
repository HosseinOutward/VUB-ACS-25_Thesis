import numpy as np
import torch

from compressor.entropy_coding import entropy_coding
from models_to_train import ResNetPLModel
from quantizer.simple import simple_quantize

# ------------------
r_keys_state = [k for k, v in ResNetPLModel(
    num_classes=10, resnet_version='resnet18').named_parameters() if v.requires_grad]

r_original_dict_list = {k: [] for k in r_keys_state}
r_byte_size_dict_list = {tt: [] for tt in ['raw', 'wz', 'entropy']}
r_decomp_error_dict_list = {k: [] for k in r_keys_state}
r_total_error = []

state_report_per_round = []


# ------------------


def report_compression_stat(func):
    from components.components import dict_to_array_and_normalize

    def wrapper(worker_grad_dict, agent_id):
        global r_original_dict_list, r_byte_size_dict_list, \
            r_decomp_error_dict_list, r_total_error, r_keys_state

        assert agent_id == len(r_original_dict_list[r_keys_state[0]]), \
            "something wrong with the order of execution, agent_id given too soon"

        r_original_dict_list = {k: v + [worker_grad_dict[k]]
                                for k, v in r_original_dict_list.items()}

        # Call the original function to get encoded data ---------------
        encoded_data, min_v, max_v, dtype = func(worker_grad_dict, agent_id)

        # simulate entropy only coding ------------------------------
        entr_grad_flat_normal = dict_to_array_and_normalize(worker_grad_dict, min_v, max_v)
        entr_quantized_data = simple_quantize(entr_grad_flat_normal)
        entr_encoded_data = entropy_coding(entr_quantized_data)

        enc_data_byte_size = len(encoded_data) / 1024**2
        org_data_byte_size = sum(v.nbytes for v in worker_grad_dict.values()) / 1024**2
        entr_enc_data_byte_size = len(entr_encoded_data)/1024**2

        r_byte_size_dict_list['raw'].append(org_data_byte_size)
        r_byte_size_dict_list['wz'].append(enc_data_byte_size)
        r_byte_size_dict_list['entropy'].append(entr_enc_data_byte_size)

        return encoded_data, min_v, max_v, dtype

    return wrapper


def report_decompression_stat(func):
    def wrapper(agent_id, worker_count, global_model_dims, previous_data, worker_broadcast_data):
        global r_original_dict_list, r_byte_size_dict_list, \
            r_decomp_error_dict_list, r_total_error, r_keys_state

        assert agent_id == len(r_decomp_error_dict_list[r_keys_state[0]]), \
            "something wrong with the order of execution, agent_id given too soon"

        result_dict = func(agent_id, worker_count, global_model_dims, previous_data, worker_broadcast_data)

        error_dict = {}
        total_error = 0
        total_element_count = sum(v[0].numel() for v in r_original_dict_list.values())
        for k, v in result_dict.items():
            v = torch.Tensor(v).cuda()
            temp = r_original_dict_list[k][agent_id]
            error_dict[k] = (v - temp).abs().sum()
            error_dict[k] = (error_dict[k] / temp.abs().sum()).item()
            total_error += error_dict[k] * (temp.numel() / total_element_count)

        r_decomp_error_dict_list = {k: v + [error_dict[k]]
                                    for k, v in r_decomp_error_dict_list.items()}
        r_total_error.append(total_error)

        # Collecting decompression error statistics and reset ---------------
        temp = agent_id == worker_count - 1
        if r_total_error is None or temp:
            if temp:
                state_report_per_round.append({
                    'KB size info per agent': r_byte_size_dict_list,
                    '% error per layer per agent': r_decomp_error_dict_list,
                    '% total error per agent': r_total_error,
                })
            r_original_dict_list = {k: [] for k in r_keys_state}
            r_byte_size_dict_list = {tt: [] for tt in ['raw', 'wz', 'entropy']}
            r_decomp_error_dict_list = {k: [] for k in r_keys_state}
            r_total_error = []

        return result_dict

    return wrapper


if __name__ == "__main__":
    from components.components import wz_reconstruction_process, wz_quantizer, wz_encoding_process
    from experiments.resnet_parameter_corr_between_worker import load_grad_files

    model_shape_dict = {
        k: v.shape for k, v in ResNetPLModel(
            num_classes=10, resnet_version='resnet18'
        ).named_parameters() if v.requires_grad
    }

    wz_quantizer.metric_report_flag = True

    temp = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    w_i_grad_dict = load_grad_files(
        temp, r_keys_state,
        [f"../experiments/exp_data/gradients_resnet"
         f"/adam/gradients_resnet_t{i}/" for i in range(2)],
        curr_round=0, current_epoch=0)
    w_i_grad_dict = [
        {k: torch.tensor(v[i][j]).to('cuda') for k, v in w_i_grad_dict.items()}
        for j in range(2) for i in range(len(list(w_i_grad_dict.values())[0]))]
    w_i_grad_dict = [
        {k: v.reshape(model_shape_dict[k]) for k, v in w_i_grad_dict[0].items()}
        for w in w_i_grad_dict]
    w_i_grad_dict = [w_i_grad_dict[:4], w_i_grad_dict[4:]]

    for ww in w_i_grad_dict:
        prev = []
        for ag_id in range(len(ww)):
            broadcast_data = wz_encoding_process(ww[ag_id], ag_id)
            decoded_agent_broadcast = wz_reconstruction_process(
                ag_id, len(ww), model_shape_dict, prev, broadcast_data)
            prev += [decoded_agent_broadcast]

    print("Compression Reporting:")
    for i, report in enumerate(state_report_per_round):
        print(f"Round {i}:")
        print("Byte Size Info per Agent:", report['byte size info per agent'])
        print("% Error per Layer per Agent:", report['% error per layer per agent'])
        print("% Total Error per Agent:", report['% total error per agent'])
        print()
