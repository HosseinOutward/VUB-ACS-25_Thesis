from typing import List, Dict
import numpy as np
import torch

from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding
from components.broadcast_components.quantizer.simple import simple_quantize, simple_dequantize
from components.broadcast_components.quantizer.wz_quant_ANN import WZQuantizer
from components.broadcast_components.reporting_utilities import ReportingUtilities

# ------------------
# todo: dont use in import file definition
wz_quantizer = WZQuantizer()
# ------------------


def dict_to_array_and_normalize(grad_dict: Dict, min_v: List, max_v: List):
    res = []
    for i, (k, v) in enumerate(grad_dict.items()):
        v = v.ravel() * 1000
        v = (v - min_v[i] * 1000) / (max_v[i] * 1000 - min_v[i] * 1000)
        v = v*2-1  # normalize to [-1, 1]
        res.append(v.to('cpu').numpy())
    res = np.concatenate(res)
    return res


def recover_shape_and_denormal_to_dict(grad_vector, org_shapes_dict, min_v: List, max_v: List):
    res = {}
    start = 0
    for i, (k, shape) in enumerate(org_shapes_dict.items()):
        end = start + np.prod(shape)
        v = grad_vector[start:end]
        v = (v+1)/2
        v = v * (max_v[i] - min_v[i]) + min_v[i]
        res[k] = v.reshape(shape)
        start = end
    return res


def wz_encoding_process(worker_grad_dict, agent_id):
    min_v, max_v = [
        [f(v).to('cpu').numpy() for k, v in worker_grad_dict.items()]
        for f in [torch.min, torch.max]]

    # worker_grad_dict={k:v*1.1 for k, v in worker_grad_dict.items()}
    # return worker_grad_dict, min_v, max_v, 0

    grad_flat_normal = dict_to_array_and_normalize(worker_grad_dict, min_v, max_v)

    quantizer = simple_quantize if agent_id <= 1 else wz_quantizer.encode
    quantized_data = quantizer(grad_flat_normal)

    dtype = quantized_data.dtype
    encoded_data = entropy_coding(quantized_data)

    return encoded_data, min_v, max_v, dtype


def wz_reconstruction_process(worker_broadcast_data, agent_id, worker_count, global_model_dims, previous_data, ):
    # return worker_broadcast_data[0]
    encoded_data, min_v, max_v, dtype = worker_broadcast_data

    quantized_decoded_data = entropy_decoding(encoded_data, dtype)

    if agent_id <= 1:
        res_vector = simple_dequantize(quantized_decoded_data, np.float32)
        if agent_id == 1:
            assert len(previous_data) == 1
            temp = [
                [f(v).to('cpu').numpy() for k, v in previous_data[0].items()]
                for f in [torch.min, torch.max]]
            prev_data = dict_to_array_and_normalize(previous_data[0], *temp)
            wz_quantizer.train_model(res_vector, prev_data, )
    else:
        res_vector = wz_quantizer.decode(quantized_decoded_data, previous_data)

    result_dict = recover_shape_and_denormal_to_dict(
        res_vector, global_model_dims, min_v, max_v)

    result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

    return result_dict

