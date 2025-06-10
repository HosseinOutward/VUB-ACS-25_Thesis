from typing import List, Dict
import numpy as np
import torch

from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding
from components.broadcast_components.quantizer.simple import simple_quantize, simple_dequantize
from components.broadcast_components.quantizer.wz_quant_ANN import WZQuantizer
from components.broadcast_components.reporting_utilities import report_compression_stat, report_decompression_stat

# ------------------
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


@report_compression_stat
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


@report_decompression_stat
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


# -----------------------------------------------------------------------------
# Component test
# -----------------------------------------------------------------------------
# def test_components():
#     """
#     Test the components of the system: quantisation, encoding, and decoding.
#     This function performs the following steps:
#     1. Load the test gradients from a pickle file.
#     2. Individually quantise the gradients using the specified quantisation method.
#     3. Individually encode the quantised gradients using the specified encoding method.
#     4. Decode the entire encoded gradients back to their original form.
#     5. De-quantise the decoded gradients back to their original form.
#     6. Calculate the sizes of the encoded data and the original data.
#     7. Calculate the errors (total error, quantisation error, compression error and error caused by combination).
#     """
#
#     print("\n== Testing components for 1 round w 5 workers ==")
#     print(f'{config.MODE_ENCODER=}, {config.MODE_quantizer=}\n')
#
#     # todo: gradiant ro baraye chnd epoch sum bokon bad bezar
#     test_grads = pickle.load(open("testing_model_grad.pkl", "rb"))
#
#     # ---- quantise & encode ---------------------------------------------------
#     quant = [grad_quantizer(g.copy()) for g in test_grads]
#     encoded = [grad_encoder(q.copy()) for q in quant]
#     decoded = grad_decoder(encoded)
#
#     template = [[name, arr.shape] for name, arr in test_grads[0]]
#     de_quant = grad_de_quantizer(decoded, template)
#
#     # ------------------------------ Report ------------------------------------
#     # ---- sizes ---------------------------------------------------------------
#     kb_data_size = sum(len(a) for a in encoded) / 1024 / 1024
#     original_size = sum(arr.nbytes for d in test_grads for _, arr in d) / 1024 / 1024
#
#     print(f"encoded size:   {kb_data_size:.3f} MB")
#     print(f"original size:  {original_size:.3f} MB")
#     print(f"ratio (enc/orig): {kb_data_size / original_size:.3f}")
#     print(f"ratio (orig/enc): {original_size / kb_data_size:.3f}\n\n")
#
#     # ---- errors --------------------------------------------------------------
#     temp = np.concatenate([arr.ravel() for d in test_grads for _, arr in d])
#     mean_v = np.percentile(temp, [0.1, 99.9])
#     mean_v = np.abs(np.clip(temp, *mean_v)).mean()
#
#     # total error
#     diff_t = np.concatenate(
#         [(de_quant[i][j][1] - test_grads[i][j][1]).ravel()
#          for i in range(len(test_grads)) for j in range(len(test_grads[i]))]
#     )
#     total_e = np.abs(diff_t).mean()
#     print(f"total error:        {total_e / mean_v:.5f}% - v: {total_e:.5f}")
#
#     # pure quantisation error
#     temp_q = grad_de_quantizer(quant, template)
#     diff_q = np.concatenate(
#         [(temp_q[i][j][1] - test_grads[i][j][1]).ravel()
#          for i in range(len(test_grads)) for j in range(len(test_grads[i]))]
#     )
#     quant_error = np.abs(diff_q).mean()
#     print(f"quantisation error: {quant_error / mean_v:.5f}% - v: {quant_error:.5f}")
#
#     # compression error
#     flat_test_grad = [np.concatenate([b.flatten() for _, b in a]) for a in test_grads]
#     temp_c = [grad_encoder(d) for d in flat_test_grad]
#     temp_c = grad_decoder(temp_c, out_dtype=test_grads[0][0][1].dtype)
#     diff_c = np.concatenate([temp_c[i] - flat_test_grad[i] for i in range(len(test_grads))])
#     comp_error = np.abs(diff_c).mean()
#     print(f"compressed error:   {comp_error / mean_v:.5f}% - v: {comp_error:.5f}")
#
#     combo = (total_e - (quant_error + comp_error))
#     print(f"combo error:        {combo / mean_v:.5f}% - v: {combo:.5f}")
#
#
# if __name__ == "__main__":
#     """
#     Test the components of the system with different configurations to see how they perform.
#     """
#
#     list_options = {
#         "encoders": ["raw", "entropy"],
#         "quants": ["raw", "8bit", "wz"],
#     }
#
#     for config.MODE_ENCODER in list_options["encoders"]:
#         for config.MODE_quantizer in list_options["quants"]:
#             print("\n============================")
#             config.dtype = np.uint8 if config.MODE_quantizer == "8bit" else np.float32
#             test_components()
