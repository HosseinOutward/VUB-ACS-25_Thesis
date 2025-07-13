from typing import List, Dict
import numpy as np
import torch
from lightning import seed_everything

from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding
from components.broadcast_components.quantizer.wz_quant_ANN import WZQuantizer, PL_EncoderDecoder_ANN
from components.broadcast_components.quantizer.wz_quant_RNN import PL_EncoderDecoder_RNN


def dict_to_array_and_normalize(grad_dict: Dict):
    # min_v, max_v = [
    #     [f(v).to('cpu').numpy() for k, v in grad_dict.items()] for f in [torch.min, torch.max]]

    clip_range_min, clip_range_max = [
        [f(v.to('cpu').numpy()) for k, v in grad_dict.items()]
        for f in [lambda x: np.percentile(x,0.002), lambda x: np.percentile(x,99.998)] ]
    grad_dict = {k: torch.clip(v, clip_range_min[i], clip_range_max[i])
                 for i, (k, v) in enumerate(grad_dict.items())}

    min_v, max_v = [
        [f(v.to('cpu').numpy()) for k, v in grad_dict.items()]
        for f in [lambda x: np.percentile(x,0.01), lambda x: np.percentile(x,99.99)] ]

    res = []
    for i, (k, v) in enumerate(grad_dict.items()):
        v = v.ravel() * 1000
        v = (v - min_v[i] * 1000) / (max_v[i] * 1000 - min_v[i] * 1000)
        v = v * 2 - 1  # normalize to [-1, 1]
        res.append(v.to('cpu').numpy())
    res = np.concatenate(res)
    return res, min_v, max_v


def recover_shape_and_denormal_to_dict(grad_vector, org_shapes_dict, min_v: List, max_v: List):
    res = {}
    start = 0
    for i, (k, shape) in enumerate(org_shapes_dict.items()):
        end = start + np.prod(shape)
        v = grad_vector[start:end]
        v = (v + 1) / 2
        v = v * (max_v[i] - min_v[i]) + min_v[i]
        res[k] = v.reshape(shape)
        start = end
    return res


class WZBroadcastProtocol:
    def __init__(self, agent_count, quantizer_type='RNN', *args, **kwargs):
        self.side_info_data_list = None
        self.agent_list_check = []
        self.warmup = True
        self.wz_pl_model_class = {'ANN': PL_EncoderDecoder_ANN, 'RNN': PL_EncoderDecoder_RNN}[quantizer_type]

        self.wz_quantizer_list: List[WZQuantizer] = [
            WZQuantizer(
                wz_pl_model=self.wz_pl_model_class(
                    inp_dim=1, side_info_size=1, **kwargs),
                count_side_info_data=0) for _ in range(agent_count)]
        self.last_recent_grads_list = [None] * agent_count

    # %%
    def wz_encoding_process(self, worker_grad_dict, agent_id):
        # worker_grad_dict={k:v*1.1 for k, v in worker_grad_dict.items()}
        # return worker_grad_dict, min_v, max_v, 0

        grad_flat_normal, min_v, max_v = dict_to_array_and_normalize(worker_grad_dict)

        quantizer = self.wz_quantizer_list[agent_id]
        quantized_data = quantizer.encoding_process(grad_flat_normal)

        dtype = quantized_data.dtype
        encoded_data = entropy_coding(quantized_data)

        return encoded_data, min_v, max_v, dtype

    def wz_reconstruction_process(self, worker_broadcast_data, agent_id,
                                  worker_count, global_model_dims, previous_data):
        # assuming that previous_data has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a==i%worker_count for i,a in enumerate(self.agent_list_check)])

        # return worker_broadcast_data[0]

        quantizer = self.wz_quantizer_list[agent_id]

        encoded_data, min_v, max_v, dtype = worker_broadcast_data

        quantized_decoded_data = entropy_decoding(encoded_data, dtype)

        model_size = np.sum([np.prod(shape) for shape in global_model_dims.values()])

        side_info_data_list = [] if self.warmup else self.side_info_data_list
        res_vector = quantizer.decoding_process(quantized_decoded_data, side_info_data_list, model_size)

        result_dict = recover_shape_and_denormal_to_dict(res_vector, global_model_dims, min_v, max_v)

        result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

        if agent_id + 1 >= worker_count:
            self.warmup = False

        if not self.warmup:
            self.prep_for_next_agent(agent_id, worker_count, res_vector, previous_data, min_v, max_v)

        return result_dict

    def prep_for_next_agent(self, agent_id, worker_count, res_vector, previous_data, min_v, max_v):
        prev_d_flat = [dict_to_array_and_normalize(pd, min_v, max_v) for pd in previous_data]
        prev_d_flat += [res_vector]

        last_recent_grads_idx = len(prev_d_flat) - worker_count
        self.side_info_data_list = prev_d_flat[:last_recent_grads_idx] + prev_d_flat[last_recent_grads_idx + 1:]
        last_recent_grads = prev_d_flat[last_recent_grads_idx]

        next_agent = (agent_id + 1) % worker_count
        qz = self.wz_quantizer_list[next_agent]
        self.wz_quantizer_list[next_agent] = WZQuantizer(
            wz_pl_model=self.wz_pl_model_class(
                inp_dim=1, side_info_size=len(self.side_info_data_list),
                lr=qz.wz_pl_model.lr,
                bins_per_plane=qz.bins_per_plane if hasattr(qz, 'bins_per_plane') else None,
                num_planes=qz.planes if hasattr(qz, 'planes') else None,
            ),
            count_side_info_data=len(self.side_info_data_list),
            metric_report_flag=qz.metric_report_flag, train_sample_size=qz.train_sample_size
        )
        self.wz_quantizer_list[next_agent].train_model(
            last_recent_grads, self.side_info_data_list, batch_size=10_000)


if __name__ == "__main__":
    # --------------------------------
    torch.set_float32_matmul_precision('medium')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    worker_count = 4
    rounds = 3

    # load testing data --------------------------------
    seed_everything(42)
    model_shape_dict = {
        f'aaa_{i}': (*np.random.randint(1, 10, size=np.random.randint(3)),
            (np.random.randint(10_000, 100_000)*1000)//1000)
        for i in range(10)
    }

    grad_test_data = [
            [{k: torch.randn(v).to('cuda') * 2 - 1 for k, v in model_shape_dict.items()}
            for _ in range(worker_count)]
        for _ in range(rounds)]

    # simulate the WZ encoding and reconstruction process --------------------------------
    broadcast_prot = WZBroadcastProtocol(worker_count,'RNN',
                train_sample_size=100_000, metric_report_flag=True, lr=1e-5, num_planes=3, bins_per_plane=2)
    prev = []
    for round, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            broadcast_data = broadcast_prot.wz_encoding_process(grad, ag_id)
            decoded_agent_broadcast = broadcast_prot.wz_reconstruction_process(
                broadcast_data, ag_id, worker_count, model_shape_dict, prev, )
            prev.append(decoded_agent_broadcast)

    # check output size and correctness
    for i, grad in enumerate(grad_test_data[-1]):
        assert all([k in grad for k in model_shape_dict.keys()])
        assert all([v.shape == model_shape_dict[k] for k, v in grad.items()])
