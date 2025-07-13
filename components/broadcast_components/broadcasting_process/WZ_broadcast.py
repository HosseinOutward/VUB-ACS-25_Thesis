from typing import List, Dict
import numpy as np
import torch
from lightning import seed_everything

from components.broadcast_components.compressor.entropy_coding import entropy_coding, entropy_decoding
from components.broadcast_components.quantizer.wz_quant_ANN import WZQuantizer, PL_EncoderDecoder_ANN
from components.broadcast_components.quantizer.wz_quant_RNN import PL_EncoderDecoder_RNN


# todo combine bias and weight into one key in dict
def dict_to_array_and_normalize(grad_dict: Dict, min_v=None, max_v=None):
    if min_v is None and max_v is None:
        # min_v, max_v = [
        #     [f(v).to('cpu').numpy() for k, v in grad_dict.items()] for f in [torch.min, torch.max]]
        min_v, max_v = [
            [f(v.to('cpu').numpy()) for k, v in grad_dict.items()]
            for f in [lambda x: np.percentile(x,0.01), lambda x: np.percentile(x,99.99)] ]
    assert (min_v is not None and max_v is not None)

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


# todo separate the wz model and use dependency injection to pass it to the protocol
# todo remove the *args, **kwargs for all related classes
class WZBroadcastProtocol:
    def __init__(self, agent_count, quantizer_type='RNN', *args, **kwargs):
        self.side_info_data_list = None
        self.agent_list_check = []
        self.warmup = True
        self.wz_pl_model_class = {'ANN': PL_EncoderDecoder_ANN, 'RNN': PL_EncoderDecoder_RNN}[quantizer_type]

        self.wz_quantizer_list: List[WZQuantizer] = [
            WZQuantizer(
                wz_pl_model=self.wz_pl_model_class(inp_dim=1, side_info_size=1, *args, **kwargs),
                count_side_info_data=0, *args, **kwargs) for _ in range(agent_count)]
        self.last_recent_grads_list = [None] * agent_count

    def to_server_from_worker_data_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        quantizer_decoder_state_dict, prob_per_bin = encoder_data_sent_by_server
        broadcast_data = self.encoding_process(agent_id, grad_dict, prob_per_bin)
        return broadcast_data

    def to_worker_from_server_data_transfer(self, agent_id):
        quantizer_decoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.decoder.state_dict()
        prob_per_bin = self.wz_quantizer_list[agent_id].get_bin_probs()
        return quantizer_decoder_state_dict, prob_per_bin

    # %%
    def encoding_process(self, agent_id, worker_grad_dict, prob_per_bin):
        # worker_grad_dict={k:v*1.1 for k, v in worker_grad_dict.items()}
        # return worker_grad_dict, min_v, max_v, 0

        grad_flat_normal, min_v, max_v = dict_to_array_and_normalize(worker_grad_dict)

        quantizer = self.wz_quantizer_list[agent_id]
        bin_data = quantizer.encoding_process(grad_flat_normal)

        encoded_grad_data = bin_data#entropy_coding(bin_data, prob_per_bin)

        return encoded_grad_data, min_v, max_v, bin_data.dtype

    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims, previous_data):
        # return worker_broadcast_data[0]

        encoded_data, min_v, max_v, dtype = worker_broadcast_data

        # assuming that previous_data has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a==i%worker_count for i,a in enumerate(self.agent_list_check)])

        quantizer = self.wz_quantizer_list[agent_id]

        quantized_decoded_data = encoded_data#entropy_decoding(encoded_data, dtype)

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
        prev_d_flat = [dict_to_array_and_normalize(pd, min_v, max_v)[0] for pd in previous_data]
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
                bins_per_plane=qz.wz_pl_model.bins_per_plane,
                num_planes=qz.wz_pl_model.num_planes,
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

    # simulate the WZ encoding and reconstruction process --------------------------------
    broadcast_prot = WZBroadcastProtocol(worker_count,'RNN',
                train_sample_size=100_000, metric_report_flag=True, lr=1e-5, num_planes=3, bins_per_plane=2)
    prev = []
    for round, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            print(f'Round {round}, Agent {ag_id}')
            server_data_sent_to_worker = broadcast_prot.to_worker_from_server_data_transfer(ag_id)
            encoded_ag_broadcast = broadcast_prot.to_server_from_worker_data_transfer(
                            ag_id, grad, server_data_sent_to_worker)

            decoded_agent_broadcast = broadcast_prot.reconstruction_process(
                ag_id, encoded_ag_broadcast, worker_count, model_shape_dict, prev, )

            prev.append(decoded_agent_broadcast)

    # check output size and correctness
    for i, grad in enumerate(grad_test_data[-1]):
        assert all([k in grad for k in model_shape_dict.keys()])
        assert all([v.shape == model_shape_dict[k] for k, v in grad.items()])
