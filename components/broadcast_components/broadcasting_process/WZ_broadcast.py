from typing import List, Dict
import numpy as np
import torch
from lightning import seed_everything
from components.broadcast_components.quantizer.wz_quant_ANN import WZQuantizer, PL_EncoderDecoder_ANN, get_real_bin_prob
from components.broadcast_components.quantizer.wz_quant_RNN import PL_EncoderDecoder_RNN
from components.broadcast_components.compressor.rans import rans_encode, rans_decode


def change_dtype_recursive(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    elif isinstance(obj, np.ndarray):
        dtype = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }[dtype]
        return obj.astype(dtype)
    elif isinstance(obj, (list, tuple)):
        return [change_dtype_recursive(x, dtype) for x in obj]
    elif isinstance(obj, dict):
        return {k:change_dtype_recursive(v, dtype) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for dtype conversion: {type(obj)}.")


# todo combine bias and weight into one key in dict
def dict_to_array_and_normalize(grad_dict: Dict, min_v=None, max_v=None):
    if min_v is None and max_v is None:
        # min_v, max_v = [
        #     [f(v).to('cpu').numpy() for k, v in grad_dict.items()] for f in [torch.min, torch.max]]
        max_v, min_v = [ np.array([torch.quantile(v, q).cpu().numpy() for k, v in grad_dict.items()])
                            for q in [0.9999,0.0001] ]
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
                wz_pl_model=self.wz_pl_model_class(
                    inp_dim=1, side_info_size=1, *args, **kwargs).to(torch.float32),
                count_side_info_data=0, *args, **kwargs) for _ in range(agent_count)]

        print('********* not loading base model from any file')
        # for i in range(agent_count):
            # self.wz_quantizer_list[i].wz_pl_model.load_from_checkpoint()

        self.last_recent_grads_list = [None] * agent_count

    # todo do compression for stuff sent to server
    def to_server_from_worker_data_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        quantizer_decoder_state_dict = encoder_data_sent_by_server

        #**********
        self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.load_state_dict(quantizer_decoder_state_dict)

        bins_vector, min_v, max_v = self.encoding_process(agent_id, grad_dict)

        # compress the bins_vector using RANS
        bin_count = self.wz_quantizer_list[agent_id].bin_count
        if self.wz_pl_model_class == PL_EncoderDecoder_RNN:
            prob_per_bin = [get_real_bin_prob(b, bin_count)[1].numpy() for b in bins_vector]
            bin_vec_compressed = [rans_encode(bv.numpy(), pp_b)
                                  for bv, pp_b in zip(bins_vector, prob_per_bin)]
        else:
            prob_per_bin = get_real_bin_prob(bins_vector, bin_count)[1].numpy()
            bin_vec_compressed = rans_encode(bins_vector.numpy(), prob_per_bin)

        # change the dtype of the encoded data to float16
        min_v, max_v, prob_per_bin = change_dtype_recursive([min_v, max_v, prob_per_bin], torch.float16)
        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float16)

        return bin_vec_compressed, min_v, max_v, prob_per_bin

    # todo do compression for stuff sent to workers
    def to_worker_from_server_data_transfer(self, agent_id):
        quantizer_encoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.state_dict()

        quantizer_encoder_state_dict = change_dtype_recursive(quantizer_encoder_state_dict, torch.float16)
        return quantizer_encoder_state_dict

    # %%
    def encoding_process(self, agent_id, worker_grad_dict):
        # worker_grad_dict={k:v*1.1 for k, v in worker_grad_dict.items()}
        # return worker_grad_dict, min_v, max_v, 0

        worker_grad_dict = change_dtype_recursive(worker_grad_dict, torch.float32)
        #**********

        grad_flat_normal, min_v, max_v = dict_to_array_and_normalize(worker_grad_dict)

        quantizer = self.wz_quantizer_list[agent_id]
        bin_data = quantizer.encoding_process(grad_flat_normal)

        return bin_data, min_v, max_v

    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims, previous_data):
        # return worker_broadcast_data[0]

        # assuming that previous_data has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a==i%worker_count for i,a in enumerate(self.agent_list_check)])

        # ****
        quantizer = self.wz_quantizer_list[agent_id]

        model_size = np.sum([np.prod(shape) for shape in global_model_dims.values()])

        # decompress the data received from the worker
        bin_vec_compressed, min_v, max_v, prob_per_bin = worker_broadcast_data
        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float32)
        min_v, max_v = change_dtype_recursive([min_v, max_v], torch.float32)

        if self.wz_pl_model_class == PL_EncoderDecoder_RNN:
            bin_data = [rans_decode(bvc, prob_per_bin[i], model_size)
                        for i, bvc in enumerate(bin_vec_compressed)]
        else:
            bin_data = rans_decode(bin_vec_compressed, prob_per_bin, model_size)

        # decode the bin data to get the vector
        side_info_data_list = [] if self.warmup else self.side_info_data_list
        res_vector = quantizer.decoding_process(bin_data, side_info_data_list, model_size)

        result_dict = recover_shape_and_denormal_to_dict(res_vector, global_model_dims, min_v, max_v)

        result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count:
            self.warmup = False

        # assuming not in warmup phase, we have at least one complete round, so we train the next quantizer
        if not self.warmup:
            self._prep_for_next_agent(agent_id, worker_count, res_vector, previous_data, min_v, max_v)

        return result_dict

    def _prep_for_next_agent(self, agent_id, worker_count, res_vector, previous_data, min_v, max_v):
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
            ).to(torch.float32),
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

    worker_count = 2
    rounds = 1
    seed_everything(42)

    # load testing data --------------------------------
    model_shape_dict = {
        f'aaa_{i}': (*np.random.randint(1, 2, size=np.random.randint(2)),
            (np.random.randint(1_000, 10_000)*1000)//1000)
        for i in range(3)
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
            print(f'>> Round {round}, Agent {ag_id}')
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
