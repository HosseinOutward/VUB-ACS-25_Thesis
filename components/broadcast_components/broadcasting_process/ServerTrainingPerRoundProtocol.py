from typing import OrderedDict
import torch
from lightning import seed_everything

from components.FL_sim import RawBroadcastProtocol
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN, get_real_bin_prob
from components.broadcast_components.compressor.rans_coding import rans_batch_decode, rans_batch_encode
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import _get_vec_slices
import pickle
import gzip
import numpy as np


# %%
def change_dtype_recursive(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    elif isinstance(obj, np.ndarray):
        numpy_dtype = torch.tensor([], dtype=dtype).numpy().dtype
        return obj.astype(numpy_dtype)
    elif isinstance(obj, (list, tuple)):
        return [change_dtype_recursive(x, dtype) for x in obj]
    elif isinstance(obj, OrderedDict):
        return OrderedDict({k: change_dtype_recursive(v, dtype) for k, v in obj.items()})
    elif isinstance(obj, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return torch.tensor([], dtype=dtype).numpy().dtype.type(obj)
    elif obj is None:
        return None
    else:
        raise TypeError(f"Unsupported type for dtype conversion: {type(obj)}.")


# %%
def make_seriable(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, (np.uint8, np.uint16, np.uint32, np.uint64, np.float16)):
        return item.item()
    elif isinstance(item, torch.Tensor):
        return item.cpu().numpy()
    elif isinstance(item, OrderedDict):
        return OrderedDict({k: make_seriable(v) for k, v in item.items()})
    elif isinstance(item, (list, tuple)):
        return [make_seriable(x) for x in item]
    elif hasattr(item, '_dtype') and hasattr(item, '__len__'):
        numpy_dtype = eval('np.' + str(item._dtype))
        return np.array(item, dtype=numpy_dtype)
    elif item is None:
        return None
    else:
        raise TypeError(f"Unsupported type for serialization: {type(item)}.")


def compress_data_list(data_list):
    # return data_list
    serializable_list = make_seriable(data_list)

    # Serialize and compress
    pickled_data = pickle.dumps(serializable_list, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = gzip.compress(pickled_data, compresslevel=6)
    return compressed_data


def decompress_data_list(compressed_data):
    # return compressed_data
    decompressed_data = gzip.decompress(compressed_data)
    data_list = pickle.loads(decompressed_data)
    return data_list


# %%
def dict_to_array(grad_dict: OrderedDict):
    res_v = []
    shapes_dict = OrderedDict()
    for k, v in grad_dict.items():
        shapes_dict[k] = v.shape

        v = v.ravel()
        res_v.append(v.to('cpu').numpy())
    res_v = np.concatenate(res_v)

    return res_v, shapes_dict


def array_to_dict_with_shapes(grad_vector, org_shapes_dict):
    res = {}
    start = 0
    for k, shape in org_shapes_dict.items():
        end = start + int(np.prod(shape))
        v = grad_vector[start:end]
        res[k] = v.reshape(shape)
        start = end
    return res


def shape_dict_to_vect_slices(dict_shapes):
    layer_groups = {}
    for k, shape in dict_shapes.items():
        if k.endswith('.weight') or k.endswith('.bias'):
            base_name = k.rsplit('.', 1)[0]  # Remove .weight or .bias
            if base_name not in layer_groups:
                layer_groups[base_name] = []
            layer_groups[base_name].append((k, shape))
        else:
            # Handle layers without .weight/.bias suffix as individual groups
            layer_groups[k] = [(k, shape)]


# %%
def _compression_protocol(grad_dict, quantizer):
    grad_dict = change_dtype_recursive(grad_dict, torch.float32)
    grad_flat, shapes_dict = dict_to_array(grad_dict)

    # **********
    temp=_get_vec_slices(shapes_dict)
    # check if the vec_slices are the same as the quantizer's
    check_same_slice_f = lambda s1,s2: s1.start==s2.start and s1.stop==s2.stop and s1.step==s2.step
    if (quantizer.vec_slices is not None) \
            and (not check_same_slice_f(quantizer.vec_slices[0], slice(None)) or len(quantizer.vec_slices)!=1):
        assert all([a == b for a, b in zip(temp, quantizer.vec_slices)])
    quantizer.vec_slices = _get_vec_slices(shapes_dict)

    # **********
    bins_vector, extra_enc_data = quantizer.encoding_process(grad_flat)

    # **********
    (norm_fact_vec, ), (outlier_positions, outlier_max, outlier_sign) = extra_enc_data

    # **********
    bin_count = quantizer.wz_pl_model.bins_per_plane
    outlier_bins_vector = torch.stack([a[outlier_positions] for a in bins_vector])
    for i in range(len(bins_vector)):
        bins_vector[i][outlier_positions] = bin_count
    bins_vector = torch.concat([bins_vector, outlier_bins_vector], dim=1)

    # **********
    # compress the bins_vector using RANS
    prob_per_bin = [get_real_bin_prob(b, bin_count + 1)[1].numpy() for b in bins_vector]
    prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float16)
    temp = change_dtype_recursive(prob_per_bin, torch.float32)
    bin_vec_compressed = [rans_batch_encode(bv.numpy(), pp_b) for bv, pp_b in zip(bins_vector, temp)]

    # **********
    norm_fact_vec, outlier_max = change_dtype_recursive([norm_fact_vec, outlier_max], torch.float16)

    outlier_count = len(outlier_positions)
    temp = [8, 16, 32, 64][np.argmax(np.array([8, 16, 32, 64]) / (np.log2(outlier_count + 1)+1e-8) > 1)]
    outlier_count = change_dtype_recursive(outlier_count, eval(f'torch.uint{temp}'))

    outlier_sign = np.packbits((outlier_sign > 0), bitorder='big') if outlier_count>0 else None

    # **********
    bin_rans_data = (bin_vec_compressed, prob_per_bin)
    normal_param = (norm_fact_vec,)
    outlier_param = (outlier_count, outlier_sign, outlier_max)
    return (
        compress_data_list((bin_rans_data, normal_param, outlier_param)),
        (bins_vector, normal_param, outlier_param)
    )


def _reconstruction_protocol(compressed_data, side_info, global_model_dims, quantizer):
    model_size = np.sum([int(np.prod(shape)) for shape in global_model_dims.values()])

    # ******
    (bin_vec_compressed, prob_per_bin), (norm_fact_vec,), (outlier_count, outlier_sign, outlier_max) = \
        decompress_data_list(compressed_data)
    prob_per_bin, norm_fact_vec, outlier_max = \
        change_dtype_recursive([prob_per_bin, norm_fact_vec, outlier_max], torch.float32)
    if outlier_count!=0:
        outlier_sign = np.unpackbits(outlier_sign, bitorder='big')[:outlier_count].astype(int) * 2 - 1
    else:
        assert outlier_sign is None

    # ******
    bin_data = [rans_batch_decode(bvc, prob_per_bin[i], model_size + outlier_count)
                for i, bvc in enumerate(bin_vec_compressed)]

    # ******
    outlier_positions=[]
    if outlier_count!=0:
        bin_count = quantizer.wz_pl_model.bins_per_plane
        outlier_positions = np.where(bin_data[0] == bin_count)[0]
        for i in range(len(bin_data)):
            bin_data[i][outlier_positions] = bin_data[i][-outlier_count:]
            bin_data[i] = bin_data[i][:-outlier_count]

    # decode the bin data to get the vector
    res_vector = quantizer.decoding_process(bin_data, side_info,
        encoding_extra_data=[(norm_fact_vec,), (outlier_positions, outlier_max, outlier_sign)])

    # ******
    result_dict = array_to_dict_with_shapes(res_vector, global_model_dims)
    result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}
    return result_dict, res_vector


def _train_model(grad_vector, side_info, to_clone_quantizer, epoch_count,
                 bins_per_plane, vec_slices, user_logger, reconst_ld=None):
    assert len(side_info) != 0

    reconst_ld = reconst_ld if reconst_ld is not None else to_clone_quantizer.wz_pl_model.reconst_ld

    qz = to_clone_quantizer
    wz_pl_model_class: PL_EncoderDecoder_RNN = qz.wz_pl_model.__class__
    new_quantizer = QuantizerWithDataPrep(
        user_logger=user_logger,
        vec_slices=vec_slices,
        count_side_info_data=len(side_info),
        wz_pl_model=wz_pl_model_class(
            inp_dim=1,
            side_info_size=len(side_info),
            num_planes=2,
            marginal=False,

            lr=qz.wz_pl_model.lr,
            bins_per_plane=bins_per_plane,
            reconst_ld=reconst_ld,
            tau=qz.wz_pl_model.tau,
            tau_rate=qz.wz_pl_model.tau_rate,
        ).to(torch.float32),
        enable_progress_bar=qz.enable_progress_bar,
        train_sample_size=qz.train_sample_size,
    )

    temp = np.random.normal(0, np.sqrt(1e-8), len(grad_vector), ).astype(np.float32)
    new_quantizer.train_model(grad_vector + temp, side_info, epoch=epoch_count, batch_size=10_000)

    return new_quantizer


class WZServerTrainingPerRoundProtocol(RawBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep,
                 epoch_count=45, no_global_quantization=False):
        assert isinstance(wz_base_quantizer, QuantizerWithDataPrep)
        self.epoch_count = epoch_count
        self.wz_basic_quantizer = wz_base_quantizer
        self.no_global_quantization = no_global_quantization

        self.wz_quantizer_list = [wz_base_quantizer for _ in range(agent_count)]

        self.si_window_size = 25
        self.past_worker_grad_recons_vec = [[] for _ in range(agent_count)]
        self.past_global_model_recons_vec = []
        self.agent_list_check = []
        self.last_global_comp = None
        self.warmup = True

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        assert agent_id == self.curr_agent_id

        # **********
        quantizer_encoder_state_dict = decompress_data_list(encoder_data_sent_by_server)
        quantizer_encoder_state_dict = {k: torch.tensor(v, dtype=torch.float32)
                                        for k, v in quantizer_encoder_state_dict.items()}

        self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.load_state_dict(
            quantizer_encoder_state_dict)

        quantizer = self.wz_quantizer_list[agent_id]
        compressed_data, _ = _compression_protocol(grad_dict, quantizer)

        return compressed_data

    def reconstruct_worker_grads(self, agent_id, worker_broadcast_data, worker_count, global_dims):
        # make sure the order of execution is correct
        assert agent_id == self.curr_agent_id
        # assuming that self.previous_data_list has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a == i % worker_count for i, a in enumerate(self.agent_list_check)])
        curr_round_id = len([a for a in self.agent_list_check if a == 0]) - 1
        assert curr_round_id == self.curr_round_id

        # **************
        quantizer = self.wz_quantizer_list[agent_id]
        side_info = self._get_side_info_for_grad_recons(agent_id)
        result_dict, result_vec = _reconstruction_protocol(worker_broadcast_data, side_info, global_dims, quantizer)

        # **************
        self._post_reconstruction_processing(agent_id, worker_count, global_dims, result_vec)

        # **************
        return result_dict

    def to_worker_prep_data_for_transfer(self, agent_id):
        assert agent_id == self.curr_agent_id
        quantizer_encoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.state_dict()
        quantizer_encoder_state_dict = change_dtype_recursive(quantizer_encoder_state_dict, torch.float16)
        return compress_data_list(quantizer_encoder_state_dict)

    # todo only send recons, seperate the compr process. change reporting too
    def model_transfer_to_worker_from_server(self, agent_id, server_model_state_dict):
        assert agent_id == self.curr_agent_id

        # *************
        if self.no_global_quantization:
            res = change_dtype_recursive(server_model_state_dict, torch.float16)
            compressed = compress_data_list(res)

            res = decompress_data_list(compressed)
            res = change_dtype_recursive(res, torch.float32)
            recons = {k: torch.tensor(v) for k, v in res.items()}
            return recons, compressed

        # *************
        # send the previous returned data as it's the same per each round for all workers
        if agent_id != 0:
            return self.last_global_comp

        # *************
        model_stat_vec, global_model_dims = dict_to_array(server_model_state_dict)
        side_info = self.past_global_model_recons_vec

        # *************
        if not self.warmup:
            print('        - training quant for global model transfer')

            quantizer = _train_model(
                model_stat_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                bins_per_plane=max(32 // (self.curr_round_id), 16),
                vec_slices=_get_vec_slices(global_model_dims),
                user_logger=None, reconst_ld=1000)
        else:
            quantizer = self.wz_basic_quantizer

        # *************
        compressed, uncompressed_encode_data = _compression_protocol(server_model_state_dict, quantizer)
        recons_dict, recons_vec = _reconstruction_protocol(
            compressed, side_info, global_model_dims, quantizer)

        # refine the reconstruction with the error diff
        copied_vec_slices = [a for a in self.wz_basic_quantizer.vec_slices]
        self.wz_basic_quantizer.vec_slices = [slice(None)]

        error_diff = OrderedDict({k: server_model_state_dict[k] - rec for k, rec in recons_dict.items()})
        compressed_diff, uncompressed_diff_encode_data = _compression_protocol(error_diff, self.wz_basic_quantizer)
        recons_diff_dict, recons_diff_vec = _reconstruction_protocol(
            compressed_diff, [], global_model_dims, self.wz_basic_quantizer)

        self.wz_basic_quantizer.vec_slices = copied_vec_slices

        recons_vec = recons_vec + recons_diff_vec
        recons_dict = OrderedDict({k: recons_dict[k] + recons_diff_dict[k] for k in recons_dict.keys()})

        # *************
        self.past_global_model_recons_vec += [change_dtype_recursive(recons_vec, torch.float16)]

        if len(self.past_global_model_recons_vec) > self.si_window_size:
            self.past_global_model_recons_vec.pop(0)

        entire_compressed_data = compress_data_list((uncompressed_encode_data, uncompressed_diff_encode_data))
        self.last_global_comp = (recons_dict, entire_compressed_data)

        return self.last_global_comp

    def _get_side_info_for_grad_recons(self, agent_id):
        if self.warmup:
            return []

        side_info = []
        for i, past_grads_agent in enumerate(self.past_worker_grad_recons_vec):
            temp = past_grads_agent
            if i == agent_id:
                temp = temp[:-1]
            side_info.extend(temp)
        return side_info

    def _post_reconstruction_processing(self, agent_id, worker_count, dict_shape, curr_recons_vector):
        assert agent_id == self.curr_agent_id

        # **************
        self.past_worker_grad_recons_vec[agent_id].append(change_dtype_recursive(curr_recons_vector, torch.float16))

        if len(self.past_worker_grad_recons_vec[agent_id]) > self.si_window_size:
            self.past_worker_grad_recons_vec[agent_id].pop(0)

        # **************
        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count:
            assert self.curr_round_id == 0
            self.warmup = False

        # **************
        # we have at least one complete round, so we train the next WZ_models
        if not self.warmup:
            next_agent = (agent_id + 1) % worker_count
            target_vec = self.past_worker_grad_recons_vec[next_agent][-1]
            side_info = self._get_side_info_for_grad_recons(next_agent)
            quantizer = _train_model(
                target_vec, side_info, self.wz_basic_quantizer, self.epoch_count,
                bins_per_plane=max(16 // (self.curr_round_id + 1), 3),
                vec_slices=_get_vec_slices(dict_shape),
                user_logger=self.wz_basic_quantizer.user_logger)
            self.wz_quantizer_list[next_agent] = quantizer


def _test_main(brod_prot_class, worker_count=2, rounds=2):
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=200_000,
                                           count_side_info_data=0, enable_progress_bar=True, vec_slices=None)

    broadcast_prot = brod_prot_class(worker_count, base_quantizer)

    # --------------------------------
    torch.set_float32_matmul_precision('medium')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    seed_everything(42)

    # load testing data --------------------------------
    model_shape_dict = {
        f'aaa_{i}': (*np.random.randint(1, 2, size=np.random.randint(2)),
                     (np.random.randint(1_000, 10_000) * 1000) // 1000)
        for i in range(3)
    }

    grad_test_data = [
        [OrderedDict({k: torch.normal(0, 1, size=v).to('cuda') for k, v in model_shape_dict.items()})
         for _ in range(worker_count)]
        for _ in range(rounds)]

    for i in range(1, rounds):
        for j in range(1, worker_count):
            for k, v in grad_test_data[i][j].items():
                grad_test_data[i][j][k] = grad_test_data[i - 1][j - 1][k] + v * 0.1

    # simulate the WZ encoding and reconstruction process --------------------------------
    for round_id, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            broadcast_prot.start_round_agent_process(ag_id, round_id)

            print(f'>> round_id {round_id}, Agent {ag_id}')
            recon_model_param = broadcast_prot.model_transfer_to_worker_from_server(ag_id, grad_per_round[0])
            recon_model_param = recon_model_param[0]

            print('          - Preparing data for transfer to worker...')
            server_data_sent_to_worker = broadcast_prot.to_worker_prep_data_for_transfer(ag_id)

            print('          - Preparing data for transfer to server...')
            encoded_ag_broadcast = broadcast_prot.to_server_prep_data_for_transfer(
                ag_id, grad, server_data_sent_to_worker)

            print('          - reconstructing data received...')
            decoded_agent_broadcast = broadcast_prot.reconstruct_worker_grads(
                ag_id, encoded_ag_broadcast, worker_count, model_shape_dict)

            # print the mspe
            grad_avg_v = np.mean(np.concat([grad[k].cpu() ** 2 for k in grad.keys()]))
            grad_mspe=[(grad[k] - v).cpu() ** 2/grad_avg_v for k, v in decoded_agent_broadcast.items()]
            grad_mspe = np.mean(np.concat(grad_mspe))

            global_avg_v = np.mean(np.concat([grad[k].cpu() ** 2 for k in grad.keys()]))
            global_mspe=[(grad_per_round[0][k] - v).cpu() ** 2/global_avg_v for k, v in recon_model_param.items()]
            global_mspe = np.mean(np.concat(global_mspe))
            print(f'     > MSPE - grad: {grad_mspe*100:.2f}%,   global: {global_mspe*100:.2f}%')

    # check output size and correctness
    for i, grad in enumerate(grad_test_data[-1]):
        assert all([k in grad for k in model_shape_dict.keys()])
        assert all([v.shape == model_shape_dict[k] for k, v in grad.items()])


if __name__ == "__main__":
    bp_f = lambda worker_count, base_quantizer: (
        WZServerTrainingPerRoundProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=3)

