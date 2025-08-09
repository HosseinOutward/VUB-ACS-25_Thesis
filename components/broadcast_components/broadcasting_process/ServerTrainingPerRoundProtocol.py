from typing import List, OrderedDict
import torch
from lightning import seed_everything

from components.FL_sim import RawBroadcastProtocol
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN, get_real_bin_prob
from components.broadcast_components.compressor.rans_coding import rans_batch_decode, rans_batch_encode
import pickle
import gzip
import numpy as np


#%%
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
    else:
        raise TypeError(f"Unsupported type for dtype conversion: {type(obj)}.")


#%%
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
        numpy_dtype = eval('np.'+str(item._dtype))
        return np.array(item, dtype=numpy_dtype)
    else:
        raise


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


#%%
def dict_to_array(grad_dict: OrderedDict):
    res_v = []
    shapes_dict = OrderedDict()
    for k, v in grad_dict.items():
        v = v.ravel()
        res_v.append(v.to('cpu').numpy())

        shapes_dict[k] = v.shape
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


#%%
class WZServerTrainingPerRoundProtocol(RawBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep):
        assert isinstance(wz_base_quantizer, QuantizerWithDataPrep)
        self.no_global_quantization = False
        self.last_global_model_recon_comp_data = None
        self.global_model_transfer_quantizer = wz_base_quantizer
        self.wz_pl_model_class = wz_base_quantizer.wz_pl_model.__class__
        self.wz_quantizer_list: List[QuantizerWithDataPrep] = [wz_base_quantizer] * agent_count

        self.last_recent_grads_list = [None] * agent_count
        self.current_side_info_list = None
        self.agent_list_check = []
        self.warmup = True
        self.prev_d_flat = []
        self.model_training_counter = [0] * agent_count
        self.past_global_model_recon_dict = []
        self.training_side_info_prev_d_flat = None
        self.epoch_count=45

    def to_worker_prep_data_for_transfer(self, agent_id):
        assert self.curr_agent_id == agent_id
        quantizer_encoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.state_dict()

        quantizer_encoder_state_dict = change_dtype_recursive(quantizer_encoder_state_dict, torch.float16)
        return compress_data_list(quantizer_encoder_state_dict)

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server,
                                         force_use_diff_model=None):
        if force_use_diff_model is None:  # *****
            assert self.curr_agent_id == agent_id

            quantizer_encoder_state_dict = decompress_data_list(encoder_data_sent_by_server)

            #**********
            quantizer_encoder_state_dict = {k: torch.tensor(v, dtype=torch.float32)
                                            for k, v in quantizer_encoder_state_dict.items()}
            self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.load_state_dict(
                quantizer_encoder_state_dict)

        #**********
        if force_use_diff_model is None:
            quantizer = self.wz_quantizer_list[agent_id]
        else:
            quantizer = force_use_diff_model  # *******

        #**********
        grad_dict = change_dtype_recursive(grad_dict, torch.float32)
        grad_flat, shapes_dict = dict_to_array(grad_dict)

        #**********
        quantizer.vec_slices = self._get_vec_slices(shapes_dict)

        # **********
        bins_vector, extra_enc_data = quantizer.encoding_process(grad_flat)

        #**********
        (norm_fact_vec), (outlier_positions, outlier_max, outlier_sign) = extra_enc_data

        #**********
        bin_count = quantizer.wz_pl_model.bins_per_plane
        outlier_bins_vector = torch.stack([a[outlier_positions] for a in bins_vector])
        for i in range(len(bins_vector)):
            bins_vector[i][outlier_positions] = bin_count
        bins_vector = torch.concat([bins_vector, outlier_bins_vector], dim=1)

        #**********
        # compress the bins_vector using RANS
        prob_per_bin = [get_real_bin_prob(b, bin_count + 1)[1].numpy() for b in bins_vector]
        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float16)
        temp = change_dtype_recursive(prob_per_bin, torch.float32)
        bin_vec_compressed = [rans_batch_encode(bv.numpy(), pp_b) for bv, pp_b in zip(bins_vector, temp)]

        #**********
        norm_fact_vec, outlier_max = change_dtype_recursive([norm_fact_vec, outlier_max], torch.float16)

        outlier_count = len(outlier_positions)
        temp = [8, 16, 32, 64][np.argmax(np.array([8, 16, 32, 64]) / np.log2(outlier_count + 1) > 1)]
        outlier_count = change_dtype_recursive(outlier_count, eval(f'torch.uint{temp}'))

        outlier_sign = np.packbits((outlier_sign>0), bitorder='big')

        #**********
        return compress_data_list(((bin_vec_compressed, prob_per_bin), (norm_fact_vec),
                                   (outlier_count, outlier_sign, outlier_max)))

    # %%
    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims,
                               force_use_diff_model=None):
        quantizer = force_use_diff_model
        if force_use_diff_model is None:  # *****
            assert self.curr_agent_id == agent_id
            # return worker_broadcast_data[0]

            # assuming that self.previous_data_list has order based on agents like 0, 1, 2, 0, 1, 2, ...
            self.agent_list_check.append(agent_id)
            assert all([a == i % worker_count for i, a in enumerate(self.agent_list_check)])
            curr_round_id = len([a for a in self.agent_list_check if a == 0]) - 1
            assert curr_round_id == self.curr_round_id
            # assert len(self.agent_list_check)-1==len(self.prev_d_flat)

            # ****
            quantizer = self.wz_quantizer_list[agent_id]

        model_size = np.sum([int(np.prod(shape)) for shape in global_model_dims.values()])

        # ******
        (bin_vec_compressed, prob_per_bin), (norm_fact_vec, ), (outlier_count, outlier_sign, outlier_max) =\
                 decompress_data_list(worker_broadcast_data)
        prob_per_bin, norm_fact_vec, outlier_max =\
            change_dtype_recursive([prob_per_bin, norm_fact_vec, outlier_max], torch.float32)
        outlier_sign = np.unpackbits(outlier_sign, bitorder='big')[:outlier_count]*2-1

        # ******
        bin_data = [rans_batch_decode(bvc, prob_per_bin[i], model_size + outlier_count)
                        for i, bvc in enumerate(bin_vec_compressed)]

        # ******
        if outlier_count > 0:
            bin_count = quantizer.wz_pl_model.bins_per_plane
            outlier_positions = np.where(bin_data[0] == bin_count)[0]
            for i in range(len(bin_data)):
                bin_data[i][outlier_positions] = bin_data[i][-outlier_count:]
                bin_data[i] = bin_data[i][:-outlier_count]

        # decode the bin data to get the vector
        if force_use_diff_model is not None: # *******
            side_info_data_list = self.past_global_model_recon_dict
        else:
            side_info_data_list = [] if self.warmup else self.current_side_info_list

        extra_enc_data = (norm_fact_vec,), (outlier_positions, outlier_max, outlier_sign)
        res_vector = quantizer.decoding_process(bin_data, side_info_data_list, encoding_extra_data=extra_enc_data)

        # ******
        result_dict = array_to_dict_with_shapes(res_vector, global_model_dims)
        result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

        # ************
        if force_use_diff_model is not None: # *******
            return result_dict, res_vector

        # ************
        self.prev_d_flat.append(change_dtype_recursive(res_vector, torch.float16))

        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count:
            self.warmup = False

        # assuming not in warmup phase, we have at least one complete round, so we train the next WZ_models
        if not self.warmup:
            self._prep_for_next_agent(agent_id, worker_count)

        # ************
        return result_dict

    # todo only send recons, seperate the compr process. change reporting too
    def model_transfer_to_worker_from_server(self, agent_id, server_model_state_dict):
        if self.no_global_quantization:
            res = change_dtype_recursive(server_model_state_dict, torch.float16)
            compressed = compress_data_list(res)

            res = decompress_data_list(compressed)
            res = change_dtype_recursive(res, torch.float32)
            recons = {k: torch.tensor(v) for k, v in res.items()}
            return recons, compressed

        # send the previous returned data as it's the same per each round for all workers
        if agent_id != 0:
            return self.last_global_model_recon_comp_data

        old_quantizer = self.global_model_transfer_quantizer
        model_stat_vec, global_model_dims = dict_to_array(server_model_state_dict)

        new_quantizer = old_quantizer
        if not self.warmup:
            print('        - training quant for global model transfer')

            si_count = len(self.past_global_model_recon_dict)
            new_quantizer = QuantizerWithDataPrep(
                wz_pl_model=self.wz_pl_model_class(
                        2, max(16 // (self.curr_round_id + 1), 2), 1,
                        si_count, 10, False, lr=1e-3, reconst_ld=400, tau=1.5
                    ).to(torch.float32),
                count_side_info_data=si_count, enable_progress_bar=old_quantizer.enable_progress_bar,
                train_sample_size=old_quantizer.train_sample_size, user_logger=None,
                vec_slices=self._get_vec_slices(global_model_dims),
            )

            model_stat_vec += np.random.normal(0, np.sqrt(1e-6), len(model_stat_vec), ).astype(np.float32)

            new_quantizer.train_model(model_stat_vec, self.past_global_model_recon_dict,
                                      epoch=self.epoch_count, batch_size=10_000)

        compressed = self.to_server_prep_data_for_transfer(
            None, server_model_state_dict, None, force_use_diff_model=new_quantizer, )

        recons, recons_vector = self.reconstruction_process(
            None, compressed, None, global_model_dims, force_use_diff_model=new_quantizer)

        self.past_global_model_recon_dict += [change_dtype_recursive(recons_vector, torch.float16)]

        if len(self.past_global_model_recon_dict) > 10:
            self.past_global_model_recon_dict.pop(0)

        self.last_global_model_recon_comp_data = (recons, compressed)

        return recons, compressed

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        assert not self.warmup

        temp = len(self.prev_d_flat) - worker_count
        self.current_side_info_list = self.prev_d_flat[:temp] + self.prev_d_flat[temp + 1:]

        next_agent = (curr_agent_id + 1) % worker_count
        qz = self.wz_quantizer_list[next_agent]
        self.wz_quantizer_list[next_agent] = QuantizerWithDataPrep(
            wz_pl_model=self.wz_pl_model_class(
                inp_dim=1, side_info_size=len(self.current_side_info_list),
                lr=qz.wz_pl_model.lr,
                bins_per_plane=max(16 // (self.curr_round_id + 1), 2),
                num_planes=2,
                reconst_ld=qz.wz_pl_model.reconst_ld,
                tau=qz.wz_pl_model.tau,
                marginal=self.curr_round_id <= 2,
            ).to(torch.float32),
            count_side_info_data=len(self.current_side_info_list), enable_progress_bar=qz.enable_progress_bar,
            train_sample_size=qz.train_sample_size, user_logger=qz.user_logger,
            vec_slices=qz.vec_slices,
        )

        last_recent_grads = self.prev_d_flat[temp] +\
                            np.random.normal(0, np.sqrt(1e-6), len(self.prev_d_flat[temp]), ).astype(np.float32)

        self.wz_quantizer_list[next_agent].train_model(
            last_recent_grads, self.current_side_info_list, epoch=self.epoch_count, batch_size=10_000)


def _test_main(broadcast_prot: WZServerTrainingPerRoundProtocol, worker_count=2, rounds=2):
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
    for round, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            broadcast_prot.start_round_agent_process(ag_id, round)

            print(f'>> Round {round}, Agent {ag_id}')
            _ = broadcast_prot.model_transfer_to_worker_from_server(ag_id, grad)

            print('          - Preparing data for transfer to worker...')
            server_data_sent_to_worker = broadcast_prot.to_worker_prep_data_for_transfer(ag_id)

            print('          - Preparing data for transfer to server...')
            encoded_ag_broadcast = broadcast_prot.to_server_prep_data_for_transfer(
                ag_id, grad, server_data_sent_to_worker)

            print('          - reconstructing data received...')
            decoded_agent_broadcast = broadcast_prot.reconstruction_process(
                ag_id, encoded_ag_broadcast, worker_count, model_shape_dict)

    # check output size and correctness
    for i, grad in enumerate(grad_test_data[-1]):
        assert all([k in grad for k in model_shape_dict.keys()])
        assert all([v.shape == model_shape_dict[k] for k, v in grad.items()])


if __name__ == "__main__":
    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=200_000,
                            count_side_info_data=0, enable_progress_bar=True, vec_slices=None)

    broadcast_prot = WZServerTrainingPerRoundProtocol(k, base_quantizer)

    _test_main(broadcast_prot, worker_count=k, rounds=2)
