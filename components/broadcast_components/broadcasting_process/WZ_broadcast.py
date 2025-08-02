from typing import List, Dict
import torch
from lightning import seed_everything

from components.FL_sim import RawBroadcastProtocol
from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer, get_real_bin_prob
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
from components.broadcast_components.compressor.rans_coding import rans_batch_decode, rans_batch_encode
import pickle
import gzip
import numpy as np


#%%
def _convert_item_recursive(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, torch.Tensor):
        return item.cpu().numpy()
    elif isinstance(item, dict):
        return {k: _convert_item_recursive(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [_convert_item_recursive(x) for x in item]
    elif hasattr(item, '_dtype') and hasattr(item, '__len__'):
        # Handle numba lists or other types by converting to numpy array
        numpy_dtype = eval('np.'+str(item._dtype))
        return np.array(item, dtype=numpy_dtype)
    else:
        # If conversion fails, keep as-is and let pickle handle it
        # print('** >> Warning: Unable to convert item of type {}. Keeping it as is. << **'.format(type(item)))
        # return item
        raise


def compress_data_list(data_list):
    # return data_list
    serializable_list = _convert_item_recursive(data_list)

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
def change_dtype_recursive(obj, dtype):
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)
    elif isinstance(obj, np.ndarray):
        numpy_dtype = torch.tensor([], dtype=dtype).numpy().dtype
        return obj.astype(numpy_dtype)
    elif isinstance(obj, (list, tuple)):
        return [change_dtype_recursive(x, dtype) for x in obj]
    elif isinstance(obj, dict):
        return {k:change_dtype_recursive(v, dtype) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for dtype conversion: {type(obj)}.")


#%%
def data_prep_function(y, side_info_data, outlier_rem=True, normalize=True):
    assert normalize or outlier_rem
    # remove outliers
    if outlier_rem:
        filt = np.percentile(y, [0.0005, 99.9995])
        filt = ((y >= filt[0]) * (y <= filt[1]))
        y = y[filt]
        side_info_data = [a[filt] for a in side_info_data]

    norm_fact = None
    if normalize:
        norm_fact = np.max(np.abs(np.percentile(y, [0.003, 99.997])))
        y = (y / norm_fact).astype(np.float32)
        side_info_data = [((a / norm_fact)).astype(np.float32) for a in side_info_data]

    return y, side_info_data, norm_fact

# todo combine bias and weight into one key in dict
def dict_to_array(grad_dict: Dict):
    """Convert dictionary of tensors to a single flattened array."""
    res = []
    for k, v in grad_dict.items():
        v = v.ravel()
        res.append(v.to('cpu').numpy())
    res = np.concatenate(res)
    return res


def normalize_array_data(data_array, org_shapes_dict, outlier_rem=False, normalize=True):
    """Normalize array data per layer using the existing data_prep_function method.
    Groups weight and bias parameters together for each layer."""
    if not normalize and not outlier_rem:
        return data_array, None

    norm_fact_vec = []
    normalized_segments = []
    start = 0

    # Group layers by their base name (removing .weight/.bias suffix)
    layer_groups = {}
    for k, shape in org_shapes_dict.items():
        if k.endswith('.weight') or k.endswith('.bias'):
            base_name = k.rsplit('.', 1)[0]  # Remove .weight or .bias
            if base_name not in layer_groups:
                layer_groups[base_name] = []
            layer_groups[base_name].append((k, shape))
        else:
            # Handle layers without .weight/.bias suffix as individual groups
            layer_groups[k] = [(k, shape)]

    # Process each layer group
    for group_name, layer_params in layer_groups.items():
        # Collect all parameters for this layer group
        group_data = []
        group_start = start

        for param_name, shape in layer_params:
            end = start + np.prod(shape)
            layer_data = data_array[start:end]
            group_data.append(layer_data)
            start = end

        # Concatenate all parameters in the group for normalization
        combined_group_data = np.concatenate(group_data)

        # Use existing data_prep_function for the entire group
        side_info_data = []  # Empty for single array normalization
        normalized_group, _, norm_fact = data_prep_function(combined_group_data, side_info_data, outlier_rem, normalize)

        # Split the normalized data back to individual parameters
        group_start_norm = 0
        for param_name, shape in layer_params:
            param_size = np.prod(shape)
            param_normalized = normalized_group[group_start_norm:group_start_norm + param_size]
            normalized_segments.append(param_normalized)
            group_start_norm += param_size

        # Store one normalization factor per group
        norm_fact_vec.append(norm_fact)

    # Concatenate all normalized segments
    normalized_data = np.concatenate(normalized_segments)
    norm_fact_vec = np.array(norm_fact_vec)

    return normalized_data, norm_fact_vec


def array_to_dict_with_shapes(grad_vector, org_shapes_dict):
    """Convert flattened array back to dictionary with original shapes."""
    res = {}
    start = 0
    for k, shape in org_shapes_dict.items():
        end = start + np.prod(shape)
        v = grad_vector[start:end]
        res[k] = v.reshape(shape)
        start = end
    return res


def denormalize_array_data(normalized_data, norm_fact_vec, org_shapes_dict):
    """Denormalize array data per layer using the normalization factors from data_prep_function.
    Groups weight and bias parameters together for each layer."""
    if norm_fact_vec is None:
        return normalized_data

    # Group layers by their base name (removing .weight/.bias suffix)
    layer_groups = {}
    for k, shape in org_shapes_dict.items():
        if k.endswith('.weight') or k.endswith('.bias'):
            base_name = k.rsplit('.', 1)[0]  # Remove .weight or .bias
            if base_name not in layer_groups:
                layer_groups[base_name] = []
            layer_groups[base_name].append((k, shape))
        else:
            # Handle layers without .weight/.bias suffix as individual groups
            layer_groups[k] = [(k, shape)]

    denormalized_segments = []
    start = 0
    group_idx = 0

    # Process each layer group
    for group_name, layer_params in layer_groups.items():
        norm_fact = norm_fact_vec[group_idx] if group_idx < len(norm_fact_vec) else None

        # Process each parameter in the group
        for param_name, shape in layer_params:
            end = start + np.prod(shape)
            layer_data = normalized_data[start:end]

            if norm_fact is not None:
                denormalized_layer = layer_data * norm_fact
            else:
                denormalized_layer = layer_data

            denormalized_segments.append(denormalized_layer)
            start = end

        group_idx += 1

    # Concatenate all denormalized segments
    denormalized_data = np.concatenate(denormalized_segments)

    return denormalized_data


class WZBroadcastProtocol(RawBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer:WZQuantizer):
        self.wz_pl_model_class = wz_base_quantizer.wz_pl_model.__class__
        self.wz_quantizer_list: List[WZQuantizer] = [wz_base_quantizer] * agent_count

        self.last_recent_grads_list = [None] * agent_count
        self.agent_list_check = []
        self.warmup = True
        self.prev_d_flat = []
        self.model_training_counter = [0] * agent_count

        self.training_side_info_prev_d_flat = None

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server):
        quantizer_encoder_state_dict = decompress_data_list(encoder_data_sent_by_server)

        #**********
        quantizer_encoder_state_dict = {k: torch.tensor(v, dtype=torch.float32)
                                         for k, v in quantizer_encoder_state_dict.items()}
        self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.load_state_dict(quantizer_encoder_state_dict)

        #**********
        grad_dict = change_dtype_recursive(grad_dict, torch.float32)

        # Get shapes dictionary before flattening
        shapes_dict = {k: v.shape for k, v in grad_dict.items()}
        grad_flat = dict_to_array(grad_dict)
        grad_flat_normal, norm_fact_vec = normalize_array_data(grad_flat, shapes_dict, outlier_rem=False, normalize=True)

        quantizer = self.wz_quantizer_list[agent_id]
        bins_vector = quantizer.encoding_process(grad_flat_normal)

        #********** compress the bins_vector using RANS
        if self.wz_pl_model_class == PL_EncoderDecoder_RNN:
            bin_count = self.wz_quantizer_list[agent_id].wz_pl_model.bins_per_plane
            prob_per_bin = [get_real_bin_prob(b, bin_count)[1].numpy() for b in bins_vector]
            prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float16)
            temp=change_dtype_recursive(prob_per_bin, torch.float32)
            bin_vec_compressed = [rans_batch_encode(bv.numpy(), pp_b) for bv, pp_b in zip(bins_vector, temp)]
        else:
            bin_count = self.wz_quantizer_list[agent_id].bin_count
            prob_per_bin = get_real_bin_prob(bins_vector, bin_count)[1].numpy()
            prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float16)
            temp=change_dtype_recursive(prob_per_bin, torch.float32)
            bin_vec_compressed = rans_batch_encode(bins_vector.numpy(), temp)

        # change the dtype of the encoded data to float16
        norm_fact_vec, prob_per_bin = change_dtype_recursive([norm_fact_vec, prob_per_bin], torch.float16)

        return compress_data_list((bin_vec_compressed, norm_fact_vec, prob_per_bin))

    def to_worker_prep_data_for_transfer(self, agent_id):
        quantizer_encoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.state_dict()

        quantizer_encoder_state_dict = change_dtype_recursive(quantizer_encoder_state_dict, torch.float16)
        return compress_data_list(quantizer_encoder_state_dict)

    # %%
    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims):
        # return worker_broadcast_data[0]

        # assuming that self.previous_data_list has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a==i%worker_count for i,a in enumerate(self.agent_list_check)])
        # assert len(self.agent_list_check)-1==len(self.prev_d_flat)

        # ****
        quantizer = self.wz_quantizer_list[agent_id]

        model_size = np.sum([np.prod(shape) for shape in global_model_dims.values()])

        # decompress the data received from the worker
        bin_vec_compressed, norm_fact_vec, prob_per_bin = decompress_data_list(worker_broadcast_data)

        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float32)
        norm_fact_vec = change_dtype_recursive(norm_fact_vec, torch.float32)

        if self.wz_pl_model_class == PL_EncoderDecoder_RNN:
            bin_data = [rans_batch_decode(bvc, prob_per_bin[i], model_size) for i, bvc in enumerate(bin_vec_compressed)]
        else:
            bin_data = rans_batch_decode(bin_vec_compressed, prob_per_bin, model_size)

        # decode the bin data to get the vector
        side_info_data_list = [] if self.model_training_counter[agent_id]==0 \
            else self.prev_d_flat[:agent_id] + self.prev_d_flat[agent_id + 1:]
        res_vector = quantizer.decoding_process(bin_data, side_info_data_list, model_size)

        # denormalize and convert back to dict
        denormalized_vector = denormalize_array_data(res_vector, norm_fact_vec, global_model_dims)
        result_dict = array_to_dict_with_shapes(denormalized_vector, global_model_dims)

        result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

        # ************
        assert len(self.prev_d_flat)<=worker_count
        if len(self.prev_d_flat)==worker_count:
            self.prev_d_flat[agent_id]=res_vector
        else:
            self.prev_d_flat.append(res_vector)
            if len(self.prev_d_flat) == worker_count:
                self.training_side_info_prev_d_flat = [a for a in self.prev_d_flat]

        if self.warmup and len(self.prev_d_flat) == worker_count:
            self._generate_models(agent_id, worker_count, res_vector, norm_fact_vec)

        # detect if we are in warmup phase
        if np.all([a == 1 for a in self.model_training_counter]) and self.warmup:
            self.warmup = False
            del self.training_side_info_prev_d_flat

        return result_dict

    # todo only send recons. change reporting to not need compressed data
    def model_transfer_to_worker_from_server(self, server_model_state_dict):
        res = change_dtype_recursive(server_model_state_dict, torch.float16)
        compressed = compress_data_list(res)

        res = decompress_data_list(compressed)
        res = change_dtype_recursive(res, torch.float32)
        recons = {k: torch.tensor(v) for k, v in res.items()}
        return recons, compressed

    def _generate_models(self, curr_agent_id, worker_count, res_vector, norm_fact_vec):
        target_id = (curr_agent_id + 1) % worker_count
        assert target_id == (curr_agent_id + 1) % worker_count and self.agent_list_check[-1]==curr_agent_id , \
            'The reporting code depends on training only the next agent.'

        # make sure the training happens in order
        curr_counter = self.model_training_counter[target_id]
        past_counters = self.model_training_counter[:target_id]
        if target_id==0:
            past_counters = self.model_training_counter[1:]
            curr_counter-=1
        assert np.all([curr_counter+1==a for a in past_counters]), 'The order of model training isn\'t compatible.'
        self.model_training_counter[target_id] += 1

        side_info = self.training_side_info_prev_d_flat[:target_id] +\
                    self.training_side_info_prev_d_flat[target_id + 1:]
        grads = self.training_side_info_prev_d_flat[target_id]
        qz = self.wz_quantizer_list[target_id]
        self.wz_quantizer_list[target_id] = WZQuantizer(
            wz_pl_model=self.wz_pl_model_class(
                inp_dim=1, side_info_size=len(side_info),
                lr=qz.wz_pl_model.lr,
                bins_per_plane=qz.wz_pl_model.bins_per_plane,
                num_planes=qz.wz_pl_model.num_planes,
                tau=qz.wz_pl_model.tau,
            ).to(torch.float32),
            count_side_info_data=len(side_info), enable_progress_bar=qz.enable_progress_bar,
            train_sample_size=qz.train_sample_size, user_logger=qz.user_logger,
        )

        self.wz_quantizer_list[target_id].train_model(grads, side_info, epoch=5, batch_size=15_000)


def _test_main(broadcast_prot, worker_count = 2, rounds = 2):
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
            (np.random.randint(1_000, 10_000)*1000)//1000)
        for i in range(3)
    }

    grad_test_data = [
            [{k: torch.normal(0,1,size=v).to('cuda') for k, v in model_shape_dict.items()}
                for _ in range(worker_count)]
        for _ in range(rounds)]

    for i in range(1,rounds):
        for j in range(1, worker_count):
            for k, v in grad_test_data[i][j].items():
                grad_test_data[i][j][k] = grad_test_data[i-1][j-1][k] + v * 0.1

    # simulate the WZ encoding and reconstruction process --------------------------------
    for round, grad_per_round in enumerate(grad_test_data):
        for ag_id, grad in enumerate(grad_per_round):
            print(f'>> Round {round}, Agent {ag_id}')
            _ = broadcast_prot.model_transfer_to_worker_from_server(grad)

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
    k=5
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=3,
                                     bins_per_plane=4, lr=1e-5).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_3plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                    count_side_info_data=0, enable_progress_bar=True)
    broadcast_prot = WZBroadcastProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=10)