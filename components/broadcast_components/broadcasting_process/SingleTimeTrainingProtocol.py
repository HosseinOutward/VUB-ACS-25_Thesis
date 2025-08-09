from typing import List
import numpy as np
import torch

from components.FL_sim import RawBroadcastProtocol
from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import outlier_normalization, \
    change_dtype_recursive, decompress_data_list, compress_data_list, array_to_dict_with_shapes, \
    outlier_de_normalization, dict_to_array
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import normalize_array_data, \
    denormalize_array_data
from components.broadcast_components.compressor.rans_coding import rans_batch_decode, rans_batch_encode


class SingleTimeTrainingProtocol(RawBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer):
        self.last_global_model_recon_comp_data = None
        self.global_model_transfer_quantizer = wz_base_quantizer
        self.wz_pl_model_class = wz_base_quantizer.wz_pl_model.__class__
        self.wz_quantizer_list: List[WZQuantizer] = [wz_base_quantizer] * agent_count

        self.last_recent_grads_list = [None] * agent_count
        self.current_side_info_list = None
        self.agent_list_check = []
        self.warmup = True
        self.prev_d_flat = []
        self.model_training_counter = [0] * agent_count
        self.past_global_model_recon_dict = []

        self.training_side_info_prev_d_flat = None

    # todo only send recons. change reporting to not need compressed data
    def model_transfer_to_worker_from_server(self, _, server_model_state_dict):
        res = change_dtype_recursive(server_model_state_dict, torch.float16)
        compressed = compress_data_list(res)

        res = decompress_data_list(compressed)
        res = change_dtype_recursive(res, torch.float32)
        recons = {k: torch.tensor(v) for k, v in res.items()}
        return recons, compressed

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
        grad_dict = change_dtype_recursive(grad_dict, torch.float32)

        # Get shapes dictionary before flattening
        shapes_dict = {k: v.shape for k, v in grad_dict.items()}
        grad_flat = dict_to_array(grad_dict)
        grad_flat_normal, norm_fact_vec = normalize_array_data(
            grad_flat, shapes_dict, outlier_rem=False, normalize=True)

        #**********
        outlier_values, outlier_positions, outlier_count, outlier_max = outlier_normalization(grad_flat_normal)
        grad_flat_normal[outlier_positions] = outlier_values

        #**********
        quantizer = self.wz_quantizer_list[agent_id] if force_use_diff_model is None else force_use_diff_model  # *******
        bin_count = quantizer.wz_pl_model.bins_per_plane
        bins_vector, extra_enc_data = quantizer.encoding_process(grad_flat_normal)

        #**********
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
        # change the dtype of the encoded data to float16
        norm_fact_vec, prob_per_bin = change_dtype_recursive([norm_fact_vec, prob_per_bin], torch.float16)

        outlier_max = outlier_max.astype(np.float16)

        temp = [8, 16, 32, 64][np.argmax(np.array([8, 16, 32, 64]) / np.log2(outlier_count + 1) > 1)]
        outlier_count = outlier_count.astype(eval(f'np.uint{temp}'))

        return compress_data_list((bin_vec_compressed, norm_fact_vec, prob_per_bin, outlier_count, outlier_max))

    # %%
    def reconstruction_process(self, agent_id, worker_broadcast_data, worker_count, global_model_dims):
        assert self.curr_agent_id == agent_id
        # return worker_broadcast_data[0]

        # assuming that self.previous_data_list has order based on agents like 0, 1, 2, 0, 1, 2, ...
        self.agent_list_check.append(agent_id)
        assert all([a==i%worker_count for i,a in enumerate(self.agent_list_check)])
        curr_round_id = len([a for a in self.agent_list_check if a==0])-1
        assert curr_round_id == self.curr_round_id
        # assert len(self.agent_list_check)-1==len(self.prev_d_flat)

        # ****
        quantizer = self.wz_quantizer_list[agent_id]

        model_size = np.sum([np.prod(shape) for shape in global_model_dims.values()])

        # decompress the data received from the worker
        bin_vec_compressed, norm_fact_vec, prob_per_bin, outlier_count, outlier_max =\
            decompress_data_list(worker_broadcast_data)

        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float32)
        norm_fact_vec = change_dtype_recursive(norm_fact_vec, torch.float32)

        bin_data = [rans_batch_decode(bvc, prob_per_bin[i], model_size+outlier_count)
                        for i, bvc in enumerate(bin_vec_compressed)]

        # ****
        bin_count = quantizer.wz_pl_model.bins_per_plane
        outlier_positions = np.where(bin_data[0]==bin_count)[0]
        for i in range(len(bin_data)):
            bin_data[i][outlier_positions] = 0

        # ****
        # decode the bin data to get the vector
        side_info_data_list = [] if self.model_training_counter[agent_id]==0 \
                            else self.prev_d_flat[:agent_id] + self.prev_d_flat[agent_id + 1:]
        side_info_data_list = [np.concatenate([a, a[outlier_positions]]) for a in side_info_data_list]
        res_vector = quantizer.decoding_process(bin_data, side_info_data_list, encoding_extra_data=extra_enc_data)

        # fix the outliers
        res_vector[outlier_positions] = outlier_de_normalization(res_vector, outlier_count, outlier_max)
        res_vector = res_vector[:-outlier_count]

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
        grads = self.training_side_info_prev_d_flat[target_id].copy()
        qz = self.wz_quantizer_list[target_id]
        self.wz_quantizer_list[target_id] = WZQuantizer(
            wz_pl_model=self.wz_pl_model_class(
                inp_dim=1, side_info_size=len(side_info),
                lr=qz.wz_pl_model.lr,
                bins_per_plane=max(16//(self.curr_round_id+1), 2),
                num_planes=2,
                reconst_ld=qz.wz_pl_model.reconst_ld,
                tau=qz.wz_pl_model.tau,
                marginal=self.curr_round_id<=2,
            ).to(torch.float32),
            count_side_info_data=len(side_info), enable_progress_bar=qz.enable_progress_bar,
            train_sample_size=qz.train_sample_size, user_logger=qz.user_logger,
        )

        grads+=np.random.normal(0, np.sqrt(1e-6), len(grads), ).astype(np.float32)

        outlier_values, outlier_positions, _, _ = outlier_normalization(grads)
        grads[outlier_positions] = outlier_values

        self.wz_quantizer_list[target_id].train_model(grads, side_info, epoch=45, batch_size=10_000)


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN, get_real_bin_prob
    from ServerTrainingPerRoundProtocol import _test_main

    k = 5
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                 count_side_info_data=0, enable_progress_bar=True)
    broadcast_prot = SingleTimeTrainingProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=10)
