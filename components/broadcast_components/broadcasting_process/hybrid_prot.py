from typing import List

import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer, get_real_bin_prob
from components.broadcast_components.broadcasting_process.WZ_broadcast import WZBroadcastProtocol, decompress_data_list, \
    change_dtype_recursive, dict_to_array, normalize_array_data, outlier_normalization, compress_data_list, \
    outlier_de_normalization, denormalize_array_data, array_to_dict_with_shapes
from components.broadcast_components.compressor.rans_coding import rans_batch_encode, rans_batch_decode


class HybridWZBroadcastProtocol(WZBroadcastProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer):
        super().__init__(agent_count, wz_base_quantizer)

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
        bins_vector = quantizer.encoding_process(grad_flat_normal)

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

    def to_worker_prep_data_for_transfer(self, agent_id):
        assert self.curr_agent_id == agent_id
        quantizer_encoder_state_dict = self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.state_dict()

        quantizer_encoder_state_dict = change_dtype_recursive(quantizer_encoder_state_dict, torch.float16)
        return compress_data_list(quantizer_encoder_state_dict)

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

        # decompress the data received from the worker
        bin_vec_compressed, norm_fact_vec, prob_per_bin, outlier_count, outlier_max = \
            decompress_data_list(worker_broadcast_data)

        prob_per_bin = change_dtype_recursive(prob_per_bin, torch.float32)
        norm_fact_vec = change_dtype_recursive(norm_fact_vec, torch.float32)

        bin_data = [rans_batch_decode(bvc, prob_per_bin[i], model_size + outlier_count)
                    for i, bvc in enumerate(bin_vec_compressed)]

        # ****
        bin_count = quantizer.wz_pl_model.bins_per_plane
        outlier_positions = np.where(bin_data[0] == bin_count)[0]
        for i in range(len(bin_data)):
            bin_data[i][outlier_positions] = 0

        # decode the bin data to get the vector
        side_info_data_list = [] if self.warmup else self.current_side_info_list
        if force_use_diff_model is not None: # *******
            side_info_data_list = self.past_global_model_recon_dict
        side_info_data_list = [np.concatenate([a, a[outlier_positions]]) for a in side_info_data_list]
        res_vector = quantizer.decoding_process(bin_data, side_info_data_list, )

        # fix the outliers
        res_vector[outlier_positions] = outlier_de_normalization(res_vector, outlier_count, outlier_max)
        res_vector = res_vector[:-outlier_count]

        # denormalize and convert back to dict
        denormalized_vector = denormalize_array_data(res_vector, norm_fact_vec, global_model_dims)
        result_dict = array_to_dict_with_shapes(denormalized_vector, global_model_dims)

        result_dict = {k: torch.tensor(v).to('cuda') for k, v in result_dict.items()}

        if force_use_diff_model is not None: # *******
            return result_dict, res_vector

        # ************

        self.prev_d_flat.append(res_vector)

        # detect if we are in warmup phase
        if agent_id + 1 >= worker_count:
            self.warmup = False

        # assuming not in warmup phase, we have at least one complete round, so we train the next WZ_models
        if not self.warmup:
            self._prep_for_next_agent(agent_id, worker_count)

        return result_dict

    # todo only send recons, seperate the compr process. change reporting too
    def model_transfer_to_worker_from_server(self, agent_id, server_model_state_dict):
        # send the previous returned data as it's the same per each round for all workers
        if agent_id != 0:
            return self.last_global_model_recon_comp_data

        old_quantizer = self.global_model_transfer_quantizer
        global_model_dims = {k: v.shape for k, v in server_model_state_dict.items()}

        new_quantizer = old_quantizer
        if not self.warmup:
            print('        - training quant for global model transfer')
            temp = len(self.past_global_model_recon_dict)
            new_quantizer = WZQuantizer(
                wz_pl_model=self.wz_pl_model_class(2, max(16 // (self.curr_round_id + 1), 2), 1,
                        temp, 10, False, lr=1e-3, reconst_ld=400, tau=1.5).to(torch.float32),
                count_side_info_data=temp, enable_progress_bar=old_quantizer.enable_progress_bar,
                train_sample_size=old_quantizer.train_sample_size, user_logger=None,
            )

            model_stat_vec = dict_to_array(server_model_state_dict)
            model_stat_vec, _ = normalize_array_data(model_stat_vec, global_model_dims, False, True)

            outlier_values, outlier_positions, _, _ = outlier_normalization(model_stat_vec)
            model_stat_vec[outlier_positions] = outlier_values

            model_stat_vec += np.random.normal(0, np.sqrt(1e-6), len(model_stat_vec), ).astype(np.float32)

            new_quantizer.train_model(model_stat_vec, self.past_global_model_recon_dict, epoch=45, batch_size=10_000)

        compressed = self.to_server_prep_data_for_transfer(
            None, server_model_state_dict, None, force_use_diff_model=new_quantizer, )

        recons, recons_vector = self.reconstruction_process(
            None, compressed, None, global_model_dims, force_use_diff_model=new_quantizer)

        self.past_global_model_recon_dict += [recons_vector]
        if len(self.past_global_model_recon_dict) > 10:
            self.past_global_model_recon_dict.pop(0)

        self.last_global_model_recon_comp_data = (recons, compressed)

        return recons, compressed

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        temp = len(self.prev_d_flat) - worker_count
        last_recent_grads = self.prev_d_flat[temp]
        self.current_side_info_list = self.prev_d_flat[:temp] + self.prev_d_flat[temp + 1:]

        next_agent = (curr_agent_id + 1) % worker_count
        qz = self.wz_quantizer_list[next_agent]
        self.wz_quantizer_list[next_agent] = WZQuantizer(
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
        )

        last_recent_grads+=np.random.normal(0, np.sqrt(1e-6), len(last_recent_grads), ).astype(np.float32)

        outlier_values, outlier_positions, _, _ = outlier_normalization(last_recent_grads)
        last_recent_grads[outlier_positions] = outlier_values

        self.wz_quantizer_list[next_agent].train_model(
            last_recent_grads, self.current_side_info_list, epoch=45, batch_size=10_000)

if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from WZ_broadcast import _test_main

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                 count_side_info_data=0, enable_progress_bar=True)
    broadcast_prot = HybridWZBroadcastProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=10)
