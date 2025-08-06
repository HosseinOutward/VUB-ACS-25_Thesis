import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer, get_real_bin_prob
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, decompress_data_list, \
    change_dtype_recursive, dict_to_array, normalize_array_data, outlier_normalization, compress_data_list, \
    outlier_de_normalization
from components.broadcast_components.compressor.rans_coding import rans_batch_encode


class HybridWZBroadcastProtocol(WZServerTrainingPerRoundProtocol):
    def __init__(self, agent_count, wz_base_quantizer: WZQuantizer, hybrid_round_num=5):
        super().__init__(agent_count, wz_base_quantizer)
        self.hybrid_round_num = hybrid_round_num
        self.is_hybrid_round_f = lambda round_id: round_id%self.hybrid_round_num == 0 and not self.warmup
        self.past_workerside_grads = [[] for _ in range(agent_count)]

    def _build_worker_side_quantizer(self, old_quantizer, training_target, side_info):
        print('****************training the workerside model')
        if old_quantizer.user_logger is not None:
            assert old_quantizer.user_logger.agent_id == self.curr_agent_id
            assert old_quantizer.user_logger.round_id == self.curr_round_id
            old_quantizer.user_logger.agent_id = (self.curr_agent_id-1)%len(self.past_workerside_grads)
            old_quantizer.user_logger.round_id = self.curr_round_id if self.curr_agent_id!=0 else self.curr_round_id-1

        new_quantizer = WZQuantizer(
            wz_pl_model=self.wz_pl_model_class(
                2, int(max(self.hybrid_round_num*16 // (self.curr_round_id + 1), 4)), 1, len(side_info), 10, False,
                lr=1e-3, reconst_ld=400, tau=1.5).to(torch.float32),
            count_side_info_data=len(side_info), enable_progress_bar=old_quantizer.enable_progress_bar,
            train_sample_size=old_quantizer.train_sample_size, user_logger=old_quantizer.user_logger,
        )
        new_quantizer.train_model(training_target, side_info, epoch=45, batch_size=10_000)

        recons_vect = new_quantizer.decoding_process(new_quantizer.encoding_process(training_target), side_info)

        if old_quantizer.user_logger is not None:
            old_quantizer.user_logger.agent_id = self.curr_agent_id
            old_quantizer.user_logger.round_id = self.curr_round_id

        return new_quantizer, recons_vect

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        next_agent = (curr_agent_id + 1) % worker_count
        coming_round = self.curr_round_id+1 if next_agent==0 else self.curr_round_id

        if coming_round==1 and next_agent == 0:
            print('********** loading first round results to the memory')
            for i, a in enumerate(self.prev_d_flat):
                self.past_workerside_grads[i]+=[a]

        if self.is_hybrid_round_f(coming_round):
            print('********** skipping training for next agent as its the hybrid node')
            return

        super()._prep_for_next_agent(curr_agent_id, worker_count)

    def to_worker_prep_data_for_transfer(self, agent_id):
        res = super().to_worker_prep_data_for_transfer(agent_id)
        if self.is_hybrid_round_f(self.curr_round_id) or (self.curr_round_id==0 and self.curr_agent_id==0):
            return res, [res]
        return res, np.array([0])

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server,
                                         force_use_diff_model=None):
        if force_use_diff_model is None:  # *****
            assert self.curr_agent_id == agent_id

            quantizer_encoder_state_dict = decompress_data_list(encoder_data_sent_by_server[0])

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


        #********** *****************************************************************************************
        if force_use_diff_model is not None:
            quantizer=force_use_diff_model
        elif self.is_hybrid_round_f(self.curr_round_id):
            self.current_side_info_list = [a for a in self.past_workerside_grads[agent_id]]

            quantizer, recons_vect = self._build_worker_side_quantizer(
                self.wz_quantizer_list[agent_id], grad_flat_normal, self.current_side_info_list)

            self.wz_quantizer_list[agent_id] = quantizer

            recons_vect[outlier_positions] = outlier_de_normalization(recons_vect, outlier_count, outlier_max)
            self.past_workerside_grads[self.curr_agent_id].append(recons_vect)
        else:
            quantizer = self.wz_quantizer_list[agent_id]
        #********** *****************************************************************************************

        #**********
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


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from ServerTrainingPerRoundProtocol import _test_main

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = WZQuantizer(wz_model, train_sample_size=100_000,
                                 count_side_info_data=0, enable_progress_bar=True)
    broadcast_prot = HybridWZBroadcastProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=10)
