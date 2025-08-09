import numpy as np
import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
    WZServerTrainingPerRoundProtocol, change_dtype_recursive, decompress_data_list, dict_to_array, compress_data_list


class HybridWZBroadcastProtocol(WZServerTrainingPerRoundProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep, hybrid_round_num=5):
        super().__init__(agent_count, wz_base_quantizer)
        self.hybrid_round_num = hybrid_round_num
        self.is_hybrid_round_f = lambda round_id: round_id % self.hybrid_round_num == 0 and not self.warmup
        self.past_workerside_grads = [[] for _ in range(agent_count)]

    def _build_worker_side_quantizer(self, old_quantizer, training_target, side_info):
        print('****************training the workerside model')
        if old_quantizer.user_logger is not None:
            assert old_quantizer.user_logger.agent_id == self.curr_agent_id
            assert old_quantizer.user_logger.round_id == self.curr_round_id
            old_quantizer.user_logger.agent_id = (self.curr_agent_id - 1) % len(self.past_workerside_grads)
            old_quantizer.user_logger.round_id = self.curr_round_id if self.curr_agent_id != 0 \
                                                                    else self.curr_round_id - 1

        new_quantizer = QuantizerWithDataPrep(
            wz_pl_model=self.wz_pl_model_class(
                2, int(max(self.hybrid_round_num * 16 // (self.curr_round_id + 1), 4)), 1, len(side_info), 10, False,
                lr=1e-3, reconst_ld=400, tau=1.5).to(torch.float32),
            count_side_info_data=len(side_info), enable_progress_bar=old_quantizer.enable_progress_bar,
            train_sample_size=old_quantizer.train_sample_size, user_logger=old_quantizer.user_logger,
            vec_slices=old_quantizer.vec_slices,
        )
        side_info = change_dtype_recursive(side_info, torch.float32)
        new_quantizer.train_model(training_target, side_info, epoch=self.epoch_count, batch_size=10_000)

        bins, extra_enc_data = new_quantizer.encoding_process(training_target)
        recons_vect = new_quantizer.decoding_process(bins, side_info, encoding_extra_data=extra_enc_data)

        if old_quantizer.user_logger is not None:
            old_quantizer.user_logger.agent_id = self.curr_agent_id
            old_quantizer.user_logger.round_id = self.curr_round_id

        return new_quantizer, recons_vect

    def _prep_for_next_agent(self, curr_agent_id, worker_count):
        next_agent = (curr_agent_id + 1) % worker_count
        coming_round = self.curr_round_id + 1 if next_agent == 0 else self.curr_round_id

        if coming_round == 1 and next_agent == 0:
            print('********** loading first round results to the memory')
            for i, a in enumerate(self.prev_d_flat):
                self.past_workerside_grads[i] += [a]

        if self.is_hybrid_round_f(coming_round):
            print('********** skipping training for next agent as its the hybrid round')
            self.prev_d_flat[-1] = self.past_workerside_grads[curr_agent_id][-1]
            return

        super()._prep_for_next_agent(curr_agent_id, worker_count)

    def to_worker_prep_data_for_transfer(self, agent_id):
        res = super().to_worker_prep_data_for_transfer(agent_id)
        if self.is_hybrid_round_f(self.curr_round_id) or (self.curr_round_id == 0 and self.curr_agent_id == 0):
            return res, [res]
        return res, np.array([0])

    def to_server_prep_data_for_transfer(self, agent_id, grad_dict, encoder_data_sent_by_server,
                                         force_use_diff_model=None):
        if force_use_diff_model is None:  # *****
            assert self.curr_agent_id == agent_id

            quantizer_encoder_state_dict = decompress_data_list(encoder_data_sent_by_server[0])
            quantizer_encoder_state_dict = {k: torch.tensor(v, dtype=torch.float32)
                                            for k, v in quantizer_encoder_state_dict.items()}
            self.wz_quantizer_list[agent_id].wz_pl_model.coding_model.encoder.load_state_dict(
                quantizer_encoder_state_dict)

            # Handle hybrid round logic
            if self.is_hybrid_round_f(self.curr_round_id):
                grad_dict = change_dtype_recursive(grad_dict, torch.float32)
                grad_flat, shapes_dict = dict_to_array(grad_dict)

                self.current_side_info_list = [a for a in self.past_workerside_grads[agent_id]]
                self.wz_quantizer_list[agent_id].vec_slices = self._get_vec_slices(shapes_dict)

                quantizer, recons_vect = self._build_worker_side_quantizer(
                    self.wz_quantizer_list[agent_id], grad_flat, self.current_side_info_list)

                self.wz_quantizer_list[agent_id] = quantizer
                self.past_workerside_grads[self.curr_agent_id].append(
                    change_dtype_recursive(recons_vect, torch.float16))

                # Use the updated quantizer for encoding
                force_use_diff_model = quantizer

                encoder_data_sent_by_server = compress_data_list(
                    quantizer.wz_pl_model.coding_model.encoder.state_dict())

        # Now delegate to parent class for the standard encoding process
        return super().to_server_prep_data_for_transfer(
            agent_id, grad_dict, encoder_data_sent_by_server, force_use_diff_model)


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    k = 2
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=100_000,
                                          count_side_info_data=0, enable_progress_bar=True, vec_slices=None)
    broadcast_prot = HybridWZBroadcastProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=10)
