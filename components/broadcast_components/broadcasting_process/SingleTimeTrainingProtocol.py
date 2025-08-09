import numpy as np
import torch

from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import\
    WZServerTrainingPerRoundProtocol, change_dtype_recursive


class SingleTimeTrainingProtocol(WZServerTrainingPerRoundProtocol):
    def __init__(self, agent_count, wz_base_quantizer: QuantizerWithDataPrep):
        super().__init__(agent_count, wz_base_quantizer)
        # Override some attributes for single-time training behavior
        self.single_time_training_completed = [False] * agent_count
        self.training_side_info_prev_d_flat = None

    def model_transfer_to_worker_from_server(self, agent_id, server_model_state_dict):
        # For single time training, we don't use quantization for global model transfer
        self.no_global_quantization = True
        return super().model_transfer_to_worker_from_server(agent_id, server_model_state_dict)

    def _get_side_info_data_list(self, agent_id, force_use_diff_model=None):
        """Override to provide single-time training specific side info logic."""
        if force_use_diff_model is not None:
            return self.past_global_model_recon_dict

        # Single time training logic: use previous gradients as side info
        if self.model_training_counter[agent_id] == 0:
            return []
        else:
            return self.prev_d_flat[:agent_id] + self.prev_d_flat[agent_id + 1:]

    def _post_reconstruction_processing(self, agent_id, worker_count, res_vector, result_dict):
        """Override to handle single-time training specific post-processing."""
        # Update tracking for single time training
        assert len(self.prev_d_flat) <= worker_count
        if len(self.prev_d_flat) == worker_count:
            self.prev_d_flat[agent_id] = change_dtype_recursive(res_vector, torch.float16)
        else:
            self.prev_d_flat.append(change_dtype_recursive(res_vector, torch.float16))
            if len(self.prev_d_flat) == worker_count:
                self.training_side_info_prev_d_flat = [a for a in self.prev_d_flat]

        # Train models if in warmup phase
        if self.warmup and len(self.prev_d_flat) == worker_count:
            self._generate_models_single_time(agent_id, worker_count, res_vector)

        # Exit warmup when all models are trained once
        if np.all([a == 1 for a in self.model_training_counter]) and self.warmup:
            self.warmup = False
            del self.training_side_info_prev_d_flat

        return result_dict

    def _generate_models_single_time(self, curr_agent_id, worker_count, res_vector):
        """Generate models for single time training - only trains each model once."""
        target_id = (curr_agent_id + 1) % worker_count

        # Ensure training happens in order
        curr_counter = self.model_training_counter[target_id]
        past_counters = self.model_training_counter[:target_id]
        if target_id == 0:
            past_counters = self.model_training_counter[1:]
            curr_counter -= 1
        assert np.all([curr_counter + 1 == a for a in past_counters]), \
            'The order of model training isn\'t compatible.'
        self.model_training_counter[target_id] += 1

        # Prepare training data
        side_info = self.training_side_info_prev_d_flat[:target_id] + \
                   self.training_side_info_prev_d_flat[target_id + 1:]
        grads = self.training_side_info_prev_d_flat[target_id].copy()

        # Create new quantizer with updated parameters
        qz = self.wz_quantizer_list[target_id]
        self.wz_quantizer_list[target_id] = QuantizerWithDataPrep(
            wz_pl_model=self.wz_pl_model_class(
                inp_dim=1, side_info_size=len(side_info),
                lr=qz.wz_pl_model.lr,
                bins_per_plane=max(16 // (self.curr_round_id + 1), 2),
                num_planes=2,
                reconst_ld=qz.wz_pl_model.reconst_ld,
                tau=qz.wz_pl_model.tau,
                marginal=self.curr_round_id <= 2,
            ).to(torch.float32),
            count_side_info_data=len(side_info),
            enable_progress_bar=qz.enable_progress_bar,
            train_sample_size=qz.train_sample_size,
            user_logger=qz.user_logger,
            vec_slices=qz.vec_slices,
        )

        # Add noise and train
        grads = change_dtype_recursive(grads, torch.float32)
        grads += np.random.normal(0, np.sqrt(1e-6), len(grads)).astype(np.float32)

        self.wz_quantizer_list[target_id].train_model(
            grads, side_info, epoch=self.epoch_count, batch_size=10_000)


if __name__ == "__main__":
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import _test_main

    k = 5
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2,
                                     bins_per_plane=16, lr=1e-5, marginal=True).to(torch.float32)
    path_to_basic = r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data\basicRNN_2plane_4bins_state.pt'
    wz_model.load_state_dict(torch.load(path_to_basic, map_location='cpu'))

    base_quantizer = QuantizerWithDataPrep(wz_model, train_sample_size=200_000,
                                          count_side_info_data=0, enable_progress_bar=True, vec_slices=None)
    broadcast_prot = SingleTimeTrainingProtocol(k, base_quantizer)
    _test_main(broadcast_prot, worker_count=k, rounds=2)
