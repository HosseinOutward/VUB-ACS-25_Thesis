# This file implements basic quantizers used in conventional protocols
from collections import OrderedDict

import numpy as np
import torch
import types
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.other_utilities.brent_wz_models import ConditionalPrior
import pytorch_lightning as pl


class PL_ConditionalPrior(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.prior_model = ConditionalPrior(*args, **kwargs)

    def forward(self, side_info):
        return self.prior_model(side_info)

    def training_step(self, batch, batch_idx):
        bins, side_info = batch
        prior_probs = self.prior_model(side_info)
        loss = torch.nn.functional.cross_entropy(prior_probs, bins)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


#%%
class _ConventionalQuantizer(QuantizerWithDataPrep):
    def __init__(self, wz_pl_model, *args, bin_count_conv=None, to_clone_wz_qz=None, **kwargs):
        # used for protocol initiation
        if to_clone_wz_qz is not None:
            kwargs = dict(
                vec_slices = to_clone_wz_qz.vec_slices,
                outlier_threshold = to_clone_wz_qz.outlier_threshold,
                no_outlier_normalization = to_clone_wz_qz.no_outlier_normalization,
                no_normalization = to_clone_wz_qz.no_normalization,
                count_side_info_data = to_clone_wz_qz.count_side_info_data,
                enable_progress_bar = to_clone_wz_qz.enable_progress_bar,
                train_sample_size = to_clone_wz_qz.train_sample_size,
                user_logger = to_clone_wz_qz.user_logger,
            )
            args = ()

        # make compatible with learned protocol setup
        bin_count = bin_count_conv if bin_count_conv is not None else wz_pl_model.bin_count
        wz_pl_model = wz_pl_model.__class__(
            bins_per_plane=bin_count,
            side_info_size=wz_pl_model.side_info_size,
            marginal=True, inp_dim=1, num_planes=1, lr=1,
            reconst_ld=1, tau=0.5, tau_rate=2,
        ).to(torch.float32)
        del wz_pl_model.coding_model.encoder
        wz_pl_model.coding_model.encoder = types.SimpleNamespace(
            state_dict=lambda: OrderedDict(), load_state_dict=lambda x: None)

        super().__init__(wz_pl_model, *args, **kwargs)

        self.is_dsc = None

    def encoding_process(self, grad_vector, *args, **kwargs):
        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)
        bins = torch.tensor(self.basic_encoding(grad_vector))
        return torch.stack([bins]), (normal_param, outlier_param)

    def decoding_process(self, quantized_data, side_info_data_list, encoding_extra_data=None, batch_size=500_000):
        res = self.basic_decoding(quantized_data[0])
        res = self._post_process_grads(res, *encoding_extra_data)
        return res.astype(np.float32)

    def get_set_training_posterior_cdf(self, grad_vector=None, side_info_data_list=None):
        res = super().get_set_training_posterior_cdf(grad_vector, side_info_data_list)

        if self.is_dsc and self.wz_pl_model.coding_model.marginal:
            self.wz_pl_model.coding_model.marginal=False

        return res

    def get_prior_and_softcodes(self, grad_vector, side_info_data_list, batch_size=500_000):
        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)
        side_info_data_list = [self._apply_pre_process(a, normal_param, outlier_param)[0] for a in side_info_data_list]
        encoded_bins = self.basic_encoding(grad_vector)

        if self.wz_pl_model.coding_model.marginal:
            # -------------- marginal prior
            unique_counts = np.bincount(encoded_bins, minlength=self.bin_count)
            prior_prob = (unique_counts / unique_counts.sum()).astype(np.float32)
            prior_prob = np.array([prior_prob]*len(grad_vector))
            prior_prob = torch.tensor(prior_prob, dtype=torch.float32)
        else:
            # -------------- learn conditional prior
            conditional_prior_model = PL_ConditionalPrior(
                code_size=self.bin_count, layers=4, hidden_dim=80, input_dim=len(side_info_data_list))

            trainer = pl.Trainer(max_epochs=10, logger=False, enable_checkpointing=False,
                                 enable_progress_bar=self.enable_progress_bar)
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(encoded_bins, dtype=torch.long),
                torch.tensor(np.stack(side_info_data_list, axis=1), dtype=torch.float32)
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
            trainer.fit(conditional_prior_model, dataloader)

            conditional_prior_model.eval()
            prior_prob = None
            with torch.no_grad():
                prior_prob = conditional_prior_model(
                    torch.tensor(np.stack(side_info_data_list, axis=1), dtype=torch.float32)
                ).detach()

        return torch.stack([prior_prob]), encoded_bins

    def train_model(self, grad_vector, side_info_data_list, *args, **kwargs):
        return

    def basic_encoding(self, grad_vector):
        raise NotImplementedError

    def basic_decoding(self, quantized_data):
        raise NotImplementedError

# --------------------------
class RoundDSCQuantizer(_ConventionalQuantizer):
    def __init__(self, wz_pl_model, *args, bin_count_conv=None, **kwargs):
        bin_count = bin_count_conv if bin_count_conv is not None else wz_pl_model.bin_count
        bin_count = max(2, int(bin_count/4)) # reduce the bin count to 1/4 due to ram issues
        super().__init__(wz_pl_model, *args, bin_count_conv=bin_count, **kwargs)
        self.is_dsc = True

    def basic_encoding(self, grad_vector):
        max_v, min_v = self.outlier_threshold, -self.outlier_threshold
        bins = np.linspace(min_v, max_v, self.bin_count + 1)
        indices = np.digitize(grad_vector, bins) - 1
        indices = np.clip(indices, 0, self.bin_count - 1)
        return indices

    def basic_decoding(self, quantized_data):
        max_v, min_v = self.outlier_threshold, -self.outlier_threshold
        bins = np.linspace(min_v, max_v, self.bin_count + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        res = bin_centers[quantized_data]
        return res

class RoundBasicQuantizer(RoundDSCQuantizer):
    is_dsc=False

class SignDSCQuantizer(_ConventionalQuantizer):
    is_dsc=True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bin_count_conv=2, **kwargs)
        self.recons_center = None

    def basic_encoding(self, grad_vector):
        temp = (grad_vector >= 0)
        indices = temp.astype(int)
        self.recons_center = [grad_vector[~temp].mean(), grad_vector[temp].mean()]
        return indices

    def basic_decoding(self, quantized_data):
        assert self.recons_center is not None
        res = np.array([self.recons_center[i] for i in quantized_data])
        return res

class SignBasicQuantizer(SignDSCQuantizer):
    is_dsc=False


if __name__ == "__main__":
    import numpy as np
    import torch
    from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
    import random

    # %%
    torch.set_float32_matmul_precision('medium')
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'train_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have")

    # %%
    temp = np.random.normal(0, np.sqrt(1), 10_000_000, ).astype(np.float32)
    y = temp + np.random.normal(0, np.sqrt(0.01), 10_000_000, ).astype(np.float32)
    side_info_data = [temp.copy()]

    # %%
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=0, num_planes=2, bins_per_plane=4, tau=1.5, tau_rate=10,
                                     reconst_ld=400, lr=1e-3, marginal=True).to(torch.float32)
    # -----------
    wz_quantizer = RoundBasicQuantizer(wz_model, train_sample_size=200_000, count_side_info_data=1,
                                       enable_progress_bar=True, vec_slices=[])
    # -----------
    wz_quantizer.train_model(y, side_info_data, epoch=50, batch_size=10_000)

    # %%
    bins, temp = wz_quantizer.encoding_process(y)
    recons = wz_quantizer.decoding_process(bins, side_info_data, temp).numpy()

    # %%
    # import matplotlib.pyplot as plt
    # temp = np.argsort(y[:1_000_000])
    # plt.figure(figsize=(16, 3))
    # plt.scatter(y[temp], np.abs(y[temp] - recons[temp]) / np.mean(np.abs(y[temp])), s=0.1)
    # plt.show()

    # %%
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import get_obj_size
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import (
        fix_outlier_in_prior, fix_zero_probabilities, compress_data_list)
    from components.broadcast_components.compressor.rans_coding import rans_batch_encode

    prior = wz_quantizer.get_set_training_posterior_cdf(y, side_info_data)
    bins_with_outlier = bins[0].numpy()

    rate_emp = get_obj_size(compress_data_list(rans_batch_encode(bins_with_outlier, prior[0]))) / len(y) * 8
    rate_th = prior[0][torch.arange(bins_with_outlier.size), bins_with_outlier]

    print("avg bit rate:", rate_emp)
    print("avg bit rate (theo):", np.mean(-np.log2(rate_th + 1e-12)))
    print("mse:", np.mean((y - recons) ** 2))
    print("mape:", np.mean(np.abs(y - recons)) / np.mean(np.abs(y)))

    # -------------
    _, _, (outlier_positions, _, _) = wz_quantizer._apply_pre_process(y)
    bins_with_outlier = np.concatenate([bins_with_outlier, bins_with_outlier[outlier_positions]])
    prior = fix_outlier_in_prior(prior, outlier_positions)
    prior = fix_zero_probabilities(prior)

    rate_emp = get_obj_size(compress_data_list(rans_batch_encode(bins_with_outlier, prior[0]))) / len(y) * 8
    rate_th = prior[0][torch.arange(bins_with_outlier.size), bins_with_outlier]

    print("with outlier bit rate:", rate_emp)
    print("with outlier bit rate (theo):", np.mean(-np.log2(rate_th + 1e-12)))
