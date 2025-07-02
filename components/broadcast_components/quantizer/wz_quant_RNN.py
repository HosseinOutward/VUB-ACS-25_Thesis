from components.broadcast_components.quantizer.wz_quant_ANN import PL_EncoderDecoder_ANN, WZQuantizerANN, plot_bins
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
import torch
import torch.nn.functional as F
import numpy as np


class PL_EncoderDecoder_RNN(PL_EncoderDecoder_ANN):
    def __init__(self, num_planes, inp_dim, side_info_size, code_size, lr=1e-4, reconst_ld=100):
        super(PL_EncoderDecoder_RNN, self).__init__(inp_dim, side_info_size, code_size, lr, reconst_ld)

        self.num_planes = num_planes

        self.coding_model = EncoderDecoderLayeredRNN(
            input_dim=inp_dim, planes=self.num_planes, side_info_size=side_info_size,
            layers=4, hidden_dim=80, code_size=code_size, marginal=False)

    def custom_steps(self, batch, batch_idx, name_prefix):
        single_grad_param, side_info = batch

        tau_t = self.tau * np.exp(self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))

        reconstruct, bins, onehot_bin, prior_probs = self.coding_model.forward(single_grad_param, side_info, tau=tau_t)

        loss = 0.0
        for i in range(self.num_planes):
            # dist component of the loss
            dist = F.mse_loss(reconstruct[i], single_grad_param)
            # loss += (i+1) * reconst_ld * dist
            loss += self.reconst_ld * dist

            # rate component of the loss
            p_ux = onehot_bin[i][torch.arange(onehot_bin[i].size(0)), bins[i]]
            p_u = prior_probs[i][torch.arange(onehot_bin[i].size(0)), bins[i]]
            loss += torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

        self.log(f'{name_prefix}_loss', loss, prog_bar=True)
        # is reconstruct[-1] correct? also the last p_u
        self.log(f'{name_prefix}_x_hat_loss', F.l1_loss(reconstruct[-1], single_grad_param), prog_bar=True)
        self.log(f'{name_prefix}_rate (bits)', torch.mean(-torch.log2(p_u + 1e-12)), prog_bar=True)

        return loss

    def encode_net(self, grad_vector):
        x, _ = self.coding_model.encode(grad_vector)
        x = torch.stack(x)
        return x

    def decode_net(self, bins, side_info):
        bin_count = self.coding_model.bins_per_plane
        codes = [F.one_hot(b, num_classes=bin_count) for b in bins]
        reconstruct = self.coding_model.decode(codes, side_info)
        return reconstruct[-1]

class WZQuantizerRNN(WZQuantizerANN):
    def __init__(self, train_sample_size=100_000, metric_report_flag=False):
        super(WZQuantizerRNN, self).__init__(metric_report_flag, train_sample_size, )
        self.wz_model = PL_EncoderDecoder_RNN(num_planes=3, inp_dim=1, side_info_size=1,
                                              code_size=2, lr=1e-4, reconst_ld=100)

    def make_model_obj(self, *args, **kwargs):
        return PL_EncoderDecoder_RNN(*args, num_planes=3, **kwargs)

    def symbol_encoding(self, bins):
        return super(WZQuantizerRNN, self).symbol_encoding(np.concat(bins.numpy()))

    def symbol_decoding(self, quantized_data, vect_size):
        res =  super(WZQuantizerRNN, self).symbol_decoding(
            quantized_data, vect_size*self.wz_model.num_planes)
        return np.split(res, self.wz_model.num_planes, axis=0)


if __name__ == "__main__":
    side_info_data = np.random.normal(0, 1, 100000)
    y = side_info_data + np.random.normal(0, 0.1, 100000)

    # %%
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many")

    # %%
    wz_quantizer = WZQuantizerRNN(train_sample_size=100_000, metric_report_flag=True)
    wz_quantizer.train_new_model(y, [side_info_data], epoch=2,
                    batch_size=10_000, code_bit_size=2, lr=1e-5, reconst_ld=100)

    # %%
    print('error ', np.mean(np.abs(y - wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), [side_info_data]))))
    # plot_bins(wz_quantizer.wz_model, y)
