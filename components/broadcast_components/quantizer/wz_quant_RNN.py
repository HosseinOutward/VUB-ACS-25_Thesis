from components.broadcast_components.quantizer.wz_quant_ANN import PL_EncoderDecoder_ANN, WZQuantizerANN, plot_bins
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
import torch
import torch.nn.functional as F
import numpy as np


class PL_EncoderDecoder_RNN(PL_EncoderDecoder_ANN):
    def __init__(self, num_planes, inp_dim, side_info_size, bins_per_plane, lr=1e-4, reconst_ld=100):
        super(PL_EncoderDecoder_RNN, self).__init__(inp_dim, side_info_size, bins_per_plane, lr, reconst_ld)

        self.coding_model = EncoderDecoderLayeredRNN(
            input_dim=inp_dim, planes=num_planes, side_info_size=side_info_size,
            layers=4, hidden_dim=80, code_size=bins_per_plane, marginal=False)

    def custom_steps(self, batch, batch_idx, name_prefix):
        single_grad_param, side_info = batch

        tau_t = self.tau * np.exp(self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))

        reconstruct, bins, onehot_bin, prior_probs = self.coding_model.forward(single_grad_param, side_info, tau=tau_t)

        loss = 0.0
        for i in range(self.coding_model.planes):
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
        bins, _ = self.coding_model.encode(grad_vector)
        bins = torch.stack(bins)

        # self.aaaa = bins.detach().cpu().clone()

        bins_per_plane = self.coding_model.bins_per_plane
        for i, bin_plane in enumerate(bins[1:], 1):
            bins[i] = bins_per_plane**i * bins[i]
        bins = torch.sum(bins, dim=0)

        return bins

    def decode_net(self, bins, side_info):
        bins_per_plane = self.coding_model.bins_per_plane

        vectors = []
        for i in range(self.coding_model.planes):
            temp = bins % (bins_per_plane**(i+1))
            bins = bins - temp
            vectors.append(temp/(bins_per_plane**i))
        bins = torch.stack(vectors)

        # assert torch.all(self.aaaa == bins.detach().cpu()), "Encoded bins do not match the expected bins."

        codes = [F.one_hot(b.to(int), num_classes=bins_per_plane) for b in bins]
        reconstruct = self.coding_model.decode(codes, side_info)
        return reconstruct[-1]


class WZQuantizerRNN(WZQuantizerANN):
    def __init__(self, code_bit_per_plane, num_planes=3, *args, **kwargs):
        self.num_planes = num_planes
        super(WZQuantizerRNN, self).__init__(*args, **kwargs, code_bit_size=code_bit_per_plane*num_planes)

    def load_basic_model(self):
        assert self.bin_count == 6
        # todo add loading from file
        # self.wz_model.load_from_checkpoint('wz_model')

    def make_model_obj(self, bin_count, *args, **kwargs):
        return PL_EncoderDecoder_RNN(
            *args, **kwargs, num_planes=self.num_planes, bins_per_plane=bin_count//self.num_planes)


if __name__ == "__main__":
    side_info_data = np.random.normal(0, 1, 100000)
    y = side_info_data + np.random.normal(0, 0.1, 100000)
    side_info_data = [side_info_data]
    # side_info_data=[]

    # %%
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'val_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many")

    # %%
    wz_quantizer = WZQuantizerRNN(train_sample_size=100_000, metric_report_flag=True,
                                  code_bit_per_plane=1, lr=1e-5, count_side_info_data=len(side_info_data))
    wz_quantizer.train_model(y, side_info_data, epoch=2, batch_size=10_000)

    # %%
    y_pred = wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), side_info_data, len(y))
    print('error ', np.mean(np.abs(y - y_pred)))
    plot_bins(wz_quantizer, y, side_info_data)
