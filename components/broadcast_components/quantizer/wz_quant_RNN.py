from components.broadcast_components.quantizer.wz_quant_ANN import PL_EncoderDecoder_ANN, WZQuantizerANN, plot_bins
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
import torch
import torch.nn.functional as F
import numpy as np


class PL_EncoderDecoder_RNN(PL_EncoderDecoder_ANN):
    def __init__(self, num_planes, bins_per_plane, inp_dim, side_info_size, *args, **kwargs):
        super(PL_EncoderDecoder_RNN, self).__init__(inp_dim, side_info_size, *args, **kwargs, bin_count=bins_per_plane)

        self.coding_model = EncoderDecoderLayeredRNN(
            input_dim=inp_dim, planes=num_planes, side_info_size=side_info_size,
            layers=3, hidden_dim=100, code_size=bins_per_plane, marginal=False)

    def compute_loss_and_log(self, batch, batch_idx, name_prefix):
        single_grad_param, side_info = batch

        tau_t = self.tau * np.exp(self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))

        reconstruct, bins, onehot_bin, prior_probs =\
            self.coding_model.forward(single_grad_param, side_info, tau=tau_t)

        loss = 0.0
        assert 1>self.reconst_ld>0
        for i in range(self.coding_model.planes):
            dist = F.mse_loss(reconstruct[i], single_grad_param)
            dist = dist / torch.mean(single_grad_param**2)
            loss = loss + self.reconst_ld*dist

            # rate component of the loss
            p_ux = onehot_bin[i][torch.arange(onehot_bin[i].size(0)), bins[i]]
            p_u = prior_probs[i][torch.arange(onehot_bin[i].size(0)), bins[i]]
            loss = loss+torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12))) * (1-self.reconst_ld)
        loss = loss / self.coding_model.planes

        self.log(f'{name_prefix}_loss', loss, prog_bar=True)
        # is reconstruct[-1] correct? also the last p_u
        self.log(f'{name_prefix}_x_hat_loss', F.l1_loss(reconstruct[-1], single_grad_param), prog_bar=True)
        self.log(f'{name_prefix}_rate (bits)', torch.mean(-torch.log2(p_u + 1e-12)), prog_bar=True)

        practical_p_u = self.unify_bins([b.detach() for b in bins]).cpu().numpy()
        bin_appearance_counts = np.unique(practical_p_u, return_counts=True)
        practical_p_u = torch.tensor(practical_p_u, dtype=torch.float32)
        for b, count in zip(*bin_appearance_counts):
            practical_p_u[practical_p_u == b] = count / len(practical_p_u)
        temp = torch.mean(-torch.log2(practical_p_u + 1e-12))
        self.log(f'{name_prefix}_real_bit_r', temp, prog_bar=True)

        return loss

    def unify_bins(self, list_bins):
        list_bins=torch.stack([l.clone() for l in list_bins])
        bins_per_plane = self.coding_model.bins_per_plane
        for i, bin_plane in enumerate(list_bins[1:], 1):
            list_bins[i] = bins_per_plane**i * list_bins[i]
        bins = torch.sum(list_bins, dim=0)
        return bins

    def deunify_bins(self, bins, bins_per_plane):
        bins=bins.clone()

        vectors = []
        for i in range(self.coding_model.planes):
            temp = bins % (bins_per_plane**(i+1))
            bins = bins - temp
            vectors.append(temp/(bins_per_plane**i))
        list_bins = torch.stack(vectors)
        return list_bins

    def encode_net(self, grad_vector):
        bins, _ = self.coding_model.encode(grad_vector)
        bins = torch.stack(bins)

        # self.aaaa = bins.detach().cpu().clone()
        bins = self.unify_bins(bins)

        assert torch.unique(bins).size(0) <= self.coding_model.bins_per_plane**self.coding_model.planes

        return bins

    def decode_net(self, bins, side_info):
        bins_per_plane = self.coding_model.bins_per_plane

        bins = self.deunify_bins(bins, bins_per_plane)
        # assert torch.all(self.aaaa == bins.detach().cpu()), "Encoded bins do not match the expected bins."

        codes = [F.one_hot(b.to(int), num_classes=bins_per_plane) for b in bins]
        reconstruct = self.coding_model.decode(codes, side_info)
        return reconstruct[-1]


class WZQuantizerRNN(WZQuantizerANN):
    def __init__(self, bins_per_plane, num_planes=3, *args, **kwargs):
        self.num_planes = num_planes
        bin_count = bins_per_plane ** num_planes
        super(WZQuantizerRNN, self).__init__(*args, **kwargs, bin_count=bin_count)

    def load_basic_model(self):
        assert self.bin_count == 6
        # todo add loading from file
        # self.wz_model.load_from_checkpoint('wz_model')

    def make_model_obj(self, bin_count, *args, **kwargs):
        bins_per_plane = int(round(bin_count ** (1 / self.num_planes)))
        return PL_EncoderDecoder_RNN(
            *args, **kwargs, num_planes=self.num_planes, bins_per_plane=bins_per_plane)


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
                                  bins_per_plane=1, lr=1e-5, count_side_info_data=len(side_info_data))
    wz_quantizer.train_model(y, side_info_data, epoch=2, batch_size=10_000)

    # %%
    y_pred = wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), side_info_data, len(y))
    print('error ', np.mean(np.abs(y - y_pred)))
    plot_bins(wz_quantizer, y, side_info_data)
