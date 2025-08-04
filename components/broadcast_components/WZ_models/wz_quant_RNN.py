from components.broadcast_components.WZ_models.wz_quant_ANN import PL_EncoderDecoder_ANN, WZQuantizer, plot_bins
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
import torch
import torch.nn.functional as F
import numpy as np


class PL_EncoderDecoder_RNN(PL_EncoderDecoder_ANN):
    def __init__(self, num_planes, bins_per_plane, inp_dim, side_info_size, marginal=False, *args, **kwargs):
        self.coding_model = None

        side_info_size = side_info_size if side_info_size != 0 else 1
        super(PL_EncoderDecoder_RNN, self).__init__(inp_dim, side_info_size, *args, **kwargs)

        self.coding_model = EncoderDecoderLayeredRNN(
            input_dim=inp_dim, side_info_size=side_info_size,
            num_planes=num_planes,  bins_per_plane=bins_per_plane,
            layers=3, hidden_dim=100, marginal=marginal)

    @property
    def num_planes(self):
        return self.coding_model.num_planes

    @property
    def bins_per_plane(self):
        return self.coding_model.bins_per_plane

    def compute_loss(self, batch, batch_idx):
        single_grad_param, side_info = batch
        training_prog = self.current_epoch / (self.trainer.max_epochs + 1)
        tau_t = self.tau * np.exp(training_prog * np.log(0.1 / self.tau))

        reconstruct, bins_no, soft_codes, prior_probs =\
            self.coding_model.forward(single_grad_param, side_info, tau=tau_t)

        loss = 0.0
        pu_vec = torch.ones(len(single_grad_param), dtype=torch.float32, device=single_grad_param.device)
        for i in range(self.num_planes):
            # reconstruction component of the loss
            dist = F.mse_loss(reconstruct[i], single_grad_param)
            dist = dist / self.mspe_denom
            loss = loss + self.reconst_ld * dist

            # rate component of the loss
            p_ux = soft_codes[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            p_u = prior_probs[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            pu_vec*=p_u
            rate_loss = torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))
            rate_weight = (np.exp(training_prog * np.log(5)))/5
            loss = loss + rate_loss * rate_weight
        loss = loss / self.num_planes
        f = lambda x: [a.detach() for a in x]
        return [loss, reconstruct[-1].detach(), single_grad_param,
                    f(bins_no), pu_vec.detach(), f(soft_codes), f(prior_probs)]

    def new_compute_loss(self, batch, batch_idx):
        single_grad_param, side_info = batch
        training_progress = self.trainer.current_epoch / (self.trainer.max_epochs + 1)

        training_timing = min(max(training_progress*1.2-0.1, 0), 1)
        tau_t = self.tau * np.exp(training_timing * np.log(0.1 / self.tau))

        temp = 0 if training_timing<=1e-6 else 4/100
        temp = [torch.quantile(single_grad_param, q) for q in [0.04 - temp, 0.96 + temp]]
        recons_target = torch.clip(single_grad_param, *temp)

        reconstruct, bins_no, soft_codes, prior_probs =\
            self.coding_model.forward(recons_target, side_info, tau=tau_t)

        pu_vec = torch.ones(len(single_grad_param), dtype=torch.float32, device=single_grad_param.device)

        # Better Loss Aggregation: Store individual layer losses for analysis
        layer_rate_losses = []
        layer_distortion_losses = []
        previous_error_vec = torch.zeros_like(single_grad_param)
        for i in range(self.num_planes):
            # Rate component of the loss for this layer
            p_ux = soft_codes[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            p_u = prior_probs[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            pu_vec *= p_u

            # Rate loss for this layer (KL divergence)
            rate_loss = torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))
            layer_rate_losses.append(rate_loss)

            # **************************

            current_error_vec = torch.abs(recons_target - reconstruct[i] + 1e-6)
            error_change = current_error_vec - previous_error_vec
            improvement_amount = F.relu(-error_change)**2
            worsen_amount = F.relu(error_change)**2

            dist_loss = (
                    0.1*(torch.mean(worsen_amount)*10-torch.mean(improvement_amount)) +
                    (torch.mean(current_error_vec**2))
                ) / self.mspe_denom

            previous_error_vec = current_error_vec.detach()

            layer_distortion_losses.append(dist_loss)

        # **************************
        layer_weights = torch.linspace(0.5, 1.5, self.num_planes).to(single_grad_param.device)
        distortion_loss = sum(w * r for w, r in zip(layer_weights, layer_distortion_losses))

        # **************************
        rate_loss = sum(layer_rate_losses) * (0.1 + 0.9 * training_timing)
        distortion_loss = distortion_loss * (self.reconst_ld * (2.0 - training_timing))

        # Better Loss Aggregation: Final loss combination with progressive weighting
        total_loss = (rate_loss + distortion_loss) / self.num_planes

        return total_loss, reconstruct[-1], single_grad_param, bins_no, pu_vec, soft_codes, prior_probs

    def log_metrics(self, name_prefix, loss, inp_rec, inp, bins_no_mat, p_u, bins_probs, prior_probs):
        unified_bins = self.unify_bins([b.detach() for b in bins_no_mat])
        super().log_metrics(name_prefix, loss, inp_rec, inp, unified_bins, p_u, bins_probs, prior_probs)

    def encode_net(self, grad_vector):
        bins_list, _ = self.coding_model.encode(grad_vector)
        bins_list = torch.stack(bins_list)

        assert torch.unique(bins_list).size(0) <= self.coding_model.bins_per_plane**self.coding_model.num_planes

        return bins_list

    def decode_net(self, bins, side_info):
        b_p_p = self.coding_model.bins_per_plane

        # if bins is a single vector, it means it was unified (wz_rnn outputs list of vectors)
        if len(bins.size()) == 1:
            bins = self.deunify_bins(bins)

        codes = [F.one_hot(b.to(int), num_classes=b_p_p) for b in bins]
        reconstruct = self.coding_model.decode(codes, side_info)
        return reconstruct[-1]

    def get_prior_and_softcodes_net(self, grad_vector, side_info=None):
        assert not self.coding_model.training
        assert self.coding_model.marginal == (side_info is None or len(side_info)==0)

        bins_list, soft_codes = self.coding_model.encode(x=grad_vector, tau=None, force_softmax=True)
        priors = self.coding_model.get_priors(codes=soft_codes, y=side_info, tau=None)

        for i in range(self.num_planes):
            soft_codes[i] = soft_codes[i][torch.arange(len(bins_list[i])), bins_list[i]]
            priors[i] = priors[i][torch.arange(len(bins_list[i])), bins_list[i]]

        return torch.stack(priors), torch.stack(soft_codes)

    def unify_bins(self, list_bins,):
        list_bins=torch.stack([l.clone() for l in list_bins])
        for i, bin_plane in enumerate(list_bins[1:], 1):
            list_bins[i] = self.coding_model.bins_per_plane**i * list_bins[i]
        bins = torch.sum(list_bins, dim=0)
        return bins

    def deunify_bins(self, bins):
        bins=bins.clone()
        vectors = []
        for i in range(self.num_planes):
            temp = bins % (self.bins_per_plane**(i+1))
            bins = bins - temp
            vectors.append(temp/(self.bins_per_plane**i))
        list_bins = torch.stack(vectors)
        return list_bins


if __name__ == "__main__":
    side_info_data = np.random.normal(0, 1, 1_000_000).astype(np.float32)
    y = side_info_data + np.random.normal(0, 0.1, 1_000_000).astype(np.float32)
    side_info_data = [side_info_data]

    # %%
    import logging
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
    warnings.filterwarnings("ignore", message="Consider setting `persistent_workers=True` in 'val_dataloader'")
    warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many")

    # %%
    side_info_size = len(side_info_data) if len(side_info_data) > 0 else 1
    pl_model = PL_EncoderDecoder_RNN(
        inp_dim=1, side_info_size=side_info_size,
        reconst_ld=100, num_planes=3, bins_per_plane=2, lr=1e-3,
    )
    wz_quantizer = WZQuantizer(wz_pl_model=pl_model,
                               count_side_info_data=len(side_info_data),
                               train_sample_size=200_000, enable_progress_bar=True)
    wz_quantizer.train_model(y, side_info_data, epoch=2, batch_size=1_000)

    # %%
    y_pred = wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), side_info_data, len(y))
    print('error ', np.mean(np.abs(y - y_pred)))
    plot_bins(wz_quantizer, y, side_info_data)
