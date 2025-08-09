import pytorch_lightning as pl
from torch.nn import functional as F

from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer, plot_bins
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN, EncoderDecoder
import torch
import torch.nn.functional as F
import numpy as np

def get_real_bin_prob(bin_no, bin_count):
    temp = bin_no.detach().clone()
    bin_appearance_counts = torch.unique(temp, return_counts=True)
    practical_p_u=temp.float()
    for b, count in zip(*bin_appearance_counts):
        practical_p_u[practical_p_u == b] = count / len(bin_no)

    bin_prob_vec = [0.0] * bin_count
    for b, count in zip(*bin_appearance_counts):
        bin_prob_vec[b] = float((count / len(bin_no)).cpu())
    return practical_p_u, torch.tensor(bin_prob_vec)


class PL_EncoderDecoder_ANN(pl.LightningModule):
    def __init__(self, inp_dim, side_info_size, bin_count=None, tau=4, lr=8e-4, reconst_ld=400, marginal=False):
        super().__init__()
        side_info_size = side_info_size if side_info_size != 0 else 1
        self.reconst_ld = reconst_ld
        self.tau = tau
        self.lr = lr
        self.lr_step = 40
        self.mspe_denom = None

        # if this is a call from super of rnn, don't make the coding model
        if not hasattr(self, 'coding_model'):
            self.coding_model = EncoderDecoder(
                input_dim=inp_dim, side_info_size=side_info_size,
                layers=4, hidden_dim=80, bin_count=bin_count, marginal=marginal)

    @property
    def bin_count(self):
        return self.coding_model.bin_count

    def compute_loss(self, batch, batch_idx):
        tau_t = self.tau * np.exp(
            self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))
        single_grad_param, side_info = batch
        reconstruct, bins_probs, prior_probs = self.coding_model.forward(single_grad_param, side_info, tau=tau_t)
        bin_no = torch.argmax(bins_probs, dim=-1)
        temp = torch.arange(bins_probs.size(0))
        p_ux = bins_probs[temp, bin_no]
        p_u = prior_probs[temp, bin_no] + 1e-12
        loss = torch.log(p_ux / p_u).mean() + self.reconst_ld * F.mse_loss(reconstruct, single_grad_param)
        return [loss, reconstruct, single_grad_param, bin_no, p_u, bins_probs, prior_probs]

    # todo reduce workload by reusing the logit bin values before softmax/gumble from loss calc
    def log_metrics(self, name_prefix, loss, inp_rec, inp, bin_no_vec, p_u, bins_probs, prior_probs):
        # train_db = 10 * np.log10(train_mse_loss / TRAIN_BATCHES)

        self.log(f'{name_prefix}_loss', loss, prog_bar=True)

        recons_loss = torch.mean((inp - inp_rec)**2) / (self.mspe_denom + 1e-8) * 100
        self.log(f'{name_prefix}_mape%', recons_loss, prog_bar=True)

        self.log(f'{name_prefix}_mse', F.mse_loss(inp_rec, inp), prog_bar=True)
        self.log(f'{name_prefix}_rate_bits', torch.mean(-torch.log2(p_u + 1e-12)), prog_bar=True)

        practical_p_u, _ = get_real_bin_prob(bin_no_vec, self.bin_count)
        self.log(f'{name_prefix}_real_bit_r', torch.mean(-torch.log2(practical_p_u + 1e-12)), prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.mspe_denom = (self.mspe_denom + torch.mean(batch[0]**2)) / 2 \
            if self.mspe_denom is not None else torch.mean(batch[0]**2)

        res = self.compute_loss(batch, batch_idx)
        loss, res = res[0], res[1:]

        # Skip logging and redundant compute_loss if no logger and no progress bar
        has_logger = self.trainer.logger is not False and self.trainer.logger is not None
        has_progress_bar = self.trainer.progress_bar_callback is not None

        if (has_logger or has_progress_bar) and batch_idx%3==0:
            self.log_metrics('train_gumble', loss.detach(), *res)

            self.eval()
            with torch.no_grad():
                assert not self.coding_model.training
                self.log_metrics('train', *self.compute_loss(batch, batch_idx))
            self.train()

        return loss

    def validation_step(self, batch, batch_idx):
        res = self.compute_loss(batch, batch_idx)
        self.log_metrics('val', *res)

        loss = res[0]
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.trainer.max_epochs*np.ceil(self.lr_step/180)), gamma=0.3)
        return [optimizer], [scheduler]

    def encode_net(self, grad_vector):
        x = self.coding_model.encoder(grad_vector)
        x = F.softmax(x, dim=-1)
        x = torch.argmax(x, dim=-1)
        return x

    def decode_net(self, bins, side_info):
        reconstruct = self.coding_model.decoder(
            F.one_hot(bins.long(), num_classes=self.bin_count), side_info)
        return reconstruct

    def get_prior_and_softcodes_net(self, grad_vector, side_info):
        from components.other_utilities.brent_wz_models import MarginalPrior
        assert not self.coding_model.training
        model_is_marginal = isinstance(self.coding_model.prior, MarginalPrior)
        assert model_is_marginal == (side_info is None or len(side_info)==0)

        prior = self.coding_model.prior(side_info)
        soft_code = F.softmax(self.coding_model.encoder(grad_vector), dim=-1)

        return prior, soft_code


class PL_EncoderDecoder_RNN(PL_EncoderDecoder_ANN):
    def __init__(self, num_planes, bins_per_plane, inp_dim, side_info_size,
                 tau_rate=10, marginal=False, *args, **kwargs):
        self.coding_model = None
        assert abs(tau_rate)>1
        self.tau_rate = tau_rate

        side_info_size = side_info_size if side_info_size != 0 else 1
        super().__init__(inp_dim, side_info_size, *args, **kwargs)

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
            rate_weight = lambda x:((x-1) + np.exp(x * np.log(abs(self.tau_rate))))/abs(self.tau_rate)
            rate_weight = rate_weight(training_prog) if self.tau_rate <= 0 else 1-rate_weight(1-training_prog)
            loss = loss + rate_loss * max(rate_weight, 0.1)
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

        assert bins.max() < b_p_p
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
    bins_vector, extra_enc_data = wz_quantizer.encoding_process(y)
    y_pred = wz_quantizer.decoding_process(bins_vector, side_info_data, encoding_extra_data=extra_enc_data)
    print('error ', np.mean(np.abs(y - y_pred)))
    plot_bins(wz_quantizer, y, side_info_data)