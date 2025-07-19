import gc
from typing import List

from pytorch_lightning.loggers import CSVLogger

from components.other_utilities.brent_wz_models import EncoderDecoder
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


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
    def __init__(self, inp_dim, side_info_size, bin_count=None, tau=1, lr=1e-4, reconst_ld=100, *args, **kwargs):
        super().__init__()
        side_info_size = side_info_size if side_info_size != 0 else 1
        self.reconst_ld = reconst_ld
        self.tau = tau
        self.lr = lr
        self.lr_step = 40

        # if this is a call from super of rnn, don't make the coding model
        if not hasattr(self, 'coding_model'):
            self.coding_model = EncoderDecoder(
                input_dim=inp_dim, side_info_size=side_info_size,
                layers=4, hidden_dim=80, bin_count=bin_count, marginal=False)

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
        return loss, reconstruct, single_grad_param, bin_no, p_u, bins_probs, prior_probs

    # todo reduce workload by reusing the logit bin values before softmax/gumble from loss calc
    def log_metrics(self, name_prefix, loss, inp_rec, inp, bin_no_vec, p_u, bins_probs, prior_probs):
        # train_db = 10 * np.log10(train_mse_loss / TRAIN_BATCHES)

        self.log(f'{name_prefix}_loss', loss, prog_bar=True)
        recons_loss = torch.mean(torch.abs((inp - inp_rec) / (inp.abs() + 1e-8))) * 100
        self.log(f'{name_prefix}_mape%', recons_loss, prog_bar=True)
        self.log(f'{name_prefix}_mse', F.mse_loss(inp_rec, inp), prog_bar=True)
        self.log(f'{name_prefix}_rate_bits', torch.mean(-torch.log2(p_u + 1e-12)), prog_bar=True)

        practical_p_u, _ = get_real_bin_prob(bin_no_vec, self.bin_count)
        self.log(f'{name_prefix}_real_bit_r', torch.mean(-torch.log2(practical_p_u + 1e-12)), prog_bar=True)

    def training_step(self, batch, batch_idx):
        res = self.compute_loss(batch, batch_idx)

        self.log_metrics('train_gumble', *res)

        self.eval()
        with torch.no_grad():
            self.log_metrics('train', *self.compute_loss(batch, batch_idx))
        self.train()

        loss = res[0]
        return loss

    def validation_step(self, batch, batch_idx):
        res = self.compute_loss(batch, batch_idx)
        self.log_metrics('val', *res)

        loss = res[0]
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=0.3)
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


# ---------------------------------------------
class WZQuantizer:
    def __init__(self, wz_pl_model, count_side_info_data,
                 enable_progress_bar=False, train_sample_size=100_000, user_logger=None, *args, **kwargs):
        from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
        assert isinstance(wz_pl_model, PL_EncoderDecoder_ANN)

        self.val_indices = None
        self.enable_progress_bar = enable_progress_bar
        self.user_logger = user_logger

        self.train_sample_size = train_sample_size

        self.wz_pl_model = wz_pl_model
        self.count_side_info_data = count_side_info_data

    @property
    def bin_count(self):
        return self.wz_pl_model.bin_count

    # def get_bin_probs(self):
    #     return None

    def encoding_process(self, grad_vector, batch_size=500_000):
        # from components.broadcast_components.WZ_models.simple import simple_quantize
        # return simple_quantize(grad_vector)

        grad_tensor = torch.tensor(grad_vector).to(torch.float32)
        total_size = len(grad_tensor)

        self.wz_pl_model.to('cuda')
        self.wz_pl_model.eval()

        all_bins = []
        with torch.no_grad():
            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)
                batch = grad_tensor[i:end_idx].unsqueeze(1).to('cuda')

                bins_batch = self.wz_pl_model.encode_net(batch).to('cpu')
                all_bins.append(bins_batch)

                batch.to('cpu')
        self.wz_pl_model.to('cpu')
        torch.cuda.empty_cache()

        # todo separate the running of the model for ann and rnn
        if len(all_bins) <= 1:
            bins = all_bins[0]
        else:
            bins = torch.cat(all_bins, dim=1) if len(all_bins[0].shape) > 1 else torch.cat(all_bins, dim=0)

        dtype = torch.uint8 if self.bin_count < 2**8 else torch.uint16
        return bins.to(dtype)

    # todo separate the running of the model for ann and rnn
    def decoding_process(self, quantized_data, side_info_data_list,
                         element_count, batch_size=500_000):
        # from components.broadcast_components.WZ_models.simple import simple_dequantize
        # return simple_dequantize(quantized_data, np.float32)

        bins_tensor = torch.tensor(np.array(quantized_data))
        total_size = len(bins_tensor[0]) if len(bins_tensor.shape) == 2 else len(bins_tensor)

        assert len(side_info_data_list) == self.count_side_info_data
        if self.count_side_info_data == 0:
            side_info_data_list = [np.zeros(element_count)]

        assert total_size == len(side_info_data_list[0])

        self.wz_pl_model.to('cuda')
        self.wz_pl_model.eval()

        all_reconstructs = []

        with torch.no_grad():
            side_info_array = torch.tensor(np.array(side_info_data_list), dtype=torch.float32).T

            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)

                if len(bins_tensor.shape) != 2:
                    bins_batch = bins_tensor[i:end_idx].to('cuda')
                else:
                    bins_batch = bins_tensor[:, i:end_idx].to('cuda')

                side_info_batch = side_info_array[i:end_idx].to('cuda')

                reconstructs_batch = self.wz_pl_model.decode_net(bins_batch, side_info_batch)
                all_reconstructs.append(reconstructs_batch.to('cpu'))

                # Clear GPU memory for this batch
                del bins_batch, side_info_batch
                torch.cuda.empty_cache()

        self.wz_pl_model.to('cpu')

        all_reconstructs = torch.cat(all_reconstructs, dim=0)
        res = all_reconstructs.squeeze()
        res = res.numpy()
        return res

    # todo have multiple input data and train on all of them in one run (change sampler)
    def train_model(self, input_data, side_info_data_list: List, epoch=10, batch_size=50_000):
        # return

        assert len(side_info_data_list) == self.count_side_info_data
        if self.count_side_info_data == 0:
            side_info_data_list = [np.zeros(len(input_data))]
        side_info_data_list = torch.tensor(np.array(side_info_data_list)).T
        input_data = torch.tensor(input_data).unsqueeze(1)

        train_dataset = torch.utils.data.TensorDataset(input_data, side_info_data_list)

        if self.train_sample_size > len(train_dataset):
            self.train_sample_size = len(train_dataset)

        # --------------- val dataloader ---------------
        val_dataloader = None
        if self.enable_progress_bar:
            all_indices = np.arange(len(train_dataset))

            self.val_indices = np.random.choice(all_indices, size=int(self.train_sample_size // 3), replace=False)
            val_dataset = torch.utils.data.Subset(train_dataset, self.val_indices)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size * 10,
                num_workers=2, pin_memory=False, persistent_workers=True)

            train_indices = np.setdiff1d(all_indices, self.val_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        # ------------------------------

        class RandomReplacementSampler(torch.utils.data.Sampler):
            def __init__(self, dataset_size, samples_per_epoch):
                super().__init__(data_source=None)
                self.dataset_size = dataset_size
                self.samples_per_epoch = samples_per_epoch
            def __iter__(self):
                temp = [np.random.randint(0, self.dataset_size) for _ in range(self.samples_per_epoch)]
                return iter(temp)
            def __len__(self):
                return self.samples_per_epoch
        train_sampler = RandomReplacementSampler(
            dataset_size=len(train_dataset), samples_per_epoch=self.train_sample_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True)

        # ------------------------------
        self.wz_pl_model.to('cpu')

        # disable val progress bar due to pl bug
        NoValidationBar = None
        if self.enable_progress_bar:
            print('          - training wz models')

            from pytorch_lightning.callbacks import TQDMProgressBar
            from tqdm import tqdm
            class NoValidationBar(TQDMProgressBar):
                def init_validation_tqdm(self): return tqdm(disable=True)

        trainer = pl.Trainer(
            accelerator="cuda",
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            log_every_n_steps=1,
            max_epochs=epoch,
            enable_progress_bar=self.enable_progress_bar,
            callbacks=[NoValidationBar()] if self.enable_progress_bar else [],
            logger=self.user_logger
        )
        trainer.fit(self.wz_pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        del train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()


def plot_bins(wz_quantizer: WZQuantizer, x_data_, side_info, step_count=1000, training_ind=False):
    from matplotlib import pyplot as plt

    def _plot_bin_regions(ax, x_range, decoded_bins, x_step, colors):
        current_bin = decoded_bins[0]
        last_idx = 0
        last_split_point = x_range[0]
        for i, x_v in enumerate(x_range[1:], start=1):
            if decoded_bins[i] != current_bin:  # change detected
                split_point = (x_v + x_range[i - 1]) / 2
                ax.axvline(split_point, color='black', linestyle='--', linewidth=0.5)
                ax.axvspan(x_range[last_idx], split_point, color=colors[current_bin], alpha=0.3)

                current_bin = decoded_bins[i]
                last_idx = i
                last_split_point = split_point
        # Final span
        ax.axvspan(last_split_point, x_range[-1], color=colors[current_bin], alpha=0.3)

    if isinstance(x_data_, torch.Tensor): x_data_ = x_data_.cpu().numpy()
    ind_list = wz_quantizer.val_indices \
        if training_ind else np.setdiff1d(np.arange(len(x_data_)), wz_quantizer.val_indices)

    min_v, max_v = np.percentile(x_data_, 0.01), np.percentile(x_data_, 99.99)
    true_min_v, true_max_v = x_data_.min(), x_data_.max()

    grad_data = np.asarray(x_data_, dtype=np.float32)[ind_list]
    side_info = [si[ind_list] for si in side_info]

    sort_idx = np.argsort(grad_data)
    grad_data = grad_data[sort_idx]
    side_info = [si[sort_idx] for si in side_info]

    x_step = (max_v - min_v) / step_count
    pointer_v = true_min_v
    spaced_idx = []
    for i, gd in enumerate(grad_data):
        # assert pointer_v + x_step * 2 > gd, f"Data has gaps larger than x_step ({(gd-pointer_v)/x_step})"
        if pointer_v + x_step * 0.95 < gd:
            pointer_v += gd
            spaced_idx.append(i)
    spaced_idx.append(len(grad_data)-1)
    spaced_idx = np.array(spaced_idx)
    grad_data = grad_data[spaced_idx]
    side_info = [si[spaced_idx] for si in side_info]

    # Create x_range for plotting
    bins = wz_quantizer.encoding_process(grad_data)
    if len(bins.shape) != 1: # if a list of bins is returned, unify them
        bins = wz_quantizer.wz_pl_model.unify_bins(bins)
    recons_for_x_range = wz_quantizer.decoding_process(
        bins, side_info, element_count=len(grad_data))

    # Setup plots
    bin_count = wz_quantizer.bin_count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Data histogram with binning analysis ------------------------
    counts, bins_edges, patches = ax1.hist(
        x_data_[np.random.choice(len(x_data_), 20_000, replace=False)],
        bins=200, alpha=0.3, color='gray', label='data histogram', density=False)
    ax1.clear()
    ax1.bar(bins_edges[:-1], counts / np.max(counts), width=np.diff(bins_edges),
            alpha=0.3, color='gray', label='data histogram (normalized)', align='edge')
    ax1.plot(grad_data, np.abs(grad_data - recons_for_x_range), label='reconstruction error')
    ax1.plot(grad_data, (np.array(bins) + 1) / bin_count, label='(normalized) encoded_bins')
    ax1.set_xlabel('x_range')

    # Visualize bin regions
    colors = plt.cm.viridis(np.linspace(0, 1, bin_count))
    _plot_bin_regions(ax1, grad_data, bins, x_step, colors)

    ax1.set_xlim(true_min_v, true_max_v)
    ax1.legend(loc="upper right")
    ax1.set_title('Binning Visualization')

    # Plot 2: Reconstruction curves for each bin (batch process) ------------------------
    # todo merge the 2 plots and change how the side_info is given (for example for all 0s as side info)
    for bin_idx in range(bin_count):
        temp = wz_quantizer.decoding_process(
            np.zeros(len(grad_data)) + bin_idx, side_info, element_count=len(grad_data))
        ax2.plot(grad_data, temp, label=f'bin={bin_idx}', linewidth=0.2)

    ax2.set_xlabel('x_range (which is forced to bin, but is paired with related side_info)')
    ax2.set_ylabel('reconstruction per bin')

    ax2.set_xlim(true_min_v, true_max_v)
    # ax2.legend(loc="upper right")
    ax2.set_title('What if x range was forced to a specific bin, plot per bin')

    plt.tight_layout()
    plt.show()

    #%%
    # x_range = np.linspace(true_min_v, true_max_v, step_count)
    # for b in range(bin_count):
    #     temp = wz_quantizer.decoding_process(np.zeros(len(x_range))+b, [x_range],
    #                                          element_count=len(x_range), symbolic_decoder=False)
    #     plt.plot(x_range, temp, label=f'bin={b}', linewidth=0.5)
    # plt.xlabel('side info')
    # plt.ylabel('reconstruction per bin')
    # plt.show()


if __name__ == "__main__":
    side_info_data = np.random.normal(0, 1, 100000).astype(np.float32)
    y = side_info_data + np.random.normal(0, 0.1, 100000).astype(np.float32)
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
    pl_model = PL_EncoderDecoder_ANN(
        inp_dim=1, side_info_size=len(side_info_data), bin_count=3, lr=1e-5
    )
    wz_quantizer = WZQuantizer(wz_pl_model=pl_model,
                               count_side_info_data=len(side_info_data),
                               train_sample_size=100_000, enable_progress_bar=True)
    wz_quantizer.train_model(y, side_info_data, epoch=2, batch_size=10_000)

    # %%
    y_pred = wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), side_info_data, len(y))
    print('error ', np.mean(np.abs(y - y_pred)))
    plot_bins(wz_quantizer, y, side_info_data)
