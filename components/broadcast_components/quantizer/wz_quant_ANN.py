import gc

from components.other_utilities.brent_wz_models import EncoderDecoder
from components.broadcast_components.compressor.arithmatic_coding import arithmetic_encode, arithmetic_decode
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class PL_EncoderDecoder_ANN(pl.LightningModule):
    def __init__(self, inp_dim, side_info_size, code_size, lr=1e-4, reconst_ld=100):
        super(PL_EncoderDecoder_ANN, self).__init__()
        self.coding_model = EncoderDecoder(
            input_dim=inp_dim, side_info_size=side_info_size,
            layers=4, hidden_dim=80, code_size=code_size, marginal=False)
        self.reconst_ld = reconst_ld
        self.tau = 1
        self.lr = lr
        self.lr_step = 40

    def custom_steps(self, batch, batch_idx, name_prefix):
        tau_t = self.tau * np.exp(
            self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))

        single_grad_param, side_info = batch
        reconstruct, bins_probs, prior_probs = self.coding_model.forward(single_grad_param, side_info, tau=tau_t)

        bin_no = torch.argmax(bins_probs, dim=-1)
        temp = torch.arange(bins_probs.size(0))
        p_ux = bins_probs[temp, bin_no]
        p_u = prior_probs[temp, bin_no] + 1e-12

        loss = torch.log(p_ux / p_u).mean() + self.reconst_ld * F.mse_loss(reconstruct, single_grad_param)

        # avoid nan by reducing loss
        # detached_loss = loss.detach().item()
        # if detached_loss>100:
        #     loss *= 100/detached_loss

        # train_db = 10 * np.log10(train_mse_loss / TRAIN_BATCHES)
        # train_mse_loss = train_mse_loss / TRAIN_BATCHES
        # train_loss = train_loss / TRAIN_BATCHES

        self.log(f'{name_prefix}_loss',
                 loss, prog_bar=True)
        self.log(f'{name_prefix}_x_hat_loss',
                 F.l1_loss(reconstruct, single_grad_param), prog_bar=True)
        # temp=len(bin_no.unique())
        # self.log(f'{name_prefix}_bits_used',
        #          -np.log2(temp + 1e-12), prog_bar=True)
        self.log(f'{name_prefix}_rate (bits)',
                 torch.mean(-torch.log2(p_u + 1e-12)), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.custom_steps(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.custom_steps(batch, batch_idx, 'val')
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
            F.one_hot(bins.long(), num_classes=self.coding_model.code_size), side_info)
        return reconstruct


# ---------------------------------------------
class WZQuantizerANN:
    def __init__(self, metric_report_flag=False, train_sample_size=100_000):
        self.metric_report_flag = metric_report_flag

        self.train_sample_size = train_sample_size
        self.side_info_data_list = []
        self.batch_size = None

        self.bin_count = None
        self.wz_model = None
        self.load_basic_wz_model()

    def make_model_obj(self, *args, **kwargs):
        return PL_EncoderDecoder_ANN(*args, **kwargs)

    def load_basic_wz_model(self):
        self.bin_count = 4
        self.wz_model = self.make_model_obj(inp_dim=1, side_info_size=1, code_size=self.bin_count)
        # todo add loading from file
        # self.wz_model.load_from_checkpoint('wz_model')

    def symbol_encoding(self, bins):
        return arithmetic_encode(bins.tolist(), self.bin_count)

    def symbol_decoding(self, quantized_data, vect_size):
        return arithmetic_decode(quantized_data, self.bin_count, vect_size)

    # todo separate the running of the model for ann and rnn
    def encoding_process(self, grad_vector, batch_size=500_000):
        # from components.broadcast_components.quantizer.simple import simple_quantize
        # return simple_quantize(grad_vector)

        grad_tensor = torch.tensor(grad_vector).float()
        total_size = len(grad_tensor)

        self.wz_model.to('cuda')
        self.wz_model.eval()

        all_bins = []
        with torch.no_grad():
            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)
                batch = grad_tensor[i:end_idx].unsqueeze(1).to('cuda')

                bins_batch = self.wz_model.encode_net(batch).to('cpu')
                all_bins.append(bins_batch)

                batch.to('cpu')
        self.wz_model.to('cpu')
        torch.cuda.empty_cache()

        # Concatenate all batches
        if len(all_bins) <= 1:
            bins = all_bins[0]
        else:
            bins = torch.cat(all_bins, dim=1) if len(all_bins[0].shape) > 1 else torch.cat(all_bins, dim=0)
        return self.symbol_encoding(bins)

    # todo separate the running of the model for ann and rnn
    def decoding_process(self, quantized_data, side_info_data_list, batch_size=500_000):
        # from components.broadcast_components.quantizer.simple import simple_dequantize
        # return simple_dequantize(quantized_data, np.float32)

        bins = self.symbol_decoding(quantized_data, len(side_info_data_list[0]))
        bins_tensor = torch.tensor(np.array(bins))
        total_size = len(bins_tensor[0]) if len(bins_tensor.shape) == 2 else len(bins_tensor)

        self.wz_model.to('cuda')
        self.wz_model.eval()

        all_reconstructs = []

        with torch.no_grad():
            side_info_array = torch.tensor(np.array(side_info_data_list), dtype=torch.float32).T

            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)

                if len(bins_tensor.shape) != 2:
                    bins_batch = bins_tensor[i:end_idx].to('cuda')
                else:
                    bins_batch = bins_tensor[:,i:end_idx].to('cuda')

                side_info_batch = side_info_array[i:end_idx].to('cuda')

                reconstructs_batch = self.wz_model.decode_net(bins_batch, side_info_batch)
                all_reconstructs.append(reconstructs_batch.to('cpu'))

                # Clear GPU memory for this batch
                del bins_batch, side_info_batch
                torch.cuda.empty_cache()

        self.wz_model.to('cpu')

        all_reconstructs = torch.cat(all_reconstructs, dim=0)
        res = all_reconstructs.squeeze()
        res = res.numpy()
        return res

    # todo have multiple input data and train on all of them in one run (change sampler)
    def train_new_model(self, input_data, side_info_data_list, epoch=10,
                        batch_size=50_000, code_bit_size=3.5, lr=1e-3, reconst_ld=100):
        # return

        self.batch_size = batch_size
        self.bin_count = int(round(2 ** code_bit_size))

        side_info_data_list = torch.tensor(np.array(side_info_data_list), dtype=torch.float32).T
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
        train_dataset = torch.utils.data.TensorDataset(input_data, side_info_data_list)

        if self.train_sample_size > len(train_dataset):
            self.train_sample_size = len(train_dataset)

        # ------------------------------
        val_dataloader = None
        if self.metric_report_flag:
            all_indices = np.arange(len(train_dataset))

            val_indices = np.random.choice(all_indices, size=int(self.train_sample_size // 3), replace=False)
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size * 10,
                num_workers=0, pin_memory=False, persistent_workers=False)

            train_indices = np.setdiff1d(all_indices, val_indices)
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
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=True, persistent_workers=True)

        # ------------------------------
        if self.wz_model is not None:
            self.wz_model.to('cpu')
        self.wz_model = self.make_model_obj(
            inp_dim=1, side_info_size=side_info_data_list.shape[1],
            lr=lr, code_size=self.bin_count, reconst_ld=reconst_ld)

        # disable val progress bar due to pl bug
        from pytorch_lightning.callbacks import TQDMProgressBar
        from tqdm import tqdm
        class NoValidationBar(TQDMProgressBar):
            def init_validation_tqdm(self): return tqdm(disable=True)

        trainer = pl.Trainer(
            accelerator="cuda",
            max_epochs=epoch,
            callbacks=[NoValidationBar()] if self.metric_report_flag else [],
            enable_progress_bar=self.metric_report_flag,
            log_every_n_steps=1 if self.metric_report_flag else None,
            enable_model_summary=False,
        )
        trainer.fit(self.wz_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        del train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()


def plot_bins(wz_model, grad_data):
    from matplotlib import pyplot as plt

    def encoding_f(grad_vector):
        x = torch.tensor(grad_vector).unsqueeze(1).to('cuda').float()
        wz_model.to('cuda')
        wz_model.eval()
        with torch.no_grad():
            bins = wz_model.encode_net(x).to('cpu')
        wz_model.to('cpu')
        return bins

    def decoding_f(bins, side_info_data_list):
        bins = torch.tensor(bins).to('cuda')

        wz_model.to('cuda')
        wz_model.eval()
        with torch.no_grad():
            side_info_list = torch.tensor(np.array(side_info_data_list), dtype=torch.float32).T
            reconstructs = [wz_model.decode_net(bins, side_info_list.to('cuda'))]
        side_info_list.to('cpu')
        wz_model.to('cpu')

        res = torch.stack(reconstructs, dim=0).mean(dim=0).squeeze()
        res = res.to('cpu').numpy()
        return res

    def _plot_bin_regions(ax, x_range, decoded_bins, x_step, colors):
        unique_bins = np.unique(decoded_bins)

        for i, bin_val in enumerate(unique_bins):
            bin_positions = x_range[np.array(decoded_bins) == bin_val]
            if len(bin_positions) == 0:
                continue

            # Mark first bin boundary
            ax.axvline(bin_positions[0], color='black', linestyle='--', linewidth=0.5)

            # Color contiguous bin regions
            start_pos = bin_positions[0]
            for j in range(1, len(bin_positions)):
                if bin_positions[j] - bin_positions[j - 1] > x_step * 1.1:  # Gap detected
                    ax.axvspan(start_pos, bin_positions[j - 1], color=colors[i], alpha=0.3)
                    start_pos = bin_positions[j]
            # Final span
            ax.axvspan(start_pos, bin_positions[-1], color=colors[i], alpha=0.3)

    if isinstance(grad_data, torch.Tensor): grad_data = grad_data.cpu().numpy()
    grad_data = np.asarray(grad_data, dtype=np.float32)

    # Setup visualization parameters
    min_v, max_v = grad_data.min(), grad_data.max()
    x_step = (max_v - min_v) / 200
    x_range = np.arange(min_v, max_v, x_step)[:200]

    # for plot 2
    bin_count = wz_model.coding_model.code_size
    reconstructed_per_bin = np.array([
        decoding_f(np.ones(200) * b, [x_range]) for b in range(bin_count)])

    # for plot 1
    bins = encoding_f(x_range)
    recons_for_x_range = reconstructed_per_bin[bins, np.arange(len(x_range))]

    # Setup plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Data histogram with binning analysis ------------------------
    counts, bins_edges, patches = ax1.hist(
        grad_data[np.random.choice(len(grad_data), 20000, replace=False)],
        bins=200, alpha=0.3, color='gray', label='data histogram', density=False)
    ax1.clear()
    ax1.bar(bins_edges[:-1], counts / np.max(counts), width=np.diff(bins_edges),
            alpha=0.3, color='gray', label='data histogram (normalized)', align='edge')
    ax1.plot(x_range, np.abs(x_range - recons_for_x_range), label='reconstruction error')
    ax1.plot(x_range, (np.array(bins) + 1) / bin_count, label='normalized encoded_bins')
    ax1.set_xlabel('x_range')

    # Visualize bin regions
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(bins))))
    _plot_bin_regions(ax1, x_range, bins, x_step, colors)

    ax1.set_xlim(min_v, max_v)
    ax1.legend()
    ax1.set_title('Binning Visualization')

    # Plot 2: Reconstruction curves for each bin (batch process) ------------------------
    # todo merge the 2 plots and change how the sideinfo is given (for example for all 0s as side info)
    for bin_idx in range(bin_count):
        ax2.plot(x_range, reconstructed_per_bin[bin_idx], label=f'bin={bin_idx}')

    ax2.set_xlabel('side info')
    ax2.set_ylabel('reconstruction')

    ax2.set_xlim(min_v, max_v)
    ax2.legend()
    ax2.set_title('Reconstruction Curves by Bin')

    plt.tight_layout()
    plt.show()


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
    wz_quantizer = WZQuantizerANN(train_sample_size=100_000, metric_report_flag=True)
    wz_quantizer.train_new_model(y, [side_info_data], epoch=2,
                                 batch_size=10_000, code_bit_size=2, lr=1e-5, reconst_ld=100)

    # %%
    print('error ', np.mean(np.abs(y - wz_quantizer.decoding_process(wz_quantizer.encoding_process(y), [side_info_data]))))
    plot_bins(wz_quantizer.wz_model, y)
