from pytorch_lightning import Trainer

from components.broadcast_components.quantizer.brent_wz_models import EncoderDecoder
from components.broadcast_components.compressor.arithmatic_coding import arithmetic_encode, arithmetic_decode
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class PL_EncoderDecoder_ANN(pl.LightningModule):
    def __init__(self, lr=1e-3, code_size=4, reconst_ld=100):
        super(PL_EncoderDecoder_ANN, self).__init__()
        self.coding_model = EncoderDecoder(
            input_dim=1, layers=3, hidden_dim=100, code_size=code_size, marginal=True)
        self.reconst_ld = reconst_ld
        self.tau = 1
        self.lr = lr
        self.lr_step = 40

    def forward(self, x, y, tau=None):
        return self.coding_model(x, y, tau=tau)

    def custom_steps(self, batch, batch_idx, name_prefix):
        tau_t = self.tau * np.exp(
            self.current_epoch / (self.trainer.max_epochs + 1) * np.log(0.1 / self.tau))

        single_grad_param, side_info = batch
        reconstruct, bins_probs, prior_probs = self.forward(
            single_grad_param, side_info, tau=tau_t)

        bin_no = torch.argmax(bins_probs, dim=-1)
        temp = torch.arange(bins_probs.size(0))
        p_ux = bins_probs[temp, bin_no]
        p_u = prior_probs[temp, bin_no] + 1e-12

        loss = torch.log(p_ux / p_u).mean() + \
               self.reconst_ld * F.mse_loss(reconstruct, single_grad_param)

        # clamp loss to avoid NaN
        loss = torch.clamp(loss, min=-5, max=5)

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
                 torch.mean(-torch.log2(p_u + 1e-12)).item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.custom_steps(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.custom_steps(batch, batch_idx, 'val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=0.3)
        return [optimizer], [scheduler]

    def encode(self, grad_vector):
        x = self.coding_model.encoder(grad_vector)
        x = F.softmax(x, dim=-1)
        x = torch.argmax(x, dim=-1)
        return x

    def decode(self, bins, side_info):
        reconstruct = self.coding_model.decoder(
            F.one_hot(bins.long(), num_classes=self.coding_model.code_size), side_info)
        return reconstruct


# ---------------------------------------------
class WZQuantizer:
    def __init__(self, batch_size=1_000,
                 code_bit_size=5.5, metric_report_flag=False):
        self.train_sample_size = 100_000
        self.metric_report_flag = metric_report_flag
        self.batch_size = batch_size

        self.bin_count = int(round(2 ** code_bit_size))
        self.wz_model = PL_EncoderDecoder_ANN(
            lr=1e-4, code_size=self.bin_count, reconst_ld=100)

        self.side_info_datasets = []

    def encode(self, grad_vector):
        # from components.broadcast_components.quantizer.simple import simple_quantize
        # return simple_quantize(grad_vector)

        x = torch.tensor(grad_vector).unsqueeze(1).to('cuda').float()
        self.wz_model.to('cuda')
        self.wz_model.eval()
        with torch.no_grad():
            bins = self.wz_model.encode(x).to('cpu')
        self.wz_model.to('cpu')
        return arithmetic_encode(bins.tolist(), self.bin_count)

    def decode(self, quantized_data, previous_data):
        # from components.broadcast_components.quantizer.simple import simple_dequantize
        # return simple_dequantize(quantized_data, np.float32)

        reconstructs=[]
        bins = arithmetic_decode(quantized_data,
                                 self.bin_count, len(self.side_info_datasets[0]))
        bins=torch.tensor(bins).to('cuda')

        side_info_list = [*self.side_info_datasets]
        for s_i in self.side_info_datasets:
            for i in range(2):
                x = torch.rand(s_i.shape).to(s_i.device) * 0.02
                x = torch.clip(s_i + x, -1, 1)
                side_info_list.append(x)

        self.wz_model.to('cuda')
        self.wz_model.eval()
        with torch.no_grad():
            for s_i in side_info_list:
                reconstructs += [self.wz_model.decode(bins, s_i.to('cuda'))]
                s_i.to('cpu')
        self.wz_model.to('cpu')

        res = torch.stack(reconstructs, dim=0).mean(dim=0).squeeze()
        res = res.to('cpu').numpy()
        return res

    # todo figure out why we have memory leak with wz
    def train_model(self, side_info_data_1, side_info_data_2):
        # return

        # todo utilize later decoded data to train the model further
        self.side_info_datasets=[]
        self.side_info_datasets.append(torch.tensor(
            side_info_data_1, dtype=torch.float32).unsqueeze(1))
        self.side_info_datasets.append(torch.tensor(
            side_info_data_2, dtype=torch.float32).unsqueeze(1))

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=10,
            enable_progress_bar=self.metric_report_flag,
            log_every_n_steps=1 if self.metric_report_flag else None,
            enable_model_summary=False,
        )

        a = self.side_info_datasets[0]
        b = self.side_info_datasets[1]
        combined_dataset = [torch.concat([a, b]), torch.concat([b, a])]
        train_dataset = torch.utils.data.TensorDataset(*combined_dataset)


        if self.train_sample_size > len(train_dataset):
            self.train_sample_size = len(train_dataset)

        val_dataloader = None
        if self.metric_report_flag:
            all_indices = np.arange(len(train_dataset))
            val_indices = np.random.choice(
                all_indices, size=int(self.train_sample_size // 2), replace=False)
            train_indices = np.setdiff1d(all_indices, val_indices)

            temp=train_dataset
            val_dataset = torch.utils.data.Subset(temp, val_indices)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size)

            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=self.train_sample_size),)

        trainer.fit(self.wz_model,
                    train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        del train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def plot_bins(self, grad_data):
        from matplotlib import pyplot as plt

        def _setup_zero_side_info(length):
            """Helper to temporarily set zero side info for visualization"""
            original_side_info = self.side_info_data_list.copy()
            self.side_info_data_list = [np.zeros(length)]
            return original_side_info

        def _restore_side_info(original_side_info):
            """Helper to restore original side info"""
            self.side_info_data_list = original_side_info

        def _plot_bin_regions(ax, x_range, decoded_bins, x_step, colors):
            """Helper to visualize bin regions with colors"""
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

        # Ensure grad_data is numpy array for consistency
        if isinstance(grad_data, torch.Tensor):
            grad_data = grad_data.cpu().numpy()
        grad_data = np.asarray(grad_data, dtype=np.float32)

        # Setup visualization parameters
        min_v, max_v = grad_data.min(), grad_data.max()
        x_step = (max_v - min_v) / 200
        x_range = np.arange(min_v, max_v, x_step)

        # Get encoding/decoding results
        bins = self.encode(x_range)
        decoded_bins = arithmetic_decode(bins, self.bin_count, len(x_range))

        # Get reconstruction with zero side info
        original_side_info = _setup_zero_side_info(len(x_range))
        reconstructed = self.decode(bins)
        _restore_side_info(original_side_info)

        # Setup plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

        # Plot 1: Data histogram with binning analysis
        sample_size = len(grad_data) // 5
        ax1.hist(grad_data[:sample_size], bins=100, alpha=0.3, color='gray',
                 label='data histogram', density=True)
        ax1.plot(x_range, np.abs(x_range - reconstructed), label='reconstruction error')
        ax1.plot(x_range, np.array(decoded_bins) / self.bin_count * 3, label='normalized bins')

        # Visualize bin regions
        colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(decoded_bins))))
        _plot_bin_regions(ax1, x_range, decoded_bins, x_step, colors)

        ax1.set_xlim(min_v, max_v)
        ax1.legend()
        ax1.set_title('Binning Visualization')

        # Plot 2: Reconstruction curves for each bin (batch process)
        original_side_info = _setup_zero_side_info(len(x_range))

        for bin_idx in range(self.bin_count):
            dummy_bins = arithmetic_encode([bin_idx] * len(x_range), self.bin_count)
            bin_reconstruction = self.decode(dummy_bins)
            ax2.plot(x_range, bin_reconstruction, label=f'bin={bin_idx}')

        _restore_side_info(original_side_info)

        ax2.set_xlim(min_v, max_v)
        ax2.legend()
        ax2.set_title('Reconstruction Curves by Bin')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    wz_quantizer = WZQuantizer(batch_size=64, code_bit_size=5.5, metric_report_flag=True)

    # Assuming side_info_data_1 and side_info_data_2 are available
    side_info_data_1 = np.random.rand(2000)*2-1
    side_info_data_2 = side_info_data_1 + np.random.rand(2000)*0.1

    wz_quantizer.train_model(side_info_data_1, side_info_data_2)

    # Example encoding
    grad_vector = side_info_data_1 + np.random.rand(2000)*0.1
    encoded_bins = wz_quantizer.encode(grad_vector)
    # print("Encoded Bins:", encoded_bins)

    # Example decoding
    decoded_data = wz_quantizer.decode(encoded_bins, [])
    # print("Decoded Data:", decoded_data)

    print('error ',np.mean(np.abs(grad_vector-decoded_data)))

    # Test the plot_bins method
    print("Testing plot_bins method...")
    wz_quantizer.plot_bins(grad_vector)
