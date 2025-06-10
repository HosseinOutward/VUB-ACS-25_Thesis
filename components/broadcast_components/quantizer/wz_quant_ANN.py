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
        # from quantizer.simple import simple_quantize
        # return simple_quantize(grad_vector)

        x = torch.tensor(grad_vector).unsqueeze(1).to('cuda').float()
        self.wz_model.to('cuda')
        self.wz_model.eval()
        with torch.no_grad():
            bins = self.wz_model.encode(x).to('cpu')
        self.wz_model.to('cpu')
        return arithmetic_encode(bins.tolist(), self.bin_count)

    def decode(self, quantized_data, previous_data):
        # from quantizer.simple import simple_dequantize
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

    def plot_bins(self, grad_data=None):
        from matplotlib import pyplot as plt
        import torch.nn.functional as F

        coding_model.to('cuda').eval()

        min_v, max_v = rtrain_x.min().numpy(), rtrain_x.max().numpy()
        # min_v, max_v = np.percentile(rtrain_x.cpu().numpy(), [1, 99])

        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(20, 5)

        x_step = ((max_v - min_v) / 200)
        x = torch.tensor(np.arange(min_v, max_v, x_step))
        x = x.to(torch.float32).to('cuda').unsqueeze(1)
        y = torch.zeros_like(x).to('cuda')
        with torch.no_grad():
            reconstruct, soft_code, prior = coding_model.forward(x, y)
            bin = soft_code.argmax(dim=1)

        bin = bin.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        ax[0].hist(rtrain_x.numpy()[:len(rtrain_x) // 5],
                   bins=100, alpha=0.3, color='gray', label='data histogram', density=True)
        ax[0].plot(x, abs(x - reconstruct.cpu().numpy()))
        ax[0].plot(x, bin / 2 ** code_size * 3)

        # Seeing the binning with colors
        unique_v = np.unique(bin)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_v)))
        for i, val in enumerate(unique_v):
            mask = (bin == val)
            ax[0].axvline(x[mask][0], color='black', linestyle='--', linewidth=0.5)
            temp = [x[mask][0, 0], x[mask][0, 0]]
            for j in range(1, len(x[mask])):
                # if the next point of same color is far enough
                if temp[1] + x_step * 1.1 < x[mask][j][0]:
                    ax[0].axvspan(*temp, color=colors[i], alpha=0.3)
                    temp[0] = x[mask][j][0]
                temp[1] = x[mask][j][0]
            # Ensure the last span is added
            ax[0].axvspan(*temp, color=colors[i], alpha=0.3)
        ax[0].set_xlim(min_v, max_v)

        # __________________________________________________________
        # Seeing the reconstruction points
        with torch.no_grad():
            for i in range(2 ** code_size):
                y = torch.tensor(np.arange(min_v, max_v, x_step))
                y = y.to(torch.float32).to('cuda').unsqueeze(1)
                codes = F.one_hot(
                    torch.ones([y.shape[0]], dtype=torch.long, device='cuda') * i,
                    num_classes=2 ** code_size).float()
                reconstruct = coding_model.coding_model.decoder(codes, y)

                ax[1].plot(y.detach().cpu().numpy(),
                           reconstruct.detach().cpu().numpy(),
                           label='bin={}'.format(i))
        ax[1].set_xlim(min_v, max_v)

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


