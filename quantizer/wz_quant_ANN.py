from pytorch_lightning import Trainer

from quantizer.brent_wz_models import EncoderDecoder
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

    # def validation_step(self, batch, batch_idx):
    #     loss = self.custom_steps(batch, batch_idx, 'val')
    #     return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=0.3)
        return [optimizer], [scheduler]


class WZQuantizer:
    def __init__(self, batch_size, dataset_length=250_000, code_bit_size=5.5):
        self.batch_size = batch_size
        self.dataset_length = int(dataset_length // 2)

        bin_count = int(round(2 ** code_bit_size))
        self.wz_model = PL_EncoderDecoder_ANN(
            lr=1e-3, code_size=bin_count, reconst_ld=100)

        self.x_info_dataset = None
        self.sampled_portion = None

    def encode(self, grad_vector):
        return simple_quantize(grad_vector)  # Replace with actual WZ encoding logic

    def decode(self, quantized_data, previous_data):
        return simple_dequantize(quantized_data, np.float32)  # Replace with actual WZ decoding logic

    def initialize_x_info_dataset(self, data):
        self.sampled_portion = np.random.choice(
            list(range(len(data))), self.dataset_length, replace=False)

        self.x_info_dataset = torch.tensor(
            data[self.sampled_portion], dtype=torch.float32).unsqueeze(1)

    def train_model(self, side_info_data):
        side_info_dataset = torch.tensor(
            side_info_data[self.sampled_portion], dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(self.x_info_dataset, side_info_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        trainer = Trainer(
            accelerator="gpu",
            max_epochs=50,
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        trainer.fit(self.wz_model, train_dataloaders=train_dataloader)

        # clear out self.x_info_dataset
        del self.x_info_dataset
        self.x_info_dataset = None
