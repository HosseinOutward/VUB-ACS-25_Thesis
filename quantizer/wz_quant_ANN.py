# todo: move the file
from quantizer.brent_wz_models import EncoderDecoder

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class PL_EncoderDecoder_ANN(pl.LightningModule):
    def __init__(self, lr=1e-3, bit_count=2):
        super(PL_EncoderDecoder_ANN, self).__init__()
        self.coding_model = EncoderDecoder(input_dim=1, layers=3, hidden_dim=100, bit_count=bit_count, marginal=True)
        self.reconst_ld = 100
        self.tau = 1
        self.lr = lr
        self.lr_step = 40

    def forward(self, x, y, tau=None):
        return self.coding_model(x, y, tau=tau)

    def custom_steps(self, batch, batch_idx, name_prefix):
        tau_t = self.tau * np.exp(self.current_epoch / (self.trainer.max_epochs+1) * np.log(0.1 / self.tau))

        single_grad_param, side_info = batch
        reconstruct, bins_probs, prior_probs = self.forward(single_grad_param, side_info, tau=tau_t)

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

    def validation_step(self, batch, batch_idx):
        loss = self.custom_steps(batch, batch_idx, 'val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=0.3)
        return [optimizer], [scheduler]


class WZ_ANN:
    def __init__(self):
        self.coding_model = PL_EncoderDecoder_ANN()

    def encoder(self, x):
        """
        Encoder function that takes an input tensor and returns a quantized tensor.
        """
        # Placeholder for actual encoding logic
        return x

    def decoder(self, x):
        """
        Decoder function that takes a quantized tensor and returns the original tensor.
        """
        # Placeholder for actual decoding logic
        return x

    def train(self, worker_models_list, train_loaders_list, epochs):
        """
        Train the model using the provided worker models and data loaders.
        """
        # Placeholder for actual training logic
        pass
