import numpy as np
import lightning as pl
import torch
from WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN


class PL_ConditionalPrior(pl.LightningModule):
    def __init__(self, cond_m, *args, **kwargs):
        super().__init__()
        self.prior_model = cond_m

    def forward(self, side_info):
        return self.prior_model(side_info, tau=0)

    def training_step(self, batch, batch_idx):
        bins, side_info = batch
        logits = self.prior_model.layers(side_info)
        loss = torch.nn.functional.cross_entropy(logits, bins)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        acc = (logits.argmax(dim=1) == bins).float().mean()
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


def train_conditional(conditionalRNN, encoded_bins, side_info_data_list,):
    con_pl_m = PL_ConditionalPrior(conditionalRNN).to(torch.float32)
    trainer = pl.Trainer(max_epochs=20, logger=False, enable_checkpointing=False)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(encoded_bins, dtype=torch.long),
        torch.tensor(np.stack(side_info_data_list, axis=1), dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=15000, shuffle=True, num_workers=8)
    trainer.fit(con_pl_m, dataloader)
    con_pl_m.eval()
    conditionalRNN.load_state_dict(con_pl_m.prior_model.state_dict())


class LearnedNonWZQuantizer(QuantizerWithDataPrep):
    def __init__(self, *args, data_folder_path=r'data', use_dsc_sw=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_folder_path = data_folder_path
        self.use_dsc_sw = use_dsc_sw

    def train_model(self, grad_vector, side_info_data_list, *args, **kwargs):
        temp = [2**1, 2**3, 3**3, 5**3]
        index = len(temp)-1
        for i, a in enumerate(temp):
            if self.bin_count<=a:
                index = i
                break
        bins_per_plane = [2,2,3,5][index]
        num_planes = 3 if self.bin_count>2 else 1

        qz = self.wz_pl_model
        self.wz_pl_model = PL_EncoderDecoder_RNN(
            inp_dim=1,
            side_info_size=len(side_info_data_list),
            marginal=qz.marginal,
            num_planes=num_planes,
            bins_per_plane=bins_per_plane,

            lr=qz.lr,
            reconst_ld=qz.reconst_ld,
            tau=qz.tau,
            tau_rate=qz.tau_rate,
        ).to(torch.float32)

        temp = ['basicRNN_1p2b.pt', 'basicRNN_3p2b.pt',
                'basicRNN_3p3b.pt', 'basicRNN_3p5b.pt'][index]
        path_to_sd = f'{self.data_folder_path}/{temp}'
        state_dict = torch.load(path_to_sd, map_location='cpu')
        self.wz_pl_model.load_state_dict(state_dict)
        self.wz_pl_model.eval()
        print(f'--- loaded pretrained model from {path_to_sd} ---')

        if not self.use_dsc_sw:
            self.wz_pl_model.marginal = True
            return

        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)
        encoded_bins = self.encoding_process(grad_vector)
        train_conditional(
            self.wz_pl_model.coding_model.conditionalRNN,
            encoded_bins, side_info_data_list,
        )

if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
        _test_main, WZServerTrainingPerRoundProtocol

    bp_f = lambda worker_count, base_quantizer: (
        WZServerTrainingPerRoundProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=50, quantizer_class=LearnedNonWZQuantizer)
