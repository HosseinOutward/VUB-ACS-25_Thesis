import numpy as np
import lightning as pl
import torch
from components.broadcast_components.WZ_models.WZQuantizerWithDataPrep import QuantizerWithDataPrep
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN


class PL_ConditionalPrior(pl.LightningModule):
    def __init__(self, coding_model, *args, **kwargs):
        super().__init__()
        self.coding_model = coding_model

    def forward(self, side_info):
        return self.prior_model(side_info, tau=0)

    def training_step(self, batch, batch_idx):
        soft_codes, side_info = batch
        soft_codes = [a for a in soft_codes.transpose(0,1)]
        bins = [torch.argmax(sc, dim=-1) for sc in soft_codes] # (planes,
        logits = self.coding_model.get_priors(
            codes=soft_codes, y=side_info, tau=None)
        loss = sum([torch.nn.functional.cross_entropy(logits[i], bins[i])
                    for i in range(len(bins))])
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        acc = sum([(torch.argmax(logits[i], dim=-1)==bins[i]).float().mean()
                   for i in range(len(bins))])/len(bins)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_conditional(coding_model, soft_c, side_info_data_list,):
    con_pl_m = PL_ConditionalPrior(coding_model).to(torch.float32)
    trainer = pl.Trainer(max_epochs=20, logger=False, enable_checkpointing=False)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(soft_c, dtype=torch.float32).transpose(0,1),
        torch.tensor(np.stack(side_info_data_list, axis=1), dtype=torch.float32)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=15000, num_workers=8, persistent_workers=True,
                sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=300_000),)
    trainer.fit(con_pl_m, dataloader)


class LearnedNonWZQuantizer(QuantizerWithDataPrep):
    def __init__(self, *args,
                 data_folder_path=r'D:\User\App Files\Projects\VUB-ACS-25_Thesis\data',
                 use_dsc_sw=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_folder_path = data_folder_path
        self.use_dsc_sw = use_dsc_sw

    def train_model(self, grad_vector, side_info_data_list, *args, **kwargs):
        temp = [2**1, 2**3, 3**3, 5**3, 16**2]
        index = len(temp)-1
        for i, a in enumerate(temp):
            if self.bin_count<=a:
                index = i
                break
        bins_per_plane = [2,2,3,5,16][index]
        num_planes = 3 if self.bin_count>2 else 1
        if index==4: num_planes = 2

        qz = self.wz_pl_model
        self.wz_pl_model = qz.__class__(
            inp_dim=1,
            side_info_size=0,
            marginal=True,
            num_planes=num_planes,
            bins_per_plane=bins_per_plane,

            lr=qz.lr,
            reconst_ld=qz.reconst_ld,
            tau=qz.tau,
            tau_rate=qz.tau_rate,
        ).to(torch.float32)

        temp = ['basicRNN_1p2b.pt', 'basicRNN_3p2b.pt',
                'basicRNN_3p3b.pt', 'basicRNN_3p5b.pt', 'basicRNN_2plane_4bins_state.pt'][index]
        path_to_sd = f'{self.data_folder_path}/{temp}'
        state_dict = torch.load(path_to_sd, map_location='cpu')
        self.wz_pl_model.load_state_dict(state_dict)
        self.wz_pl_model.eval()
        print(f'--- loaded pretrained model from {path_to_sd} ---')

        if not self.use_dsc_sw:
            return

        temp = qz.coding_model.conditionalRNN
        cond_m = temp.__class__(
            bins_per_plane + len(side_info_data_list),
            temp.hidden_dim, temp.layers, output_activation=False)
        self.wz_pl_model.coding_model.marginal = False
        self.wz_pl_model.coding_model.conditionalRNN = cond_m
        _, soft_code = self.get_prior_and_softcodes(grad_vector, side_info_data_list)
        train_conditional(
            self.wz_pl_model.coding_model, soft_code, side_info_data_list,)

        _ = self.get_set_training_posterior_cdf(grad_vector, side_info_data_list)
        self.count_side_info_data = 0

    def decoding_process(self, quantized_data, side_info_data_list, encoding_extra_data=None,):
        side_info_data_list = []
        return super().decoding_process(quantized_data, side_info_data_list, encoding_extra_data)


if __name__ == "__main__":
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import \
        _test_main, WZServerTrainingPerRoundProtocol
    from components.broadcast_components.broadcasting_process.CancerProt import \
        CancerProtocol

    bp_f = lambda worker_count, base_quantizer: (
        CancerProtocol(worker_count, base_quantizer, epoch_count=1))
    _test_main(bp_f, worker_count=2, rounds=50,
               no_global_quant=True, quantizer_class=LearnedNonWZQuantizer)
