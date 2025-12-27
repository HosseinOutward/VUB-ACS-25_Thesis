from typing import List, TYPE_CHECKING
import numpy as np
import torch
import torch.nn.functional as F

from FL_reworked.run_fl import FLConfig
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from FL_reworked.cancer_protocol import CancerConfig


class WZQuantizerCancer:
    def __init__(self, c_cfg: 'CancerConfig', fl_cfg: FLConfig, num_planes: int, bins_per_plane: int,
                 train_x_vec: torch.Tensor, side_info_list: List[torch.Tensor], pretrained: bool = False):
        self.si_vec_size = None
        self.c_cfg = c_cfg
        self.fl_cfg = fl_cfg
        self.coding_model = EncoderDecoderLayeredRNN(
            num_planes=num_planes, bins_per_plane=bins_per_plane,
            side_info_size=max(1, len(side_info_list)), input_dim=1,
            layers=3, hidden_dim=100, marginal=pretrained)

        self.side_info_list_used = None

        if pretrained:
            weight_path = c_cfg.pretrain_pth_dir + f'bpp{bins_per_plane}_np{num_planes}_pretrained_wzq_rnn.pth'
            self.coding_model.load_state_dict(torch.load(weight_path), strict=False)
            self.side_info_list_used = []
            assert train_x_vec is None and side_info_list == [], "Pretrained model should not have training data."
            return

        # train the model
        self.mspe_denom = torch.mean(train_x_vec ** 2).item()
        noise = torch.from_numpy(np.random.normal(0, np.sqrt(1e-8), len(train_x_vec)).astype(np.float32))
        self.train_model(train_x_vec + noise, side_info_list)

    @property
    def num_planes(self):
        return self.coding_model.num_planes

    @property
    def bins_per_plane(self):
        return self.coding_model.bins_per_plane

    @property
    def bin_count(self):
        return self.coding_model.bin_count

    def compute_loss(self, x_vec, side_info, current_epoch):
        training_prog = current_epoch / (self.c_cfg.train_epochs + 1)
        tau_t = self.c_cfg.tau * np.exp(training_prog * np.log(0.1 / self.c_cfg.tau))

        reconstruct, bins_no, soft_codes, prior_probs = \
            self.coding_model.forward(x_vec, side_info, tau=tau_t)

        loss = 0.0
        pu_vec = [None for _ in range(self.num_planes)]
        for i in range(self.num_planes):
            # reconstruction component of the loss
            dist = F.mse_loss(reconstruct[i], x_vec)
            dist = dist / self.mspe_denom
            loss = loss + self.c_cfg.reconst_ld * dist

            # rate component of the loss
            p_ux = soft_codes[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            p_u = prior_probs[i][torch.arange(soft_codes[i].size(0)), bins_no[i]]
            pu_vec[i] = p_u.detach()
            rate_loss = torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

            rate_weight = lambda x: ((x - 1) + np.exp(x * np.log(abs(self.c_cfg.tau_rate)))) / abs(
                self.c_cfg.tau_rate) * 1.25
            rate_weight = rate_weight(training_prog) if self.c_cfg.tau_rate <= 0 else 1 - rate_weight(1 - training_prog)

            loss = loss + rate_loss * max(rate_weight, 0.2)
        loss = loss / self.num_planes

        # f = lambda x: [a.detach() for a in x]
        # loss = loss
        # inp_rec = reconstruct[-1].detach()
        # inp = single_grad_param.detach()
        # bin_no_vec = f(bins_no)
        # p_u = pu_vec
        # bins_probs = f(soft_codes)
        # prior_probs = f(prior_probs)
        # mape = torch.mean((inp - inp_rec)**2) / (self.mspe_denom + 1e-8) * 100
        # mse = F.mse_loss(inp_rec, inp)
        # rate_bits = np.sum([torch.mean(-torch.log2(a + 1e-12)).cpu().numpy() for a in p_u])
        # practical_p_u, _ = get_real_bin_prob(bin_no_vec, self.bin_count)
        # real_bit_r = torch.mean(-torch.log2(practical_p_u + 1e-12))

        return loss

    def get_x_dataset(self, x_vec, ):
        x_vec = x_vec.cuda(non_blocking=True).unsqueeze(1).to(torch.float32).contiguous()
        if self.si_vec_size is None:
            self.si_vec_size = x_vec.shape[0]
        return x_vec

    def get_si_dataset(self, ):
        if self.side_info_list_used == []:
            self.side_info_list_used = [torch.zeros(self.si_vec_size)]
        side_info_list = torch.stack(self.side_info_list_used).cuda(non_blocking=True).T.to(torch.float32).contiguous()
        return side_info_list

    def train_model(self, x_vec, side_info_list, batch_size=50_000):
        self.side_info_list_used = side_info_list
        assert len(side_info_list) != 0, "Side information is required for training."

        # Enable TF32 for faster matmul on Ampere+ GPUs
        if self.fl_cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Use fused AdamW for better performance
        optimizer = torch.optim.AdamW(self.coding_model.parameters(), fused=self.fl_cfg.fused_optimizer,
                                      lr=self.c_cfg.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.c_cfg.train_epochs*np.ceil(self.c_cfg.lr_step/180)), gamma=0.3)

        x_vec = self.get_x_dataset(x_vec)
        side_info_list = self.get_si_dataset()
        train_dataset = torch.utils.data.TensorDataset(x_vec, side_info_list)

        self.coding_model.cuda()

        # Compile model for JIT optimization (PyTorch 2.0+)
        if self.fl_cfg.compile_mode and hasattr(torch, 'compile'):
            compiled_model = torch.compile(self.coding_model, mode=self.fl_cfg.compile_mode)
        else:
            compiled_model = self.coding_model

        compiled_model.train()

        # Mixed precision training with GradScaler
        use_amp = self.fl_cfg.mixed_precision and torch.cuda.is_available()
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        for epoch in range(self.c_cfg.train_epochs):
            indices = torch.randint(0, len(train_dataset), (self.c_cfg.train_sample_size,), dtype=torch.long)
            subset_dataset = torch.utils.data.Subset(train_dataset, indices)
            for start_i in range(0, len(subset_dataset), batch_size):
                end_i = min(start_i + batch_size, len(subset_dataset))
                x_batch, si_batch = subset_dataset[start_i:end_i]

                optimizer.zero_grad()

                if use_amp:
                    # Automatic mixed precision
                    with torch.amp.autocast('cuda'):
                        loss = self.compute_loss(x_batch, si_batch, epoch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self.compute_loss(x_batch, si_batch, epoch)
                    loss.backward()
                    optimizer.step()

            # Step scheduler after all batches in epoch (after optimizer.step() calls)
            scheduler.step()

        # Move back to CPU and cleanup
        self.coding_model.cpu()
        torch.cuda.empty_cache()

    def _batch_loop(self, func, batch_size):
        self.coding_model.to('cuda', non_blocking=True)
        self.coding_model.eval()

        with torch.inference_mode():
            # Pre-allocate list with estimated capacity
            num_batches = (self.si_vec_size + batch_size - 1) // batch_size
            all_res = [None] * num_batches

            batch_idx = 0
            for start_i in range(0, self.si_vec_size, batch_size):
                end_idx = min(start_i + batch_size, self.si_vec_size)
                res = func(start_i, end_idx)
                all_res[batch_idx] = res
                batch_idx += 1
        concat_res = torch.cat(all_res, dim=1) if all_res[0].shape[1] > 1 else torch.cat(all_res, dim=0)
        self.coding_model.to('cpu')
        torch.cuda.empty_cache()
        return concat_res

    def encoding_process(self, grad_vector, batch_size=500_000):
        # return torch.round(grad_vector*1000).to(torch.int16)

        grad_vector = self.get_x_dataset(grad_vector)

        assert grad_vector.shape[0] == self.si_vec_size

        def func(start_i, end_idx):
            x_batch = grad_vector[start_i:end_idx]
            bins_list, _ = self.coding_model.encode(x_batch)
            bins_list = torch.stack(bins_list)
            assert torch.unique(bins_list).size(0) <= self.coding_model.bins_per_plane ** self.coding_model.num_planes
            return bins_list.to('cpu', non_blocking=True)
        bins = self._batch_loop(func, batch_size)

        dtype = torch.uint8 if self.bins_per_plane < 2**8 else torch.uint16

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size

        return bins.to(dtype)

    def decoding_process(self, bins, batch_size=500_000):
        # return torch.from_numpy(quantized_data).float()/1000.0

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size
        b_p_p = self.coding_model.bins_per_plane
        assert bins.float().max() < b_p_p

        side_info = self.get_si_dataset()
        def func(start_i, end_idx):
            bins_batch = bins[:, start_i:end_idx].to('cuda', non_blocking=True)
            side_info_batch = side_info[start_i:end_idx].to('cuda', non_blocking=True)

            codes = [F.one_hot(b.to(int), num_classes=b_p_p) for b in bins_batch]
            reconstructs_batch = self.coding_model.decode(codes, side_info_batch)[-1]

            return reconstructs_batch.to('cpu', non_blocking=True)
        all_reconstructs = self._batch_loop(func, batch_size)

        res = all_reconstructs.squeeze()

        assert res.shape[0] == self.si_vec_size

        return res

    # def get_prior_and_softcodes_net(self, grad_vector, side_info=None):
    #     assert not self.coding_model.training
    #     assert self.coding_model.marginal == (side_info is None or len(side_info)==0)
    #
    #     bins_list, soft_codes = self.coding_model.encode(x=grad_vector, tau=None, force_softmax=True)
    #     priors = self.coding_model.get_priors(codes=soft_codes, y=side_info, tau=None)
    #
    #     return torch.stack(priors), torch.stack(soft_codes)
    #
    # def get_set_training_posterior_cdf(self, grad_vector=None, side_info_data_list=None):
    #     if grad_vector is not None or side_info_data_list is not None:
    #         assert grad_vector is not None and side_info_data_list is not None
    #
    #         self.training_si = side_info_data_list
    #
    #         if len(side_info_data_list) == 0:
    #             bins = self.encoding_process(grad_vector)[0] # (num_planes, N)
    #             probs_per_plane = []
    #             for b_vec in bins:
    #                 counts = np.bincount(b_vec, minlength=self.bins_per_plane)
    #                 probs = counts / counts.sum()
    #                 probs_per_plane.append(probs)
    #             probs_per_plane = np.array(probs_per_plane) # (num_planes, bin_count)
    #
    #             self.training_posterior_cdf = torch.stack(
    #                 [torch.tensor([a]*len(grad_vector), dtype=torch.float32)
    #                  for a in probs_per_plane]).numpy()
    #         else:
    #             assert self.training_posterior_cdf is None
    #             self.training_posterior_cdf = self.get_prior_and_softcodes(grad_vector, side_info_data_list)[0].numpy()
    #
    #     assert self.training_posterior_cdf is not None
    #     return self.training_posterior_cdf
    #
    # def get_prior_and_softcodes(self, grad_vector, side_info_data_list, batch_size=500_000):
    #     if type(grad_vector) != torch.Tensor:
    #         grad_tensor = torch.tensor(grad_vector, dtype=torch.float32)
    #     else:
    #         grad_tensor = grad_vector.to(torch.float32)
    #
    #     side_info_array = torch.tensor(np.array(side_info_data_list), dtype=torch.float32)
    #     if self.count_side_info_data != 0:
    #         side_info_array = side_info_array.T
    #
    #     total_size = len(grad_tensor)
    #
    #     def func(start_i, end_idx):
    #         grad_batch = grad_tensor[start_i:end_idx].unsqueeze(1).to('cuda', non_blocking=True)
    #         side_info_batch = side_info_array[start_i:end_idx].to('cuda', non_blocking=True)
    #         prior_batch, soft_code_batch = self.get_prior_and_softcodes_net(grad_batch, side_info_batch)
    #         return (prior_batch.to('cpu', non_blocking=True), soft_code_batch.to('cpu', non_blocking=True))
    #     all_priors = self._batch_loop(func, batch_size, total_size)
    #
    #     prior, soft_codes = zip(*all_priors)
    #     prior, soft_codes = [torch.cat(a, dim=1) for a in [prior, soft_codes]]
    #
    #     bins_vector = [torch.argmax(sc, dim=-1) for sc in soft_codes]
    #     for i in range(prior.shape[0]):
    #         prior[i, np.arange(prior.shape[1]), bins_vector[i]] += 1e-6
    #         prior[i] /= prior[i].sum(dim=-1, keepdim=True)
    #
    #     return prior, soft_codes


if __name__ == "__main__":
    from FL_reworked.cancer_protocol import CancerConfig

    # Create synthetic data: base signal + noise
    base_signal = torch.from_numpy(np.random.normal(0, 1, 10_000_000).astype(np.float32))
    y = base_signal + torch.from_numpy(np.random.normal(0, 0.1, 10_000_000).astype(np.float32))
    side_info = [base_signal.clone()]

    # Test with side info (from run_sim.py: num_planes=2, bins_per_plane=16)
    print("Training quantizer (num_planes=2, bins_per_plane=16)...")
    quantizer = WZQuantizerCancer(c_cfg=CancerConfig(),
        fl_cfg=FLConfig(num_clients=1), num_planes=3, bins_per_plane=16,
        train_x_vec=y, side_info_list=side_info, pretrained=False
    )

    # Encode and decode
    bins = quantizer.encoding_process(y)
    recons = quantizer.decoding_process(bins)

    # Calculate metrics
    mse = torch.mean((y - recons) ** 2).item()
    mape = torch.mean(torch.abs(y - recons) / (torch.abs(y) + 1e-8)).item() * 100

    print(f"MSE: {mse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bins shape: {bins.shape}")
    print(f"Unique bins used per plane: {[torch.unique(bins[i]).numel() for i in range(bins.shape[0])]}")
