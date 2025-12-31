from typing import List, Optional, Callable, TYPE_CHECKING, Dict
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from FL_reworked.prior_calculator import PriorCalculator
from FL_reworked.run_fl import FLConfig
from FL_reworked.utils import create_training_progress_bar
from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from FL_reworked.cancer_protocol import CancerConfig


class WZQuantizerCancer:
    def __init__(self, c_cfg: 'CancerConfig', fl_cfg: FLConfig, num_planes: int, bins_per_plane: int, si_size: int, ) -> None:
        self.si_vec_size: Optional[int] = None
        self.c_cfg: 'CancerConfig' = c_cfg
        self.fl_cfg: FLConfig = fl_cfg

        self.coding_model: EncoderDecoderLayeredRNN = EncoderDecoderLayeredRNN(
            num_planes=num_planes, bins_per_plane=bins_per_plane,
            side_info_size=max(1, si_size), input_dim=1,
            layers=3, hidden_dim=100, marginal=(si_size==0))

        self.side_info_list_used: List[torch.Tensor] | str | None
        if si_size==0:
            self.side_info_list_used = 'P' # default assume that its pretrained marginal model
        else:
            self.side_info_list_used = None

        self.cached_priors_dict: Dict[str, torch.Tensor] = {}
        self.mspe_denom: float | None = None

    @property
    def num_planes(self) -> int:
        return self.coding_model.num_planes

    @property
    def bins_per_plane(self) -> int:
        return self.coding_model.bins_per_plane

    @property
    def bin_count(self) -> int:
        return self.coding_model.bin_count

    def compute_loss(self, x_vec: torch.Tensor, side_info: torch.Tensor, current_epoch: int) -> torch.Tensor:
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
            temp = torch.arange(soft_codes[i].size(0))
            p_ux = soft_codes[i][temp, bins_no[i]]
            p_u = prior_probs[i][temp, bins_no[i]]
            pu_vec[i] = p_u.detach()
            rate_loss = torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

            rate_weight = lambda x: (((x - 1) + np.exp(x * np.log(abs(self.c_cfg.tau_rate))))
                                     / abs(self.c_cfg.tau_rate) * 1.25)
            rate_weight = rate_weight(training_prog) if self.c_cfg.tau_rate <= 0 else 1 - rate_weight(1 - training_prog)

            loss = loss + rate_loss * max(rate_weight, 0.2)
        loss = loss / self.num_planes

        return loss

    def get_x_data(self, x_vec: torch.Tensor) -> torch.Tensor:
        x_vec = x_vec.cuda().unsqueeze(1).to(torch.float32).contiguous()
        if self.si_vec_size is None:
            self.si_vec_size = x_vec.shape[0]
        return x_vec

    def get_si_data(self) -> torch.Tensor:
        if self.side_info_list_used in [[], 'P']:
            self.side_info_list_used = [torch.zeros(self.si_vec_size)]
        side_info_list = torch.stack(self.side_info_list_used).cuda().T.to(torch.float32).contiguous()
        return side_info_list

    def train_model(self, x_vec: torch.Tensor, side_info_list: Optional[List[torch.Tensor]],
                    batch_size: int = 50_000) -> None:
        if self.coding_model.marginal:
            assert side_info_list is None, "Marginal model expects an empty side_info_list for training."
            side_info_list = []
        else:
            assert len(side_info_list) > 0, "Conditional model requires side info for training."
        assert self.side_info_list_used in [None, 'P'], "This quantizer instance has already been trained."
        assert x_vec is not None, "Training data x_vec must be provided for training."

        self.side_info_list_used = side_info_list

        self.mspe_denom:float = torch.mean(x_vec ** 2).item() + 1e-8
        x_vec = self.get_x_data(x_vec)
        side_info_list = self.get_si_data()
        self._train_model(x_vec, side_info_list, batch_size)

    def _train_model(self, x_vec: torch.Tensor, side_info_list: torch.Tensor, batch_size: int = 50_000) -> None:
        # Enable TF32 for faster matmul on Ampere+ GPUs
        if self.fl_cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Use fused AdamW for better performance
        optimizer = torch.optim.AdamW(self.coding_model.parameters(), fused=self.fl_cfg.fused_optimizer,
                                      lr=self.c_cfg.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.c_cfg.train_epochs*np.ceil(self.c_cfg.lr_step/180)), gamma=0.3)

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

        # Single progress bar for all training
        total_samples = min(self.c_cfg.train_sample_size, len(train_dataset))
        total_iterations = self.c_cfg.train_epochs * ((total_samples + batch_size - 1) // batch_size)
        pbar = create_training_progress_bar(
            total_iterations,
            desc="Training Quantizer",
            disable=not self.fl_cfg.training_progress_bar
        )

        for epoch in range(self.c_cfg.train_epochs):
            indices = torch.randint(0, len(train_dataset), (total_samples,), dtype=torch.long)
            subset_dataset = torch.utils.data.Subset(train_dataset, indices)

            for start_i in range(0, len(subset_dataset), batch_size):
                end_i = min(start_i + batch_size, len(subset_dataset))
                x_batch, si_batch = subset_dataset[start_i:end_i]

                noise = torch.randn_like(x_batch, device='cuda') * (1e-5 * x_batch.abs().mean())
                x_batch += noise

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = self.compute_loss(x_batch, si_batch, epoch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = self.compute_loss(x_batch, si_batch, epoch)
                    loss.backward()
                    optimizer.step()

                if self.fl_cfg.training_progress_bar:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.2f}',
                    })
                    pbar.update(1)
            scheduler.step()

        if self.fl_cfg.training_progress_bar:
            pbar.close()

        # mark this x_vec as related to the current model to avoid retraining prior models
        self.cached_priors_dict[PriorCalculator.get_hash(x_vec)] = np.array(['flag_no_retrain'])

        # Move back to CPU and cleanup
        self.coding_model.cpu()
        torch.cuda.empty_cache()

    @staticmethod
    def _batch_loop(func: Callable[[int, int], torch.Tensor], coding_model, input_size: int,
                    batch_size: int, training_mode: bool = False) -> torch.Tensor:
        coding_model.cuda()
        coding_model.train() if training_mode else coding_model.eval()

        # Pre-allocate list with estimated capacity
        num_batches = (input_size + batch_size - 1) // batch_size
        all_res: List[torch.Tensor | None] = [None] * num_batches

        # Use inference_mode context only when not in training mode
        ctx = nullcontext() if training_mode else torch.inference_mode()

        with ctx:
            batch_idx = 0
            for start_i in range(0, input_size, batch_size):
                end_idx = min(start_i + batch_size, input_size)
                res = func(start_i, end_idx)
                all_res[batch_idx] = res
                batch_idx += 1

        concat_res = torch.cat(all_res, dim=1) if all_res[0].shape[1] > 1 else torch.cat(all_res, dim=0)
        if not training_mode:
            coding_model.cpu()
        torch.cuda.empty_cache()
        return concat_res

    def encoding_process(self, grad_vector: torch.Tensor) -> torch.Tensor:
        # Keep on CPU, batch processing will handle GPU transfers
        grad_vector_reshaped = self.get_x_data(grad_vector)
        bins = self._encoding_process(grad_vector_reshaped)
        return bins

    def _encoding_process(self, grad_vector: torch.Tensor, batch_size: int = 500_000) -> torch.Tensor:
        # return torch.round(grad_vector*1000).to(torch.int16)

        assert grad_vector.shape[0] == self.si_vec_size

        def func(start_i, end_idx):
            x_batch = grad_vector[start_i:end_idx].cuda()
            bins_list, _ = self.coding_model.encode(x_batch)
            bins_list = torch.stack(bins_list)
            assert torch.unique(bins_list).size(0) <= self.coding_model.bins_per_plane ** self.coding_model.num_planes
            return bins_list.cpu()
        bins = self._batch_loop(func, self.coding_model, self.si_vec_size, batch_size)

        dtype = torch.uint8 if self.bins_per_plane < 2**8 else torch.uint16

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size

        return bins.to(dtype)

    def decoding_process(self, payload_content: np.ndarray, batch_size: int = 500_000) -> torch.Tensor:
        # return torch.from_numpy(quantized_data).float()/1000.0

        bins = torch.from_numpy(payload_content)
        side_info = self.get_si_data()

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size
        b_p_p = self.bins_per_plane
        assert bins.float().max() < b_p_p

        def func(start_i, end_idx):
            bins_batch = bins[:, start_i:end_idx].cuda()
            side_info_batch = side_info[start_i:end_idx].cuda()

            codes = [F.one_hot(b.to(int), num_classes=b_p_p) for b in bins_batch]
            reconstructs_batch = self.coding_model.decode(codes, side_info_batch)[-1]

            return reconstructs_batch.cpu()
        all_reconstructs = self._batch_loop(func, self.coding_model, self.si_vec_size, batch_size)

        res = all_reconstructs.squeeze()

        assert res.shape[0] == self.si_vec_size

        return res

    def _get_posterior(self, x_vec, bins_vec_save_compute=None):
        data_hash_str = PriorCalculator.get_hash(x_vec)
        hash_exists = data_hash_str in self.cached_priors_dict
        use_coding_model = hash_exists and self.cached_priors_dict[data_hash_str][0] == 'flag_no_retrain'
        if hash_exists and not use_coding_model:
            return self.cached_priors_dict[data_hash_str]

        bins_vec = self.encoding_process(x_vec) if bins_vec_save_compute is None else bins_vec_save_compute

        if self.coding_model.marginal:
            prior = PriorCalculator.compute_marginal_prior(bins_vec, self.bins_per_plane, self.num_planes)
        else:
            side_info = self.get_si_data()

            if use_coding_model:
                q_model = self.coding_model
            else:
                q_model = PriorCalculator.train_prior_model(
                    bins_vec, side_info, self.num_planes, self.bins_per_plane,)
            prior = PriorCalculator._compute_prior_from_network(q_model, bins_vec, side_info)

        self.cached_priors_dict[data_hash_str] = prior.to(torch.float16)
        return self.cached_priors_dict[data_hash_str]


if __name__ == "__main__":
    import time
    from FL_reworked.cancer_protocol import CancerConfig
    from FL_reworked.cancer_preprocess_protocol import WZQuantizerCancerWithDataPrep

    base_signal = torch.from_numpy(np.random.normal(0, 1, 10_000_000).astype(np.float32))
    y = base_signal + torch.from_numpy(np.random.normal(0, 0.1, 10_000_000).astype(np.float32))

    pretrained_path = CancerConfig().pretrain_pth_dir+"/bpp16_np3_pretrained_wzq_rnn.pth"

    quantizer_class = WZQuantizerCancer
    def test(quantizer):
        bins = quantizer.encoding_process(y)
        recons = quantizer.decoding_process(bins.numpy())
        prior = quantizer._get_posterior(y, bins_vec_save_compute = bins)

        mape = torch.mean(torch.abs(y - recons) / (torch.abs(y) + 1e-8)).item() * 100
        rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)
        print(f"MAPE: {mape:.2f}%", f"Prior rate: {rate:.4f} bits/symbol")
        return bins, recons, prior

    # quantizer_class = lambda **kargs: WZQuantizerCancerWithDataPrep(
    #         **kargs, vec_slices=[slice(i, None, 3) for i in range(3)])
    # def test(quantizer):
    #     bins,temp = quantizer.encoding_process(y)
    #     recons = quantizer.decoding_process([bins.numpy(), temp])
    #     prior = quantizer._get_posterior(y, bins_vec_save_compute = bins)
    #
    #     mape = torch.mean(torch.abs(y - recons) / (torch.abs(y) + 1e-8)).item() * 100
    #     rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)
    #     print(f"MAPE: {mape:.2f}%", f"Prior rate: {rate:.4f} bits/symbol")
    #     return bins, recons, prior

    t_s = time.time()

    print("(num_planes=3, bins_per_plane=16)")

    print("\nwithout side info - pretrained model (P)")
    quantizer = quantizer_class(c_cfg=CancerConfig(),
        fl_cfg=FLConfig(num_clients=1), num_planes=3, bins_per_plane=16, si_size=0,)
    quantizer.coding_model.load_state_dict(torch.load(pretrained_path), strict=False)
    test(quantizer)

    print("\nwithout side info - marginal model (M)")
    quantizer = quantizer_class(c_cfg=CancerConfig(),
        fl_cfg=FLConfig(num_clients=1), num_planes=3, bins_per_plane=16, si_size=0,)
    quantizer.train_model(y, side_info_list=None, batch_size=500_000)
    test(quantizer)

    print("\nTraining quantizer with side info (R or T)...")
    side_info = [base_signal.clone()]
    quantizer = quantizer_class(c_cfg=CancerConfig(),
        fl_cfg=FLConfig(num_clients=1), num_planes=3, bins_per_plane=16, si_size=len(side_info),)
    quantizer.train_model(y, side_info_list=side_info, batch_size=500_000)
    test(quantizer)

    # Test for unseen data
    print("\nTesting quantizer on unseen data...")
    y = base_signal + torch.from_numpy(np.random.normal(0, 0.1, 10_000_000).astype(np.float32))
    bins, recons, prior = test(quantizer)

    t_e = time.time()
    print(f"\nTotal time: {t_e - t_s:.2f} seconds")

    print(f"Prior shape: {prior.shape}")
    print(f"Bins shape: {bins.shape}")
    print(f"Recons shape: {recons.shape}")
    print(f"Unique bins used per plane: {[torch.unique(bins[i]).numel() for i in range(bins.shape[0])]}")
