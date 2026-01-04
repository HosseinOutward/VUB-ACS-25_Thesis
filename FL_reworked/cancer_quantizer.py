from typing import List, Optional, Callable, TYPE_CHECKING, Dict, Tuple
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from prior_calculator import PriorCalculator
from run_fl import FLConfig
from utils import create_training_progress_bar
from brent_wz_models import EncoderDecoderLayeredRNN

if TYPE_CHECKING:
    from cancer_protocol import CancerConfig


def get_normalization_factor(y: torch.Tensor) -> float:
    """Calculate normalization factor based on quantiles."""
    num_samples = 5
    sample_size = min(200_000, len(y))
    norm_facts = []
    grav_centers = []
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y), size=sample_size, replace=True)
        y_sample = y[sample_indices].float()
        y_sample = y_sample[(y_sample >= torch.quantile(y_sample, 0.02)) &
                                  (y_sample <= torch.quantile(y_sample, 0.98))]
        g=torch.mean(y_sample).item()
        grav_centers.append(g)
    g = float(np.mean(grav_centers))

    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y), size=sample_size, replace=True)
        y_sample = y[sample_indices].float()

        norm_fact_99 = torch.abs(torch.quantile(y_sample - g, 0.99)).item()
        norm_fact_1 = torch.abs(torch.quantile(y_sample - g, 0.01)).item()

        norm_facts.append([norm_fact_1, norm_fact_99])
    norm_fact = float(np.mean(norm_facts))

    assert norm_fact != 0

    return norm_fact, g


def get_outlier_factor(grad_flat_normal: torch.Tensor, outlier_threshold: float) -> Tuple:
    """Extract outlier information from a tensor."""
    outlier_mask = torch.abs(grad_flat_normal) > outlier_threshold
    outlier_count = torch.sum(outlier_mask)

    if outlier_count == 0:
        return np.array([], dtype=int), None, torch.tensor([])

    outlier_sign: np.ndarray = torch.sign(grad_flat_normal[outlier_mask]).cpu().numpy()
    outlier_max: float = float(
        torch.quantile(torch.abs(grad_flat_normal[outlier_mask]).float() - outlier_threshold, .99)) / outlier_threshold
    outlier_positions: np.ndarray = np.where(outlier_mask.cpu().numpy())[0]

    assert outlier_max != 0

    return outlier_positions, outlier_max, outlier_sign


class WZQuantizerCancer:
    def __init__(self, c_cfg: 'CancerConfig', fl_cfg: FLConfig, num_planes: int,
                 bins_per_plane: int, si_size: int, marginal_loss=False,
                 norm_slices: List[slice]|bool|None = False, outlier_threshold: float|bool = False) -> None:
        # Data preprocessing parameters - defaults to single slice (no partitioning)
        self.vec_slices: List[slice]|bool = norm_slices if norm_slices is not None else [slice(0, None)]
        self.outlier_threshold: float|bool = outlier_threshold

        self.c_cfg: 'CancerConfig' = c_cfg
        self.fl_cfg: FLConfig = fl_cfg

        self.no_si: bool = (si_size == 0)
        marginal_loss = marginal_loss or self.no_si
        self.coding_model: EncoderDecoderLayeredRNN = EncoderDecoderLayeredRNN(
            num_planes=num_planes, bins_per_plane=bins_per_plane,
            side_info_size=max(1, si_size), input_dim=1,
            layers=3, hidden_dim=100, marginal=marginal_loss,)

        # default assume that its pretrained marginal model if si_size==0 unless trained otherwise
        self.side_info_list_used: List[torch.Tensor] | str | None
        self.side_info_list_used = 'P' if si_size==0 else None

        self.cached_priors_dict: Dict[str, torch.Tensor] = {}
        self.mspe_denom: float | None = None
        self.si_vec_size: Optional[int] = None

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
            dist = F.mse_loss(reconstruct[i]/self.mspe_denom, x_vec/self.mspe_denom)
            loss = loss + self.c_cfg.reconst_ld * dist

            # rate component of the loss
            temp = torch.arange(soft_codes[i].size(0))
            p_ux = soft_codes[i][temp, bins_no[i]]
            p_u = prior_probs[i][temp, bins_no[i]]
            pu_vec[i] = p_u.detach()
            rate_loss = torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

            # rate_weight = lambda x: (((x - 1) + np.exp(x * np.log(abs(self.c_cfg.tau_rate))))
            #                          / abs(self.c_cfg.tau_rate) * 1.25)
            # rate_weight = rate_weight(training_prog) if self.c_cfg.tau_rate <= 0 else 1 - rate_weight(1 - training_prog)

            loss = loss + rate_loss #* max(rate_weight, 0.2)
        loss = loss / self.num_planes

        return loss

    def get_x_data(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[float], Tuple]]:
        if self.si_vec_size is None:
            self.si_vec_size = x_raw.shape[0]

        # Apply preprocessing (normalization + outlier handling)
        x_prep, norm_factors, outlier_param = self._apply_pre_process(x_raw)
        x_prep = x_prep.cuda().unsqueeze(1).to(torch.float32).contiguous()

        if self.vec_slices is False:
            assert norm_factors == [1], "norm_factors should be [1] when vec_slices is False."
        if self.outlier_threshold is False:
            assert outlier_param[0].size == 0, "No outliers should be detected when outlier_threshold is False."

        return x_prep, (norm_factors, outlier_param)

    def get_si_data(self) -> torch.Tensor:
        if self.side_info_list_used in [[], 'P']:
            self.side_info_list_used = [torch.zeros(self.si_vec_size)]

        si_trans = self.side_info_list_used
        if not (len(si_trans) == 1 and torch.all(si_trans[0] == 0)): # if not zeros
            si_trans = [self._apply_pre_process(si, True)[0] for si in self.side_info_list_used]

        si_trans = torch.stack(si_trans).cuda().T.to(torch.float32).contiguous()
        return si_trans

    def train_model(self, x_raw: torch.Tensor, si_raw_list: Optional[List[torch.Tensor]],
                    batch_size: int = 50_000) -> None:
        if self.no_si:
            assert si_raw_list is None, "Marginal model expects an empty si_raw_list for training."
            si_raw_list = []
        else:
            assert len(si_raw_list) > 0, "require side info for training."
        assert self.side_info_list_used in [None, 'P'], "This quantizer instance has already been trained."
        assert x_raw is not None, "Training data x_raw must be provided for training."

        self.side_info_list_used = si_raw_list

        # Convert to model format (preprocessing applied)
        self.mspe_denom: float = x_raw.abs().mean().item()/2 + 1e-8
        x_prep, _ = self.get_x_data(x_raw)
        si_trans = self.get_si_data()
        self._train_model(x_prep, si_trans, batch_size)

        # Mark this x_prep as related to the current model to avoid retraining prior models
        self.cached_priors_dict[PriorCalculator.get_hash(x_raw)] = np.array(['flag_no_retrain'])


    def _train_model(self, x_prep: torch.Tensor, si_trans: torch.Tensor, batch_size: int = 50_000) -> None:
        # Enable TF32 for faster matmul on Ampere+ GPUs
        if self.fl_cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Use fused AdamW for better performance
        optimizer = torch.optim.AdamW(self.coding_model.parameters(), fused=self.fl_cfg.fused_optimizer,
                                      lr=self.c_cfg.lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.c_cfg.train_epochs*np.ceil(self.c_cfg.lr_step/180)), gamma=0.3)

        train_dataset = torch.utils.data.TensorDataset(x_prep, si_trans)

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

    def encoding_process(self, grad_raw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[float], Tuple]]:
        # Keep on CPU, batch processing will handle GPU transfers
        grad_prep, prep_metadata = self.get_x_data(grad_raw)
        bins = self._encoding_process(grad_prep)
        return bins, prep_metadata

    def _encoding_process(self, grad_prep: torch.Tensor, batch_size: int = 500_000) -> torch.Tensor:
        # return torch.round(grad_vector*1000).to(torch.int16)

        assert grad_prep.shape[0] == self.si_vec_size

        def func(start_i, end_idx):
            x_batch = grad_prep[start_i:end_idx].cuda()
            bins_list, _ = self.coding_model.encode(x_batch)
            bins_list = torch.stack(bins_list)
            assert torch.unique(bins_list).size(0) <= self.coding_model.bins_per_plane ** self.coding_model.num_planes
            return bins_list.cpu()
        bins = self._batch_loop(func, self.coding_model, self.si_vec_size, batch_size)

        dtype = torch.uint8 if self.bins_per_plane < 2**8 else torch.uint16

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size

        return bins.to(dtype)

    def decoding_process(self, payload_content: Tuple[torch.Tensor, Tuple[List[float], Tuple]],
                         batch_size: int = 500_000) -> torch.Tensor:
        # return torch.from_numpy(quantized_data).float()/1000.0

        bins, (norm_factors, outlier_param) = payload_content
        si_trans = self.get_si_data()

        assert self.num_planes == bins.shape[0]
        assert bins.shape[1] == self.si_vec_size
        b_p_p = self.bins_per_plane
        assert bins.float().max() < b_p_p

        def func(start_i, end_idx):
            bins_batch = bins[:, start_i:end_idx].cuda()
            si_batch = si_trans[start_i:end_idx].cuda()

            codes = [F.one_hot(b.to(int), num_classes=b_p_p) for b in bins_batch]
            reconstructs_batch = self.coding_model.decode(codes, si_batch)[-1]

            return reconstructs_batch.cpu()
        all_reconstructs = self._batch_loop(func, self.coding_model, self.si_vec_size, batch_size)

        grad_prep = all_reconstructs.squeeze()

        assert grad_prep.shape[0] == self.si_vec_size

        # Apply post-processing to restore original scale
        grad_raw = self._post_process(grad_prep, norm_factors, outlier_param)

        return grad_raw

    def _get_posterior(self, x_raw: torch.Tensor, bins_vec_save_compute: torch.Tensor = None):
        data_hash_str = PriorCalculator.get_hash(x_raw)
        hash_exists = data_hash_str in self.cached_priors_dict
        use_coding_model = hash_exists and self.cached_priors_dict[data_hash_str][0] == 'flag_no_retrain'

        # comment out to force training prior model every time
        if hash_exists and not use_coding_model:
            return self.cached_priors_dict[data_hash_str]

        bins_vec = self.encoding_process(x_raw)[0] if bins_vec_save_compute is None else bins_vec_save_compute

        si_trans = self.get_si_data()
        q_model = self.coding_model
        if not use_coding_model:
            q_model = PriorCalculator.train_prior_model(
                bins_vec, si_trans, self.num_planes, self.bins_per_plane,)

        prior = PriorCalculator._compute_prior_from_network(q_model, bins_vec, si_trans)

        self.cached_priors_dict[data_hash_str] = prior.to(torch.float16)
        return self.cached_priors_dict[data_hash_str]

    def _apply_pre_process(self, x_raw: torch.Tensor, force_no_outlier_handling=False,
                           ) -> Tuple[torch.Tensor, List[float], Tuple]:
        x_prep = x_raw.clone()

        no_normal_handling = (self.vec_slices == False)
        no_outlier_handling = (self.outlier_threshold == False) or force_no_outlier_handling

        norm_factors: List[Tuple[float,float]] = [(1,0)]
        if not no_normal_handling:
            norm_factors = []
            for v_slc in self.vec_slices:
                norm_fact, grav_center = get_normalization_factor(x_prep[v_slc])
                norm_factors.append((norm_fact, grav_center))
                x_prep[v_slc] = (x_prep[v_slc] - grav_center) / norm_fact
        norm_factors:torch.Tensor = torch.Tensor(norm_factors).to(torch.float16)

        # Outlier handling (if enabled)
        outlier_positions: np.ndarray = np.array([], dtype=int)
        outlier_max: Optional[float] = None
        outlier_sign: np.ndarray = np.array([])
        outlier_param = (outlier_positions, outlier_max, outlier_sign)

        if not no_outlier_handling:
            outlier_positions, outlier_max, outlier_sign = get_outlier_factor(x_prep, self.outlier_threshold)
            if len(outlier_positions) != 0:
                temp = x_prep[outlier_positions]
                x_prep[outlier_positions] = (torch.abs(temp) - self.outlier_threshold) * torch.sign(temp) / outlier_max

        return x_prep, norm_factors, outlier_param

    def _post_process(self, recons_raw: torch.Tensor, norm_factors: List[float], outlier_param: Tuple) -> torch.Tensor:
        recons_prep = recons_raw.clone()
        # Restore outliers
        outlier_positions: np.ndarray = outlier_param[0]
        if len(outlier_positions) != 0:
            outlier_max: float = outlier_param[1]
            outlier_sign: torch.Tensor = torch.from_numpy(outlier_param[2])

            assert len(np.unique(outlier_sign)) in [1, 2]
            assert outlier_sign.max() in [1, -1] and outlier_sign.min() in [1, -1]
            recons_prep[outlier_positions] = \
                (torch.abs(recons_prep[outlier_positions]) * outlier_max + self.outlier_threshold) * outlier_sign

        # Denormalize per slice
        if type(self.vec_slices) is list:
            for i, v_slc in enumerate(self.vec_slices):
                norm_fact, grav_center = norm_factors[i]
                recons_prep[v_slc] = recons_prep[v_slc] * norm_fact + grav_center

        return recons_prep


if __name__ == "__main__":
    import time
    from cancer_protocol import CancerConfig

    base_signal = torch.from_numpy(np.random.normal(0, 1, 1_000_000).astype(np.float32))
    y = base_signal + torch.from_numpy(np.random.normal(0, 0.1, 1_000_000).astype(np.float32))
    side_info = [base_signal.clone()]

    pretrained_path = CancerConfig().pretrain_pth_dir+"/bpp16_np3_pretrained_wzq_rnn.pth"

    def test(quantizer):
        bins, metadata = quantizer.encoding_process(y)
        recons = quantizer.decoding_process((bins.numpy(), metadata))
        prior = quantizer._get_posterior(y, bins_vec_save_compute = bins)

        mape = torch.mean(torch.abs(y - recons) / (torch.abs(y) + 1e-8)).item() * 100
        rate = PriorCalculator.compute_rate_from_prior_tensor(prior, bins, quantizer.num_planes)
        print(f"MAPE: {mape:.2f}%", f"Prior rate: {rate:.4f} bits/symbol")
        return bins, recons, prior

    t_s = time.time()

    print("(num_planes=3, bins_per_plane=16)")

    print("\n1. Without side info - pretrained model (P)")
    quantizer = WZQuantizerCancer(
        c_cfg=CancerConfig(), fl_cfg=FLConfig(num_clients=1),
        num_planes=3, bins_per_plane=16, si_size=0,)
    quantizer.coding_model.load_state_dict(torch.load(pretrained_path), strict=False)
    test(quantizer)

    print("\n2. With side info - marginal model (M)")
    quantizer = WZQuantizerCancer(
        c_cfg=CancerConfig(), fl_cfg=FLConfig(num_clients=1),
        num_planes=3, bins_per_plane=16, si_size=1, marginal_loss=True
    )
    quantizer.train_model(y, si_raw_list=side_info, batch_size=500_000)
    test(quantizer)

    print("\n2.B Without side info - marginal model (M)")
    quantizer = WZQuantizerCancer(
        c_cfg=CancerConfig(), fl_cfg=FLConfig(num_clients=1),
        num_planes=3, bins_per_plane=16, si_size=0
    )
    quantizer.train_model(y, si_raw_list=None, batch_size=500_000)
    test(quantizer)

    print("\n3. Training quantizer with side info (R or T)...")
    quantizer = WZQuantizerCancer(
        c_cfg=CancerConfig(), fl_cfg=FLConfig(num_clients=1),
        num_planes=3, bins_per_plane=16, si_size=len(side_info)
    )
    quantizer.train_model(y, si_raw_list=side_info, batch_size=500_000)
    test(quantizer)

    # Test for unseen data
    print("\n4. Testing quantizer on unseen data...")
    y = base_signal + torch.from_numpy(np.random.normal(0, 0.1, 1_000_000).astype(np.float32))
    bins, recons, prior = test(quantizer)

    print("\n" + "="*70)
    print("Testing WITH vec_slices and outlier handling")
    print("="*70)

    print("\n5. With side info + vec_slices + outlier handling...")
    quantizer_advanced = WZQuantizerCancer(
        c_cfg=CancerConfig(), fl_cfg=FLConfig(num_clients=1),
        num_planes=3, bins_per_plane=16, si_size=len(side_info),
        norm_slices=[slice(i, None, 3) for i in range(3)],
        outlier_threshold=1.4
    )
    quantizer_advanced.train_model(y, si_raw_list=side_info, batch_size=500_000)
    test(quantizer_advanced)

    t_e = time.time()
    print(f"\nTotal time: {t_e - t_s:.2f} seconds")
    print(f"\nFinal results:")
    print(f"  Prior shape: {prior.shape}")
    print(f"  Bins shape: {bins.shape}")
    print(f"  Recons shape: {recons.shape}")
    print(f"  Unique bins used per plane: {[torch.unique(bins[i]).numel() for i in range(bins.shape[0])]}")
