import gc
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch

from components.other_utilities.user_logger import UnifiedLoggingClass


class WZQuantizer:
    def __init__(self, wz_pl_model, count_side_info_data,
                 enable_progress_bar=False, train_sample_size=100_000, user_logger:UnifiedLoggingClass=None,):
        from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN

        self.val_indices = None
        self.training_posterior_cdf = None
        self.enable_progress_bar = enable_progress_bar
        self.user_logger = user_logger

        self.train_sample_size = train_sample_size

        self.wz_pl_model:PL_EncoderDecoder_RNN = wz_pl_model
        self.count_side_info_data = count_side_info_data

    @property
    def bin_count(self):
        return self.wz_pl_model.bin_count

    def _batch_loop(self, func, batch_size, total_size):
        self.wz_pl_model.to('cuda')
        self.wz_pl_model.eval()
        with torch.no_grad():
            all_res = []
            for start_i in range(0, total_size, batch_size):
                end_idx = min(start_i + batch_size, total_size)
                res = func(start_i, end_idx)
                all_res.append(res)
        self.wz_pl_model.to('cpu')
        torch.cuda.empty_cache()
        return all_res

    def get_set_training_posterior_cdf(self, grad_vector=None, side_info_data_list=None):
        if self.wz_pl_model.coding_model.marginal:
            self.training_posterior_cdf = self.get_prior_and_softcodes(grad_vector, side_info_data_list)[0].numpy()
            return self.training_posterior_cdf

        if grad_vector is None or side_info_data_list is None:
            assert grad_vector is None and side_info_data_list is None
            assert self.training_posterior_cdf is not None
        else:
            self.training_posterior_cdf = self.get_prior_and_softcodes(grad_vector, side_info_data_list)[0].numpy()

        return self.training_posterior_cdf

    def get_prior_and_softcodes(self, grad_vector, side_info_data_list, batch_size=500_000):
        if type(grad_vector) != torch.Tensor:
            grad_tensor = torch.tensor(grad_vector).to(torch.float32)
        else:
            grad_tensor = grad_vector.to(torch.float32)

        side_info_array = torch.tensor(np.array(side_info_data_list), dtype=torch.float32)
        if self.count_side_info_data != 0:
            side_info_array = side_info_array.T

        total_size = len(grad_tensor)

        def func(start_i, end_idx):
            grad_batch = grad_tensor[start_i:end_idx].unsqueeze(1).to('cuda')
            side_info_batch = side_info_array[start_i:end_idx].to('cuda')
            prior_batch, soft_code_batch = self.wz_pl_model.get_prior_and_softcodes_net(grad_batch, side_info_batch)
            return (prior_batch.to('cpu'), soft_code_batch.to('cpu'))
        all_priors = self._batch_loop(func, batch_size, total_size)

        prior, soft_codes = zip(*all_priors)
        prior, soft_codes = [torch.cat(a, dim=1) for a in [prior, soft_codes]]

        bins_vector = [torch.argmax(sc, dim=-1) for sc in soft_codes]
        for i in range(prior.shape[0]):
            prior[i, np.arange(prior.shape[1]), bins_vector[i]] += 1e-6
            prior[i] /= prior[i].sum(axis=-1, keepdims=True)

        return prior, soft_codes

    def encoding_process(self, grad_vector, batch_size=500_000):
        # from components.broadcast_components.WZ_models.simple import simple_quantize
        # return simple_quantize(grad_vector)

        grad_tensor = torch.tensor(grad_vector).to(torch.float32)
        total_size = len(grad_tensor)

        def func(start_i, end_idx):
            grad_batch = grad_tensor[start_i:end_idx].unsqueeze(1).to('cuda')
            bins_batch = self.wz_pl_model.encode_net(grad_batch)
            return bins_batch.to('cpu')
        all_bins = self._batch_loop(func, batch_size, total_size)

        bins = torch.cat(all_bins, dim=1) if len(all_bins[0].shape) > 1 else torch.cat(all_bins, dim=0)

        dtype = torch.uint8 if self.wz_pl_model.bins_per_plane < 2**8 else torch.uint16
        return bins.to(dtype), None

    # todo remove element_count
    def decoding_process(self, quantized_data, side_info_data_list, encoding_extra_data=None, batch_size=500_000):
        # from components.broadcast_components.WZ_models.simple import simple_dequantize
        # return simple_dequantize(quantized_data, np.float32)

        bins_tensor = torch.tensor(np.asarray(quantized_data))
        total_size = len(bins_tensor[0])

        assert len(side_info_data_list) == self.count_side_info_data
        if self.count_side_info_data == 0:
            side_info_data_list = [np.zeros(len(quantized_data[0]))]

        assert total_size == len(side_info_data_list[0])

        side_info_array = torch.tensor(np.array(side_info_data_list), dtype=torch.float32).T

        def func(start_i, end_idx):
            if len(bins_tensor.shape) != 2:
                bins_batch = bins_tensor[start_i:end_idx].to('cuda')
            else:
                bins_batch = bins_tensor[:, start_i:end_idx].to('cuda')

            side_info_batch = side_info_array[start_i:end_idx].to('cuda')

            reconstructs_batch = self.wz_pl_model.decode_net(bins_batch, side_info_batch)
            return reconstructs_batch.to('cpu')

        all_reconstructs = self._batch_loop(func, batch_size, total_size)

        all_reconstructs = torch.cat(all_reconstructs, dim=0)
        res = all_reconstructs.squeeze()
        res = res.numpy()
        return res

    # todo have multiple input data and train on all of them in one run (change sampler)
    def train_model(self, input_data_, side_info_data_list_: List, epoch=10, batch_size=50_000, device=[0]):
        # return

        assert len(side_info_data_list_) == self.count_side_info_data
        side_info_data_list = side_info_data_list_
        if self.count_side_info_data == 0:
            side_info_data_list = [np.zeros(len(input_data_))]
        input_data = torch.tensor(input_data_).unsqueeze(1).to(torch.float32)
        side_info_data_list = torch.tensor(np.array(side_info_data_list)).T.to(torch.float32)

        train_dataset = torch.utils.data.TensorDataset(input_data, side_info_data_list)

        if self.train_sample_size > len(train_dataset):
            self.train_sample_size = len(train_dataset)

        # --------------- val dataloader ---------------
        all_indices = np.arange(len(train_dataset))

        self.val_indices = np.random.choice(all_indices, size=int(self.train_sample_size // 3), replace=False)
        val_dataset = torch.utils.data.Subset(train_dataset, self.val_indices)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size * 10,
            num_workers=2, pin_memory=False, persistent_workers=True)

        train_indices = np.setdiff1d(all_indices, self.val_indices)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        # ------------------------------

        class RandomReplacementSampler(torch.utils.data.Sampler):
            def __init__(self, dataset_size, samples_per_epoch):
                super().__init__(data_source=None)
                self.dataset_size = dataset_size
                self.samples_per_epoch = samples_per_epoch
            def __iter__(self):
                return iter(np.random.randint(0, self.dataset_size, self.samples_per_epoch))
            def __len__(self):
                return self.samples_per_epoch
        train_sampler = RandomReplacementSampler(
            dataset_size=len(train_dataset), samples_per_epoch=self.train_sample_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=False, persistent_workers=True)

        # ------------------------------
        self.wz_pl_model.to('cpu')

        # disable val progress bar due to pl bug
        NoValidationBar = None
        if self.enable_progress_bar:
            print('        - training wz model')

            from pytorch_lightning.callbacks import TQDMProgressBar
            from tqdm import tqdm
            class NoValidationBar(TQDMProgressBar):
                def init_validation_tqdm(self): return tqdm(disable=True)

        logger = False
        if self.user_logger:
            logger = self.user_logger.get_wz_csv_logger()

        trainer = pl.Trainer(
            # accelerator="cuda",
            devices=device,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            log_every_n_steps=1 if self.enable_progress_bar or logger else False,
            max_epochs=epoch,
            enable_progress_bar=self.enable_progress_bar,
            callbacks=[NoValidationBar()] if self.enable_progress_bar else [],
            logger=logger,
        )
        trainer.fit(self.wz_pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        _ = self.get_set_training_posterior_cdf(input_data_, side_info_data_list_)

        # Safer cleanup approach that doesn't interfere with PyTorch Lightning's teardown
        try:
            # Let PyTorch Lightning finish its own cleanup first
            if hasattr(trainer, 'state') and trainer.state.finished:
                # Only do manual cleanup if training completed normally
                self._cleanup_dataloaders([train_dataloader, val_dataloader], used_persistent_workers=True)
        except Exception as e:
            print(f'Post-training cleanup encountered issue: {e}')

        # Clean up trainer and dataloaders
        del trainer, train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    def _cleanup_dataloaders(self, dataloaders, used_persistent_workers):
        """Safely cleanup DataLoader workers without interfering with PyTorch Lightning"""
        if not used_persistent_workers:
            return  # No persistent workers to clean up

        cleanup_successful = True
        for i, dl in enumerate(dataloaders):
            dl_name = ['train', 'val'][i]
            try:
                # Method 1: Check if iterator exists and is in a clean state
                if hasattr(dl, '_iterator') and dl._iterator is not None:
                    # Only attempt cleanup if iterator is not already being cleaned up
                    if hasattr(dl._iterator, '_shutdown_workers') and not getattr(dl._iterator, '_shutdown', False):
                        dl._iterator._shutdown_workers()

            except Exception as e:
                print(f'Safe cleanup failed for {dl_name} dataloader: {e}')
                cleanup_successful = False

        if not cleanup_successful:
            print('WARNING: Some DataLoader cleanup failed - minor memory leak possible')
            # Light garbage collection attempt
            gc.collect()


def plot_bins(wz_quantizer: WZQuantizer, x_data_, side_info, step_count=1000, training_ind=False):
    if isinstance(x_data_, torch.Tensor):
        x_data_ = x_data_.cpu().numpy()
    ind_list = wz_quantizer.val_indices if wz_quantizer.val_indices is not None\
                    else np.arange(len(x_data_))
    ind_list = ind_list if training_ind \
                    else np.setdiff1d(np.arange(len(x_data_)), wz_quantizer.val_indices)

    min_v, max_v = np.percentile(x_data_, 0.1), np.percentile(x_data_, 99.9)
    true_min_v, true_max_v = x_data_.min(), x_data_.max()

    grad_data = np.asarray(x_data_, dtype=np.float32)[ind_list]
    side_info = [si[ind_list] for si in side_info]

    sort_idx = np.argsort(grad_data)
    grad_data = grad_data[sort_idx]
    side_info = [si[sort_idx] for si in side_info]

    # Create x_range for plotting
    deunif_bins, encoding_extra = wz_quantizer.encoding_process(grad_data)
    recons_for_x_range = wz_quantizer.decoding_process(deunif_bins, side_info, encoding_extra_data=encoding_extra)

    x_step = (max_v - min_v) / step_count
    pointer_v = true_min_v
    spaced_idx = []
    for i, gd in enumerate(grad_data):
        if pointer_v + x_step * 0.95 < gd:
            pointer_v = gd
            spaced_idx.append(i)
    spaced_idx.append(len(grad_data)-1)
    spaced_idx = np.array(spaced_idx)

    grad_data = grad_data[spaced_idx]
    side_info = [si[spaced_idx] for si in side_info]
    recons_for_x_range = recons_for_x_range[spaced_idx]
    deunif_bins = [deunif_bins[b][spaced_idx] for b in range(len(deunif_bins))]

    bins = wz_quantizer.wz_pl_model.unify_bins(deunif_bins)
    print('bins used:', len(np.unique(bins)))

    # Setup plots
    from matplotlib import pyplot as plt
    def _plot_bin_regions(ax, x_range, decoded_bins, x_step, colors):
        current_bin = decoded_bins[0]
        last_idx = 0
        last_split_point = x_range[0]
        for i, x_v in enumerate(x_range[1:], start=1):
            if decoded_bins[i] != current_bin:  # change detected
                split_point = (x_v + x_range[i - 1]) / 2
                ax.axvline(split_point, color='black', linestyle='--', linewidth=0.5)
                ax.axvspan(x_range[last_idx], split_point, color=colors[current_bin], alpha=0.3)

                current_bin = decoded_bins[i]
                last_idx = i
                last_split_point = split_point
        # Final span
        ax.axvspan(last_split_point, x_range[-1], color=colors[current_bin], alpha=0.3)

    bin_count = wz_quantizer.bin_count
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # Plot 1: Data histogram with binning analysis ------------------------
    counts, bins_edges, patches = ax1.hist(
        x_data_[np.random.choice(len(x_data_), min(len(x_data_), 20_000), replace=False)],
        bins=200, alpha=0.3, color='gray', label='data histogram', density=True)
    ax1.clear()
    ax1.bar(bins_edges[:-1], counts / np.max(counts), width=np.diff(bins_edges),
            alpha=0.3, color='gray', label='data histogram (normalized)', align='edge')
    ax1.scatter(grad_data, np.abs(grad_data - recons_for_x_range), label='reconstruction error', s=0.2, alpha=0.5)
    temp = [wz_quantizer.wz_pl_model.bins_per_plane, wz_quantizer.wz_pl_model.num_planes]
    for b in range(temp[1]):
        ax1.plot(grad_data, ((np.asarray(deunif_bins[b]) + 1 + b) / (b + temp[0]) + b)/temp[1],
                 color='orange', label='(normalized) encoded_bins' if b == 0 else None, linewidth=0.5)
    for i in range(bin_count):
        ax1.hlines((i + 1) / bin_count, min(grad_data), max(grad_data), linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('x_range')

    # Visualize bin regions
    colors = plt.cm.viridis(np.linspace(0, 1, bin_count))
    _plot_bin_regions(ax1, grad_data, bins, x_step, colors)

    ax1.set_xlim(true_min_v, true_max_v)
    ax1.set_ylim(-0.01, 1.01)
    ax1.legend(loc="upper right")
    ax1.set_title('Binning Visualization')

    # Plot 2: Reconstruction curves for each bin (batch process) ------------------------
    # todo merge the 2 plots and change how the side_info is given (for example for all 0s as side info)
    for bin_idx in np.unique(bins):
        temp = wz_quantizer.wz_pl_model.deunify_bins(torch.zeros(len(x_data_)) + bin_idx)
        temp = wz_quantizer.decoding_process(temp, side_info, encoding_extra_data=encoding_extra)
        temp = temp[spaced_idx]
        ax2.scatter(grad_data, temp, label=f'bin={bin_idx}', s=0.1)

    ax2.set_xlabel('x_range (which is forced to bin, but is paired with related side_info)')
    ax2.set_ylabel('reconstruction per bin')

    ax2.set_xlim(true_min_v, true_max_v)
    # ax2.legend(loc="upper right")
    ax2.set_title('What if x range was forced to a specific bin, plot per bin')

    plt.tight_layout()
    plt.show()

    #%%
    # x_range = np.linspace(true_min_v, true_max_v, step_count)
    # for b in range(bin_count):
    #     temp = wz_quantizer.decoding_process(np.zeros(len(x_range))+b, [x_range],
    #                                          element_count=len(x_range), symbolic_decoder=False)
    #     plt.plot(x_range, temp, label=f'bin={b}', linewidth=0.5)
    # plt.xlabel('side info')
    # plt.ylabel('reconstruction per bin')
    # plt.show()
