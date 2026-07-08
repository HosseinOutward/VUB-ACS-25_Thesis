from __future__ import annotations

import gc
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from FL_code.cancer_protocol import CancerCodec, CancerConfig, CancerRecord
from FL_code.cancer_quantizer import WZQuantizerCancer
from FL_code.codec import Access, CompressionRecord, get_obj_compressed_size, record_reconstruction_metrics
from FL_code.prior_calculator import PriorCalculator
from FL_code.utils import create_training_progress_bar


class SampledCancerRecord(CancerRecord):
    """Compression record for sampled Cancer rounds, including first-pass sample telemetry."""

    def __init__(
        self,
        round_id: int,
        client_id: int,
        method: str = "cancer|sampled",
        phase: str | None = None,
        round_type: str | None = None,
        bits_per_plane: int | None = None,
        num_planes: int | None = None,
    ) -> None:
        super().__init__(round_id, client_id, method, phase, round_type, bits_per_plane, num_planes)
        self.sampled_seed: int | None = None
        self.sampled_count: int | None = None
        self.sampled_fraction: float | None = None
        self.sampled_data_size: float | None = None
        self.sampled_entropy_real_rate: float | None = None
        self.sampled_symbol_rate: float | None = None
        self.sampled_mse: float | None = None
        self.sampled_mape: float | None = None
        self.sampled_mspe_sqrt: float | None = None
        self.sampled_wmape: float | None = None
        self.sampled_wmspe_sqrt: float | None = None
        self.head_state_size: float | None = None
        self.head_train_loss: float | None = None


class SampledWZQuantizerCancer(WZQuantizerCancer):
    """WZ quantizer variant that supports random reconstruction targets and encoder-head updates."""

    def train_model_from_random_reconstruction_targets(
        self,
        target_raw_list: Sequence[torch.Tensor],
        si_raw_list: Sequence[torch.Tensor],
        batch_size: int = 50_000,
    ) -> None:
        """Train the full quantizer with all reconstructions as SI and random reconstruction targets per batch."""
        assert target_raw_list, "Sampled quantizer training requires at least one reconstruction target."
        assert len(si_raw_list) > 0, "Sampled quantizer training requires side information."
        assert self.side_info_list_used in [None, "P"], "This quantizer instance has already been trained."

        vec_size = target_raw_list[0].numel()
        assert all(target.numel() == vec_size for target in target_raw_list), (
            "All sampled quantizer targets must have the same vector size."
        )
        assert all(side_info.numel() == vec_size for side_info in si_raw_list), (
            "All sampled quantizer side-information tensors must have the same vector size."
        )

        self.side_info_list_used = list(si_raw_list)
        self.si_vec_size = vec_size
        self.wmspe_denom = (
            float(np.mean([(target.float() ** 2).mean().item() for target in target_raw_list])) / 2 + 1e-8
        )

        target_prep_list = [
            self._apply_pre_process(target)[0].unsqueeze(1).to(dtype=torch.float16).contiguous()
            for target in target_raw_list
        ]
        si_trans = self.get_si_data()

        num_planes = self.coding_model.num_planes
        bins_per_plane = self.coding_model.bins_per_plane
        marginal_loss = self.coding_model.marginal

        model_losses: list[float] = []
        model_list: list[Any] = []
        tries = 0
        while len(model_losses) < self.c_cfg.quantizer_train_repeats:
            assert tries <= self.c_cfg.quantizer_train_repeats * 5, "Too many failed sampled training attempts."
            tries += 1

            qz_model = self.get_new_RNN_model(num_planes, bins_per_plane, si_trans.shape[1], marginal_loss)
            qz_loss = self._train_random_target_attempt(
                qz_model, target_prep_list, si_trans, self.c_cfg, num_planes,
                self.wmspe_denom, batch_size,
            )
            if qz_loss is None or np.isnan(qz_loss) or np.isinf(qz_loss):
                continue

            model_losses.append(qz_loss)
            model_list.append(qz_model)

        best_idx = int(np.argmin(model_losses))
        self.coding_model = model_list[best_idx]

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _gather_random_targets(
        target_prep_list: Sequence[torch.Tensor],
        indices: torch.Tensor,
        target_choices: torch.Tensor,
    ) -> torch.Tensor:
        batch = torch.empty((indices.numel(), 1), dtype=torch.float16)
        for target_idx in torch.unique(target_choices):
            mask = target_choices == target_idx
            batch[mask] = target_prep_list[int(target_idx)][indices[mask]]
        return batch.to(device="cuda", dtype=torch.float32)

    @classmethod
    def _train_random_target_attempt(
        cls,
        rnn_model: Any,
        target_prep_list: Sequence[torch.Tensor],
        si_trans: torch.Tensor,
        c_cfg: CancerConfig,
        num_planes: int,
        mspe_denom: float,
        batch_size: int,
    ) -> float | None:
        if c_cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        optimizer = torch.optim.AdamW(
            rnn_model.parameters(), fused=c_cfg.fused_optimizer,
            lr=c_cfg.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(c_cfg.train_epochs * np.ceil(c_cfg.lr_step / 180)), gamma=0.3,
        )

        rnn_model.cuda().train()
        use_amp = c_cfg.mixed_precision and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        vec_size = si_trans.shape[0]
        total_samples = min(c_cfg.train_sample_size, vec_size)
        total_iterations = c_cfg.train_epochs * ((total_samples + batch_size - 1) // batch_size)
        pbar = create_training_progress_bar(
            total_iterations,
            desc="Training Sampled Quantizer",
            disable=not c_cfg.training_progress_bar,
        )

        epoch_loss = float("inf")
        for epoch in range(c_cfg.train_epochs):
            indices = torch.randint(0, vec_size, (total_samples,), dtype=torch.long)
            target_choices = torch.randint(0, len(target_prep_list), (total_samples,), dtype=torch.long)

            epoch_loss = 0.0
            batch_count = 0
            for start_i in range(0, total_samples, batch_size):
                end_i = min(start_i + batch_size, total_samples)
                idx_batch = indices[start_i:end_i]
                x_batch = cls._gather_random_targets(
                    target_prep_list, idx_batch, target_choices[start_i:end_i],
                )
                si_batch = si_trans[idx_batch]
                x_batch = x_batch + torch.randn_like(x_batch, device="cuda") * (1e-5 * x_batch.abs().mean())

                optimizer.zero_grad()
                if use_amp:
                    assert scaler is not None
                    with torch.amp.autocast("cuda"):
                        loss = WZQuantizerCancer.compute_loss(
                            rnn_model, x_batch, si_batch, epoch, c_cfg, num_planes, mspe_denom,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = WZQuantizerCancer.compute_loss(
                        rnn_model, x_batch, si_batch, epoch, c_cfg, num_planes, mspe_denom,
                    )
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                if c_cfg.training_progress_bar:
                    pbar.set_postfix({"loss": f"{loss.item():.2f}"})
                    pbar.update(1)
            scheduler.step()
            epoch_loss /= batch_count

        pbar.close()
        rnn_model.cpu()
        torch.cuda.empty_cache()
        return epoch_loss

    def encoder_head_state_dict(self) -> dict[str, Any]:
        """Return only the trainable encoder output head state sent back to the worker."""
        if self.coding_model.shared_encoder is False:
            return {"binners": self.coding_model.binners.state_dict()}
        return {"binner": self.coding_model.binner.state_dict()}

    def train_encoder_head_from_sample(
        self,
        sample_x_prep: torch.Tensor,
        sample_bins: torch.Tensor,
        batch_size: int = 50_000,
    ) -> float:
        """Fine-tune only the encoder binning head from decoded sample values and their transmitted bins."""
        assert sample_x_prep.ndim == 2 and sample_x_prep.shape[1] == 1, (
            "sample_x_prep must have shape [sample_count, 1]."
        )
        assert sample_bins.shape == (self.num_planes, sample_x_prep.shape[0]), (
            "sample_bins must have shape [num_planes, sample_count]."
        )

        model = self.coding_model
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        head = model.binners if model.shared_encoder is False else model.binner
        head_params = list(head.parameters())
        for parameter in head_params:
            parameter.requires_grad_(True)

        optimizer = torch.optim.AdamW(head_params, lr=self.c_cfg.sampled_head_lr, weight_decay=1e-4)
        model.cuda().train()

        total_samples = sample_x_prep.shape[0]
        epoch_loss = 0.0
        for _ in range(self.c_cfg.sampled_head_train_epochs):
            indices = torch.randperm(total_samples)
            epoch_loss = 0.0
            batch_count = 0
            for start_i in range(0, total_samples, batch_size):
                batch_indices = indices[start_i:start_i + batch_size]
                x_batch = sample_x_prep[batch_indices].to(device="cuda", dtype=torch.float32)
                bins_batch = sample_bins[:, batch_indices].to(device="cuda", dtype=torch.long)

                rnn_inputs = x_batch.unsqueeze(1).repeat(1, self.num_planes, 1)
                rnn_out, _ = model.encoder(rnn_inputs)
                if model.shared_encoder is False:
                    logits = [binner(rnn_out[:, idx]) for idx, binner in enumerate(model.binners)]
                else:
                    logits = [model.binner(rnn_out[:, idx]) for idx in range(self.num_planes)]

                loss = sum(
                    F.cross_entropy(plane_logits, bins_batch[idx])
                    for idx, plane_logits in enumerate(logits)
                ) / self.num_planes

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            epoch_loss /= batch_count

        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.cpu().eval()
        torch.cuda.empty_cache()
        return epoch_loss


class SampledCancerCodec(CancerCodec):
    """Cancer protocol variant with S rounds that sample first, retrain the encoder head, then finish coding."""

    def __init__(
        self,
        c_cfg: CancerConfig,
        quantizer_kwargs: dict[str, Any] | None = None,
        codec_name: str = "cancer|sampled",
    ) -> None:
        super().__init__(c_cfg, quantizer_kwargs, codec_name)
        self.c_cfg.warmup_phase = tuple(
            ("S" if phase_type == "R" else phase_type, bins_per_plane, num_planes)
            for phase_type, bins_per_plane, num_planes in self.c_cfg.warmup_phase
        )
        self.c_cfg.routine_phase = tuple(
            ("S" if phase_type == "R" else phase_type, bins_per_plane, num_planes)
            for phase_type, bins_per_plane, num_planes in self.c_cfg.routine_phase
        )

    def create_record(self, round_id: int, client_id: int) -> SampledCancerRecord:
        base_record = super().create_record(round_id, client_id)
        return SampledCancerRecord(
            round_id=base_record.round_id,
            client_id=base_record.client_id,
            method=base_record.codec_class_used,
            phase=base_record.phase,
            round_type=base_record.round_type,
            bits_per_plane=base_record.bins_per_plane,
            num_planes=base_record.num_planes,
        )

    def _train_quantizer_or_load(self, delta_vec: torch.Tensor, record: CancerRecord) -> None:
        """Train regular rounds through CancerCodec and S rounds with random reconstruction targets."""
        assert record.round_type is not None
        assert record.bins_per_plane is not None
        assert record.num_planes is not None

        round_type, force_marginal_loss, include_training_si_in_prior = self._base_round_type(record.round_type)
        if round_type != "S":
            super()._train_quantizer_or_load(delta_vec, record)
            return

        client_idx = record.client_id
        self._ensure_client_state(client_idx)
        server_recons = self.reconstruction_history.view(Access.SERVER, record)
        assert server_recons, "Sampled retrain requires server reconstruction history."

        prior_recons = server_recons if include_training_si_in_prior else ()
        quantizer = SampledWZQuantizerCancer(
            c_cfg=self.c_cfg,
            num_planes=record.num_planes,
            bins_per_plane=record.bins_per_plane,
            si_size=len(server_recons),
            marginal_loss=force_marginal_loss,
            **self.quantizer_kwargs,
            extra_si_for_prior=[recon.tensor for recon in prior_recons],
        )
        recon_tensors = [recon.tensor for recon in server_recons]
        quantizer.train_model_from_random_reconstruction_targets(recon_tensors, recon_tensors)
        self.frozen_quantizers[client_idx] = quantizer

        gc.collect()
        torch.cuda.empty_cache()

    def _compress(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        self._ensure_client_state(record.client_id)
        assert record.round_type is not None

        round_type = self._base_round_type(record.round_type)[0]
        if round_type != "S":
            return super()._compress(delta_vec, record)

        self._train_quantizer_or_load(delta_vec, record)
        quantizer = self.frozen_quantizers[record.client_id]
        assert isinstance(quantizer, SampledWZQuantizerCancer)
        assert isinstance(record, SampledCancerRecord)

        grad_prep, prep_metadata = quantizer.get_x_data(delta_vec)
        sample_count = self._sample_count(delta_vec.numel())
        seed = self._sample_seed(record, delta_vec.numel())
        sample_indices, remaining_indices = self._split_indices(seed, delta_vec.numel(), sample_count)

        sample_bins = quantizer._encoding_process(grad_prep, indices=sample_indices)
        sample_recon_prep = quantizer._decoding_process_prep(sample_bins, indices=sample_indices)
        sample_recon = quantizer._post_process_indexed(
            sample_recon_prep.squeeze(1), *prep_metadata, sample_indices, delta_vec.numel(),
        )
        self._record_sample_metrics(record, delta_vec[sample_indices], sample_recon)

        record.head_train_loss = quantizer.train_encoder_head_from_sample(sample_recon_prep, sample_bins)
        head_state = quantizer.encoder_head_state_dict()
        remaining_bins = quantizer._encoding_process(grad_prep, indices=remaining_indices)
        full_bins = self._merge_bins(sample_bins, remaining_bins, sample_indices, remaining_indices)

        payload_content = {
            "sample_seed": seed,
            "sample_count": sample_count,
            "sample_bins": sample_bins,
            "remaining_bins": remaining_bins,
            "prep_metadata": prep_metadata,
            "encoder_head_state": head_state,
        }

        sample_payload = {
            "sample_seed": seed,
            "sample_count": sample_count,
            "sample_bins": sample_bins,
            "prep_metadata": prep_metadata,
        }
        record.sampled_seed = seed
        record.sampled_count = sample_count
        record.sampled_fraction = sample_count / delta_vec.numel()
        record.sampled_data_size = get_obj_compressed_size(sample_payload) / (1024 ** 2)
        record.sampled_entropy_real_rate = record.sampled_data_size * (1024 ** 2) * 8 / delta_vec.numel()
        record.sampled_symbol_rate = record.sampled_data_size * (1024 ** 2) * 8 / sample_count
        record.head_state_size = get_obj_compressed_size(head_state) / (1024 ** 2)
        record.encoder_decoder_size = record.head_state_size
        record.meta_data_size = get_obj_compressed_size(prep_metadata) / (1024 ** 2)

        prior = quantizer._get_posterior(delta_vec, bins_vec_save_compute=full_bins)
        record.prior_rate = PriorCalculator.compute_rate_from_prior_tensor(prior, full_bins, quantizer.num_planes)
        marginal_prior = PriorCalculator.compute_marginal_prior(
            full_bins, quantizer.bins_per_plane, quantizer.num_planes,
        )
        record.marginal_rate = PriorCalculator.compute_rate_from_prior_tensor(
            marginal_prior, full_bins, quantizer.num_planes,
        )

        return payload_content

    def _decompress(self, payload: dict[str, Any], record: CancerRecord) -> torch.Tensor:
        if "sample_seed" not in payload:
            return super()._decompress(payload, record)

        client_idx = record.client_id
        self._ensure_client_state(client_idx)
        quantizer = self.frozen_quantizers[client_idx]
        assert quantizer is not None, f"Missing quantizer for client {client_idx}."

        sample_count = int(payload["sample_count"])
        sample_bins = payload["sample_bins"]
        remaining_bins = payload["remaining_bins"]
        vector_size = sample_bins.shape[1] + remaining_bins.shape[1]
        sample_indices, remaining_indices = self._split_indices(
            int(payload["sample_seed"]), vector_size, sample_count,
        )
        bins = self._merge_bins(sample_bins, remaining_bins, sample_indices, remaining_indices)

        reconst = quantizer.decoding_process((bins, payload["prep_metadata"]))
        self.reconstruction_history.commit(
            reconst.detach().to(device="cpu", dtype=torch.float16),
            record,
            self._reconstruction_access(record),
        )
        return reconst

    def _sample_count(self, vector_size: int) -> int:
        sampled_count = int(vector_size * self.c_cfg.sampled_round_fraction)
        sampled_count = max(self.c_cfg.sampled_round_min_count, sampled_count)
        sampled_count = min(self.c_cfg.sampled_round_max_count, sampled_count, vector_size)
        return sampled_count

    @staticmethod
    def _sample_seed(record: CancerRecord, vector_size: int) -> int:
        return int((record.round_id * 1_000_003 + record.client_id * 9_176 + vector_size) % (2**31 - 1))

    @staticmethod
    def _sample_indices(seed: int, vector_size: int, sample_count: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randperm(vector_size, generator=generator)[:sample_count].sort().values

    @classmethod
    def _split_indices(cls, seed: int, vector_size: int, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_indices = cls._sample_indices(seed, vector_size, sample_count)
        sample_mask = torch.zeros(vector_size, dtype=torch.bool)
        sample_mask[sample_indices] = True
        remaining_indices = torch.arange(vector_size, dtype=torch.long)[~sample_mask]
        return sample_indices, remaining_indices

    @staticmethod
    def _merge_bins(
        sample_bins: torch.Tensor,
        remaining_bins: torch.Tensor,
        sample_indices: torch.Tensor,
        remaining_indices: torch.Tensor,
    ) -> torch.Tensor:
        bins = torch.empty(
            (sample_bins.shape[0], sample_bins.shape[1] + remaining_bins.shape[1]),
            dtype=sample_bins.dtype,
        )
        bins[:, sample_indices] = sample_bins
        bins[:, remaining_indices] = remaining_bins
        return bins

    @staticmethod
    def _record_sample_metrics(
        record: SampledCancerRecord,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> None:
        temp_record = CompressionRecord(record.round_id, record.client_id, record.codec_class_used)
        record_reconstruction_metrics(original, reconstructed, temp_record)
        record.sampled_mse = temp_record.mse
        record.sampled_mape = temp_record.mape
        record.sampled_mspe_sqrt = temp_record.mspe_sqrt
        record.sampled_wmape = temp_record.wmape
        record.sampled_wmspe_sqrt = temp_record.wmspe_sqrt
