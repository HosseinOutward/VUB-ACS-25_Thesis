from typing import List, Optional, Tuple, Any
import numpy as np
import torch

from FL_reworked.cancer_protocol import CancerCodec
from FL_reworked.cancer_quantizer import WZQuantizerCancer
from FL_reworked.run_fl import FLConfig


def get_normalization_factor(y: torch.Tensor) -> float:
    num_samples = 5
    sample_size = min(200_000, len(y))
    norm_facts = []
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y), size=sample_size, replace=True)
        y_sample = y[sample_indices]
        norm_fact_99 = torch.max(torch.abs(torch.quantile(y_sample.float(), .99))).item()
        norm_fact_1 = torch.max(torch.abs(torch.quantile(y_sample.float(), .01))).item()
        norm_facts.append([norm_fact_1, norm_fact_99])
    norm_fact = float(np.mean(norm_facts))

    assert norm_fact!=0

    return norm_fact


def get_outlier_factor(grad_flat_normal: torch.Tensor, outlier_threshold: float) -> Tuple:
    outlier_mask = torch.abs(grad_flat_normal) > outlier_threshold
    outlier_count = torch.sum(outlier_mask)

    if outlier_count==0:
        return np.array([], dtype=int), np, torch.array([])

    outlier_sign: np.ndarray = torch.sign(grad_flat_normal[outlier_mask]).cpu().numpy()
    outlier_max: float = float(
        torch.quantile(torch.abs(grad_flat_normal[outlier_mask])-outlier_threshold, .99)) / outlier_threshold
    outlier_positions: np.ndarray = np.where(outlier_mask.cpu().numpy())[0]

    assert outlier_max!=0

    return outlier_positions, outlier_max, outlier_sign


class WZQuantizerCancerWithDataPrep(WZQuantizerCancer):
    def __init__(self, vec_slices: List[slice], outlier_threshold:float=1.4, **kargs: Any) -> None:
        self.outlier_threshold: float = outlier_threshold
        self.vec_slices: List[slice] = vec_slices
        super().__init__(**kargs)

    def get_x_data(self, x_vec: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, Tuple]]:
        x_vec = super().get_x_data(x_vec)
        x_vec, norm_factors, outlier_param = self._apply_pre_process(
            x_vec, False, False)
        return x_vec, (norm_factors, outlier_param)

    def encoding_process(self, grad_vector: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        grad_vector_reshaped, data_prep_info = self.get_x_data(grad_vector)
        bins:torch.Tensor = self._encoding_process(grad_vector_reshaped)
        data_prep_info: Tuple[np.ndarray, float, np.ndarray]
        return bins, data_prep_info

    def decoding_process(self, payload_content: Tuple[np.ndarray, Any], batch_size: int = 500_000) -> torch.Tensor:
        bins_payload, encoding_extra_data = payload_content
        recons_grad = super().decoding_process(bins_payload, batch_size)
        res = self._post_process_grads(recons_grad, *encoding_extra_data)
        return res

    def train_model(self, x_vec: torch.Tensor, side_info_list: Optional[List[torch.Tensor]], batch_size: int = 50_000) -> None:
        if self.coding_model.marginal:
            assert side_info_list is None, "Marginal model expects an empty side_info_list for training."
            side_info_list = []
        else:
            assert len(side_info_list) > 0, "Conditional model requires side info for training."
        assert self.side_info_list_used in [None, 'P'], "This quantizer instance has already been trained."
        assert x_vec is not None, "Training data x_vec must be provided for training."

        self.side_info_list_used = side_info_list

        self.mspe_denom:float = torch.mean(x_vec ** 2).item() + 1e-8
        x_vec, _ = self.get_x_data(x_vec)
        side_info_list = self.get_si_data()
        self._train_model(x_vec, side_info_list, batch_size)

    def _apply_pre_process(self, _vector: torch.Tensor, ignore_normalization: bool, ignore_outliers: bool) -> Tuple:
        vector=_vector.clone()

        # normalization ----------
        norm_factors:List[float] = [1 for _ in self.vec_slices]
        if not ignore_normalization:
            norm_factors = [get_normalization_factor(vector[v_slc]) for v_slc in self.vec_slices]

        for i, v_slc in enumerate(self.vec_slices):
            vector[v_slc] /= norm_factors[i]

        # outlier ----------
        outlier_positions:np.ndarray
        outlier_max:Optional[float]
        outlier_sign:np.ndarray

        outlier_positions, outlier_max, outlier_sign = (np.array([], dtype=int), None, np.array([]))
        outlier_param = (outlier_positions, outlier_max, outlier_sign)
        if not ignore_outliers:
            outlier_positions, outlier_max, outlier_sign = get_outlier_factor(vector, self.outlier_threshold)
            outlier_param = (outlier_positions, outlier_max, outlier_sign)

        if len(outlier_positions) != 0:
            temp = vector[outlier_positions]
            vector[outlier_positions] = (torch.abs(temp) - self.outlier_threshold) * torch.sign(temp) / outlier_max

        return vector, norm_factors, outlier_param

    def _post_process_grads(self, vector: torch.Tensor, norm_factors: List, outlier_param: Tuple) -> torch.Tensor:
        outlier_positions:np.ndarray = outlier_param[0]
        outlier_max:float = outlier_param[1]
        outlier_sign:torch.Tensor = torch.from_numpy(outlier_param[2])

        # outlier ----------
        if len(outlier_positions)!=0:
            assert len(np.unique(outlier_sign)) in [1,2]
            assert outlier_sign.max() in [1,-1] and outlier_sign.min() in [1,-1]
            vector[outlier_positions] =\
                (torch.abs(vector[outlier_positions]) * outlier_max + self.outlier_threshold) * outlier_sign

        # normalization ----------
        for i, v_slc in enumerate(self.vec_slices):
            vector[v_slc] *= norm_factors[i]

        return vector

    def _get_posterior(self, x_vec: torch.Tensor, bins_vec_save_compute: Optional[Tuple] = None):
        """Override to use preprocessed data for hashing and handle tuple return from encoding."""
        # Extract bins from tuple if needed (this class returns (bins, extra_data))
        bins_only = bins_vec_save_compute[0] if isinstance(bins_vec_save_compute, tuple) else bins_vec_save_compute

        # Use preprocessed data for hashing to ensure consistency
        x_vec_preprocessed, _ = self.get_x_data(x_vec)

        # Call parent with preprocessed data
        return super()._get_posterior(x_vec_preprocessed.squeeze(), bins_only)


class CancerDataPrepCodec(CancerCodec):
    def __init__(self, fl_cfg: FLConfig, vec_slices: List[slice]) -> None:
        super().__init__(fl_cfg)
        self.vec_slices:List[slice] = vec_slices

    def get_new_quantizer(self, **kargs: Any) -> WZQuantizerCancer:
        return WZQuantizerCancerWithDataPrep(self.vec_slices, **kargs)

