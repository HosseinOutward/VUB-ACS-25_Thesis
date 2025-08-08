import numpy as np

from components.broadcast_components.WZ_models.WZ_quantizer import WZQuantizer


def get_normalization_factor(y: np.ndarray):
    num_samples = 5
    sample_size = min(200_000, len(y))
    norm_facts = []
    for _ in range(num_samples):
        sample_indices = np.random.choice(len(y), size=sample_size, replace=True)
        y_sample = y[sample_indices]
        norm_fact_sample = np.max(np.abs(np.percentile(y_sample, [1, 99.])))
        norm_facts.append(norm_fact_sample)
    norm_fact = np.mean(norm_facts)

    return norm_fact


#%%
def get_outlier_factor(grad_flat_normal, outlier_threshold=1.5):
    outlier_mask = np.abs(grad_flat_normal) > outlier_threshold
    outlier_count = np.sum(outlier_mask)

    if outlier_count==0:
        return [], None, []

    outlier_sign = np.sign(grad_flat_normal[outlier_mask])
    outlier_max = np.percentile(np.abs(grad_flat_normal[outlier_mask])-outlier_threshold, 99) / outlier_threshold
    outlier_positions = np.where(outlier_mask)[0]

    return outlier_positions, outlier_max, outlier_sign


#%%
class QuantizerWithDataPrep(WZQuantizer):
    def __init__(self, *args, outlier_threshold=1.5, vec_slices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.outlier_threshold = outlier_threshold
        self.vec_slices = vec_slices if vec_slices is not None else [slice(None)]

    def _apply_pre_process(self, vector, normal_param=None, outlier_param=None):
        vector = vector.copy()

        # normalization ----------
        if normal_param is not None:
            norm_factors = normal_param
        else:
            norm_factors = [get_normalization_factor(vector[v_slc]) for v_slc in self.vec_slices]
            normal_param = norm_factors

        for i, v_slc in enumerate(self.vec_slices):
            vector[v_slc] /= norm_factors[i]

        # outlier ----------
        if outlier_param is not None:
            outlier_positions, outlier_max, _ = outlier_param
        else:
            outlier_positions, outlier_max, outlier_sign = get_outlier_factor(vector, self.outlier_threshold)
            outlier_param = (outlier_positions, outlier_max, outlier_sign)

        if len(outlier_positions) != 0:
            temp = vector[outlier_positions]
            vector[outlier_positions] = (np.abs(temp) - self.outlier_threshold) * np.sign(temp) / outlier_max

        return vector, normal_param, outlier_param

    def _post_process_grads(self, vector, normal_param, outlier_param):
        norm_factors = normal_param
        outlier_positions, outlier_max, outlier_sign = outlier_param
        # outlier ----------
        if len(outlier_positions) != 0:
            vector[outlier_positions] =\
                (np.abs(vector[outlier_positions]) * outlier_max + self.outlier_threshold)*outlier_sign

        # normalization ----------
        for i, v_slc in enumerate(self.vec_slices):
            vector[v_slc] *= norm_factors[i]

        return vector

    def encoding_process(self, grad_vector, *args, **kwargs):
        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)

        bins, temp = super().encoding_process(grad_vector, *args, **kwargs)
        assert temp is None

        return bins, (normal_param, outlier_param)

    def decoding_process(self, quantized_data, side_info_data_list, encoding_extra_data=None, batch_size=500_000):
        res = super().decoding_process(quantized_data, side_info_data_list, None, batch_size)
        res = self._post_process_grads(res, *encoding_extra_data)
        return res

    def get_prior_and_softcodes(self, grad_vector, side_info_data_list, batch_size=500_000):
        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)
        return super().get_prior_and_softcodes(grad_vector, side_info_data_list, batch_size)

    def train_model(self, grad_vector, side_info_data_list, *args, **kwargs):
        grad_vector, normal_param, outlier_param = self._apply_pre_process(grad_vector)
        super().train_model(grad_vector, side_info_data_list, *args, **kwargs)
