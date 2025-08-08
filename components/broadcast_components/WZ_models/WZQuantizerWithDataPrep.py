import numpy as np

from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol \
    import outlier_normalization, outlier_de_normalization


def outlier_normalization(grad_flat_normal, outlier_threshold=1.5, margin_gap=0.25):
    # if outlier_threshold != 1.5: print('warning, using non default outlier threshold!')
    assert outlier_threshold>margin_gap>0

    # Separate outliers with abs value > outlier_threshold
    outlier_mask = np.abs(grad_flat_normal) > outlier_threshold

    if sum(outlier_mask)!=0:
        return [], [], 0, None

    # normalize the outlier values
    if isinstance(grad_flat_normal, np.ndarray):
        outlier_values = grad_flat_normal[outlier_mask].copy()
    else:
        outlier_values = grad_flat_normal[outlier_mask].detach().cpu().clone()

    # close the gap, but leave a small gap to prevent zeroing outliers
    sign_v = np.sign(outlier_values)
    normalized_outlier_values = np.abs(outlier_values) - outlier_threshold
    outlier_max = np.percentile(normalized_outlier_values, 99.99) / (outlier_threshold-margin_gap)
    normalized_outlier_values /= outlier_max
    normalized_outlier_values += margin_gap
    normalized_outlier_values *= sign_v

    outlier_count = np.sum(outlier_mask)

    outlier_positions = np.where(outlier_mask)[0]

    return normalized_outlier_values, outlier_positions, outlier_count, outlier_max


def outlier_de_normalization(res_vector, outlier_count, outlier_max, outlier_threshold=1.5, margin_gap=0.25):
    # if outlier_threshold != 1.5: print('warning, using non default outlier threshold!')
    assert outlier_threshold>margin_gap>0

    outlier_values = res_vector[-outlier_count:].copy() if isinstance(res_vector, np.ndarray) \
                    else res_vector[-outlier_count:].detach().cpu().clone()

    sign_v = np.sign(outlier_values)
    outlier_values = np.abs(outlier_values) - margin_gap
    outlier_values *= outlier_max
    outlier_values = outlier_values + outlier_threshold
    outlier_values *= sign_v

    return outlier_values


class QuantizerWithDataPrep:
    def __init__(self, wz_quantizer, outlier_threshold):
        self.wz_quantizer = wz_quantizer
        self.outlier_threshold=1.5
        self.margin_gap=0.25

    def get_prior_and_softcodes(self, *args, **kwargs):
        return prior, soft_codes

    def encoding_process(self, grad_vector, *args, **kwargs):
        return bins.to(dtype)

    def decoding_process(self, quantized_data, *args, **kwargs):
        return res

    def train_model(self, input_data, *args, **kwargs):
        pass

    def encoding_process(self, y, ):
        y = y.copy()
        normalized_outlier_values, outlier_positions, outlier_count, outlier_max = outlier_normalization(
            y, outlier_threshold=outlier_threshold, margin_gap=margin_gap)
        outlier_positions = outlier_positions
        outlier_count = outlier_count
        outlier_max = outlier_max
        y[outlier_positions] = normalized_outlier_values
        return y

    def decoding_process(self, y_pred, ):
        y_pred = y_pred.copy()

        normalized_outlier_values = y_pred[outlier_positions]
        denormalized_outliers = outlier_de_normalization(
            normalized_outlier_values, outlier_count, outlier_max,
            outlier_threshold=outlier_threshold, margin_gap=margin_gap)
        y_pred[outlier_positions] = denormalized_outliers

        return y_pred

    def outlier_normalization(self, grad_flat_normal, outlier_threshold=1.5, margin_gap=0.25):
        # if outlier_threshold != 1.5: print('warning, using non default outlier threshold!')
        assert outlier_threshold > margin_gap > 0

        # Separate outliers with abs value > outlier_threshold
        outlier_mask = np.abs(grad_flat_normal) > outlier_threshold

        # normalize the outlier values
        outlier_values = grad_flat_normal[outlier_mask].copy() if isinstance(grad_flat_normal, np.ndarray) \
            else grad_flat_normal[outlier_mask].detach().cpu().clone()

        assert len(outlier_values) != 0

        # close the gap, but leave a small gap to prevent zeroing outliers
        sign_v = np.sign(outlier_values)
        normalized_outlier_values = np.abs(outlier_values) - outlier_threshold
        outlier_max = np.percentile(normalized_outlier_values, 99.99) / (outlier_threshold - margin_gap)
        normalized_outlier_values /= outlier_max
        normalized_outlier_values += margin_gap
        normalized_outlier_values *= sign_v

        outlier_count = np.sum(outlier_mask)

        outlier_positions = np.where(outlier_mask)[0]

        return normalized_outlier_values, outlier_positions, outlier_count, outlier_max

    def outlier_de_normalization(self, res_vector, outlier_count, outlier_max, outlier_threshold=1.5, margin_gap=0.25):
        # if outlier_threshold != 1.5: print('warning, using non default outlier threshold!')
        assert outlier_threshold > margin_gap > 0

        outlier_values = res_vector[-outlier_count:].copy() if isinstance(res_vector, np.ndarray) \
            else res_vector[-outlier_count:].detach().cpu().clone()

        sign_v = np.sign(outlier_values)
        outlier_values = np.abs(outlier_values) - margin_gap
        outlier_values *= outlier_max
        outlier_values = outlier_values + outlier_threshold
        outlier_values *= sign_v

        return outlier_values