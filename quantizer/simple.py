import numpy as np

def simple_quantize(data:np.ndarray):
    """
    :param data: normalized data to be quantized (0-1)
    :return: quantized data as float16
    """
    return data.astype(np.float16)

def simple_dequantize(quantized_data, dtype):
    de_q = quantized_data.astype(dtype)
    return de_q
