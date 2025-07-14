import numpy as np

def simple_quantize(data:np.ndarray):
    return data.astype(np.float32)

def simple_dequantize(quantized_data, dtype):
    de_q = quantized_data.astype(dtype)
    return de_q
