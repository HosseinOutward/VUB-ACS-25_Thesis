import numpy as np

min_v,max_v = -0.017, +0.017

def simple_quantize(data, **kwargs):
    quantized = np.clip(data, min_v, max_v)
    quantized = (quantized - min_v) / (max_v - min_v)
    quantized = np.round(quantized * 255)
    return quantized.astype(np.int8)


def simple_dequantize(quantized_data, **kwargs):
    de_quantized = quantized_data.astype(np.float32)
    de_quantized = (de_quantized / 255) * (max_v - min_v) + min_v
    return de_quantized