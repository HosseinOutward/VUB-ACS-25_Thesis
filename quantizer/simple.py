import numpy as np
import warnings

# Clamp range for your gradients
#todo: make this configurable as an input
min_v, max_v = -0.017, +0.017


def simple_quantize(data, **kwargs):
    """
    Uniform 8‑bit quantisation that emits **unsigned** bytes (0‒255).
    """
    warnings.warn("**********************************************************************\n"
            "**********************************************************************\n"
            "the maximum value of the quantizer is hardcoded to 0.017, this should be configurable")

    q = np.clip(data, min_v, max_v)
    q = (q - min_v) / (max_v - min_v)      # → 0‒1
    q = np.round(q * 255).astype(np.uint8) # → 0‒255 as uint8
    return q


def simple_dequantize(quantized_data, **kwargs):
    """
    Inverse of `simple_quantize`.
    """
    warnings.warn("**********************************************************************\n"
            "**********************************************************************\n"
            "the maximum value of the quantizer is hardcoded to 0.017, this should be configurable")

    de_q = quantized_data.astype(np.float32)
    de_q = (de_q / 255) * (max_v - min_v) + min_v
    return de_q
