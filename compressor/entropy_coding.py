import zlib

import numpy as np


def entropy_coding(data, **kwargs):
    return zlib.compress(data.tobytes())

def entropy_decoding(encoded_data, dtype, **kwargs):
    return np.frombuffer(zlib.decompress(encoded_data), dtype=dtype)