import zlib

import numpy as np


def entropy_coding(data):
    return zlib.compress(data.tobytes())

def entropy_decoding(encoded_data, dtype):
    return np.frombuffer(zlib.decompress(encoded_data), dtype=dtype)