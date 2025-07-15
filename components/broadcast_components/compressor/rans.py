from typing import List
import numpy as np
import rans.rANSCoder as rans


def rans_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    encoder = rans.Encoder()
    for s in data_symbols:
        encoder.encode_symbol(probs_per_bin, s)
    return encoder.get_encoded()


def rans_decode(encoded_state:np.ndarray, freqs:np.ndarray, length_decoded:int):
    decoder = rans.Decoder(encoded_state.copy())
    decoded_data = [decoder.decode_symbol(freqs) for _ in range(length_decoded)][::-1]
    return decoded_data
