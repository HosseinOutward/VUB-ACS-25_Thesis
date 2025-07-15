import numpy as np
import rans.rANSCoder as rans
import numba.typed


def rans_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    encoder = rans.Encoder()
    for s in data_symbols.copy():
        encoder.encode_symbol(probs_per_bin.copy(), s)
    return encoder.get_encoded()


def rans_decode(encoded_state, freqs:np.ndarray, length_decoded:int):
    if isinstance(encoded_state, np.ndarray):
        encoded_state = numba.typed.List(encoded_state)

    decoder = rans.Decoder(encoded_state.copy())
    decoded_data = [decoder.decode_symbol(freqs.copy()) for _ in range(length_decoded)][::-1]
    return decoded_data
