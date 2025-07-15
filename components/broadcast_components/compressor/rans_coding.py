import numpy as np
import rans.rANSCoder as rans
import numba.typed


def rans_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    # return data_symbols

    encoder = rans.Encoder()
    for s in data_symbols.copy():
        encoder.encode_symbol(probs_per_bin.copy(), s)
    return encoder.get_encoded()


def rans_decode(encoded_state, freqs:np.ndarray, length_decoded:int):
    # return encoded_state

    if isinstance(encoded_state, np.ndarray):
        encoded_state = numba.typed.List(encoded_state)
    assert str(encoded_state._dtype)=='uint32'

    decoder = rans.Decoder(encoded_state.copy())
    decoded_data = []
    for _ in range(length_decoded):
        decoded_data.insert(0, decoder.decode_symbol(freqs.copy()))
    return decoded_data


if __name__ == '__main__':
    data = np.array([0, 1, 2, 3, 4])
    probs = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

    encoded = rans_encode(data, probs)
    print("Encoded Data:", encoded)

    encoded = np.array(encoded, dtype=eval('np.'+str(encoded._dtype)))

    decoded = rans_decode(encoded, probs, len(data))
    print("Decoded Data:", decoded)