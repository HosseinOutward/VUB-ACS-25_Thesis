
import numpy as np
import rans.rANSCoder as rans
import numba.typed

batch_size=100_000


def rans_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    # return data_symbols

    encoder = rans.Encoder()
    for s in data_symbols.copy():
        encoder.encode_symbol(probs_per_bin.copy(), s)
    res = encoder.get_encoded()
    return np.array(res, dtype=eval(f'np.{res._dtype}'))


def rans_decode(encoded_state, freqs:np.ndarray, length_decoded:int):
    # return encoded_state

    if isinstance(encoded_state, np.ndarray):
        encoded_state = numba.typed.List(encoded_state)
    assert str(encoded_state._dtype)=='uint32'

    decoder = rans.Decoder(encoded_state.copy())
    decoded_data = []
    for _ in range(length_decoded):
        decoded_data.insert(0, decoder.decode_symbol(freqs.copy()))
    return np.array(decoded_data)


def rans_batch_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    from multiprocessing import Pool
    from functools import partial
    import os

    batches = [data_symbols[i:i+batch_size] for i in range(0, len(data_symbols), batch_size)]

    max_cpu_processes = os.cpu_count() - 1
    with Pool(max_cpu_processes) as p:
        encoded_batches = p.map(partial(rans_encode, probs_per_bin=probs_per_bin), batches)
    return encoded_batches


def rans_batch_decode(encoded_state, freqs:np.ndarray, length_decoded:int) -> np.ndarray:
    from multiprocessing import Pool
    import os

    num_batches = len(encoded_state)
    batch_sizes = [batch_size] * (num_batches - 1)
    last_batch_size = length_decoded % batch_size
    if last_batch_size == 0 and length_decoded > 0:
        last_batch_size = batch_size
    if num_batches > 0:
        batch_sizes.append(last_batch_size)

    args_for_starmap = [(e, freqs, bs) for e, bs in zip(encoded_state, batch_sizes)]

    max_cpu_processes = os.cpu_count() - 1
    with Pool(max_cpu_processes) as p:
        decoded_batches = p.starmap(rans_decode, args_for_starmap)

    return np.concatenate(decoded_batches)


if __name__ == '__main__':
    import time
    from components.broadcast_components.reporting_utilities import get_obj_size
    from components.broadcast_components.broadcasting_process.WZ_broadcast import compress_data_list

    data = np.abs(np.random.normal(0,3, size=1_831_000)//1)
    data = np.clip(data, 0, 10).astype(np.uint8)
    probs = np.array(
            [uq/len(data) for uq in np.unique(data, return_counts=True)[1]]
        ).astype(np.float16).astype(np.float32)

    t0 = time.time()
    # encoded = rans_encode(data, probs)
    # decoded = rans_decode(encoded, probs, len(data))

    temp=rans_batch_encode(data, probs)
    decoded = rans_batch_decode(temp, probs, len(data))
    t1 = time.time()

    byte_size = get_obj_size(compress_data_list(temp)) / (1024 * 1024)
    org_byte_size = get_obj_size(compress_data_list(data)) / (1024 * 1024)

    print("Decoded Data:", np.sum(np.abs(data-decoded)),
          "speed:", int(len(data) // (t1 - t0)), 'sym/sec', f'({t1 - t0:.2f}s)',
          f"encoded size: {byte_size:.2f}MB (org: {org_byte_size:.2f}MB)",)
