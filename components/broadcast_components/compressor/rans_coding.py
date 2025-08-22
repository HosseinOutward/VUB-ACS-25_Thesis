import numpy as np
import rans.rANSCoder as rans
import numba.typed

batch_size=1_000_000


def rans_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    assert len(data_symbols)==len(probs_per_bin), "data_symbols and probs_per_bin must have the same length"

    encoder = rans.Encoder()
    for s, probs in zip(data_symbols, probs_per_bin):
        encoder.encode_symbol(probs, s)
    res = encoder.get_encoded()
    return np.array(res, dtype=eval(f'np.{res._dtype}'))


def rans_decode(encoded_state, freqs:np.ndarray, length_decoded:int):
    if isinstance(encoded_state, np.ndarray):
        encoded_state = numba.typed.List(encoded_state)
    assert str(encoded_state._dtype)=='uint32'

    decoder = rans.Decoder(encoded_state.copy())

    freqs=freqs.copy()
    decoded_data = [None] * length_decoded
    for i in range(length_decoded - 1, -1, -1):
        decoded_data[i] = decoder.decode_symbol(freqs[i])
    return np.array(decoded_data)


def rans_batch_encode(data_symbols:np.ndarray, probs_per_bin:np.ndarray) -> np.ndarray:
    # return data_symbols

    from multiprocessing import Pool
    import os

    batches = [(data_symbols[i:i+batch_size], probs_per_bin[i:i+batch_size])
                    for i in range(0, len(data_symbols), batch_size)]

    max_cpu_processes = os.cpu_count() - 1
    with Pool(max_cpu_processes) as p:
        encoded_batches = p.starmap(rans_encode, batches)
    return encoded_batches


def rans_batch_decode(encoded_state, freqs:np.ndarray, length_decoded:int) -> np.ndarray:
    # return encoded_state

    from multiprocessing import Pool
    import os

    num_batches = len(encoded_state)
    batch_sizes = [batch_size] * (num_batches - 1)
    last_batch_size = length_decoded % batch_size
    if last_batch_size == 0 and length_decoded > 0:
        last_batch_size = batch_size
    if num_batches > 0:
        batch_sizes.append(last_batch_size)

    freqs_batches = [freqs[i:i+batch_size] for i in range(0, len(freqs), batch_size)]

    args_for_starmap = [(e, freqs_batches[i], bs) for i, (e, bs) in enumerate(zip(encoded_state, batch_sizes))]

    max_cpu_processes = os.cpu_count() - 1
    with Pool(max_cpu_processes) as p:
        decoded_batches = p.starmap(rans_decode, args_for_starmap)

    return np.concatenate(decoded_batches)


if __name__ == '__main__':
    import time
    from components.broadcast_components.broadcasting_process.broadcast_reporting_utilities import get_obj_size
    from components.broadcast_components.broadcasting_process.ServerTrainingPerRoundProtocol import compress_data_list

    np.random.seed(0)

    data = np.abs(np.random.normal(0,3, size=2_831_000)//1)
    data = np.clip(data, 0, 10).astype(np.uint8)
    probs = np.array(
            [uq/len(data) for uq in np.unique(data, return_counts=True)[1]]
        ).astype(np.float16).astype(np.float32)
    probs = np.array([probs]*len(data))

    probs += np.random.random(size=probs.shape)
    probs /= np.sum(probs, axis=1, keepdims=True)

    # add a row of zeros to individual probs (second dim) to test its effects
    # probs = np.concatenate([probs, np.zeros((len(probs), 1), dtype=probs.dtype)], axis=1)

    org_byte_size = get_obj_size(compress_data_list(data)) / (1024 * 1024)
    print(f"\n      org: {org_byte_size:.2f}MB")

    # Test single encode/decode ----------------
    print('\n\n************\ntesting pooling...')
    t0 = time.time()
    encoded=rans_batch_encode(data.copy(), probs.copy())
    decoded = rans_batch_decode(encoded, probs.copy(), len(data))
    byte_size = get_obj_size(compress_data_list(encoded)) / (1024 * 1024)
    t1 = time.time()

    print("Decoded Data error:", np.sum(np.abs(data-decoded)),
          "\nspeed:", int(len(data) // (t1 - t0)), 'sym/sec', f'({t1 - t0:.2f}s)',
          f"\nencoded size: {byte_size:.2f}MB (x{org_byte_size/byte_size:.2f})",)


    # Test single encode/decode ----------------
    print('\n\n************\nTesting single encode/decode...')
    t0 = time.time()
    encoded = rans_encode(data.copy(), probs.copy())
    decoded = rans_decode(encoded, probs.copy(), len(data))
    byte_size = get_obj_size(compress_data_list(encoded)) / (1024 * 1024)
    t1 = time.time()

    print("Decoded Data error:", np.sum(np.abs(data-decoded)),
          "\nspeed:", int(len(data) // (t1 - t0)), 'sym/sec', f'({t1 - t0:.2f}s)',
          f"\nencoded size: {byte_size:.2f}MB (x{org_byte_size/byte_size:.2f})",)

    # Test single encode/decode ----------------
    print('\n\n************\nTesting batching without parallelization encode/decode...')
    t0 = time.time()
    byte_size=0
    temp=0
    for i in range(0, len(data)+batch_size-1, batch_size):
        encoded = rans_encode(data[i:i+batch_size], probs[i:i+batch_size])
        decoded = rans_decode(encoded, probs[i:i+batch_size], len(data[i:i+batch_size]))
        byte_size+= get_obj_size(compress_data_list(encoded)) / (1024 * 1024)
        temp+=np.sum(np.abs(data[i:i+batch_size] - decoded))
    t1 = time.time()

    print("Decoded Data error:", temp,
          "\nspeed:", int(len(data) // (t1 - t0)), 'sym/sec', f'({t1 - t0:.2f}s)',
          f"\nencoded size: {byte_size:.2f}MB (x{org_byte_size/byte_size:.2f})",)
