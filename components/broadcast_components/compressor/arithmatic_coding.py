import numpy as np


def arithmetic_encode(list_of_ints, symbol_count, out_element_bit_size=64):
    encode_dtype = {'64': np.uint64, '32': np.uint32,
                    '16': np.uint16, '8': np.uint8}[str(out_element_bit_size)]
    # symbol_count = len(np.unique(list_of_ints))
    bit_per_symbol = np.log2(symbol_count)
    symbol_per_out_element = int(out_element_bit_size // bit_per_symbol)
    out_element_count = int((len(list_of_ints) + symbol_per_out_element - 1) // symbol_per_out_element)
    out_res = np.zeros(out_element_count, dtype=encode_dtype)

    for i in range(0, len(list_of_ints), symbol_per_out_element):
        chunk = list_of_ints[i:i + symbol_per_out_element]
        out_idx = i // symbol_per_out_element
        acc = 0
        for j, v in enumerate(chunk):
            assert v < symbol_count, "Index out of bounds for symbol count"
            acc += int(v) * (symbol_count ** j)
        out_res[out_idx] = acc
    return out_res


def arithmetic_decode(encoded_array, symbol_count, orig_size):
    bit_per_symbol = np.log2(symbol_count)
    out_element_bit_size = encoded_array.dtype.itemsize * 8
    symbol_per_out_element = int(out_element_bit_size // bit_per_symbol)

    orig_array = []
    for x in encoded_array:
        temp = int(x)
        for _ in range(symbol_per_out_element):
            orig_array.append(temp % symbol_count)
            temp //= symbol_count
    return np.array(orig_array[:orig_size], dtype=int)

# encod = arithmetic_encode(array, out_element_bit_size=64)
# recons = arithmetic_decode(encod, sym_count, len(array))