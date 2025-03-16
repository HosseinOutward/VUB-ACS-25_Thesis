import pickle
from typing import List
import numpy as np
from quantizer.simple import simple_quantize, simple_dequantize
from compressor.entropy_coding import entropy_coding, entropy_decoding

config = {
    "MODE_ENCODER": "entropy",
    "MODE_quantizer": "8bit",
}
config = type('config', (object,), config)  # Convert dict to a class for attribute access


# ---------- SW Encoding/Decoding (Simulated) ----------

def grad_encoder(data, **kwargs):
    if config.MODE_ENCODER == 'sw':
        raise NotImplementedError("SW encoding not implemented yet.")

    elif config.MODE_ENCODER == 'entropy':
        encoded_data = entropy_coding(data, **kwargs)

    elif "raw" in config.MODE_ENCODER:
        encoded_data = data

    return encoded_data


def grad_decoder(encoded_data: List[np.ndarray], **kwargs):
    if config.MODE_ENCODER == 'sw':
        raise NotImplementedError("SW encoding not implemented yet.")

    elif config.MODE_ENCODER == 'entropy':
        if kwargs.get('dtype') is None:
            kwargs['dtype'] = np.float64 if config.MODE_quantizer != '8bit' else np.uint8
        decoded_data = np.array([entropy_decoding(a, **kwargs) for a in encoded_data])

    elif "raw" in config.MODE_ENCODER:
        decoded_data = np.array(encoded_data)

    return decoded_data


# ---------- WZ Quantization/Dequantization ----------

def grad_quantizer(data, **kwargs):
    data = np.concatenate([b.flatten() for a, b in data])

    if config.MODE_quantizer == '8bit':
        quantized = simple_quantize(data, **kwargs)

    elif config.MODE_quantizer == 'wz':
        raise NotImplementedError("WZ quantization not implemented yet.")

    elif config.MODE_quantizer == 'raw':
        quantized = data

    return quantized


def grad_de_quantizer(quantized_data, original_struct, **kwargs):
    if config.MODE_quantizer == '8bit':
        de_quantized = np.array([simple_dequantize(a, **kwargs) for a in quantized_data])

    elif config.MODE_quantizer == 'wz':
        raise NotImplementedError("WZ de_quantization not implemented yet.")

    elif config.MODE_quantizer == 'raw':
        de_quantized = quantized_data

    final = []
    for i in range(len(de_quantized)):
        start = 0
        final.append([])
        for j, (name, org_shape) in enumerate(original_struct):
            size = np.prod(org_shape)
            final[-1].append([original_struct[j][0], de_quantized[i][start:start + size].reshape(org_shape)])
            start += size

    return final


# ---------- Testing the Components ----------

def test_components():
    print("\n== Testing Components ==")

    test_grads = pickle.load(open("testing_model_grad.pkl", "rb"))

    quant, encoded = [], []
    for i, grad_w in enumerate(test_grads):
        quant.append(grad_quantizer(grad_w.copy()))
        encoded.append(grad_encoder(quant[i].copy()))
    quant = np.array(quant).astype(np.float64)

    decoded = grad_decoder(encoded).astype(np.float64)
    temp = [[a, b.shape] for a, b in test_grads[0].copy()]
    de_quant = grad_de_quantizer(decoded, temp)

    # Calculate sizes
    kb_data_size = sum([len(a) / 1024 / 1024 for a in encoded])
    original_size = sum([a.nbytes / 1024 / 1024 for d in test_grads for _, a in d])

    # Calculate errors
    temp = grad_de_quantizer(quant, temp)
    temp = np.concatenate([(temp[i][j][1] - test_grads[i][j][1]).flatten()
                           for i in range(len(test_grads)) for j in range(len(test_grads[i]))])
    quanti_error = abs(temp).mean()

    temp = np.array([np.concatenate([(test_grads[i][j][1]).flatten()
                                     for j in range(len(test_grads[i]))]) for i in range(len(test_grads))])
    encode_error = [grad_encoder(a) for a in temp]
    encode_error = abs(grad_decoder(encode_error, dtype=np.float32) - temp).mean()

    max_abs_v = np.percentile(abs(temp), 99.85)

    temp = np.concatenate([(de_quant[i][j][1] - test_grads[i][j][1]).flatten()
                           for i in range(len(de_quant)) for j in range(len(test_grads[i]))])
    total_error = abs(temp).mean()

    # Print results
    print(f"size of encoded data: {kb_data_size:.3f} KB",
          f"original data size: {original_size:.3f} KB",
          f"\nratio e/o: {kb_data_size / original_size:.3f}; o/e {original_size / kb_data_size:.3f}", )

    print(f"\nquantization error: {quanti_error / max_abs_v:.5f}%",
          f"\nencoding error: {encode_error / max_abs_v:.5f}%",
          f"\ncombo error: {max(total_error - quanti_error - encode_error, 0) / max_abs_v:.5f}%", )
    print(f"\nTotal error: {total_error / max_abs_v:.5f}%")


# ---------- Main Routine (for direct execution) ----------

if __name__ == '__main__':
    test_components()
