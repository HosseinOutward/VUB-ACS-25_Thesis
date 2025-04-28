import pickle
from typing import List, Tuple
import numpy as np

from quantizer.simple import simple_dequantize, simple_quantize
# from quantizer.wz_quant import wz_quantizer, wz_de_quantizer

from compressor.entropy_coding import entropy_coding, entropy_decoding

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer

# todo: remove this block
config = {
    "MODE_ENCODER": "raw",
    "MODE_quantizer": "raw",
}
config['dtype'] = np.float32 if config['MODE_quantizer'] == 'raw' else np.uint8
config = type("config", (object,), config)  # for attribute-style access


# -----------------------------------------------------------------------------
# Federated model wrapper
# -----------------------------------------------------------------------------
class FederatedModelWrapper(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super(FederatedModelWrapper, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.latest_parameters = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.loss_fn(output, target)
        return loss

    def on_before_optimizer_step(self, optimizer: Optimizer): # --->  important part
        """
        This is the hook that is called before the optimizer step.
        """
        self.latest_parameters = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            self.latest_parameters.append([name, param.grad.cpu().detach().numpy()])

        return super().on_before_optimizer_step(optimizer)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0005)


# -----------------------------------------------------------------------------
# Encoding / decoding
# -----------------------------------------------------------------------------
def grad_encoder(data: np.ndarray, method=None, **kwargs) -> bytes:
    """
    Worker encoder to compresses the data using the specified encoding method from config.
    :param data: 1D array of quantized data to be encoded. (gradients flattened)
    :param method: Encoding method. If None, uses the default from config.
    :param kwargs: Additional arguments for the specified encoding method.
    :return: Encoded data as byte stream.
    """

    method = config.MODE_ENCODER if method is None else method

    # switch
    if method == "sw":
        raise NotImplementedError("SW encoding not fully implemented yet. "
                                  "(for test use entropy since sw is lossless too)")
    elif method == "entropy":
        return entropy_coding(data, **kwargs)
    elif "raw" in method:
        return data.tobytes()


def grad_decoder(encoded_data: List[bytes], out_dtype=None, method=None, **kwargs) -> np.ndarray:
    """
    Server decoder to decompresses the data using the specified decoding method from config.
    :param encoded_data: List of encoded data (byte streams).
    :param out_dtype: Output data type. If None, uses the default from config.
    :param method: Decoding method. If None, uses the default from config.
    :param kwargs: Additional arguments for the specified decoding method.
    :return: Decoded data as a numpy array.
    """

    method = config.MODE_ENCODER if method is None else method
    out_dtype = config.dtype if out_dtype is None and config.dtype is not None else out_dtype

    # switch
    if method == "sw":
        raise NotImplementedError("SW encoding not fully implemented yet. "
                                  "(for test use entropy since sw is lossless too)")
    elif method == "entropy":
        return np.array([entropy_decoding(a, dtype=out_dtype, **kwargs) for a in encoded_data])
    elif "raw" in method:
        return np.array([np.frombuffer(a, dtype=out_dtype) for a in encoded_data])


# -----------------------------------------------------------------------------
# Quantisation / de‑quantisation
# -----------------------------------------------------------------------------
def grad_quantizer(data: List[Tuple[str, np.ndarray]], method=None, **kwargs) -> np.ndarray:
    """
    Worker quantiser to quantise the data using the specified quantisation method from config.
    :param data: List of gradients (name, array) to be quantised.
    :param method: Quantisation method. If None, uses the default from config.
    :param kwargs: Additional arguments for the specified quantisation method.
    :return: Quantised data as a numpy array.
    """

    method = config.MODE_quantizer if method is None else method

    data = np.concatenate([b.flatten() for _, b in data])

    # switch
    if method == "8bit":
        return simple_quantize(data, **kwargs)  # <-- now uint8
    elif method == "wz":
        raise NotImplementedError("WZ quantisation not implemented yet.")
        # return wz_quantizer(data)
    elif method == "raw":
        return data


def grad_de_quantizer(quantized_data: List[np.ndarray],
                      original_struct: List[Tuple[str, Tuple[int, int]]],
                      method=None, **kwargs) -> List[Tuple[str, np.ndarray]]:
    """
    Server de‑quantiser to de‑quantise the data using the specified de‑quantisation method from config.
    :param quantized_data: List of quantised data (1D arrays).
    :param original_struct: List of tuples containing the original structure (name, shape).
    :param method: De‑quantisation method. If None, uses the default from config.
    :param kwargs: Additional arguments for the specified de‑quantisation method.
    :return: De‑quantised data as a list of tuples (name, array).
    """

    method = config.MODE_quantizer if method is None else method

    # switch
    if method == "8bit":
        de_q = np.array([simple_dequantize(a, **kwargs) for a in quantized_data])
    elif method == "wz":
        raise NotImplementedError("WZ de‑quantisation not implemented yet.")
        # return wz_de_quantizer(data)
    elif method == "raw":
        de_q = quantized_data

    # rebuild original list‑of‑lists structure
    final = []
    for i in range(len(de_q)):
        start = 0
        final.append([])
        for j, (_, org_shape) in enumerate(original_struct):
            size = np.prod(org_shape)
            final[-1].append([
                original_struct[j][0],
                de_q[i][start: start + size].reshape(org_shape)
            ])
            start += size
    return final


# -----------------------------------------------------------------------------
# Component test
# -----------------------------------------------------------------------------
def test_components():
    """
    Test the components of the system: quantisation, encoding, and decoding.
    This function performs the following steps:
    1. Load the test gradients from a pickle file.
    2. Individually quantise the gradients using the specified quantisation method.
    3. Individually encode the quantised gradients using the specified encoding method.
    4. Decode the entire encoded gradients back to their original form.
    5. De-quantise the decoded gradients back to their original form.
    6. Calculate the sizes of the encoded data and the original data.
    7. Calculate the errors (total error, quantisation error, compression error and error caused by combination).
    """

    print("\n== Testing Components for 1 round w 5 workers ==")
    print(f'{config.MODE_ENCODER=}, {config.MODE_quantizer=}\n')

    test_grads = pickle.load(open("testing_model_grad.pkl", "rb"))

    # ---- quantise & encode ---------------------------------------------------
    quant = [grad_quantizer(g.copy()) for g in test_grads]
    encoded = [grad_encoder(q.copy()) for q in quant]
    decoded = grad_decoder(encoded)

    template = [[name, arr.shape] for name, arr in test_grads[0]]
    de_quant = grad_de_quantizer(decoded, template)

    # ------------------------------ Report ------------------------------------
    # ---- sizes ---------------------------------------------------------------
    kb_data_size = sum(len(a) for a in encoded) / 1024 / 1024
    original_size = sum(arr.nbytes for d in test_grads for _, arr in d) / 1024 / 1024

    print(f"encoded size:   {kb_data_size:.3f} MB")
    print(f"original size:  {original_size:.3f} MB")
    print(f"ratio (enc/orig): {kb_data_size / original_size:.3f}")
    print(f"ratio (orig/enc): {original_size / kb_data_size:.3f}\n\n")

    # ---- errors --------------------------------------------------------------
    temp = np.concatenate([arr.ravel() for d in test_grads for _, arr in d])
    mean_v = np.percentile(temp, [0.1, 99.9])
    mean_v = np.abs(np.clip(temp, *mean_v)).mean()

    # total error
    diff_t = np.concatenate(
        [(de_quant[i][j][1] - test_grads[i][j][1]).ravel()
         for i in range(len(test_grads)) for j in range(len(test_grads[i]))]
    )
    total_e = np.abs(diff_t).mean()
    print(f"total error:        {total_e / mean_v:.5f}% - v: {total_e:.5f}")

    # pure quantisation error
    temp_q = grad_de_quantizer(quant, template)
    diff_q = np.concatenate(
        [(temp_q[i][j][1] - test_grads[i][j][1]).ravel()
         for i in range(len(test_grads)) for j in range(len(test_grads[i]))]
    )
    quant_error = np.abs(diff_q).mean()
    print(f"quantisation error: {quant_error / mean_v:.5f}% - v: {quant_error:.5f}")

    # compression error
    flat_test_grad = [np.concatenate([b.flatten() for _, b in a]) for a in test_grads]
    temp_c = [grad_encoder(d) for d in flat_test_grad]
    temp_c = grad_decoder(temp_c, out_dtype=test_grads[0][0][1].dtype)
    diff_c = np.concatenate([temp_c[i] - flat_test_grad[i] for i in range(len(test_grads))])
    comp_error = np.abs(diff_c).mean()
    print(f"compressed error:   {comp_error / mean_v:.5f}% - v: {comp_error:.5f}")

    combo = (total_e - (quant_error + comp_error))
    print(f"combo error:        {combo / mean_v:.5f}% - v: {combo:.5f}")


if __name__ == "__main__":
    """
    Test the components of the system with different configurations to see how they perform.
    """

    list_options={
        "encoders": ["raw", "entropy"],
        "quants": ["raw", "8bit", "wz"],
    }

    for config.MODE_ENCODER in list_options["encoders"]:
        for config.MODE_quantizer in list_options["quants"]:
            print("\n============================")
            config.dtype = np.uint8 if config.MODE_quantizer == "8bit" else np.float32
            test_components()
