# Federated Learning Framework

A high-performance, distributed federated learning implementation using PyTorch with multiprocessing.

## Quick Start
run the run_fl.py script with default settings. change config values as needed.

## Architecture

### File Structure

- **`run_fl.py`**: Main entry point, spawns server and client processes
- **`server.py`**: Server logic (broadcasts model, aggregates updates)
- **`client.py`**: Client logic (local training, sends updates)
- **`models.py`**: Model definitions (ResNet18 by default)
- **`dataset.py`**: Dataset loading and preprocessing (SVHN)
- **`utils.py`**: Utilities (evaluation, BN recalibration, state dict management)
- **`codec.py`**: Gradient compression codec framework

### How It Works

1. **Initialization**: Server and clients spawn as separate processes using `torch.distributed`
2. **Round Loop**:
   - Server evaluates global model
   - Server broadcasts model parameters to all clients
   - [on client] Clients receive model, train locally for `local_epochs`
   - Clients send model deltas back to server
   - Server aggregates deltas using FedAvg (weighted by sample counts)
   - Server updates global model
3. **Completion**: After `rounds`, server evaluates final model and terminates

### Communication Protocol

All communication uses `torch.distributed` primitives:
- **Broadcast**: Server → Clients (model parameters, round number)
- **Send/Recv**: Clients → Server (client ID, sample counts, deltas, optional metrics)

## Performance Optimizations

### Memory
- **Shared memory tensors**: Zero-copy dataset sharing across processes
- **In-place operations**: Aggregation uses in-place tensor operations
- **Efficient flattening**: Uses `view()` instead of `reshape()` when possible
- **Channels-last format**: Better memory locality on modern GPUs

### Compute
- **Automatic Mixed Precision (AMP)**: Enabled on CUDA devices
- **Fused optimizers**: Uses fused Adam on CUDA
- **TF32**: Enabled on Ampere+ GPUs
- **Pin memory**: Faster CPU→GPU transfers
- **Persistent workers**: DataLoader workers stay alive across epochs

### Throughput
- **Non-blocking transfers**: Asynchronous data movement
- **Efficient dataloaders**: Configurable number of workers
- **Minimal overhead**: Optimized communication and aggregation

## Compression Codecs

The framework supports pluggable compression codecs for gradient communication:

### Available Codecs

| Codec | Description | Use Case |
|-------|-------------|----------|
| `IdentityCodec` | No compression (baseline) | Debugging, benchmarking |
| `BasicCompressionCodec` | Float16 + gzip | Simple compression |
| `CancerCodec` | Wyner-Ziv learned compression | Production, research |

### Cancer Codec (Wyner-Ziv)

The Cancer codec implements learned compression using:
- **Layered RNN quantizer**: Progressive refinement across planes
- **Side information**: Exploits temporal/cross-client correlations
- **Adaptive training**: Different strategies per round type (P/T/R/F)

See [Quantizer System](docs/QUANTIZER_SYSTEM.md) and [Cancer Protocol](docs/CANCER_PROTOCOL.md) for details.

### Validation

Validate the quantizer rate-distortion performance:

```bash
cd experiments
python quantizer_check.py              # Full validation
python quantizer_check.py --plot-only  # Re-plot existing results
python debug_rate.py                   # Verify rate calculations
```

## Extending the Framework

### Adding a New Model

Edit `models.py`:
```python
class MyModel(FLModelTemplate):
    def __init__(self, num_classes, lr, weight_decay):
        super().__init__()
        # Define your model
        
    def configure_optimizer(self, device):
        # Return optimizer
        
    def training_step(self, batch, device, cfg):
        # Return loss
```

### Adding a New Compression Codec

Edit `codec.py`:
```python
class MyCodec(FederatedCodec):
    def create_record(self, round_id, client_id):
        return MyRecord(round_id, client_id)
    
    def encode(self, delta_vec, record):
        # Compress and update record
        return compressed_data
    
    def decode(self, payload, record):
        # Decompress
        return reconstructed_vec
```

### Adding a New Dataset

Edit `dataset.py`:
```python
def precompute_mydataset_to_shared(data_folder, split, dtype, fraction):
    # Load and preprocess dataset
    # Return shared memory tensors (X, y)
```

## Troubleshooting

**Issue**: "RuntimeError: Address already in use"
- **Solution**: Change `master_port` in config or kill existing processes

**Issue**: Slow training
- **Solution**: Increase `num_loader_workers`, enable `tf32`, reduce `batch_size`

**Issue**: Out of memory
- **Solution**: Reduce `batch_size`, use `dataset_fraction < 1.0`, reduce `num_clients`

**Issue**: Debugging multiprocessing errors
- **Solution**: Check stderr for detailed tracebacks with client/server prefixes
- **Tip**: To debug with breakpoints in multiprocessing mode, configure your debugger to **stop on raise (outside libraries)** instead of **on termination**. This allows you to inspect the actual line that caused the exception in the worker process, rather than stopping at the `spawn.py` in the main process.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- numpy
- scikit-learn
- scipy (for SVHN dataset)

## License

MIT License

