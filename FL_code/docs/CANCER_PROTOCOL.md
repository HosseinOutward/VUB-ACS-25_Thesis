# Cancer Compression Protocol Documentation

## Overview

The Cancer protocol is a learned compression scheme for federated learning that uses Wyner-Ziv coding principles to achieve efficient gradient compression by exploiting temporal and cross-client correlations as side information.

## Protocol Phases

### Round Types

The protocol operates in different round types, each with its own quantizer training strategy:

```
Timeline: ──P──T──R──R──R──T──R──F──F──F──F──F──...
            │  │  │        │     │
            │  │  │        │     └─ Frozen: Reuse existing quantizer
            │  │  │        └─ Temporal: Train with client-side history
            │  │  └─ Retrain: Train with server-side history (all reconst.)
            │  └─ Temporal: Train with client-side history (reconst. available at client only)
            └─ Pretrained/Marginal: No side info
```

### Phase Configuration
The warmup phase happens only once at the start, followed by the routine phase which repeats.
```python
@dataclass
class CancerConfig:
    # Warmup phase: (round_type, bins_per_plane, num_planes)
    warmup_phase = (
        ('P', 16, 3),  # Start with marginal (no side info)
        ('T', 8, 3),   # Temporal side info
        ('R', 4, 3),   # Retrain with client history
        ('R', 4, 3),
        ('R', 4, 3),
    )
    
    # Routine phase: Lower rates after warmup
    routine_phase = (
        ('T', 2, 3),
        ('R', 2, 3),
        ('F', 2, 3),  # Frozen rounds for efficiency
        ('F', 2, 3),
        ('F', 2, 3),
        ('F', 2, 3),
        ('F', 2, 3),
    )
```

## Side Information Management

### Server-Side History (`srvr_past_reconst`)

Stores reconstructed gradients on the server, available to all clients:
- Updated after every decode
- Used for **T (Temporal)** rounds
- Provides cross-client correlation

### Client-Side History (`client_past_reconst`)

Stores reconstructed gradients per client:
- Updated only for **T** and **P** rounds
- Used for **R (Retrain)** rounds  
- Provides temporal correlation within client

### History Management

```python
def _update_history(history, item):
    history.append(item.to(torch.float16))  # Save memory
    if len(history) > max_side_info_count:
        history.pop(0)  # FIFO queue
```

## Compression Flow

### Encoding (Client → Server)

```python
def _compress(delta_vec, record):
    # 1. Train quantizer if needed (P, T, R rounds)
    if record.round_type != 'F':
        _train_quantizer(delta_vec, record)
    
    # 2. Encode gradients
    quantizer = frozen_quantizers[client_id]
    bins = quantizer.encoding_process(delta_vec)
    
    # 3. Build payload with model state for decoder sync
    payload = {
        'payload_content': bins,
        'encoder_state': ...,  # or 'decoder_state' for T/P
    }
    
    # 4. Compute rate metrics
    prior = quantizer._get_posterior(delta_vec, bins)
    record.prior_rate = compute_rate_from_prior(prior, bins)
    record.marginal_rate = compute_marginal_rate(bins)
    
    return payload
```

### Decoding (Server)

```python
def _decompress(payload, record):
    quantizer = frozen_quantizers[client_id]
    
    # Reconstruct gradients
    reconst = quantizer.decoding_process(payload['payload_content'])
    
    # Update histories
    _update_history(srvr_past_reconst[client_id], reconst)
    if record.round_type in ('T', 'P'):
        _update_history(client_past_reconst[client_id], reconst)
    
    return reconst
```

## Quantizer Training by Round Type

### P Rounds (Pretrained/Marginal)

```python
if round_type == 'P':
    if train_marginal:
        # Train marginal model with zero side-info
        train_si = [torch.zeros_like(delta_vec)]
        target_x = delta_vec
        marginal = True
    else:
        # Load pretrained model
        train_si, target_x = [], None
        pretrained = True
```

### T Rounds (Temporal)

```python
if round_type == 'T':
    # Use all server-side reconstructions as side info
    # (excluding current client's latest)
    train_si = [item for cid, reconst_list in srvr_past_reconst
                for item in (reconst_list[:-1] if cid == client_idx 
                            else reconst_list)]
    target_x = srvr_past_reconst[client_idx][-1]
```

### R Rounds (Retrain)

```python
if round_type == 'R':
    # Use client's own history as side info
    train_si = [item for reconst_list in client_past_reconst 
                for item in reconst_list]
    target_x = delta_vec
```

### F Rounds (Frozen)

```python
if round_type == 'F':
    # Skip training, reuse existing quantizer
    pass
```

## Compression Records

### CancerRecord Fields

| Field | Description |
|-------|-------------|
| `round_id` | Current round number |
| `client_id` | Client identifier |
| `round_type` | P, T, R, or F |
| `bits_per_plane` | Quantization bins per plane |
| `num_planes` | Number of quantization planes |
| `compressed_bytes` | Compressed payload size (MB) |
| `basic_raw_bytes` | Uncompressed size (MB) |
| `compression_ratio` | raw / compressed |
| `entropy_real_rate` | Actual bits per symbol |
| `prior_rate` | Rate using conditional prior |
| `marginal_rate` | Rate using marginal prior |
| `mse` | Mean squared error |
| `mape` | Mean absolute percentage error |
| `encoder_decoder_size` | Model state overhead (MB) |

### Saving Records

```python
record.save_to_csv(save_dir)  # Appends to compression_records.csv
```

## Rate Metrics

### Prior Rate

Uses the trained conditional prior model:
```python
prior = quantizer._get_posterior(delta_vec, bins)
prior_rate = sum(
    -log2(prior[plane, samples, bins[plane]]).mean()
    for plane in range(num_planes)
)
```

### Marginal Rate

Uses only bin histograms (ignores side info):
```python
marginal_prior = compute_marginal_prior(bins, bpp, np)
marginal_rate = sum(
    -log2(marginal_prior[plane, samples, bins[plane]]).mean()
    for plane in range(num_planes)
)
```

### Real Entropy Rate

Actual compressed size:
```python
real_rate = compressed_bytes * 8 / model_size
```

## Integration with FL Framework

### In Client (`client.py`)

```python
# After local training
delta_vec = compute_delta(local_model, global_model)

# Compress
record = codec.create_record(round_id, client_id)
compressed = codec.encode(delta_vec, record)

# Send to server
send(compressed, record)
```

### In Server (`server.py`)

```python
# Receive from client
compressed, record = recv()

# Decompress
delta_vec = codec.decode(compressed, record)

# Aggregate
aggregate_deltas(delta_vec, ...)

# Log metrics
record.save_to_csv(log_dir)
```

## Performance Considerations

### Memory

- Side info stored as `float16` to reduce memory
- History limited by `max_side_info_count`
- Quantizers cached in `frozen_quantizers` dict

### Compute

- F rounds skip training entirely
- Batch processing for large vectors
- GPU acceleration for RNN operations

### Communication

- Bins encoded as `uint8` or `uint16`
- Model states compressed with gzip
- Only encoder OR decoder state sent (not both)

## Example Usage

```python
from FL_code.cancer_protocol import CancerCodec, CancerConfig
from FL_code.run_fl import FLConfig

# Configure
c_cfg = CancerConfig()
c_cfg.train_epochs = 60
c_cfg.reconst_ld = 400.0

fl_cfg = FLConfig(num_clients=5)

# Create codec
codec = CancerCodec(fl_cfg)
codec.c_cfg = c_cfg

# Use in FL loop
for round_id in range(num_rounds):
    for client_id in range(num_clients):
        record = codec.create_record(round_id, client_id)
        compressed = codec.encode(delta_vec, record)
        reconstructed = codec.decode(compressed, record)
        record.save_to_csv("./logs")
```

