# Wyner-Ziv Quantizer System Documentation

## Overview

This document provides comprehensive documentation for the Wyner-Ziv (WZ) quantizer system used in the Cancer compression protocol. The system implements a learned, layered RNN-based quantizer that leverages side information for efficient compression of model gradients in federated learning.

## Table of Contents

1. [Architecture](#architecture)
2. [Key Components](#key-components)
3. [Data Flow](#data-flow)
4. [Training Process](#training-process)
5. [Rate Calculation](#rate-calculation)
6. [Configuration](#configuration)
7. [Common Issues & Solutions](#common-issues--solutions)

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    WZQuantizerCancer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Encoder    │───▶│   Bins/Codes │───▶│   Decoder    │      │
│  │  (RNN-based) │    │  (Quantized) │    │  (RNN-based) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   ▲               │
│         │                   ▼                   │               │
│         │            ┌──────────────┐           │               │
│         └───────────▶│ Prior Model  │───────────┘               │
│                      │  (Cond. RNN) │                           │
│                      └──────────────┘                           │
│                             ▲                                   │
│                             │                                   │
│                      ┌──────────────┐                           │
│                      │ Side Info (Y)│                           │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Model: EncoderDecoderLayeredRNN

The core model (`EncoderDecoderLayeredRNN` in `brent_wz_models.py`) consists of three RNN networks:

1. **Encoder RNN**: Maps input `x` to soft codes (probability distributions over bins)
2. **Decoder RNN**: Reconstructs `x` from quantized codes and side information `y`
3. **Conditional Prior RNN**: Predicts bin probabilities given side information (autoregressive across planes)

### Layered Quantization

The quantizer uses multiple "planes" for progressive refinement:
- Each plane has `bins_per_plane` quantization levels
- Planes are processed autoregressively (plane i's prior depends on planes 0..i-1)
- Total codebook size: `bins_per_plane ^ num_planes`

---

## Key Components

### WZQuantizerCancer (`cancer_quantizer.py`)

Main quantizer class that wraps the RNN model and provides:
- `encoding_process(x)`: Quantize input to bin indices
- `decoding_process(bins)`: Reconstruct from bins using side info
- `_get_posterior(x, bins)`: Compute prior probabilities for rate calculation

### CancerCodec (`cancer_protocol.py`)

Protocol handler that manages:
- Round types: P (Pretrained/Marginal), T (Temporal), R (Retrain), F (Frozen)
- Quantizer training and caching
- Side information management across rounds
- Compression records for analysis

### PriorCalculator (`prior_calculator.py`)

Utility class for:
- Computing marginal priors (histogram-based)
- Computing conditional priors from the trained model
- Training separate prior models (if needed)

---

## Data Flow

### Training Flow

```
Input: x (gradients), y (side information)
                │
                ▼
┌───────────────────────────────────┐
│  forward(x, y, tau)               │
│  ├── encode(x, tau)               │
│  │   └── bins, soft_codes         │
│  ├── decode(soft_codes, y)        │
│  │   └── reconstructions          │
│  └── get_priors(soft_codes, y)    │
│      └── prior_probs              │
└───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  Loss = λ·MSE(recon, x)           │
│       + rate_weight·KL_loss       │
│                                   │
│  where KL_loss = log(p_ux/p_u)    │
│  p_ux = encoder soft probability  │
│  p_u  = prior probability         │
└───────────────────────────────────┘
```

### Inference Flow

```
Encoding:
  x ──▶ encode(x) ──▶ bins [num_planes, N]

Rate Calculation:
  bins, y ──▶ get_priors(one_hot(bins), y) ──▶ prior_probs
  rate = Σ -log2(prior_probs[actual_bins])

Decoding:
  bins, y ──▶ decode(one_hot(bins), y) ──▶ x_reconstructed
```

---

## Training Process

### Loss Function

The training loss combines reconstruction and rate terms:

```python
for each plane i:
    # Reconstruction loss (normalized MSE)
    dist = MSE(reconstruct[i], x) / mspe_denom
    loss += reconst_ld * dist
    
    # Rate loss (KL divergence proxy)
    p_ux = soft_codes[i][actual_bins]  # Encoder probability
    p_u = prior_probs[i][actual_bins]  # Prior probability
    rate_loss = mean(log(p_ux / p_u))
    loss += rate_weight * rate_loss

loss /= num_planes
```

### Temperature Annealing

The Gumbel-Softmax temperature `tau` is annealed during training:
```python
tau_t = tau * exp(progress * log(0.1 / tau))
```
- High tau (start): Soft, differentiable codes
- Low tau (end): Hard, discrete-like codes

### Rate Weight Schedule

The rate component weight increases during training:
```python
rate_weight = f(training_progress, tau_rate)
# Starts low, increases to emphasize rate later in training
```

---

## Rate Calculation

Three rate metrics are computed:

### 1. Prior Rate (Conditional)
Rate achievable using the trained conditional prior model:
```python
prior = model.get_priors(codes=one_hot(bins), y=side_info)
rate = Σ mean(-log2(prior[plane][samples, bins[plane]]))
```

### 2. Marginal Rate
Rate using only bin histograms (no side information):
```python
for each plane:
    counts = histogram(bins[plane])
    probs = counts / total
rate = Σ entropy(probs)
```

### 3. Real Entropy Rate
Actual compressed size divided by number of symbols:
```python
real_rate = compressed_bytes * 8 / num_parameters
```

### Wyner-Ziv Theoretical Bound

For Gaussian source X = Y + N:
```python
cond_var = (noise_var * side_info_var) / (noise_var + side_info_var)
WZ_rate = 0.5 * log2(cond_var / MSE)
```

Lattice quantization adds ~1.53 dB to the bound.

---

## Configuration

### CancerConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_phase` | `(('P',16,3), ('T',8,3), ...)` | Phase schedule: (type, bpp, np) |
| `routine_phase` | `(('T',2,3), ('R',2,3), ...)` | Routine phase schedule |
| `max_side_info_count` | `5` | Max historical reconstructions to keep |
| `train_marginal` | `False` | Train marginal models instead of loading pretrained |
| `train_epochs` | `60` | Training epochs per quantizer |
| `reconst_ld` | `400.0` | Reconstruction loss weight (λ) |
| `train_sample_size` | `300_000` | Samples used for training |
| `lr` | `1e-3` | Learning rate |
| `tau` | `1.3` | Initial Gumbel-Softmax temperature |
| `tau_rate` | `10.0` | Rate weight schedule parameter |

### Round Types

| Type | Description | Side Info | Training |
|------|-------------|-----------|----------|
| **P** | Pretrained/Marginal | None (zeros) | Load or train marginal |
| **T** | Temporal | Server reconstructions | Train with temporal SI |
| **R** | Retrain | Client reconstructions | Train with client SI |
| **F** | Frozen | Same as previous | No training, reuse model |

---

## Common Issues & Solutions

### Issue: All-Zero Bins from Encoding

**Cause**: Using `non_blocking=True` in CUDA transfers causes data to be used before transfer completes.

**Solution**: Use synchronous transfers:
```python
# Bad
x_batch = data.cuda(non_blocking=True)
result = model(x_batch)
return result.cpu(non_blocking=True)  # Data may be incomplete!

# Good
x_batch = data.cuda()
result = model(x_batch)
return result.cpu()  # Guaranteed complete
```

### Issue: Rate Calculated as 0.0

**Cause**: Converting prior probabilities to `float16` rounds small values to 0, making all probability mass go to one bin.

**Solution**: Keep priors in `float32`:
```python
# Bad
cached_prior = prior.to(torch.float16)  # Small probs → 0

# Good
cached_prior = prior  # Keep float32
```

### Issue: Marginal Model Uses Side Info

**Cause**: The `marginal` flag not being passed correctly when `train_marginal=True`.

**Solution**: Explicitly pass the `marginal` parameter:
```python
quantizer = WZQuantizerCancer(
    ...,
    pretrained=False,
    marginal=True  # Explicitly set for marginal models
)
```

### Issue: Prior Rate Much Lower Than Expected

**Cause**: `get_priors()` receives incorrect input format or the model is in wrong mode.

**Solution**: Ensure:
1. Model is in `eval()` mode for inference
2. Codes are `float32` tensors (not `int64` from `one_hot`)
3. Side info shape matches training: `[batch, side_info_size]`

---

## Validation

Use `experiments/quantizer_check.py` to validate the quantizer:

```bash
# Run full validation
python quantizer_check.py

# Only plot existing results
python quantizer_check.py --plot-only
```

Use `experiments/debug_rate.py` to verify rate calculations match:

```bash
python debug_rate.py
```

Expected output:
```
Method 1 (_get_posterior):           1.1170
Method 2 (forward pass):             1.1170
Method 3 (encode+get_priors):        1.1170
Method 4 (_compute_prior_from_net):  1.1170
Max diff: 0.0000
```

---

## References

- Wyner-Ziv coding: [Wyner & Ziv, 1976]
- Gumbel-Softmax: [Jang et al., 2017]
- Neural compression: [Ballé et al., 2018]

