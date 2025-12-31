"""Debug script to compare rate calculation methods"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from FL_reworked.cancer_quantizer import WZQuantizerCancer
from FL_reworked.cancer_protocol import CancerConfig
from FL_reworked.run_fl import FLConfig
from FL_reworked.prior_calculator import PriorCalculator

# Config
DATA_SIZE = 100_000
NOISE_POWER = 0.1
BPP = 4
NP = 2
BATCH_SIZE = 50_000

torch.manual_seed(42)
np.random.seed(42)

# Generate data
base = torch.randn(DATA_SIZE)
y = base + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)
side_info = [base.clone()]

# Create and train quantizer
print("Training quantizer...")
c_cfg = CancerConfig(train_epochs=40, train_sample_size=50_000)
fl_cfg = FLConfig(num_clients=1, training_progress_bar=True, compile_mode=False)
quantizer = WZQuantizerCancer(
    c_cfg=c_cfg, fl_cfg=fl_cfg, num_planes=NP, bins_per_plane=BPP, si_size=len(side_info))
quantizer.train_model(y, side_info)

print("\nEncoding...")

bins = quantizer.encoding_process(y)
print(f"Bins shape: {bins.shape}")  # Should be [num_planes, N]
print(f"Final bins unique: {[torch.unique(bins[i]).tolist() for i in range(bins.shape[0])]}")

# --- Rate Calculation Methods ---
rates = {}
quantizer.coding_model.to('cuda').eval()

with torch.inference_mode():
    # Prepare data on GPU
    x_gpu = y.unsqueeze(1).to('cuda', non_blocking=True).float()
    si_gpu = base.unsqueeze(1).to('cuda', non_blocking=True).float()

    # Method 1: Using _get_posterior (current implementation)
    print("\nMethod 1: _get_posterior...")
    prior1 = quantizer._get_posterior(y, bins_vec_save_compute=bins)
    rates['Method 1 (_get_posterior)'] = PriorCalculator.compute_rate_from_prior_tensor(prior1, bins, NP)

    # Method 2: Direct forward pass (like training)
    print("Method 2: Direct forward pass...")
    quantizer.coding_model.to('cuda').eval()
    _, bins_fwd, _, prior_probs = quantizer.coding_model.forward(x_gpu, si_gpu, tau=0.001)
    rates['Method 2 (forward pass)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(prior_probs), torch.stack(bins_fwd), NP)

    # Method 3: encode + get_priors (like main_layered.py eval)
    print("Method 3: encode + get_priors...")
    bins3, hard_codes3 = quantizer.coding_model.encode(x_gpu)
    priors3 = quantizer.coding_model.get_priors(codes=hard_codes3, y=si_gpu)
    rates['Method 3 (encode+get_priors)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(priors3), torch.stack(bins3), NP)

    # Method 4: _compute_prior_from_network directly
    print("Method 4: _compute_prior_from_network...")
    prior4 = PriorCalculator._compute_prior_from_network(
        quantizer.coding_model, bins, torch.stack(side_info).T, batch_size=BATCH_SIZE)
    rates['Method 4 (_compute_prior_from_net)'] = PriorCalculator.compute_rate_from_prior_tensor(prior4, bins, NP)

    # Method 5: Manual get_priors with encoded bins
    print("Method 5: Manual get_priors...")
    hard_codes_manual = [F.one_hot(bins[i].long(), num_classes=BPP).float().to('cuda') for i in range(NP)]
    quantizer.coding_model.cuda()
    priors5 = quantizer.coding_model.get_priors(codes=hard_codes_manual, y=si_gpu)
    rates['Method 5 (manual get_priors)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(priors5), bins.cuda(), NP)

quantizer.coding_model.cpu()

print("\n=== Summary ===")
for name, rate in rates.items():
    print(f"{name:<35}: {rate:.4f} bits/symbol")

rate_values = list(rates.values())
max_diff = max(rate_values) - min(rate_values)
print(f"\nMax difference between methods: {max_diff:.4f} bits/symbol")
