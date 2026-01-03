"""Debug script to compare rate calculation methods"""
import sys
from pathlib import Path

# Add project root to path (works both in PyCharm and terminal)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # FL_reworked/experiments/debuging -> VUB-ACS-25_Thesis
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np

from FL_reworked.cancer_quantizer import WZQuantizerCancer
from FL_reworked.cancer_protocol import CancerConfig
from FL_reworked.run_fl import FLConfig
from FL_reworked.prior_calculator import PriorCalculator

# Config
DATA_SIZE = 1_000_000
NOISE_POWER = 0.1
BPP = 4
NP = 2

torch.manual_seed(42)
np.random.seed(42)

# Generate data
base = torch.randn(DATA_SIZE)
y = base + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)
side_info = [base.clone()]

# Create and train quantizer
print("Training quantizer...")
c_cfg = CancerConfig()
fl_cfg = FLConfig(num_clients=1, training_progress_bar=True, compile_mode=False)
quantizer = WZQuantizerCancer(
    c_cfg=c_cfg, fl_cfg=fl_cfg, num_planes=NP, bins_per_plane=BPP, si_size=len(side_info))
quantizer.train_model(y, side_info)

prep_y,_ = quantizer.get_x_data(y)
prep_si = quantizer.get_si_data()

print("\nEncoding...")

bins,_ = quantizer.encoding_process(y)
print(f"Bins shape: {bins.shape}")  # Should be [num_planes, N]
print(f"Final bins unique: {[torch.unique(bins[i]).tolist() for i in range(bins.shape[0])]}")

# --- Rate Calculation Methods ---
rates = {}
quantizer.coding_model.cuda().eval()

with torch.inference_mode():
    # Method 1: Using _get_posterior (current implementation)
    print("\nMethod 1: quantizer _get_posterior...")
    prior1 = quantizer._get_posterior(y, bins_vec_save_compute=bins)
    rates['Method 1 (_get_posterior)'] = PriorCalculator.compute_rate_from_prior_tensor(prior1, bins, NP)
    print('rate: ', rates['Method 1 (_get_posterior)'])

    # Method 2: Direct forward pass (like training)
    print("Method 2: Direct net forward pass...")
    quantizer.coding_model.cuda().eval()
    _, bins_fwd, _, prior_probs = quantizer.coding_model.forward(prep_y, prep_si, tau=0.001)
    rates['Method 2 (forward pass)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(prior_probs), torch.stack(bins_fwd), NP)
    print('rate: ', rates['Method 2 (forward pass)'])

    # Method 3: encode + get_priors (like main_layered.py eval)
    print("Method 3: direct net encode + direct net get_priors...")
    bins3, hard_codes3 = quantizer.coding_model.encode(prep_y)
    priors3 = quantizer.coding_model.get_priors(codes=hard_codes3, y=prep_si)
    rates['Method 3 (encode+get_priors)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(priors3), torch.stack(bins3), NP)
    print('rate: ', rates['Method 3 (encode+get_priors)'])

    # Method 4: _compute_prior_from_network directly
    print("Method 4: _compute_prior_from_network...")
    prior4 = PriorCalculator._compute_prior_from_network(
        quantizer.coding_model, bins, prep_si)
    rates['Method 4 (_compute_prior_from_net)'] = PriorCalculator.compute_rate_from_prior_tensor(prior4, bins, NP)
    print('rate: ', rates['Method 4 (_compute_prior_from_net)'])

    # Method 5: Manual get_priors with encoded bins
    print("Method 5: Manual bins for direct net get_priors...")
    hard_codes_manual = [F.one_hot(bins[i].long(), num_classes=BPP).float().cuda() for i in range(NP)]
    quantizer.coding_model.cuda()
    priors5 = quantizer.coding_model.get_priors(codes=hard_codes_manual, y=prep_si)
    rates['Method 5 (manual get_priors)'] = PriorCalculator.compute_rate_from_prior_tensor(
        torch.stack(priors5), bins.cuda(), NP)
    print('rate: ', rates['Method 5 (manual get_priors)'])

print("Method 6: train_prior_model via quantizer _get_posterior...")
quantizer.cached_priors_dict = {}
prior6 = quantizer._get_posterior(y, bins_vec_save_compute=bins)
rates['Method 6 (train_prior_model)'] = PriorCalculator.compute_rate_from_prior_tensor(prior6, bins, NP)
print('rate: ', rates['Method 6 (train_prior_model)'])

print("Method 7: direct train_prior_model...")
q_model = PriorCalculator.train_prior_model(
    bins, torch.stack(side_info).T, NP, BPP, )
prior7 = PriorCalculator._compute_prior_from_network(q_model, bins, torch.stack(side_info).T)
rates['Method 7 (direct train_prior_model)'] = PriorCalculator.compute_rate_from_prior_tensor(prior7, bins, NP)
print('rate: ', rates['Method 7 (direct train_prior_model)'])

rate_values = list(rates.values())
max_diff = max(rate_values) - min(rate_values)
print(f"\nMax difference between methods: {max_diff:.4f} bits/symbol")
