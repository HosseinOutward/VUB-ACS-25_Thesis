"""Quantizer Long-term Performance Test

This script tests a single quantizer configuration over multiple iterations
where the data distribution gradually changes. Unlike quantizer_check.py which
compares different configurations, this focuses on tracking performance over time.

The script:
1. Initializes a Cancer codec with a specific configuration
2. Generates data from a normal distribution that evolves over iterations
3. Runs the Cancer protocol for multiple rounds
4. Records rate-distortion metrics for each iteration
5. Plots RD curves showing performance over time
"""
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# import pydevd_pycharm
# pydevd_pycharm.settrace('ETROFLOCK', port=32112,
#         stdout_to_server=True, stderr_to_server=True, suspend=True)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent  # -> VUB-ACS-25_Thesis/FL_code

from FL_code.FL_core.codec import create_codec
from FL_code.cancer_protocol import CancerCodec

# ============== EXPERIMENT CONFIG ==============
# Test configuration
NUM_ITERATIONS = 0
# NUM_ITERATIONS = 20  # Number of rounds to test
DATA_SIZE = 1_000_000  # Signal size
INITIAL_MEAN = 0.0
INITIAL_STD = 1.0
NOISE_POWER = 0.1  # Variance of noise added to side information

# Distribution drift parameters
MEAN_DRIFT_PER_ITER = 0.05  # How much mean shifts each iteration
STD_DRIFT_PER_ITER = 0.02   # How much std changes each iteration

# Codec configuration (single config to test)
CODEC_NAME = 'cancer|non_wz_worker|no_model_slices'
# ===============================================


def setup_codec(codec_name: str) -> CancerCodec:
    """Initialize codec with custom configuration for long-term testing."""
    codec: CancerCodec = create_codec(codec_name, None)
    codec.c_cfg.pretrain_pth_dir = str(project_root / "data/pre_trained_pth")+'/'
    codec.c_cfg.training_progress_bar = True
    return codec


def run_long_term_test(out_path: Path):
    """Run long-term quantizer test with evolving distribution."""
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "longrun_results.csv"

    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    codec = setup_codec(CODEC_NAME)
    client_id = 0

    # Initialize side information
    side_info = torch.randn(DATA_SIZE) * INITIAL_STD + INITIAL_MEAN

    print(f"=== Long-term Quantizer Test ===")
    print(f"Codec: {CODEC_NAME}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Data size: {DATA_SIZE:,}\n")

    # Write CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iteration', 'round_type', 'phase', 'mse', 'distortion_db',
            'prior_rate', 'marginal_rate', 'entropy_real_rate',
            'compression_ratio', 'signal_mean', 'signal_std'
        ])

    # Run iterations
    for iter_idx in range(NUM_ITERATIONS):
        # Evolve distribution parameters
        current_mean = INITIAL_MEAN + iter_idx * MEAN_DRIFT_PER_ITER
        current_std = INITIAL_STD + iter_idx * STD_DRIFT_PER_ITER

        # Generate signal: side_info + noise, with evolving statistics
        signal = side_info + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)
        signal = signal * (current_std / signal.std()) + (current_mean - signal.mean())

        # Create record and compress/decompress
        record = codec.create_record(round_id=iter_idx, client_id=client_id)
        record.model_size = DATA_SIZE

        compressed = codec.encode(signal, record)
        print('[before dec] si size:', len(codec.frozen_quantizers[0].side_info_list_used),
              'extra si size:', len(codec.frozen_quantizers[0].extra_si_for_prior),
              'frozen quantizers:', len(codec.frozen_quantizers))
        reconstructed = codec.decode(compressed, record)
        print('[after dec] si size:', len(codec.frozen_quantizers[0].side_info_list_used),
              'extra si size:', len(codec.frozen_quantizers[0].extra_si_for_prior))

        # Update side info for next iteration
        side_info = reconstructed.clone()

        # Compute metrics
        distortion_db = 10 * np.log10(record.wmape) * (14.21/12.87) if record.wmape > 0 else -np.inf

        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iter_idx,
                record.round_type,
                record.phase,
                record.wmape,
                distortion_db,
                record.prior_rate,
                record.marginal_rate,
                record.entropy_real_rate,
                record.compression_ratio,
                current_mean,
                current_std
            ])

        print(f"[{iter_idx+1}/{NUM_ITERATIONS}] "
              f"{record.phase[0]}|{record.round_type} | "
              f"wmape={record.wmape:.6f} ({distortion_db:.2f}dB) | "
              f"Rate: P={record.prior_rate:.3f} M={record.marginal_rate:.3f} E={record.entropy_real_rate:.3f}\n")

    print(f"\n✓ Results saved to: {csv_path}")
    return csv_path


def plot_longrun_results(csv_path: Path):
    """Create plots showing performance over iterations."""
    # Load data
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding='utf-8')

    if len(data) == 0:
        print("❌ No data to plot")
        return

    iterations = data['iteration']
    distortion_db = data['distortion_db']
    prior_rate = data['prior_rate']
    marginal_rate = data['marginal_rate']
    entropy_rate = data['entropy_real_rate']
    phase = data['phase']
    round_type = data['round_type']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Rate vs Iteration
    ax1 = axes[0, 0]
    ax1.plot(iterations, prior_rate, 'o-', label='Conditional Rate', alpha=0.7)
    ax1.plot(iterations, marginal_rate, 's-', label='Marginal Rate', alpha=0.7)
    ax1.plot(iterations, entropy_rate, '^-', label='Entropy Rate', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Rate (bits/symbol)')
    ax1.set_title('Rate Evolution Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Color-code background by round type
    warmup_mask = phase == 'warmup'
    if np.any(warmup_mask):
        ax1.axvspan(iterations[warmup_mask].min(), iterations[warmup_mask].max(),
                    alpha=0.1, color='yellow', label='Warmup')

    # Plot 2: Distortion vs Iteration
    ax2 = axes[0, 1]
    ax2.plot(iterations, distortion_db, 'o-', color='darkred', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Distortion (dB)')
    ax2.set_title('Distortion Over Time')
    ax2.grid(True, alpha=0.3)

    if np.any(warmup_mask):
        ax2.axvspan(iterations[warmup_mask].min(), iterations[warmup_mask].max(),
                    alpha=0.1, color='yellow')

    # Plot 3: Rate-Distortion curve (colored by iteration)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(prior_rate, distortion_db, c=iterations,
                         cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Conditional Rate (bits/symbol)')
    ax3.set_ylabel('Distortion (dB)')
    ax3.set_title('Rate-Distortion Trajectory (Conditional)')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Iteration')

    # Plot 4: All rates RD curves
    ax4 = axes[1, 1]
    ax4.scatter(prior_rate, distortion_db, marker='o',
               label='Conditional', alpha=0.6, edgecolors='black', linewidths=0.5)
    ax4.scatter(marginal_rate, distortion_db, marker='s',
               label='Marginal', alpha=0.6, edgecolors='black', linewidths=0.5)
    ax4.scatter(entropy_rate, distortion_db, marker='^',
               label='Entropy', alpha=0.6, edgecolors='black', linewidths=0.5)
    ax4.set_xlabel('Rate (bits/symbol)')
    ax4.set_ylabel('Distortion (dB)')
    ax4.set_title('Rate-Distortion (All Metrics)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average Distortion: {np.mean(distortion_db):.2f} dB")
    print(f"Average Conditional Rate: {np.mean(prior_rate):.3f} bits/symbol")
    print(f"Average Marginal Rate: {np.mean(marginal_rate):.3f} bits/symbol")
    print(f"Average Entropy Rate: {np.mean(entropy_rate):.3f} bits/symbol")
    print(f"\nDistortion range: [{np.min(distortion_db):.2f}, {np.max(distortion_db):.2f}] dB")
    print(f"Rate range (conditional): [{np.min(prior_rate):.3f}, {np.max(prior_rate):.3f}]")


if __name__ == '__main__':
    import argparse

    out_dir = Path(__file__).parent / "results"
    csv_path = out_dir / "longrun_results.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-run', action='store_true', help='Skip experiment, only plot')
    args = parser.parse_args()

    if not args.skip_run and NUM_ITERATIONS!=0:
        print("=== Running Long-term Quantizer Test ===\n")
        csv_path = run_long_term_test(out_dir)

    if csv_path.exists():
        print("\n=== Plotting Results ===")
        plot_longrun_results(csv_path)
    else:
        print(f"❌ No data found at {csv_path}")
