"""Quantizer Rate-Distortion Validation"""
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

# Add project root to path (works both in PyCharm and terminal)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from FL_reworked.codec import create_codec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from FL_reworked.cancer_protocol import CancerCodec, CancerConfig
from FL_reworked.run_fl import FLConfig

# ============== CONFIG ==============
NUM_REPEATS = 5
DATA_SIZE = 10_000_000
NOISE_POWER = 0.01
# (round_type, bins_per_plane, num_planes) - M=marginal, T=with side info
CONFIGS = []
CONFIGS += [('T', 4, 2),('T', 4, 3),('T', 8, 3), ('T', 16, 2), ('T', 32, 3),]
CONFIGS += [('TM', 4, 2),('TM', 8, 3)]
# ====================================

def run_experiments(out_path: Path):
    """Run experiments and save to CSV."""
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "quantizer_check.csv"

    c_cfg = CancerConfig()
    c_cfg.train_epochs = 40
    c_cfg.train_sample_size = min(200_000, int(DATA_SIZE * 0.8))

    fl_cfg = FLConfig(num_clients=1, training_progress_bar=True, compile_mode=False)
    fl_cfg.codec='cancer'

    first_write = True
    for round_type, bpp, np_ in CONFIGS:
        print(f"\n=== {round_type} | bpp={bpp} | np={np_} ===")

        for rep in range(NUM_REPEATS):
            torch.manual_seed(rep * 42)
            np.random.seed(rep * 42)

            # Generate WZ data: Y = X + N
            base = torch.randn(DATA_SIZE)
            y = base + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)

            # Setup codec
            codec:CancerCodec = create_codec(fl_cfg, None)
            codec.c_cfg = c_cfg
            codec.c_cfg.warmup_phase = ((round_type, bpp, np_),)

            codec.client_past_reconst[0] = [base.clone().to(torch.float16)]

            # Run compression
            record = codec.create_record(round_id=0, client_id=0)
            record.model_size = DATA_SIZE
            compressed = codec.encode(y, record)
            _ = codec.decode(compressed, record)

            print(f"  [{rep+1}/{NUM_REPEATS}] MSE={record.mse:.6f} Prior={record.prior_rate:.3f} "
                  f"Marginal={record.marginal_rate:.3f}")

            # Save incrementally
            row = record.to_dict()
            row['repeat'] = rep

            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if first_write:
                    writer.writeheader()
                    first_write = False
                writer.writerow(row)

    print(f"\nSaved to {csv_path}")



def load_records(csv_path: Path):
    """Load records from CSV."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return [{k: float(v) if k in ['mse', 'prior_rate', 'marginal_rate', 'entropy_real_rate'] else v
                 for k, v in row.items()} for row in reader]


def plot_results(csv_path: Path):
    """Plot rate-distortion curves."""
    records = load_records(csv_path)

    # WZ bound
    cond_var = NOISE_POWER / (NOISE_POWER + 1.0)
    mse_range = np.linspace(1e-6, cond_var * 0.999, 500)
    bound_rate = 0.5 * np.log2(cond_var / mse_range)
    bound_dist = 10 * np.log10(mse_range)
    ptp_bound = 0.5 * np.log2((1+NOISE_POWER) / mse_range)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    rate_attrs = ['prior_rate', 'marginal_rate', 'entropy_real_rate']
    titles = ['Prior Model Rate', 'Marginal Rate', 'Real Entropy Rate']

    # Plot individual rate types
    for ax, rate_attr, title in zip(axes[:3], rate_attrs, titles):
        ax.plot(bound_rate, bound_dist, 'k-', lw=2, label='WZ Bound')
        ax.plot(bound_rate, bound_dist + 1.53, 'k:', lw=2, label='+ Lattice')
        ax.plot(ptp_bound, bound_dist, 'g--', lw=1, label='PTP Bound')

        for r in records:
            rate = r[rate_attr]
            mse_db = 10 * np.log10(r['mse'])
            is_marginal = r['round_type'] == 'M'
            color, marker = ('red', 'x') if is_marginal else ('blue', 'o')
            ax.scatter(rate, mse_db, c=color, marker=marker, s=40, alpha=0.6)

        ax.set_xlabel('Rate (bits/symbol)')
        ax.set_ylabel('Distortion (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 12)
        ax.set_ylim(-30, 12)

    # Combined plot
    ax_all = axes[3]
    ax_all.plot(bound_rate, bound_dist, 'k-', lw=2, label='WZ Bound')
    ax_all.plot(bound_rate, bound_dist + 1.53, 'k:', lw=2, label='+ Lattice')
    ax_all.plot(ptp_bound, bound_dist, 'g--', lw=1, label='PTP Bound')

    markers = {'prior_rate': 'o', 'marginal_rate': 's', 'entropy_real_rate': '^'}
    for r in records:
        mse_db = 10 * np.log10(r['mse'])
        is_marginal = r['round_type'] == 'M'
        color = 'red' if is_marginal else 'blue'

        for rate_attr in rate_attrs:
            rate = r[rate_attr]
            facecolor = color if rate_attr == 'prior_rate' else 'none'
            ax_all.scatter(rate, mse_db, c=color, marker=markers[rate_attr],
                          s=40, alpha=0.6, facecolors=facecolor)

    ax_all.set_title('All Rates Combined')
    ax_all.set_xlabel('Rate (bits/symbol)')
    ax_all.set_ylabel('Distortion (dB)')
    ax_all.grid(True, alpha=0.3)
    ax_all.set_xlim(-0.1, 12)
    ax_all.set_ylim(-30, 12)

    # Legends
    axes[0].legend(handles=[
        Line2D([0], [0], color='k', lw=2, label='WZ Bound'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='+ Lattice'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', ms=8, label='With SI (R)'),
        Line2D([0], [0], marker='x', color='red', ms=8, ls='', label='Marginal (M)'),
    ], fontsize=8)

    ax_all.legend(handles=[
        Line2D([0], [0], color='k', lw=2, label='WZ Bound'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='+ Lattice'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', ms=8, label='With SI (R)'),
        Line2D([0], [0], marker='x', color='red', ms=8, ls='', label='Marginal (M)'),
        Line2D([0], [0], marker='o', color='gray', ms=6, ls='', label='Prior Rate'),
        Line2D([0], [0], marker='s', color='gray', ms=6, ls='', label='Marginal Rate', fillstyle='none'),
        Line2D([0], [0], marker='^', color='gray', ms=6, ls='', label='Real Rate', fillstyle='none')
    ], fontsize=8)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    import argparse

    out_dir = Path(__file__).parent / "results"
    csv_path = out_dir / "quantizer_check.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-run', action='store_true', help='Skip experiments, only plot')
    args = parser.parse_args()

    if not args.skip_run:
        print("=== Running Experiments ===\n")
        run_experiments(out_dir)

    if csv_path.exists():
        print("\n=== Plotting Results ===\n")
        plot_results(csv_path)
    else:
        print(f"No records found at {csv_path}")

