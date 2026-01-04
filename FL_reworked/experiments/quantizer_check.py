"""Quantizer Rate-Distortion Validation"""
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import sys

# Add project root to path (works both in PyCharm and terminal)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from FL_reworked.codec import create_codec
from FL_reworked.cancer_protocol import CancerCodec, CancerConfig
from FL_reworked.run_fl import FLConfig

# ============== CONFIG ==============
NUM_REPEATS = 2
DATA_SIZE = 2_000_000
NOISE_POWER = 0.1
# (round_type, bins_per_plane, num_planes) - M=marginal, T=with side info, TM=with side info + marginal prior
CONFIGS = []
# CONFIGS += [('T', 4, 2),('T', 4, 3),('T', 16, 2),]
# CONFIGS += [('TM', 4, 2),('TM', 4, 3),]
# ====================================

def run_experiments(out_path: Path):
    """Run experiments and save to CSV."""
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "quantizer_check.csv"

    c_cfg = CancerConfig()

    fl_cfg = FLConfig(num_clients=1, training_progress_bar=True, compile_mode=False)
    # "identity", "basic", "cancer", "cancer_raw", "cancer_only_normalize"
    fl_cfg.codec='cancer_only_normalize'

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
            record.method = fl_cfg.codec
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


def plot_bounds(ax, bound_rate, bound_dist, ptp_bound):
    """Helper to plot theoretical bounds."""
    ax.plot(bound_rate, bound_dist, 'k-', lw=2, label='WZ Bound')
    ax.plot(bound_rate, bound_dist + 1.53, 'k:', lw=2, label='+ Lattice')
    ax.plot(ptp_bound, bound_dist, 'g--', lw=1, label='PTP Bound')


def setup_axis(ax, title=None):
    """Configure axis with common settings."""
    ax.set_xlabel('Rate (bits/symbol)')
    ax.set_ylabel('Distortion (dB)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 8)
    ax.set_ylim(-30, -10)
    if title:
        ax.set_title(title)


def plot_scatter(ax, records, rate_attrs, rate_styles=None, combined=False):
    """Plot scatter points with appropriate styling."""
    rate_attrs = [rate_attrs] if isinstance(rate_attrs, str) else rate_attrs

    for r in records:
        mse_db = 10 * np.log10(r['mse'])
        is_marginal = 'M' in r['round_type']

        for rate_attr in rate_attrs:
            rate = r[rate_attr]

            if combined and rate_styles:
                # Combined plot: distinct markers per rate type
                style = rate_styles[rate_attr]
                fill_color = 'darkred' if is_marginal else 'darkblue'
                edge_color = 'orangered' if is_marginal else 'navy'
                is_filled = (rate_attr == 'prior_rate')

                ax.scatter(rate, mse_db, marker=style['marker'], s=style['size'],
                          facecolors=fill_color if is_filled else 'none',
                          edgecolors=edge_color, linewidths=style['edge_width'],
                          alpha=0.8 if is_filled else 0.9, zorder=3)
            else:
                # Simple plot: basic markers
                color, marker = ('red', 'x') if is_marginal else ('blue', 'o')
                ax.scatter(rate, mse_db, c=color, marker=marker, s=50, alpha=0.7)


def create_legend(ax, combined=False):
    """Create legend with appropriate entries."""
    base_handles = [
        Line2D([0], [0], color='k', lw=2, label='WZ Bound'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='+ Lattice'),
    ]

    if combined:
        base_handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
                   markeredgecolor='navy', markeredgewidth=2, ms=10, label='With SI (R)', ls=''),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
                   markeredgecolor='orangered', markeredgewidth=2, ms=10, label='Marginal (M)', ls=''),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='gray', markeredgewidth=2, ms=9, label='Prior Rate', ls=''),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, ms=8, label='Marginal Rate', ls=''),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, ms=9, label='Real Rate', ls='')
        ]
    else:
        base_handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', ms=8, label='With SI (R)'),
            Line2D([0], [0], marker='x', color='red', ms=8, ls='', label='Marginal (M)'),
        ]

    ax.legend(handles=base_handles, fontsize=8 if not combined else 9, loc='best')


def plot_results(csv_path: Path):
    """Plot rate-distortion curves."""
    records = load_records(csv_path)

    # Calculate WZ bounds
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
        plot_bounds(ax, bound_rate, bound_dist, ptp_bound)
        plot_scatter(ax, records, rate_attr)
        setup_axis(ax, title)

    # Combined plot with distinct markers
    rate_styles = {
        'prior_rate': {'marker': 'o', 'size': 100, 'edge_width': 2},
        'marginal_rate': {'marker': 's', 'size': 80, 'edge_width': 2},
        'entropy_real_rate': {'marker': '^', 'size': 90, 'edge_width': 2}
    }

    plot_bounds(axes[3], bound_rate, bound_dist, ptp_bound)
    plot_scatter(axes[3], records, rate_attrs, rate_styles, combined=True)
    setup_axis(axes[3], 'All Rates Combined')

    create_legend(axes[0])
    create_legend(axes[3], combined=True)

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

