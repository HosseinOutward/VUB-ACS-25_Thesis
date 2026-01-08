"""Quantizer Rate-Distortion Validation

This script validates Wyner-Ziv (WZ) quantizer performance by:
1. Training quantizer models (conditional or marginal)
2. Compressing test signals with side information at decoder
3. Comparing three rate metrics against theoretical WZ bounds

Model Types:
- 'T' (Temporal/Conditional): Encoder/decoder both use side information
- 'TM' (Temporal-Marginal): Marginal prior model (encoder has no side info access)

Rate Metrics:
- Conditional Rate: Estimated from posterior P(bin|side_info) using trained model
- Marginal Rate: Estimated from bin histogram P(bin) empirically
- Actual Entropy Rate: Real compression rate from entropy coding
"""
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

from FL_code.codec import create_codec
from FL_code.cancer_protocol import CancerCodec, CancerConfig
from FL_code.run_fl import FLConfig

# ============== EXPERIMENT CONFIG ==============
NUM_REPEATS = 2  # Number of trials per configuration
DATA_SIZE = 2_000_000  # Number of signal samples
NOISE_POWER = 0.1  # Variance of additive noise: signal = side_info + noise
CANCER_CODEC_NAME = 'cancer' # "identity", "basic", "cancer", "cancer_raw"

# Each config: (model_type, bins_per_plane, num_planes)
# model_type: 'T' = conditional model, 'TM' = marginal model
QUANTIZER_CONFIGS = []
QUANTIZER_CONFIGS += [('T', 2, 2), ('T', 4, 3),]
QUANTIZER_CONFIGS += [('TM', 2, 2), ('TM', 4, 3),]
# ===============================================

def run_experiments(out_path: Path):
    """Run compression experiments across different quantizer configurations.

    For each configuration:
    1. Generate synthetic WZ data (signal = side_info + noise)
    2. Train quantizer model (conditional or marginal type)
    3. Compress signal with side_info available at decoder
    4. Record distortion and rate metrics
    """
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "quantizer_check.csv"

    cancer_cfg = CancerConfig()
    fl_cfg = FLConfig(num_clients=1, training_progress_bar=True, compile_mode=False)

    fl_cfg.codec=CANCER_CODEC_NAME

    first_write = True
    for model_type, bins_per_plane, num_planes in QUANTIZER_CONFIGS:
        print(f"\n=== Model={model_type} | BinsPerPlane={bins_per_plane} | NumPlanes={num_planes} ===")

        for trial_idx in range(NUM_REPEATS):
            torch.manual_seed(trial_idx * 42)
            np.random.seed(trial_idx * 42)

            # Generate WZ scenario: signal = side_info + noise
            side_info = torch.randn(DATA_SIZE)
            signal = side_info + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)

            # Configure codec with current quantizer settings
            codec: CancerCodec = create_codec(fl_cfg, None)
            codec.c_cfg = cancer_cfg
            codec.c_cfg.warmup_phase = ((model_type, bins_per_plane, num_planes),)

            # Provide side information to decoder (stored as past reconstruction)
            codec.client_past_reconst[0] = [side_info.clone().to(torch.float16)]

            # Execute compression pipeline
            record = codec.create_record(round_id=0, client_id=0)
            record.codec_class_used = fl_cfg.codec
            record.model_size = DATA_SIZE
            compressed_payload = codec.encode(signal, record)
            _ = codec.decode(compressed_payload, record)

            print(f"  Trial [{trial_idx+1}/{NUM_REPEATS}] "
                  f"MSE={record.mse:.6f} | "
                  f"ConditionalRate={record.prior_rate:.3f} | "
                  f"MarginalRate={record.marginal_rate:.3f}")

            # Save results incrementally to CSV
            row = record.to_dict()
            row['trial_number'] = trial_idx

            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if first_write:
                    writer.writeheader()
                    first_write = False
                writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")



def load_records(csv_path: Path):
    """Load compression records from CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return [{k: float(v) if k in ['mse', 'prior_rate', 'marginal_rate', 'entropy_real_rate'] else v
                 for k, v in row.items()} for row in reader]


def plot_theoretical_bounds(ax, wz_rate, wz_distortion_db, point_to_point_rate):
    """Plot theoretical rate-distortion bounds.

    Args:
        ax: Matplotlib axis
        wz_rate: Wyner-Ziv bound rates
        wz_distortion_db: Distortion in dB
        point_to_point_rate: Point-to-point (no side info) bound rates
    """
    ax.plot(wz_rate, wz_distortion_db, 'k-', lw=2, label='WZ Bound (with SI)')
    ax.plot(wz_rate, wz_distortion_db + 1.53, 'k:', lw=2, label='WZ + Lattice Loss')
    ax.plot(point_to_point_rate, wz_distortion_db, 'g--', lw=1, label='Point-to-Point Bound (no SI)')


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
    """Plot scatter points for rate-distortion performance.

    Args:
        ax: Matplotlib axis
        records: List of experiment records
        rate_attrs: Rate attribute name(s) to plot ('prior_rate', 'marginal_rate', or 'entropy_real_rate')
        rate_styles: Dict mapping rate_attr to style config (for combined plots)
        combined: If True, use distinct markers for different rate types
    """
    rate_attrs = [rate_attrs] if isinstance(rate_attrs, str) else rate_attrs

    for record in records:
        distortion_db = 10 * np.log10(record['mse'])
        is_marginal_model = 'M' in record['round_type']  # TM = marginal model, T = conditional model

        for rate_attr in rate_attrs:
            rate_value = record[rate_attr]

            if combined and rate_styles:
                # Combined plot: use distinct markers for different rate types
                style = rate_styles[rate_attr]
                base_color = 'darkred' if is_marginal_model else 'darkblue'
                edge_color = 'orangered' if is_marginal_model else 'navy'

                # Fill only conditional rate points to distinguish from other rates
                is_filled = (rate_attr == 'prior_rate')

                ax.scatter(rate_value, distortion_db, marker=style['marker'], s=style['size'],
                          facecolors=base_color if is_filled else 'none',
                          edgecolors=edge_color, linewidths=style['edge_width'],
                          alpha=0.8 if is_filled else 0.9, zorder=3)
            else:
                # Simple plot: just distinguish model types
                color = 'red' if is_marginal_model else 'blue'
                marker = 'x' if is_marginal_model else 'o'
                ax.scatter(rate_value, distortion_db, c=color, marker=marker, s=50, alpha=0.7)


def create_legend(ax, combined=False):
    """Create legend with appropriate entries.

    Args:
        ax: Matplotlib axis
        combined: If True, include rate type markers in legend
    """
    base_handles = [
        Line2D([0], [0], color='k', lw=2, label='WZ Bound'),
        Line2D([0], [0], color='k', lw=2, ls=':', label='+ Lattice'),
    ]

    if combined:
        # Combined plot legend: show both model types and rate types
        base_handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
                   markeredgecolor='navy', markeredgewidth=2, ms=10,
                   label='Conditional Model (T)', ls=''),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
                   markeredgecolor='orangered', markeredgewidth=2, ms=10,
                   label='Marginal Model (TM)', ls=''),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markeredgecolor='gray', markeredgewidth=2, ms=9,
                   label='Conditional Rate', ls=''),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, ms=8,
                   label='Marginal Rate', ls=''),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='none',
                   markeredgecolor='gray', markeredgewidth=2, ms=9,
                   label='Actual Entropy Rate', ls='')
        ]
    else:
        # Simple plot legend: only show model types
        base_handles += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', ms=8,
                   label='Conditional Model (T)'),
            Line2D([0], [0], marker='x', color='red', ms=8, ls='',
                   label='Marginal Model (TM)'),
        ]

    ax.legend(handles=base_handles, fontsize=8 if not combined else 9, loc='best')


def plot_results(csv_path: Path):
    """Plot rate-distortion curves comparing different rate metrics against theoretical bounds."""
    records = load_records(csv_path)

    # Calculate theoretical Wyner-Ziv bounds
    conditional_variance = NOISE_POWER / (NOISE_POWER + 1.0)
    mse_range = np.linspace(1e-6, conditional_variance * 0.999, 500)
    wz_rate_bound = 0.5 * np.log2(conditional_variance / mse_range)
    distortion_db = 10 * np.log10(mse_range)
    point_to_point_rate_bound = 0.5 * np.log2((1 + NOISE_POWER) / mse_range)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    rate_attributes = ['prior_rate', 'marginal_rate', 'entropy_real_rate']
    plot_titles = [
        'Conditional Rate (from posterior model)',
        'Marginal Rate (from bin histogram)',
        'Actual Entropy Rate (from compressed bits)'
    ]

    # Plot individual rate types in first 3 subplots
    for ax, rate_attr, title in zip(axes[:3], rate_attributes, plot_titles):
        plot_theoretical_bounds(ax, wz_rate_bound, distortion_db, point_to_point_rate_bound)
        plot_scatter(ax, records, rate_attr)
        setup_axis(ax, title)
        create_legend(ax, combined=False)

    # Combined plot in 4th subplot with distinct markers for each rate type
    rate_marker_styles = {
        'prior_rate': {'marker': 'o', 'size': 100, 'edge_width': 2},
        'marginal_rate': {'marker': 's', 'size': 80, 'edge_width': 2},
        'entropy_real_rate': {'marker': '^', 'size': 90, 'edge_width': 2}
    }

    plot_theoretical_bounds(axes[3], wz_rate_bound, distortion_db, point_to_point_rate_bound)
    plot_scatter(axes[3], records, rate_attributes, rate_marker_styles, combined=True)
    setup_axis(axes[3], 'All Rate Metrics Combined')
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

