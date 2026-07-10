"""Quantizer Rate-Distortion Validation - SIMPLIFIED VERSION

Removed all error handling, fallbacks, and defensive checks.
Let it crash to find edge cases!
"""
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import itertools


from FL_code.FL_core.codec import create_codec
from FL_code.cancer_protocol import Access, CancerCodec, CancerRecord

# import pydevd_pycharm
# pydevd_pycharm.settrace('ETROFLOCK', port=32112,
#         stdout_to_server=True, stderr_to_server=True, suspend=True)

@dataclass
class ExperimentConfig:
    codec_name: str
    model_type: str
    bins_per_plane: int
    num_planes: int

    def __hash__(self):
        return hash((self.codec_name, self.model_type, self.bins_per_plane, self.num_planes))

    def to_dict_key(self):
        return f"{self.codec_name}_{self.model_type}_{self.bins_per_plane}b_{self.num_planes}p"


@dataclass
class PlotStyle:
    color: str
    marker: str
    label: str


# ============== CONFIG ==============
NUM_REPEATS = 2
DATA_SIZE = 2_000_000
NOISE_POWER = 0.1

# Define codec and quantizer configurations to test
CODEC_NAMES = []
CODEC_NAMES += ['cancer|no_model_slices']
MODEL_TYPES = []
MODEL_TYPES += ['M','TM','T','R','RM']
QUANTIZER_SETTINGS = []
# QUANTIZER_SETTINGS += [(2,1)]
# QUANTIZER_SETTINGS += [(2, 2), (4, 3)]

EXPERIMENT_CONFIGS = [
    ExperimentConfig(codec, model, bins, planes)
    for codec, model, (bins, planes) in itertools.product(
        CODEC_NAMES, MODEL_TYPES, QUANTIZER_SETTINGS)
]
# ====================================


def generate_colors(n: int) -> List[str]:
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    return [base_colors[i % len(base_colors)] for i in range(n)]


def create_config_key_from_record(rec: dict) -> str:
    codec = rec['codec_class_used']
    model = rec['round_type']
    bins = int(float(rec['bins_per_plane']))
    planes = int(float(rec['num_planes']))
    return f"{codec}_{model}_{bins}b_{planes}p"


def get_plot_styles_from_data(records: List[dict]) -> Dict[str, PlotStyle]:
    codec_model_pairs = set()
    config_to_codec_model = {}

    for rec in records:
        codec_name = rec['codec_class_used']
        model_type = rec['round_type']
        codec_model_key = f"{codec_name}_{model_type}"
        codec_model_pairs.add(codec_model_key)
        full_config_key = create_config_key_from_record(rec)
        config_to_codec_model[full_config_key] = codec_model_key

    codec_model_list = sorted(codec_model_pairs)
    pair_colors = dict(zip(codec_model_list, generate_colors(len(codec_model_list))))

    model_markers = {
        'T': 'o',
        'TM': 'x',
        'R': 's',
        'M': '^',
        'RM': 'v',
    }

    styles = {}
    seen_labels = set()

    for rec in records:
        config_key = create_config_key_from_record(rec)
        if config_key in styles:
            continue

        codec_name = rec['codec_class_used']
        model_type = rec['round_type']
        codec_model_key = f"{codec_name}_{model_type}"

        color = pair_colors[codec_model_key]
        marker = model_markers[model_type]
        label = f"{codec_name}|{model_type}"

        if label in seen_labels:
            label = None
        else:
            seen_labels.add(label)

        styles[config_key] = PlotStyle(color=color, marker=marker, label=label)

    return styles


def append_to_csv(csv_path: Path, record: dict):
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def run_single_experiment(config: ExperimentConfig, trial_idx: int) -> dict:
    torch.manual_seed(trial_idx * 42)
    np.random.seed(trial_idx * 42)

    side_info = torch.randn(DATA_SIZE)
    signal = side_info + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)

    codec: CancerCodec = create_codec(config.codec_name, None)
    codec.c_cfg.training_progress_bar = True

    codec.c_cfg.warmup_phase = ((config.model_type, config.bins_per_plane, config.num_planes),)

    record = codec.create_codec_record(round_id=0, client_id=0)

    temp = signal + torch.randn(DATA_SIZE) * np.sqrt(NOISE_POWER)
    possible_si = [
        side_info.detach().clone().to(device="cpu", dtype=torch.float16),
        temp.detach().to(device="cpu", dtype=torch.float16),
    ]
    if config.model_type in ['T', 'TM']:
        seed_record = CancerRecord(-1, 0, config.codec_name, phase="seed", round_type="T")
        codec.reconstruction_history.commit(possible_si[0], seed_record, Access.TEMPORAL_TOO)
    elif config.model_type in ['R', 'RM']:
        for seed_round, seed_tensor in enumerate(possible_si):
            seed_record = CancerRecord(seed_round, 0, config.codec_name, phase="seed", round_type="R")
            codec.reconstruction_history.commit(seed_tensor, seed_record, Access.SERVER_ONLY)
        record.round_id = 2
    elif config.model_type == 'M':
        seed_record = CancerRecord(-1, 0, config.codec_name, phase="seed", round_type="P")
        codec.reconstruction_history.commit(possible_si[0], seed_record, Access.SERVER_ONLY)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    record.codec_class_used = config.codec_name
    record.model_size = DATA_SIZE
    compressed_payload = codec.encode(signal, record)
    _ = codec.decode(compressed_payload, record)

    print(f'used si count: {len(codec.frozen_quantizers[0].side_info_list_used)}, '
          f'extra si count: {len(codec.frozen_quantizers[0].extra_si_for_prior)}')

    result = record.to_dict()
    result['trial_number'] = trial_idx
    result['config_key'] = config.to_dict_key()

    return result


def run_experiments(out_path: Path):
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / "quantizer_check.csv"

    total_runs = len(EXPERIMENT_CONFIGS) * NUM_REPEATS
    current_run = 0

    for config in EXPERIMENT_CONFIGS:
        print(f"\n=== {config.codec_name} | {config.model_type} | "
              f"{config.bins_per_plane}bins × {config.num_planes}planes ===")

        for trial_idx in range(NUM_REPEATS):
            current_run += 1
            result = run_single_experiment(config, trial_idx)

            print(f"  [{current_run}/{total_runs}] "
                  f"MSE={result['mse']:.6f} | "
                  f"Cond={result['prior_rate']:.3f} | "
                  f"Marg={result['marginal_rate']:.3f}")

            append_to_csv(csv_path, result)

    print(f"\n✓ Results saved to: {csv_path}")


def load_records(csv_path: Path) -> List[dict]:
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            numeric_fields = ['mse', 'prior_rate', 'marginal_rate', 'entropy_real_rate',
                            'bins_per_plane', 'num_planes']
            for key in numeric_fields:
                row[key] = float(row[key]) if row[key] is not None else 0.0
            records.append(row)
    return records


def compute_theoretical_bounds():
    conditional_variance = NOISE_POWER / (NOISE_POWER + 1.0)
    mse_range = np.linspace(1e-3, conditional_variance * 0.999, 500)

    return {
        'distortion_db': 10 * np.log10(mse_range),
        'wz_rate': 0.5 * np.log2(conditional_variance / mse_range),
        'p2p_rate': 0.5 * np.log2((1 + NOISE_POWER) / mse_range)
    }


def plot_theoretical_bounds(ax, bounds: dict):
    ax.plot(bounds['wz_rate'], bounds['distortion_db'], 'k-', lw=2, label='WZ Bound')
    ax.plot(bounds['wz_rate'], bounds['distortion_db'] + 1.53, 'k:', lw=2, label='+ Lattice')
    ax.plot(bounds['p2p_rate'], bounds['distortion_db'], 'g--', lw=1, alpha=0.5, label='P2P (no SI)')


def plot_rate_comparison(csv_path: Path):
    records = load_records(csv_path)
    bounds = compute_theoretical_bounds()
    styles = get_plot_styles_from_data(records)

    rate_types = {
        'prior_rate': 'Conditional Rate',
        'marginal_rate': 'Marginal Rate',
        'entropy_real_rate': 'Actual Entropy Rate'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    # Individual rate plots
    for ax, (rate_key, rate_name) in zip(axes[:3], rate_types.items()):
        plot_theoretical_bounds(ax, bounds)

        config_groups = {}
        for rec in records:
            key = create_config_key_from_record(rec)
            config_groups.setdefault(key, []).append(rec)

        for config_key, group_records in config_groups.items():
            style = styles[config_key]
            rates = [r[rate_key] for r in group_records]
            distortions = [10 * np.log10(r['mse']) for r in group_records]

            ax.scatter(rates, distortions,
                       color=style.color,
                       marker=style.marker,
                       s=60,
                       alpha=0.7,
                       label=style.label,
                       edgecolors='black',
                       linewidths=0.5)

        ax.set_xlabel('Rate (bits/symbol)')
        ax.set_ylabel('Distortion (dB)')
        ax.set_title(rate_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best', ncol=2)

    # Combined plot
    plot_theoretical_bounds(axes[3], bounds)

    rate_markers = {'prior_rate': 'o', 'marginal_rate': 's', 'entropy_real_rate': '^'}

    for rec in records:
        config_key = create_config_key_from_record(rec)
        style = styles[config_key]
        distortion_db = 10 * np.log10(rec['mse'])

        for rate_key, marker in rate_markers.items():
            axes[3].scatter(rec[rate_key], distortion_db,
                            color=style.color,
                            marker=marker,
                            s=50,
                            alpha=0.6,
                            edgecolors='black',
                            linewidths=0.5)

    axes[3].set_xlabel('Rate (bits/symbol)')
    axes[3].set_ylabel('Distortion (dB)')
    axes[3].set_title('All Metrics Combined')
    axes[3].grid(True, alpha=0.3)

    # Custom legend
    handles = [Line2D([0], [0], color='k', lw=2, label='WZ Bound')]
    for config_key, style in styles.items():
        if style.label:
            handles.append(Line2D([0], [0], marker=style.marker, color='w',
                                  markerfacecolor=style.color,
                                  markeredgecolor='black',
                                  markeredgewidth=0.5,
                                  ms=8, label=style.label))
    axes[3].legend(handles=handles, fontsize=6, loc='best', ncol=2)

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
        print("\n=== Plotting Results ===")
        plot_rate_comparison(csv_path)
    else:
        print(f"❌ No data found at {csv_path}")
