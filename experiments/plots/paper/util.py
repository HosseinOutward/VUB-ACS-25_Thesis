import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Reusable constants
PARAM_COUNT = 11_191_262
MB_TO_BITS = 1024 ** 2 * 8
# Default reports root. Can be overridden in load_data(reports_root=...).
reports_dir = Path("reports")


def preprocess_tensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tensor-like strings (e.g., 'tensor(0.9123)') to float values in common metric columns.

    This does not mutate the original DataFrame.
    """
    df_processed = df.copy()
    tensor_columns = ['train_auc', 'val_auc', 'train_acc', 'val_acc']

    for col in tensor_columns:
        if col in df_processed.columns:
            df_processed[col] = (
                df_processed[col]
                .astype(str)
                .apply(lambda x: float(re.findall(r"[\d\.]+", x)[0]) if 'tensor' in x else float(x))
            )
    return df_processed


def calculate_bit_rate(mbytes_received: float, *, param_count: int = PARAM_COUNT) -> float:
    """Calculate bit rate from megabytes received and model parameter count."""
    return (mbytes_received * MB_TO_BITS) / param_count


def load_data(
    reports_root: Optional[Path | str] = None,
    *,
    param_count: int = PARAM_COUNT,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load and process experiment data from the given reports directory.

    Arguments:
    - reports_root: Path to the 'reports' folder containing experiment subfolders. Defaults to util.reports_dir.
    - param_count: Total model parameter count used for bit-rate normalization.

    Returns:
    - auc_data:     {experiment -> DataFrame with AUC/ACC metrics per round}
    - broadcast_data:{experiment -> DataFrame with averaged broadcast metrics per round (+ bit_rate)}
    - wz_training_data:{experiment -> DataFrame with averaged theoretical rate/mape per round}
    """
    root = Path(reports_root) if reports_root is not None else reports_dir

    auc_data: Dict[str, pd.DataFrame] = {}
    broadcast_data: Dict[str, pd.DataFrame] = {}
    wz_training_data: Dict[str, pd.DataFrame] = {}

    if not root.exists():
        print(f"Reports directory {root} not found!")
        return auc_data, broadcast_data, wz_training_data

    for exp_folder in root.iterdir():
        if not exp_folder.is_dir():
            continue

        exp_name = exp_folder.name

        # Load AUC data
        global_metrics_file = exp_folder / "_global_metrics_before_round_start.csv"
        if global_metrics_file.exists():
            try:
                df = pd.read_csv(global_metrics_file)
                auc_data[exp_name] = preprocess_tensor_columns(df)
            except Exception:
                # Skip malformed files but continue other experiments
                pass

        # Load broadcast data
        wz_file = exp_folder / "_broadcast_protocol_stats" / "wz.csv"
        if wz_file.exists():
            try:
                df = pd.read_csv(wz_file)
                numeric_cols = ['mbytes_recived', 'mse', 'mape%', 'mae']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df_averaged = df.groupby('round_id')[numeric_cols].mean().reset_index()
                if 'mbytes_recived' in df_averaged.columns:
                    df_averaged['bit_rate'] = df_averaged['mbytes_recived'].apply(
                        lambda mb: calculate_bit_rate(mb, param_count=param_count)
                    )
                broadcast_data[exp_name] = df_averaged
            except Exception:
                pass

        # Load WZ training data
        wz_logs_folder = exp_folder / "wz_training_logs"
        if wz_logs_folder.exists():
            round_data = []
            for round_folder in wz_logs_folder.iterdir():
                if round_folder.is_dir() and round_folder.name.startswith("round_"):
                    try:
                        parts = round_folder.name.split("_")
                        round_id, agent_id = int(parts[1]), int(parts[3])

                        metrics_file = round_folder / "metrics.csv"
                        if metrics_file.exists():
                            df = pd.read_csv(metrics_file)
                            val_data = df.dropna(subset=['val_mape%', 'val_rate_bits'])
                            if len(val_data) > 0:
                                last_val = val_data.iloc[-1]
                                round_data.append({
                                    'round_id': round_id, 'agent_id': agent_id,
                                    'val_mape%': last_val['val_mape%'],
                                    'val_rate_bits': last_val['val_rate_bits']
                                })
                    except (ValueError, IndexError, KeyError):
                        continue
                    except Exception:
                        continue

            if round_data:
                df_rounds = pd.DataFrame(round_data)
                df_averaged = (
                    df_rounds.groupby('round_id')[['val_mape%', 'val_rate_bits']]
                    .mean()
                    .reset_index()
                )
                df_averaged['theoretical_bit_rate'] = df_averaged['val_rate_bits']
                wz_training_data[exp_name] = df_averaged

    print(
        f"Loaded data: {len(auc_data)} AUC, {len(broadcast_data)} broadcast, {len(wz_training_data)} WZ training"
    )
    return auc_data, broadcast_data, wz_training_data


def compare_final_metrics(
    auc_data: Dict[str, pd.DataFrame],
    broadcast_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create a comparison DataFrame of final/average metrics across experiments.

    Returns a DataFrame and prints a friendly summary.
    """
    all_experiments = set(list(auc_data.keys()) + list(broadcast_data.keys()))
    comparison_data = []

    for exp_name in all_experiments:
        row = {'experiment': exp_name}

        # AUC/accuracy metrics
        if exp_name in auc_data and len(auc_data[exp_name]) > 0:
            final_row = auc_data[exp_name].iloc[-1]
            for col, key in [('val_auc', 'final_val_auc'), ('val_acc', 'final_val_acc'), ('train_loss', 'final_train_loss')]:
                row[key] = final_row.get(col, np.nan)

        # Communication metrics
        if exp_name in broadcast_data and len(broadcast_data[exp_name]) > 0:
            bc_df = broadcast_data[exp_name]
            for col, key_avg, key_final in [('bit_rate', 'avg_bit_rate', None), ('mape%', 'avg_mape', 'final_mape')]:
                if col in bc_df.columns:
                    row[key_avg] = bc_df[col].mean()
                    if key_final:
                        row[key_final] = bc_df[col].iloc[-1]

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\n=== FINAL METRICS COMPARISON ===")
    # Use to_string for aligned display; fallback if very large
    try:
        print(comparison_df.to_string(index=False, float_format='%.4f'))
    except Exception:
        print(comparison_df)
    return comparison_df

