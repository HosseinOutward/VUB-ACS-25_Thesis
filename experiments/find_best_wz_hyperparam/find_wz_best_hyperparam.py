import numpy as np
import torch
import csv
import os
import sys
import argparse
import time
from itertools import product
from copy import deepcopy
from datetime import datetime
from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN

torch.set_float32_matmul_precision('medium')

csv_file_path = r'hyperparameter_search_results.csv'


def run_single_test(y, side_info_data, wz_config, seed_i):
    torch.manual_seed(42 + seed_i)
    np.random.seed(42 + seed_i)

    wz_model = PL_EncoderDecoder_RNN(
        inp_dim=wz_config['inp_dim'],
        side_info_size=len(side_info_data),
        num_planes=wz_config['num_planes'],
        bins_per_plane=wz_config['bins_per_plane'],
        lr=wz_config['lr'],
        tau=wz_config['tau'],
        reconst_ld=wz_config['reconst_ld']
    ).to(torch.float32)

    wz_quantizer = WZQuantizer(
        wz_model,
        train_sample_size=wz_config['train_sample_size'],
        count_side_info_data=len(side_info_data),
        enable_progress_bar=False
    )

    # try:
    wz_quantizer.train_model(
        y,
        side_info_data,
        epoch=wz_config['epochs'],
        batch_size=wz_config['batch_size']
    )

    # Get validation loss or performance metric
    res = {}
    keys = ['loss', 'mape%', 'rate_bits','real_bit_r', 'mse']
    for k in keys:
        k='val_' + k
        value = wz_model.trainer.callback_metrics.get(k)
        res[k] = float(value) if value is not None else float('nan')

    return res


def add_line_to_csv(sample_id, wz_config, data_type, res, csv_path):
    fieldnames = ['sample_id', 'y_type', 'no_si'] + list(res.keys()) + [f'cnfg_{k}' for k in wz_config.keys()]

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Write one row for each result tuple
        row = {
            'sample_id': sample_id,
            **res,
            'y_type': data_type[0],
            'no_si': data_type[1],
            **{f'cnfg_{k}': v for k, v in wz_config.items()},
        }
        writer.writerow(row)


def get_data(y_option, no_si):
    """Generate different types of data for testing"""
    if y_option == 'normal_dist':
        y = np.random.normal(0, 1, 1_000_000).astype(np.float32)
    elif y_option == 'complete_random':
        y = np.random.uniform(-1, 1, 1_000_000).astype(np.float32)
    elif y_option == 'exponential':
        y = np.random.exponential(0.5, 1_000_000).astype(np.float32)
    else:
        raise ValueError(f"Unknown data type: {y_option}")
        
    # Normalize to [-1, 1]
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1

    # Generate side info data
    if no_si:
        side_info_data = []
    else:
        # Add normal noise to side info
        side_info_data = [y + np.random.normal(0, 0.1, len(y))]

    # Remove outliers
    temp = np.percentile(y, [0.0003, 99.9997])
    mask = (y >= temp[0]) & (y <= temp[1])
    y = y[mask]
    side_info_data = [a[mask] for a in side_info_data]
    
    return y, side_info_data


def main():
    base_wz_config = {
        'inp_dim': 1,
        'num_planes': 3,
        'bins_per_plane': 4,
        'lr': 1e-4,
        'tau': 5.0,
        'reconst_ld': 100.0,
        'train_sample_size': 100_000,
        'epochs': 25,
        'batch_size': 10_000
    }

    what_to_check = {
        'tau': [1.0, 5.0, 10.0, 20.0],
        'reconst_ld': [1, 3, 5],
        'y_si_data': [('normal_dist', 1),#('complete_random', 1),('exponential', 1),
                      ('normal_dist', 0),#('complete_random', 0),('exponential', 0)
                      ]
    }

    num_samples = 3
    max_retries = 5
    output_dir = '.'

    # Generate all possible configurations
    param_combinations = []
    hyperparams = {k: v for k, v in what_to_check.items() if k != 'y_si_data'}
    
    # Create cartesian product of hyperparameters
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    
    for param_combo in product(*values):
        param_dict = dict(zip(keys, param_combo))
        for y_type, no_si in what_to_check['y_si_data']:
            config = {
                'params': param_dict,
                'data_type': (y_type, no_si)
            }
            param_combinations.append(config)
    
    print(f"Total configurations: {len(param_combinations)}")
    
    for config_id, config in enumerate(param_combinations):
        # Get the specific configuration for this job
        if config_id >= len(param_combinations):
            raise ValueError(f"Config ID {config_id} out of range. Max: {len(param_combinations)-1}")
        
        # Create the actual config by merging base config with specific params
        wz_config = deepcopy(base_wz_config)
        wz_config.update(config['params'])
        
        data_type = config['data_type']
        y_type, no_si = data_type
        
        print(f"Running config {config_id}: {wz_config}")
        print(f"Data type: {data_type}")
        
        # Generate data
        y, side_info_data = get_data(y_type, no_si)
        
        # Run multiple samples for this configuration
        csv_path = os.path.join(output_dir, 'hyperparameter_search_results.csv')
        
        print(f"Configuration will run {num_samples} samples with up to {max_retries} retries each")
        
        for sample_i in range(num_samples):
            print(f"Running sample {sample_i + 1}/{num_samples}")
            
            success = False
            for retry in range(max_retries):
                try:
                    if retry > 0:
                        print(f"  Retry {retry}/{max_retries - 1} for sample {sample_i + 1}")
                    
                    res = run_single_test(y, side_info_data, wz_config, sample_i)
                    
                    # Create unique sample ID
                    sample_id = f"config_{config_id}_sample_{sample_i}"
                    if retry > 0:
                        sample_id += f"_retry_{retry}"
                    
                    # Add results to CSV
                    add_line_to_csv(sample_id, wz_config, data_type, res, csv_path)
                    
                    print(f"Sample {sample_i + 1} completed successfully" + (f" (after {retry} retries)" if retry > 0 else ""))
                    success = True
                    break
                    
                except Exception as e:
                    print(f"  Attempt {retry + 1} failed with error: {e}")
                    if retry < max_retries - 1:
                        print(f"  Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"  All {max_retries} attempts failed for sample {sample_i + 1}")
            
            # If all retries failed, log the failure
            if not success:
                print(f"Sample {sample_i + 1} failed after {max_retries} attempts")
                res = {f'val_{k}': float('nan') for k in ['loss', 'mape%', 'rate_bits', 'real_bit_r', 'mse']}
                sample_id = f"config_{config_id}_sample_{sample_i}_FAILED"
                add_line_to_csv(sample_id, wz_config, data_type, res, csv_path)
        
        print(f"Configuration {config_id} completed")


if __name__ == "__main__":
    main()

