import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
from experiments.make_basic_rnn_model import utilities

torch.set_float32_matmul_precision('medium')


def dict_to_csv(csv_dict, param, mode):
    if not os.path.exists(param):
        with open(param, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
            writer.writeheader()
    with open(param, mode=mode) as f:
        writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
        writer.writerow(csv_dict)


def test_single_run(y, side_info_data, a, b, c):
    """Single test run that can be executed in a separate process"""
    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=1, num_planes=2, bins_per_plane=4, **a).to(torch.float32)
    wz_quantizer = WZQuantizer(wz_model, count_side_info_data=1, enable_progress_bar=False, **b)
    wz_quantizer.train_model(y, side_info_data, epoch=60, **c)

    mse, mspe, real_bit_rate, prior_bit_rate, softcodes_bit_rate =\
        utilities.get_metrics(y, side_info_data, wz_quantizer)

    return mse, mspe, real_bit_rate, prior_bit_rate, softcodes_bit_rate


def objective(y, side_info_data, lr, tau, ld, bs):
    sample_count = 3

    a = {
        'lr': lr,
        'tau': tau,
        'reconst_ld': ld,
    }
    b = {
        'train_sample_size': 200_000
    }
    c = {
        'batch_size': bs
    }

    # Execute test() calls in parallel using processes
    # with ProcessPoolExecutor(max_workers=sample_count) as executor:
    #     futures = [executor.submit(test_single_run, y, side_info_data, a, b, c) for _ in range(sample_count)]
    #
    #     for i, future in enumerate(futures):
    #         res = future.result()
    for i in range(sample_count):
        res = test_single_run(y, side_info_data, a, b, c)
        print(f"Test {i+1}/{sample_count} completed for lr={lr}, tau={tau}, ld={ld}")
        csv_dict = {
            'mse': res[0],
            'mspe': res[1],
            'real_bit_rate': res[2],
            'prior_bit_rate': res[3],
            'softcodes_bit_rate': res[4],
            'lr': a['lr'],
            'tau': a['tau'],
            'reconst_ld': a['reconst_ld'],
            'train_sample_size': b['train_sample_size'],
            'batch_size': c['batch_size'],
        }
        dict_to_csv(csv_dict, 'wz_rnn_batch_size.csv', mode='a')


if __name__ == '__main__':
    temp = np.random.normal(0, np.sqrt(1), 1_000_000, ).astype(np.float32)
    y = temp + np.random.normal(0, np.sqrt(0.01), 1_000_000, ).astype(np.float32)
    side_info_data = [temp]

    # Define hyperparameters to test
    lrs = [5e-3]
    taus = [4]
    reconst_ld = [400]
    batch_size = [1000, 10000, 50000]

    test_count = len(lrs) * len(taus) * len(reconst_ld) * len(batch_size)
    print(f"Test count: {test_count}*sample_count")
    i=0
    for lr in lrs:
        for tau in taus:
            for ld in reconst_ld:
                for bs in batch_size:
                    i+=1
                    print(f'Running grid search {i/test_count*100:.1f}% ({i}/{test_count}): ')
                    objective(y, side_info_data, lr, tau, ld, bs)
    print('Grid search completed. Results saved to wz_rnn_batch_size.csv')
