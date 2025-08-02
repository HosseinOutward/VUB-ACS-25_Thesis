import csv
import os
import numpy as np
import torch

from components.broadcast_components.WZ_models.wz_quant_ANN import WZQuantizer
from components.broadcast_components.WZ_models.wz_quant_RNN import PL_EncoderDecoder_RNN
from experiments.make_basic_rnn_model import utilities

torch.set_float32_matmul_precision('medium')

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


def dict_to_csv(csv_dict, param, mode):
    if not os.path.exists(param):
        with open(param, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
            writer.writeheader()
    with open(param, mode=mode) as f:
        writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
        writer.writerow(csv_dict)

def objective(y, side_info_data, lr, tau, ld, cs, n_p, marginal):
    a = {
        'lr': lr,
        'tau': tau,
        'reconst_ld': ld,
        'num_planes': n_p,
        'bins_per_plane': cs,
        'marginal': marginal,
    }
    b = {
        'train_sample_size': 200_000
    }
    c = {
        'batch_size': 1_000
    }

    wz_model = PL_EncoderDecoder_RNN(inp_dim=1, side_info_size=1, **a).to(torch.float32)
    wz_quantizer = WZQuantizer(wz_model, count_side_info_data=1, enable_progress_bar=False, **b)
    wz_quantizer.train_model(y, side_info_data, epoch=180, **c)

    mse, _, real_bit_rate, prior_bit_rate, _ =\
        utilities.get_metrics(y, side_info_data, wz_quantizer)

    return 10*np.log10(mse), real_bit_rate, prior_bit_rate


if __name__ == '__main__':
    lrs = [1e-3, 1e-2]
    taus = [1, 5]
    reconst_ld = [50, 100, 400, 1000]
    code_size = [2,3,4,8,16]
    num_planes = [1,2,3]
    sample_count = 3
    marginals = [False, True]

    temp = np.random.normal(0, np.sqrt(1), 20_000_000, ).astype(np.float32)
    y = temp + np.random.normal(0, np.sqrt(0.01), 20_000_000, ).astype(np.float32)
    side_info_data = [temp]

    prod_list = [(lr, tau, ld, cs, n_p, marg)
                 for lr in lrs      for tau in taus
                 for ld in reconst_ld        for cs in code_size
                 for n_p in num_planes for marg in marginals]
    test_count = len(prod_list)
    print(f"Test count: {test_count}*{sample_count}")
    for i, (lr, tau, ld, cs, n_p, marg) in enumerate(prod_list):

        if n_p != 1 and cs > 4:
            continue

        print(f'  ** >> Running grid search {i/test_count*100:.1f}% ({i+1}/{test_count}): \n'
              f'         {prod_list[i]}"')

        for j in range(sample_count):
            eval_db, eval_rate, prior_rate = objective(y, side_info_data, lr, tau, ld, cs, n_p, marg)
            print(f"Test {j+1}/{sample_count} completed")

            csv_dict = {
                'mse': eval_db,
                'real_bit_rate': eval_rate,
                'prior_bit_rate': prior_rate,
                'whole_code_size': cs**n_p,
                'num_planes': n_p,
                'lr': lr,
                'tau': tau,
                'reconst_ld': ld,
            }
            cond = 'marginal' if marg else 'conditional'
            mono = 'monolithic' if n_p==1 else 'layered'
            dict_to_csv(csv_dict, f'res/hos_{mono},{cond}.csv', mode='a')

    print('Grid search completed. Results saved to wz_rnn_batch_size.csv')
