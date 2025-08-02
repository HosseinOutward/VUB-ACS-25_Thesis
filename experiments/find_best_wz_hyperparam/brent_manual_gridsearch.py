import csv

from components.other_utilities.brent_wz_models import EncoderDecoderLayeredRNN
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dict_to_csv(csv_dict, param, mode):
    if not os.path.exists(param):
        with open(param, mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
            writer.writeheader()
    with open(param, mode=mode) as f:
        writer = csv.DictWriter(f, fieldnames=csv_dict.keys())
        writer.writerow(csv_dict)

def parse_args(layered):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--samples_per_epoch', type=int, default=2e5)
    parser.add_argument('--noise_power', type=float, default=0.01)
    parser.add_argument('--y_std', type=float, default=1.0)
    parser.add_argument('--log_name', type=str, default='temp')
    parser.add_argument('--debug', action='store_true')
    if layered:
        parser.add_argument('--entropy_coder', action='store_true')
        parser.add_argument('--eval_every_epochs', type=int, default=5)

    return parser.parse_args()


def main(code_size, num_planes, lr, tau, recon_ld, marginal = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args(layered=True)

    model = EncoderDecoderLayeredRNN(input_dim=1, bins_per_plane=code_size, num_planes=num_planes,
                                     layers=2, hidden_dim=100, side_info_size=1,
                                     shared_encoder=False, shared_decoder=False,
                                     shared_priors=False, rnn_type='rnn', marginal=marginal)
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80 if marginal else 40, gamma=0.3)
    mse_loss = nn.MSELoss()

    TRAIN_BATCHES = int(args.samples_per_epoch // args.batch_size)

    for epoch in range(args.epochs):

        # Train
        model.train()

        train_loss = 0.0
        train_mse_loss = [0.0] * num_planes

        tau_t = tau * np.exp(epoch / args.epochs * np.log(0.1 / tau))
        # print('tau={:.04f}'.format(tau_t))

        for batch_idx in range(TRAIN_BATCHES):

            optimizer.zero_grad()

            # Source model:
            #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
            y = torch.empty([args.batch_size, 1], device=device).normal_(mean=0, std=args.y_std)
            x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

            reconstruct, bin, out, prior = model.forward(x, y, tau=tau_t)

            loss = 0.0

            dists = []
            for i in range(num_planes):
                dist = mse_loss(reconstruct[i], x)
                dists.append(dist)
                # loss += (i+1) * LAMBDA * dist
                loss += recon_ld * dist

            for i in range(num_planes):
                p_ux = out[i][torch.arange(out[i].size(0)), bin[i]]
                p_u = prior[i][torch.arange(out[i].size(0)), bin[i]]  # it is also p_u|y for the conditional model

                loss += torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

            train_loss += loss.item()
            for i in range(num_planes):
                train_mse_loss[i] += dists[i].item()

            loss.backward()

            optimizer.step()

        scheduler.step()

        # Eval: *********************************************

    eval_mse_loss = [0.0] * num_planes
    eval_rate = [0.0] * num_planes
    bins_array = []

    test_batches = int(1e7 // args.batch_size)

    model.eval()
    with torch.no_grad():
        for batch_idx in range(test_batches):
            # Source model:
            #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
            y = torch.empty([args.batch_size, 1], device=device).normal_(mean=0, std=args.y_std)
            x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

            bins, hard_codes = model.encode(x=x)
            bins_array.append(torch.stack(bins))
            priors = model.get_priors(codes=hard_codes, y=y)
            reconstructed = model.decode(codes=hard_codes, y=y)

            for p in range(num_planes):
                dist = mse_loss(reconstructed[p], x)
                eval_mse_loss[p] += dist.item()

                p_u = priors[p][torch.arange(hard_codes[p].size(0)), bins[p]]  # it is also p_u|y for the conditional model
                rate = torch.mean(-torch.log2(p_u + 1e-12))
                eval_rate[p] += rate.item()

    bins_array = torch.cat(bins_array, dim=1)
    real_rate = 0
    # replace each bin with its prob
    for p in range(num_planes):
        per_bin_probs = torch.stack(bins_array[p].unique(return_counts=True)).T
        stage_bins_probs = bins_array[p].cpu().clone().float()
        for bin_name, bin_count in per_bin_probs:
            temp = bins_array[p].to(int).cpu().numpy()==bin_name.to(int).cpu().numpy()
            stage_bins_probs[temp] = (bin_count / bins_array[p].size(0)).item()
        real_rate += torch.mean(-torch.log2(stage_bins_probs + 1e-12)).cpu().numpy()

    # train_db = 10 * np.log10(train_mse_loss[-1] / TRAIN_BATCHES)
    eval_db = 10 * np.log10(eval_mse_loss[-1] / test_batches)
    eval_rate = [er / test_batches for er in eval_rate]

    return sum(eval_rate), real_rate, eval_db



if __name__ == '__main__':
    lrs = [1e-3, 1e-2]
    taus = [1, 5]
    reconst_ld = [50, 100, 400, 1000]
    code_size = [2,3,4,8,16]
    num_planes = [1,2,3]
    sample_count = 3
    marginals = [False, True]

    #120 per monol, 288 per layered

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
            prior_rate, eval_rate, eval_db = main(cs, n_p, lr, tau, ld, marginal=marg)
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
            dict_to_csv(csv_dict, f'res/brent_{mono},{cond}.csv', mode='a')

    print('Grid search completed. Results saved to wz_rnn_batch_size.csv')

