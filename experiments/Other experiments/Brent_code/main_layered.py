import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import parse_args
from model_layered import EncoderDecoderLayeredRNN


def main(args):
    args.debug=True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creating model
    model = EncoderDecoderLayeredRNN(input_dim=args.sample_dim, bins_per_plane=args.code_size, planes=args.planes,
                                     layers=args.layers, hidden_dim=args.hidden_units,
                                     shared_encoder=args.shared_encoder, shared_decoder=args.shared_decoder,
                                     shared_priors=args.shared_priors, rnn_type=args.rnn_type, marginal=args.marginal)
    model = model.to(device)

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.3)
    mse_loss = nn.MSELoss()

    TRAIN_BATCHES = args.samples_per_epoch // args.batch_size

    for epoch in range(args.epochs):

        # Train
        model.train()

        train_loss = 0.0
        train_mse_loss = [0.0] * args.planes

        tau_t = args.tau * np.exp(epoch / args.epochs * np.log(0.1 / args.tau))
        print('tau={:.04f}'.format(tau_t))

        for batch_idx in range(int(TRAIN_BATCHES)):

            optimizer.zero_grad()

            # Source model:
            #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
            y = torch.empty([args.batch_size, args.sample_dim], device=device).normal_(mean=0, std=args.y_std)
            x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

            reconstruct, bin, out, prior = model.forward(x, y, tau=tau_t)

            loss = 0.0

            dists = []
            for i in range(args.planes):
                dist = mse_loss(reconstruct[i], x)
                dists.append(dist)
                # loss += (i+1) * LAMBDA * dist
                loss += args.ld * dist

            for i in range(args.planes):
                p_ux = out[i][torch.arange(out[i].size(0)), bin[i]]
                p_u = prior[i][torch.arange(out[i].size(0)), bin[i]]  # it is also p_u|y for the conditional model

                loss += torch.mean(torch.log((p_ux + 1e-12) / (p_u + 1e-12)))

            train_loss += loss.item()
            for i in range(args.planes):
                train_mse_loss[i] += dists[i].item()

            loss.backward()

            optimizer.step()

        scheduler.step()

        # Eval:

        eval_mse_loss = [0.0] * args.planes
        eval_rate = [0.0] * args.planes

        test_samples = args.samples_per_epoch
        if epoch == args.epochs-1 or (args.debug is False and epoch % args.eval_every_epochs == 0):
            test_samples = args.test_samples
        test_batches = int(test_samples // args.batch_size)

        model.eval()
        with torch.no_grad():
            for batch_idx in range(test_batches):
                # Source model:
                #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
                y = torch.empty([args.batch_size, args.sample_dim], device=device).normal_(mean=0, std=args.y_std)
                x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

                bins, hard_codes = model.encode(x=x)
                priors = model.get_priors(codes=hard_codes, y=y)
                if args.entropy_coder and args.marginal:
                    strings = model.entropy_encode(bins=bins, priors=priors)
                reconstructed = model.decode(codes=hard_codes, y=y)

                for p in range(args.planes):
                    dist = mse_loss(reconstructed[p], x)
                    eval_mse_loss[p] += dist.item()

                    if args.entropy_coder and args.marginal:
                        rate = len(strings[p]) * 32 / x.numel()
                        eval_rate[p] += rate
                    else:
                        p_u = priors[p][torch.arange(hard_codes[p].size(0)), bins[p]]  # it is also p_u|y for the conditional model
                        rate = torch.mean(-torch.log2(p_u + 1e-12))
                        eval_rate[p] += rate.item()

        train_db = 10 * np.log10(train_mse_loss[-1] / TRAIN_BATCHES)
        eval_db = 10 * np.log10(eval_mse_loss[-1] / test_batches)
        eval_rate = [er / test_batches for er in eval_rate]
        print('Epoch {}: train loss={:.06f}, train_distortion={:.06f} dB, eval_distortion={:.06f} dB'
              .format(epoch, train_loss / TRAIN_BATCHES, train_db, eval_db))
        for i in range(args.planes):
            print('PLANE {}: train_distortion={:.06f}, eval_distortion={:.06f}, rate: {:.06f} bits'
                  .format(i, train_mse_loss[i] / TRAIN_BATCHES,
                          eval_mse_loss[i] / test_batches, eval_rate[i]))

        os.makedirs(args.log_name, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'lambda': args.ld,
            'config': args,
        }, os.path.join(args.log_name, 'checkpoint.pth'))

        model.eval()
        with torch.no_grad():

            if (args.debug and epoch % args.eval_every_epochs == 0) or (args.debug is False and epoch == args.epochs-1):
                # 1D source:
                # Seeing if there is binning:
                fig, ax = plt.subplots(nrows=1, ncols=args.planes + 2, figsize=(4 * args.planes + 8, 4))

                x = torch.arange(-3.0, 3.0, 0.005).to(device).unsqueeze(1)
                y = torch.zeros_like(x)
                with torch.no_grad():
                    reconstruct, bins, _, _ = model.forward(x, y)

                [
                    ax[0].plot(
                        x.detach().cpu().numpy(), bin.detach().cpu().numpy(),
                        label='Plane {}'.format(bin_idx), linewidth=1.0
                    )
                        for bin_idx, bin in reversed(list(enumerate(bins)))
                ]
                ax[0].legend()

                # Visualizing the joined binner

                combined = torch.sum(torch.stack([bin * args.code_size**(args.planes-i-1)
                                                  for i, bin in enumerate(bins)], dim=0), dim=0)
                ax[1].plot(x.detach().cpu().numpy(), combined.detach().cpu().numpy(), linewidth=1.0)

                # Seeing the reconstruction points
                for j in range(args.planes):
                    for i in range(args.code_size):

                        y = torch.arange(-3.0, 3.0, 0.01).to(device).unsqueeze(1)
                        if j>0:
                            codes_input = [F.one_hot(torch.zeros([y.shape[0]], dtype=torch.long, device=device),
                                                     num_classes=args.code_size).float() for _ in range(j)]
                        else:
                            codes_input = []
                        hard_codes = F.one_hot(torch.ones([y.shape[0]], dtype=torch.long, device=device) * i,
                                          num_classes=args.code_size).float()
                        codes_input.append(hard_codes)
                        if j < args.planes:
                            codes_input.extend([F.one_hot(torch.zeros([y.shape[0]], dtype=torch.long, device=device),
                                            num_classes=args.code_size).float() for _ in range(args.planes - j - 1)])

                        reconstruct = model.decode(codes_input, y)

                        ax[j+2].plot(y.detach().cpu().numpy(), reconstruct[j].detach().cpu().numpy(), label='bin {}{}{}'.format(
                            ''.join(['0' for _ in range(j)]), i, ''.join(['X' for _ in range(args.planes - j - 1)])
                        ))

                [ax[p + 2].legend() for p in range(args.planes)]
                if args.debug:
                    plt.show()
                else:
                    plt.savefig(os.path.join(args.log_name, 'compressor_{}.png'.format(epoch)))

        if epoch == args.epochs-1 or (args.debug is False and epoch % args.eval_every_epochs == 0):
            with open(os.path.join(args.log_name, 'results.txt'), 'a') as myfile:
                myfile.write('Epoch={}\n'.format(epoch))
                for i in range(args.planes):
                    myfile.write('PLANE {}: train_distortion={:.08f}, eval_distortion={:.08f}, rate: {:.06f} bits\n'
                                 .format(i, train_mse_loss[i] / TRAIN_BATCHES,
                                         eval_mse_loss[i] / test_batches, eval_rate[i]))


if __name__ == '__main__':
    main(parse_args(layered=True))
