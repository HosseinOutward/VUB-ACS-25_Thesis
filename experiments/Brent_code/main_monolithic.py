#####################################################################################
## THIS CODE REPLICATES THE PAPER "Learned Wyner-Ziv Compressors Recover Binning" ###
#####################################################################################
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import parse_args
from model import EncoderDecoder


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creating model
    model = EncoderDecoder(input_dim=args.sample_dim, layers=args.layers, hidden_dim=args.hidden_units,
                           code_size=args.code_size, marginal=args.marginal)
    model = model.to(device)

    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.3)
    mse_loss = nn.MSELoss()

    TRAIN_BATCHES = args.samples_per_epoch // args.batch_size

    for epoch in range(args.epochs):

        # Train
        model.train()

        train_loss = 0.0
        train_mse_loss = 0.0

        tau_t = args.tau * np.exp(epoch/args.epochs * np.log(0.1 / args.tau))
        print('tau={:.04f}'.format(tau_t))

        for batch_idx in range(int(TRAIN_BATCHES)):

            optimizer.zero_grad()

            # Source model:
            #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
            y = torch.empty([args.batch_size, args.sample_dim], device=device).normal_(mean=0, std=args.y_std)
            x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

            reconstruct, bin, out, prior = model.forward(x, y, tau=tau_t)

            p_ux = out[torch.arange(out.size(0)), bin]
            p_u = prior[torch.arange(out.size(0)), bin] # it is also p_u|y for the conditional model

            dist = mse_loss(reconstruct, x)
            loss = torch.mean(torch.log(p_ux/(p_u + 1e-12))) + args.ld * dist

            train_loss += loss.item()
            train_mse_loss += dist.item()

            loss.backward()

            optimizer.step()

        scheduler.step()

        train_db = 10 * np.log10(train_mse_loss / TRAIN_BATCHES)
        train_mse_loss = train_mse_loss / TRAIN_BATCHES
        train_loss = train_loss / TRAIN_BATCHES

        # Eval:

        eval_mse_loss = 0.0
        eval_rate = 0.0

        test_samples = args.samples_per_epoch
        if epoch == args.epochs - 1 or (args.debug is False and epoch % 5 == 0):
            test_samples = args.test_samples
        test_batches = test_samples // args.batch_size

        model.eval()
        with torch.no_grad():
            for batch_idx in range(test_batches):
                # Source model:
                #   X = Y+N,    Y~N(0, Y_STD),  N~N(0, NOISE_STD)
                y = torch.empty([args.batch_size, args.sample_dim], device=device).normal_(mean=0, std=args.y_std)
                x = y + torch.empty_like(y).normal_(mean=0, std=np.sqrt(args.noise_power))

                reconstruct, bin, code_probs, prior = model.forward(x, y)

                p_u = prior[torch.arange(code_probs.size(0)), bin] # also the conditional one for the second model

                dist = mse_loss(reconstruct, x)

                eval_mse_loss += dist.item()

                eval_rate += torch.mean(-torch.log2(p_u + 1e-12)).item()

        eval_db = 10*np.log10(eval_mse_loss / test_batches)
        eval_rate = eval_rate / test_batches
        print('Epoch {}: train loss={:.06f}, train_distortion={:.06f} dB, eval_distortion={:.06f} dB'
              .format(epoch, train_loss, train_db, eval_db))
        print('train_distortion={:.06f}, eval_distortion={:.06f}'
              .format(train_mse_loss, eval_mse_loss / test_batches))
        print('rate: {:.06f} bits'.format(eval_rate))

        os.makedirs(args.log_name, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'lambda': args.ld,
            'config': args,
        }, os.path.join(args.log_name, 'checkpoint.pth'))

        model.eval()
        with torch.no_grad():
            # 1D source:
            # Seeing if there is binning:
            if (args.debug and epoch % 5 == 0) or (args.debug is False and epoch == args.epochs-1):
                fig, ax = plt.subplots(nrows=1, ncols=2)

                x = torch.arange(-4.0, 4.0, 0.01).to(device).unsqueeze(1)
                y = torch.zeros_like(x)
                with torch.no_grad():
                    reconstruct, bin, _, _ = model.forward(x, y)

                ax[0].plot(x.detach().cpu().numpy(), bin.detach().cpu().numpy())

                # Seeing the reconstruction points
                for i in range(args.code_size):
                    y = torch.arange(-4.0, 4.0, 0.01).to(device).unsqueeze(1)
                    codes = F.one_hot(torch.ones([y.shape[0]], dtype=torch.long, device=device)*i,
                                      num_classes=args.code_size).float()
                    reconstruct = model.decoder(codes, y)
                    ax[1].plot(y.detach().cpu().numpy(), reconstruct.detach().cpu().numpy(), label='bin={}'.format(i))

                ax[1].legend()
                if args.debug:
                    plt.show()
                else:
                    plt.savefig(os.path.join(args.log_name, 'compressor_{}.png'.format(epoch)))

            if epoch == args.epochs - 1 or (args.debug is False and epoch % 5 == 0):
                with open(os.path.join(args.log_name, 'results.txt'), 'a') as myfile:
                    myfile.write('Epoch={}\n'.format(epoch))
                    myfile.write('train_distortion={:.08f}, eval_distortion={:.08f}, rate: {:.06f} bits\n'
                                 .format(train_mse_loss / TRAIN_BATCHES,
                                         eval_mse_loss / test_batches, eval_rate))


if __name__ == '__main__':
    main(parse_args(layered=False))
