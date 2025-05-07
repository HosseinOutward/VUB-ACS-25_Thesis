import argparse

def parse_args(layered):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_step', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--samples_per_epoch', type=int, default=2e5)
    parser.add_argument('--test_samples', type=int, default=1e7)
    parser.add_argument('--ld', type=float, default=100.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--sample_dim', type=int, default=1)
    parser.add_argument('--hidden_units', type=int, default=100)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--code_size', type=int, default=2)
    parser.add_argument('--noise_power', type=float, default=0.01)
    parser.add_argument('--y_std', type=float, default=1.0)
    parser.add_argument('--log_name', type=str, default='temp')
    parser.add_argument('--marginal', action='store_true')
    parser.add_argument('--debug', action='store_true')
    if layered:
        parser.add_argument('--planes', type=int, default=3)
        parser.add_argument('--rnn_type', type=str, default='rnn')
        parser.add_argument('--shared_encoder', default=False, action='store_true')
        parser.add_argument('--shared_decoder', default=False, action='store_true')
        parser.add_argument('--shared_priors', default=False, action='store_true')
        parser.add_argument('--entropy_coder', action='store_true')
        parser.add_argument('--eval_every_epochs', type=int, default=5)

    import sys
    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
