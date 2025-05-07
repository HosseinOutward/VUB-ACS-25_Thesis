from functools import partial

import numpy as np
import rans.rANSCoder as rans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU

from wyner_ziv.customrnn import CustomRNN


class EncoderDecoderLayeredRNN(nn.Module):

    def __init__(self, input_dim, bins_per_plane, planes, layers, hidden_dim, rnn_type='rnn',
                 shared_encoder=True, shared_decoder=True, shared_priors=False, marginal=False, output_activation=None):
        super().__init__()

        self.bins_per_plane = bins_per_plane
        self.planes = planes
        self.shared_decoder = shared_decoder
        self.shared_encoder = shared_encoder
        self.shared_priors = shared_priors
        self.marginal = marginal
        self.output_activation = output_activation

        cond_input_dim = bins_per_plane if marginal else bins_per_plane + input_dim
        if rnn_type.lower() == 'rnn':
            self.encoder = CustomRNN(input_dim, hidden_dim, layers, output_activation=True)
            self.decoder = CustomRNN(bins_per_plane + input_dim, hidden_dim, layers, output_activation=True)
            self.conditionalRNN = CustomRNN(cond_input_dim, hidden_dim, layers, output_activation=False)
        elif rnn_type.lower() == 'lstm':
            self.encoder = LSTM(input_dim, hidden_dim, layers, batch_first=True)
            self.decoder = LSTM(bins_per_plane + input_dim, hidden_dim, layers, batch_first=True)
            self.conditionalRNN = LSTM(cond_input_dim, hidden_dim, layers)
        elif rnn_type.lower() == 'gru':
            self.encoder = GRU(input_dim, hidden_dim, layers, batch_first=True)
            self.decoder = GRU(bins_per_plane + input_dim, hidden_dim, layers, batch_first=True)
            self.conditionalRNN = GRU(cond_input_dim, hidden_dim, layers)

        if shared_encoder is False:
            self.binners = nn.ModuleList([nn.Linear(hidden_dim, bins_per_plane) for p in range(planes)])
        else:
            self.binner = nn.Linear(hidden_dim, bins_per_plane)

        if shared_decoder is False:
            self.reconstructors = nn.ModuleList([nn.Linear(hidden_dim, input_dim) for p in range(planes)])
        else:
            self.reconstructor = nn.Linear(hidden_dim, input_dim)

        if shared_priors is False:
            self.conditionalPriors = nn.ModuleList([nn.Linear(hidden_dim, bins_per_plane) for p in range(planes)])
        else:
            self.conditionalPrior = nn.Linear(hidden_dim, bins_per_plane)

    @property
    def code_size(self):
        return self.bins_per_plane

    def encode(self, x, tau=1.):
        if self.training:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        rnn_inputs = x.unsqueeze(1).repeat(1, self.planes, 1)
        rnn_out, _ = self.encoder(rnn_inputs)
        if self.shared_encoder is False:
            soft_codes = [softmax(binner(rnn_out[:, binner_idx]), dim=-1)
                          for binner_idx, binner in enumerate(self.binners)]
        else:
            soft_codes = [softmax(self.binner(rnn_out[:, binner_idx]), dim=-1) for binner_idx in range(self.planes)]
        bins = [torch.argmax(sc, dim=-1) for sc in soft_codes]

        if self.training:
            codes = soft_codes
        else:
            # hard codes
            codes = [F.one_hot(bin, num_classes=self.bins_per_plane) for bin in bins]
        return bins, codes

    def forward(self, x, y, tau=1.):
        # Encoder
        bins, codes = self.encode(x=x, tau=tau)

        # Decoder
        reconstructed = self.decode(codes=codes, y=y)

        # Priors
        priors = self.get_priors(codes=codes, y=y, tau=tau)

        return reconstructed, bins, codes, priors

    @staticmethod
    def entropy_encode(bins, priors):
        strings = []
        for bins_i, prior in zip(bins, priors):

            ans_encoder = rans.Encoder()

            symbols = bins_i.detach().cpu().numpy().flatten().astype(np.int32)
            probs = prior.detach().cpu().numpy().astype(np.float32)

            for s_idx, s in enumerate(symbols):
                ans_encoder.encode_symbol(probs[s_idx], s)

            strings.append(ans_encoder.get_encoded())

        return strings

    def entropy_decode(self, strings, y, broadcast_size):
        # Difference with the other scheme: everything has to be done sequentially because we need to decode the strings

        reconstructed = []
        hard_codes = []

        # Initialize priors inputs
        if self.marginal:
            rnn_inputs_prior = torch.zeros((broadcast_size[0], self.bins_per_plane),
                                           device=y.device).float().unsqueeze(1)
        else:
            rnn_inputs_prior = torch.cat([y, torch.zeros((broadcast_size[0], self.bins_per_plane), device=y.device)],
                                         dim=-1).unsqueeze(1)

        for p, string in enumerate(strings):
            # get probabilities
            prior_logits, _ = self.conditionalRNN(rnn_inputs_prior)
            if self.shared_priors is False:
                prior = F.softmax(self.conditionalPriors[p](prior_logits[:, p]), dim=-1)
            else:
                prior = F.softmax(self.conditionalPrior(prior_logits[:, p]), dim=-1)

            # Entropy decode
            decoder = rans.Decoder(string.copy())
            probs = prior.detach().cpu().numpy().astype(np.float32)
            data = []
            for i in range(broadcast_size[0]-1, -1, -1):
                # Decoding happens in reverse order !
                data.append(decoder.decode_symbol(probs[i]))

            # Reverse order of decoded data
            data.reverse()
            symbols = torch.tensor(data, device=y.device)
            hard_code = F.one_hot(symbols, num_classes=self.bins_per_plane)
            hard_codes.append(hard_code)

            # Decode with side-information
            rnn_d_inputs = torch.stack([torch.cat([hc, y], dim=-1) for hc in hard_codes], dim=1)
            rnn_d_out, _ = self.decoder(rnn_d_inputs)
            if self.shared_decoder is False:
                decoded = self.reconstructors[p](rnn_d_out[:, p])
            else:
                decoded = self.reconstructor(rnn_d_out[:, p])

            if self.output_activation is not None:
                decoded = self.output_activation(decoded)

            reconstructed.append(decoded)

            # Update prior inputs
            if p < self.planes:
                if self.marginal:
                    rnn_inputs_prior = torch.cat([torch.zeros_like(hard_codes[0]).unsqueeze(1),
                                                  torch.stack(hard_codes, dim=1)], dim=1).float()
                else:
                    rnn_inputs_prior = torch.cat([y.unsqueeze(1).repeat(1, self.planes, 1),
                                                  torch.cat([torch.zeros_like(hard_codes[0]).unsqueeze(1),
                                                             torch.stack(hard_codes, dim=1)], dim=1).float()
                                                  ], dim=-1)

        return reconstructed

    def decode(self, codes, y):
        rnn_d_inputs = torch.stack([torch.cat([c, y], dim=-1) for c in codes], dim=1)
        rnn_d_out, _ = self.decoder(rnn_d_inputs)
        if self.shared_decoder is False:
            reconstructed = [rec(rnn_d_out[:, rec_idx]) for rec_idx, rec in enumerate(self.reconstructors)]
        else:
            reconstructed = [self.reconstructor(rnn_d_out[:, rec_idx]) for rec_idx in range(self.planes)]

        if self.output_activation is not None:
            reconstructed = [self.output_activation(rec) for rec in reconstructed]

        return reconstructed

    def get_priors(self, codes, y=None, tau=1.):
        # y is only needed if the prior is conditional
        if self.training:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        if self.marginal:
            rnn_inputs_prior = torch.cat([torch.zeros_like(codes[0]).unsqueeze(1),
                                          torch.stack(codes[:-1], dim=1)], dim=1)
        else:
            rnn_inputs_prior = torch.cat([y.unsqueeze(1).repeat(1, self.planes, 1),
                                          torch.cat([torch.zeros_like(codes[0]).unsqueeze(1),
                                                     torch.stack(codes[:-1], dim=1)], dim=1)
                                          ], dim=-1)
        if self.training:
            rnn_inputs_prior = rnn_inputs_prior.detach()
        else:
            rnn_inputs_prior = rnn_inputs_prior.float()  # hard codes are LongTensors
        prior_logits, _ = self.conditionalRNN(rnn_inputs_prior)
        if self.shared_priors is False:
            priors = [cp(prior_logits[:, cp_idx]) for cp_idx, cp in enumerate(self.conditionalPriors)]
            priors = [softmax(p, dim=-1) for p in priors]
        else:
            priors = [self.conditionalPrior(prior_logits[:, cp_idx]) for cp_idx in range(self.planes)]
            priors = [softmax(p, dim=-1) for p in priors]
        return priors
