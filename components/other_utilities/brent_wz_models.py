import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU
from functools import partial
import numpy as np
import rans.rANSCoder as rans


class CustomRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, output_activation=False):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.output_activation = output_activation

        input_layers = [nn.Linear(input_dim, hidden_dim)]
        input_layers.extend([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layers - 1)
        ])
        self.input_layers = nn.ModuleList(input_layers)
        self.transition_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])

    def to(self, *args, **kwargs):
        super(CustomRNN, self).to(*args, **kwargs)
        self.input_layers.to(*args, **kwargs)
        self.transition_layers.to(*args, **kwargs)

    def forward(self, x, h_0=None):

        B, T, D = x.shape

        if h_0 is None:
            temp=self.transition_layers[0].weight
            h_0 = [torch.zeros(B, self.hidden_dim).to(temp.device).to(temp.dtype) for _ in
                   range(self.layers)]

        h_outputs = [h_0]

        for t in range(T):
            h_outputs.append([])
            for l in range(self.layers):
                if l == 0:
                    y = F.leaky_relu(self.input_layers[l](x[:, t]) + self.transition_layers[l](h_outputs[-2][l]))
                elif l < self.layers - 1 or self.output_activation:
                    y = F.leaky_relu(
                        self.input_layers[l](h_outputs[-1][l - 1]) + self.transition_layers[l](h_outputs[-2][l]))
                else:
                    y = self.input_layers[l](h_outputs[-1][l - 1]) + self.transition_layers[l](h_outputs[-2][l])
                h_outputs[-1].append(y)

        all_h_output = torch.stack([torch.stack(ht, dim=1) for ht in h_outputs], dim=1)

        return all_h_output[:, 1:, -1, :], all_h_output[:, 1:, :, :]


class EncoderDecoderLayeredRNN(nn.Module):
    def __init__(self, input_dim, side_info_size, bins_per_plane, num_planes, layers, hidden_dim, rnn_type='rnn',
                 shared_encoder=False, shared_decoder=False, shared_priors=False, marginal=False,
                 output_activation=None):
        super().__init__()

        assert bins_per_plane > 1, 'need at least single bit (2 bins) per plane'

        self.bins_per_plane:int = bins_per_plane
        self.num_planes:int = num_planes
        self.shared_decoder = shared_decoder
        self.shared_encoder = shared_encoder
        self.shared_priors = shared_priors
        self.marginal = marginal
        self.output_activation = output_activation

        cond_input_dim = bins_per_plane if marginal else bins_per_plane + side_info_size
        if rnn_type.lower() == 'rnn':
            self.encoder = CustomRNN(input_dim, hidden_dim, layers, output_activation=True)
            self.decoder = CustomRNN(bins_per_plane + side_info_size, hidden_dim, layers, output_activation=True)
            self.conditionalRNN = CustomRNN(cond_input_dim, hidden_dim, layers, output_activation=False)
        elif rnn_type.lower() == 'lstm':
            self.encoder = LSTM(input_dim, hidden_dim, layers, batch_first=True)
            self.decoder = LSTM(bins_per_plane + side_info_size, hidden_dim, layers, batch_first=True)
            self.conditionalRNN = LSTM(cond_input_dim, hidden_dim, layers)
        elif rnn_type.lower() == 'gru':
            self.encoder = GRU(input_dim, hidden_dim, layers, batch_first=True)
            self.decoder = GRU(bins_per_plane + side_info_size, hidden_dim, layers, batch_first=True)
            self.conditionalRNN = GRU(cond_input_dim, hidden_dim, layers)

        if shared_encoder is False:
            self.binners = nn.ModuleList([nn.Linear(hidden_dim, bins_per_plane) for p in range(num_planes)])
        else:
            self.binner = nn.Linear(hidden_dim, bins_per_plane)

        if shared_decoder is False:
            self.reconstructors = nn.ModuleList([nn.Linear(hidden_dim, input_dim) for p in range(num_planes)])
        else:
            self.reconstructor = nn.Linear(hidden_dim, input_dim)

        if shared_priors is False:
            self.conditionalPriors = nn.ModuleList([nn.Linear(hidden_dim, bins_per_plane) for p in range(num_planes)])
        else:
            self.conditionalPrior = nn.Linear(hidden_dim, bins_per_plane)

    # def to(self, *args, **kwargs):
    #     super(EncoderDecoderLayeredRNN, self).to(*args, **kwargs)
    #     self.encoder.to(*args, **kwargs)
    #     self.decoder.to(*args, **kwargs)
    #     self.conditionalRNN.to(*args, **kwargs)
    #     self.binner.to(*args, **kwargs)
    #     self.reconstructor.to(*args, **kwargs)
    #     self.conditionalPriors.to(*args, **kwargs)

    @property
    def bin_count(self):
        return self.bins_per_plane**self.num_planes

    def encode(self, x, tau=1., force_softmax=False):
        if self.training:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        rnn_inputs = x.unsqueeze(1).repeat(1, self.num_planes, 1)
        rnn_out, _ = self.encoder(rnn_inputs)
        if self.shared_encoder is False:
            soft_codes = [softmax(binner(rnn_out[:, binner_idx]), dim=-1)
                          for binner_idx, binner in enumerate(self.binners)]
        else:
            soft_codes = [softmax(self.binner(rnn_out[:, binner_idx]), dim=-1) for binner_idx in range(self.num_planes)]
        bins = [torch.argmax(sc, dim=-1) for sc in soft_codes]

        if self.training or force_softmax:
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
            for i in range(broadcast_size[0] - 1, -1, -1):
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
            if p < self.num_planes:
                if self.marginal:
                    rnn_inputs_prior = torch.cat([torch.zeros_like(hard_codes[0]).unsqueeze(1),
                                                  torch.stack(hard_codes, dim=1)], dim=1).float()
                else:
                    rnn_inputs_prior = torch.cat([y.unsqueeze(1).repeat(1, self.num_planes, 1),
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
            reconstructed = [self.reconstructor(rnn_d_out[:, rec_idx]) for rec_idx in range(self.num_planes)]

        if self.output_activation is not None:
            reconstructed = [self.output_activation(rec) for rec in reconstructed]

        return reconstructed

    def get_priors(self, codes, y=None, tau=1.):
        # y is only needed if the prior is conditional
        if self.training:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        rnn_inputs_prior = torch.zeros_like(codes[0]).unsqueeze(1)
        if len(codes)!=1:
            rnn_inputs_prior = torch.cat([rnn_inputs_prior, torch.stack(codes[:-1], dim=1)], dim=1)
        if not self.marginal:
            temp = y.unsqueeze(1).repeat(1, self.num_planes, 1)
            rnn_inputs_prior = torch.cat([temp, rnn_inputs_prior], dim=-1)

        if self.training:
            rnn_inputs_prior = rnn_inputs_prior.detach()
        else:
            rnn_inputs_prior = rnn_inputs_prior.float()
        prior_logits, _ = self.conditionalRNN(rnn_inputs_prior)
        if self.shared_priors is False:
            priors = [cp(prior_logits[:, cp_idx]) for cp_idx, cp in enumerate(self.conditionalPriors)]
            priors = [softmax(p, dim=-1) for p in priors]
        else:
            priors = [self.conditionalPrior(prior_logits[:, cp_idx]) for cp_idx in range(self.num_planes)]
            priors = [softmax(p, dim=-1) for p in priors]
        return priors


#%% ANN model
class ProbabilisticModel(nn.Module):
    def __init__(self, input_dim=1, hidden_units=100, output_dim=1, layers=3):
        super().__init__()

        modules = []
        assert layers > 1, 'ProbabilisticModel requires more than one hidden layer'
        for i in range(layers):
            if i == 0:
                modules.append(nn.Linear(input_dim, hidden_units))
            elif i == layers - 1:
                modules.append(nn.Linear(hidden_units, output_dim))
            else:
                modules.append(nn.Linear(hidden_units, hidden_units))

            if i != layers - 1:
                modules.append(nn.LeakyReLU())

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class Decoder(ProbabilisticModel):
    def __init__(self, output_dim=1, side_info_size=1, code_size=4, layers=3, hidden_dim=100):
        super().__init__(input_dim=code_size + side_info_size, output_dim=output_dim, layers=layers,
                         hidden_units=hidden_dim)

    def forward(self, x, y):
        return super().forward(torch.cat([x, y], dim=-1))


class Encoder(ProbabilisticModel):
    def __init__(self, input_dim=1, code_size=4, layers=3, hidden_dim=100):
        super().__init__(input_dim=input_dim, output_dim=code_size, layers=layers, hidden_units=hidden_dim)

    def forward(self, x, resample=True, tau=1.0, hard=False):
        out = super().forward(x)
        return out


class MarginalPrior(nn.Module):
    def __init__(self, code_size=4):
        super().__init__()

        self.probs = nn.Parameter(torch.rand(code_size) / code_size)

    def forward(self, x, tau=1.0):
        bs = x.shape[0]
        probs = self.probs.unsqueeze(0).repeat(bs, 1)
        logits = -torch.log(probs + 1e-9)
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            return F.softmax(logits, dim=-1)


class ConditionalPrior(ProbabilisticModel):
    def __init__(self, input_dim=1, code_size=4, layers=3, hidden_dim=100):
        super().__init__(input_dim=input_dim, output_dim=code_size, layers=layers, hidden_units=hidden_dim)

    def forward(self, x, tau=1.0):
        logits = super().forward(x)
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            return F.softmax(logits, dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, side_info_size, layers, hidden_dim, bin_count, marginal=True):
        super().__init__()

        self.encoder = Encoder(input_dim=input_dim, code_size=bin_count, layers=layers, hidden_dim=hidden_dim)
        self.decoder = Decoder(code_size=bin_count, side_info_size=side_info_size,
                               output_dim=input_dim, layers=layers, hidden_dim=hidden_dim)
        if marginal is True:
            self.prior = MarginalPrior(code_size=bin_count)
        else:
            self.prior = ConditionalPrior(code_size=bin_count, layers=layers, hidden_dim=hidden_dim,
                                          input_dim=side_info_size)

    @property
    def bin_count(self):
        return self.encoder.layers[-1].out_features

    def forward(self, x, y, tau=1.0):

        enc_out = self.encoder(x)

        if self.training:
            soft_code = F.gumbel_softmax(enc_out, hard=False, tau=tau, dim=-1)
        else:
            soft_code = F.softmax(enc_out, dim=-1)

        if self.training:
            reconstruct = self.decoder(soft_code, y)
        else:
            bin = torch.argmax(soft_code, dim=-1)
            reconstruct = self.decoder(F.one_hot(bin, num_classes=self.bin_count), y)

        prior = self.prior(y)

        return reconstruct, soft_code, prior

    def decode(self, bins_list, y):
        return [self.decoder(bins, y) for bins in bins_list]
