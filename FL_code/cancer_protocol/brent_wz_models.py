from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


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
    def __init__(self, input_dim, side_info_size, bins_per_plane, num_planes, layers, hidden_dim,
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
        self.encoder = CustomRNN(input_dim, hidden_dim, layers, output_activation=True)
        self.decoder = CustomRNN(bins_per_plane + side_info_size, hidden_dim, layers, output_activation=True)
        self.conditionalRNN = CustomRNN(cond_input_dim, hidden_dim, layers, output_activation=False)

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

    @property
    def bin_count(self):
        return self.bins_per_plane**self.num_planes

    @staticmethod
    def _transmission_state_dict(
        state_dict: dict[str, torch.Tensor],
        prefixes: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Return the quantizer parameters in the dtype used on the wire."""
        return {
            key: tensor.detach().cpu().to(
                torch.float16 if tensor.is_floating_point() else tensor.dtype
            ).clone()
            for key, tensor in state_dict.items()
            if key.startswith(prefixes)
        }

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Encoder-side parameters: the encoder RNN plus the per-plane binner heads."""
        return self._transmission_state_dict(self.state_dict(), ('encoder.', 'binner'))

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Decoder-side parameters: decoder RNN, reconstructor heads, and the conditional prior network."""
        return self._transmission_state_dict(self.state_dict(), ('decoder.', 'reconstructor', 'conditional'))

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
            # hard codes; float so they concatenate with float side information downstream
            codes = [F.one_hot(bin, num_classes=self.bins_per_plane).float() for bin in bins]
        return bins, codes

    def forward(self, x, y, tau=1.):
        # Encoder
        bins, codes = self.encode(x=x, tau=tau)

        # Decoder
        reconstructed = self.decode(codes=codes, y=y)

        # Priors
        priors = self.get_priors(codes=codes, y=y, tau=tau)

        return reconstructed, bins, codes, priors

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
