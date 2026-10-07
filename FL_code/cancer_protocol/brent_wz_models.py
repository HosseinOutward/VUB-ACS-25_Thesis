from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from typing import Any

from FL_code.FL_core.numerical_policy import NumericalPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        output_activation: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.layers: int = layers
        self.output_activation: bool = output_activation

        input_layers = [nn.Linear(input_dim, hidden_dim)]
        input_layers.extend([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layers - 1)
        ])
        self.input_layers = nn.ModuleList(input_layers)
        self.transition_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])

    def forward(self, x: torch.Tensor, h_0: Sequence[torch.Tensor] | None = None) -> torch.Tensor:
        """Run the layered recurrence and return the top layer's output per timestep.

        Only the previous timestep's hidden states are retained, because the recurrence never
        reaches further back. Materializing every (timestep, layer) state costs gigabytes at
        inference batch sizes and no caller ever consumed it.
        """
        batch_size, timesteps, _ = x.shape

        if h_0 is None:
            reference = self.transition_layers[0].weight
            h_0 = [
                torch.zeros(batch_size, self.hidden_dim, device=reference.device, dtype=reference.dtype)
                for _ in range(self.layers)
            ]

        previous = list(h_0)
        outputs = []
        for t in range(timesteps):
            current: list[torch.Tensor] = []
            for l in range(self.layers):
                y = (
                    self.input_layers[l](x[:, t] if l == 0 else current[l - 1])
                    + self.transition_layers[l](previous[l])
                )
                if l == 0 or l < self.layers - 1 or self.output_activation:
                    y = F.leaky_relu(y)
                current.append(y)
            outputs.append(current[-1])
            previous = current

        return torch.stack(outputs, dim=1)


class EncoderDecoderLayeredRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        side_info_size: int,
        bins_per_plane: int,
        num_planes: int,
        layers: int,
        hidden_dim: int,
        shared_encoder: bool = False,
        shared_decoder: bool = False,
        shared_priors: bool = False,
        marginal: bool = False,
        output_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        encoder_input_dim: int | None = None,
        numerical_policy: NumericalPolicy | None = None,
    ) -> None:
        super().__init__()

        assert bins_per_plane > 1, 'need at least single bit (2 bins) per plane'

        self.numerical_policy: NumericalPolicy = numerical_policy or NumericalPolicy()
        self.training_provenance: dict[str, Any] | None = None
        self.bins_per_plane: int = bins_per_plane
        self.num_planes: int = num_planes
        self.shared_decoder: bool = shared_decoder
        self.shared_encoder: bool = shared_encoder
        self.shared_priors: bool = shared_priors
        self.marginal: bool = marginal
        self.output_activation: Callable[[torch.Tensor], torch.Tensor] | None = output_activation

        cond_input_dim = bins_per_plane if marginal else bins_per_plane + side_info_size
        self.encoder = CustomRNN(
            input_dim if encoder_input_dim is None else encoder_input_dim,
            hidden_dim,
            layers,
            output_activation=True,
        )
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

    @staticmethod
    def _transmission_state_dict(
        state_dict: dict[str, torch.Tensor],
        prefixes: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        """Return the quantizer parameters that go on the wire, at their trained precision."""
        return {
            key: tensor.detach().cpu().clone()
            for key, tensor in state_dict.items()
            if key.startswith(prefixes)
        }

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Encoder-side parameters: the encoder RNN plus the per-plane binner heads."""
        return self._transmission_state_dict(self.state_dict(), ('encoder.', 'binner'))

    def decoder_state_dict(self, include_prior: bool = False) -> dict[str, torch.Tensor]:
        """Return decoder parameters, optionally including the conditional prior network."""
        prefixes = ('decoder.', 'reconstructor', 'conditional') if include_prior else ('decoder.', 'reconstructor')
        return self._transmission_state_dict(self.state_dict(), prefixes)

    def copy_for_prior_retraining(self) -> EncoderDecoderLayeredRNN:
        """Copy this quantizer's conditional prior and release unrelated network parameters."""
        prior_model = deepcopy(self)
        prior_model.encoder = None
        prior_model.decoder = None
        if prior_model.shared_encoder:
            prior_model.binner = None
        else:
            prior_model.binners = None
        if prior_model.shared_decoder:
            prior_model.reconstructor = None
        else:
            prior_model.reconstructors = None
        return prior_model

    def encode(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
        force_softmax: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Map scalar inputs to one symbol and code vector per refinement stage."""
        if self.training:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        rnn_inputs = x.unsqueeze(1).repeat(1, self.num_planes, 1)
        rnn_out = self.encoder(rnn_inputs)
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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        tau: float = 1.0,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Run the encoder, decoder, and in-model symbol prior for training."""
        # Encoder
        bins, codes = self.encode(x=x, tau=tau)

        # Decoder
        reconstructed = self.decode(codes=codes, y=y)

        # Priors
        priors = self.get_priors(codes=codes, y=y, tau=tau)

        return reconstructed, bins, codes, priors

    def decode(
        self,
        codes: Sequence[torch.Tensor],
        y: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Reconstruct every refinement stage from code vectors and model side information."""
        assert y is not None, "The decoder requires its formatted side-information tensor."
        rnn_d_inputs = torch.stack([torch.cat([c, y], dim=-1) for c in codes], dim=1)
        rnn_d_out = self.decoder(rnn_d_inputs)
        if self.shared_decoder is False:
            reconstructed = [rec(rnn_d_out[:, rec_idx]) for rec_idx, rec in enumerate(self.reconstructors)]
        else:
            reconstructed = [self.reconstructor(rnn_d_out[:, rec_idx]) for rec_idx in range(self.num_planes)]

        if self.output_activation is not None:
            reconstructed = [self.output_activation(rec) for rec in reconstructed]

        return reconstructed

    def get_priors(
        self,
        codes: Sequence[torch.Tensor],
        y: torch.Tensor | None = None,
        tau: float = 1.0,
        force_softmax: bool = False,
    ) -> list[torch.Tensor]:
        # y is only needed if the prior is conditional
        if self.training and not force_softmax:
            softmax = partial(F.gumbel_softmax, tau=tau, hard=False)
        else:
            softmax = F.softmax

        rnn_inputs_prior = torch.zeros_like(codes[0]).unsqueeze(1)
        if len(codes)!=1:
            rnn_inputs_prior = torch.cat([rnn_inputs_prior, torch.stack(codes[:-1], dim=1)], dim=1)
        if self.training:
            # The prior must not train the encoder through earlier-stage symbols, but
            # conditional side information may itself come from trainable features.
            rnn_inputs_prior = rnn_inputs_prior.detach()
        if not self.marginal:
            assert y is not None, "A conditional prior requires decoder side information."
            temp = y.unsqueeze(1).repeat(1, self.num_planes, 1)
            rnn_inputs_prior = torch.cat([temp, rnn_inputs_prior], dim=-1)

        if not self.training:
            rnn_inputs_prior = rnn_inputs_prior.float()
        prior_logits = self.conditionalRNN(rnn_inputs_prior)
        if self.shared_priors is False:
            priors = [cp(prior_logits[:, cp_idx]) for cp_idx, cp in enumerate(self.conditionalPriors)]
            priors = [softmax(p, dim=-1) for p in priors]
        else:
            priors = [self.conditionalPrior(prior_logits[:, cp_idx]) for cp_idx in range(self.num_planes)]
            priors = [softmax(p, dim=-1) for p in priors]
        return priors
