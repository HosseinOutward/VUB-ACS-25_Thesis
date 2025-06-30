import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, output_dim=1, code_size=4, layers=3, hidden_dim=100):
        super().__init__(input_dim=code_size + output_dim, output_dim=output_dim, layers=layers,
                         hidden_units=hidden_dim)

    def forward(self, x, y):
        return super().forward(torch.cat([x, *y.transpose(0,1)], dim=-1))


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

    def __init__(self, input_dim=1, layers=3, hidden_dim=100, code_size=2**4, marginal=True):

        super().__init__()

        self.code_size = code_size

        self.encoder = Encoder(input_dim=input_dim, code_size=self.code_size, layers=layers, hidden_dim=hidden_dim)
        self.decoder = Decoder(code_size=self.code_size, output_dim=input_dim, layers=layers, hidden_dim=hidden_dim)
        if marginal is True:
            self.prior = MarginalPrior(code_size=self.code_size)
        else:
            self.prior = ConditionalPrior(code_size=self.code_size, layers=layers, hidden_dim=hidden_dim,
                                          input_dim=input_dim)

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
            reconstruct = self.decoder(F.one_hot(bin, num_classes=self.code_size), y)

        prior = self.prior(y)

        return reconstruct, soft_code, prior

    def decode(self, bins_list, y):
        return [self.decoder(bins, y) for bins in bins_list]
