import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, layers, output_activation=False):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.output_activation = output_activation

        input_layers = [nn.Linear(input_dim, hidden_dim)]
        input_layers.extend([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(self.layers-1)
        ])
        self.input_layers = nn.ModuleList(input_layers)
        self.transition_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])

    def forward(self, x, h_0=None):

        B, T, D = x.shape

        if h_0 is None:
            h_0 = [torch.zeros(B, self.hidden_dim).to(self.transition_layers[0].weight.device) for _ in range(self.layers)]

        h_outputs = [h_0]

        for t in range(T):
            h_outputs.append([])
            for l in range(self.layers):
                if l == 0:
                    y = F.leaky_relu(self.input_layers[l](x[:, t]) + self.transition_layers[l](h_outputs[-2][l]))
                elif l < self.layers-1 or self.output_activation:
                    y = F.leaky_relu(self.input_layers[l](h_outputs[-1][l-1]) + self.transition_layers[l](h_outputs[-2][l]))
                else:
                    y = self.input_layers[l](h_outputs[-1][l-1]) + self.transition_layers[l](h_outputs[-2][l])
                h_outputs[-1].append(y)

        all_h_output = torch.stack([torch.stack(ht, dim=1) for ht in h_outputs], dim=1)

        return all_h_output[:, 1:, -1, :], all_h_output[:, 1:, :, :]
