import torch
import torch.nn as nn

class apply_gating_layer(nn.Module):
    """
    apply Gated Linear Unit to input
    """
    def __init__(self, input_dim, hidden_dim):
        super(apply_gating_layer, self).__init__()
        self.activation_layer = nn.Linear(input_dim, hidden_dim)
        self.gated_layer = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        activation_output = self.activation_layer(x)
        gated_output = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation_output, gated_output), self.gated_layer

class gated_residual_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, droupout_rate=0.4, output_dim=None, additional_context=False):
        super(gated_residual_network, self).__init__()

        self.hidden_dim = hidden_dim
        if output_dim is None:
            self.output_dim = hidden_dim
        else:
            self.output_dim = output_dim
        self.output_linear = nn.Linear(input_dim, self.output_dim)

        self.dense0 = nn.Linear(input_dim, hidden_dim)
        if additional_context == True:
            self.add_linear = nn.Linear(input_dim, hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.glu = apply_gating_layer(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=droupout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x, additional_context=None):
        if self.output_dim == self.hidden_dim:
            skip = x
        else:
            skip = self.output_linear(x)
        output = self.elu(self.dense0(x))
        if additional_context is not None:
            add_output = self.add_linear(additional_context)
            output = output + add_output
        output = self.dropout(self.dense1(output))
        output, gated_layer = self.glu(output)
        output += skip
        return self.layer_norm(output)

if __name__ == "__main__":
    x = torch.rand(32, 2048)
    add_context = torch.rand(32, 8 * 256)
    grn = gated_residual_network(input_dim=8 * 256,
                                 hidden_dim=256,
                                 additional_context=True)
    output = grn(x, add_context)
    print(output.shape)