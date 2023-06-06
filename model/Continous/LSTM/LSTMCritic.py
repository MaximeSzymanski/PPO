import torch
from torch import nn as nn
from model.Utils import extract_LSTM_features

class LSTMCritic(nn.Module):
    def __init__(self, lstm_hidden_size: int = 16, state_size: int = 0,
                 hidden_size=None) -> None:
        super(LSTMCritic, self).__init__()

        # Validation
        if hidden_size is None:
            hidden_size = {"layer": [16], "activ": "tanh"}
        if 'layer' not in hidden_size or 'activ' not in hidden_size:
            raise ValueError("Input dictionary must contain 'layer' and 'activ' keys")

        if not isinstance(hidden_size['layer'], list) or not all(isinstance(i, int) for i in hidden_size['layer']):
            raise ValueError("'layer' key must be a list of integers")

        if not isinstance(hidden_size['activ'], str) or any(
                i not in ['relu', 'tanh'] for i in hidden_size['activ'].split(',')
        ):
            raise ValueError(
                "'activ' key must be a string of activation function names ('relu', 'tanh') separated by comma")

        layers = []
        layer_sizes = [lstm_hidden_size, *hidden_size['layer'], 1]
        activ_funcs = hidden_size['activ'].split(',')

        if len(activ_funcs) == 1:
            activ_funcs = activ_funcs * len(hidden_size['layer'])

        if len(activ_funcs) != len(layer_sizes) - 2:  # Subtract 2 for LSTM hidden and output sizes
            raise ValueError("The number of activation functions must be equal to the number of layers")

        # Create LSTM layer
        self.lstm = nn.LSTM(input_size=state_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < len(layer_sizes) - 2:  # Skip adding an activation function after the last layer
                if activ_funcs[i] == 'relu':
                    layers.append(nn.ReLU())
                elif activ_funcs[i] == 'tanh':
                    layers.append(nn.Tanh())
                # Add more activation functions if needed

        self.Dense = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The input to LSTM must be of shape (batch_size, seq_len, input_size)
        output = self.lstm(x)
        x = extract_LSTM_features(output)
        x = self.Dense(x)
        return x

    def _init_weights(self, module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)