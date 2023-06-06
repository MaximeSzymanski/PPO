import torch
from torch import nn as nn

from model.Utils import extract_LSTM_features


class LSTMActor(nn.Module):
    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(LSTMActor, self).__init__()
        self.LSTM = nn.Sequential(
            nn.LSTM(input_size=state_size, hidden_size= 64, num_layers=1, batch_first=True),
            extract_LSTM_features(),
            nn.Tanh(),
            nn.Linear(64 * 50, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.LSTM(x)

        x = nn.Softmax(dim=-1)(x)
        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
