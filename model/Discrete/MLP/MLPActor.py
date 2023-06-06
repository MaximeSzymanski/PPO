import torch
from torch import nn as nn


class MLPActor(nn.Module):

    def __init__(self, state_size: int = 0, action_size: int = 0, hidden_size: int = 0) -> None:
        super(MLPActor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)
        x = nn.Softmax(dim=-1)(x)
        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
