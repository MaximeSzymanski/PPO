import torch
from torch import nn as nn


class MLPCritic(nn.Module):

    def __init__(self, state_size: int = 0,  hidden_size: int = 0) -> None:
        super(MLPCritic, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)


        return x

    def _init_weights(self,module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
