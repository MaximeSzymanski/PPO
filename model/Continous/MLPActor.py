import torch
from torch import nn as nn


class MLPActor(nn.Module):

    def __init__(self, state_size: int = 0, action_size: int = 1, hidden_size: int = 0) -> None:
        super(MLPActor, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size * 2),
        )
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Dense(x)

        half_size = x.shape[-1] // 2
        means = x[..., :half_size]
        log_stds = x[..., half_size:]
        stds = torch.exp(log_stds).clamp(min=-20, max=2)
        return means, stds

    def _init_weights(self, module) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
