import torch
from torch import nn as nn


class extract_LSTM_features(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor, _ = x
        shape = tensor.size()
        if len(shape) == 2:
            reshaped_tensor = torch.flatten(tensor)
        elif len(shape) == 3:
            reshaped_tensor = torch.flatten(tensor, start_dim=1)
        else:
            raise ValueError("Unsupported tensor shape")
        return reshaped_tensor
