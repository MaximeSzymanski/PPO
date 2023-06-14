import torch
from torch import nn as nn


class extract_LSTM_features(nn.Module):
    """Extract the last output of a LSTM layer
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the last output of a LSTM layer
        Arguments
        ---------

        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size)
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, hidden_size)
        """

        if len(x.shape) == 2:
            # add a dimension to the tensor
            x = x.unsqueeze(1)
        # keep only the last output of the LSTM
        x = x[:, -1, :]

        return x
