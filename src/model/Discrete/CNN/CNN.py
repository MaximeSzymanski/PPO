import torch
from torch import nn
class CNN(nn.Module):

    def __init__(self,channels, action_size: int = 1, hidden_size=None) -> None:
        super(CNN, self).__init__()

        # Convolutional layers
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=10),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),

        )




    def forward(self, x):
            # if dimension is 4, remove the first dimensionpr
            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            if len(x.shape) == 4:
                x = x.permute(1, 0, 2, 3)
            batch_size = x.shape[0]
            """if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(2, 0, 1)"""
            x = self.conv_layers(x)
            x = x.reshape(batch_size, -1)
            return x