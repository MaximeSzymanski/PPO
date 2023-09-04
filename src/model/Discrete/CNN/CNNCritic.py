import torch
from torch import nn
class CNNCritic(nn.Module):
    def __init__(self,channels,  hidden_size=None) -> None:
        super(CNNCritic, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 52 * 40, 256),  # Adjust input size based on the output shape from conv layers
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, x):
            # if dimension is 4, remove the first dimension

            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            x = x.permute(0, 3, 1, 2)

            x = self.conv_layers(x)

            x = x.reshape(x.shape[0], -1)
            x = self.fc_layers(x)
            return x