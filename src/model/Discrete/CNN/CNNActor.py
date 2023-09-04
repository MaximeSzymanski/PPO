import torch
from torch import nn
class CNNActor(nn.Module):

    def __init__(self,channels, action_size: int = 1, hidden_size=None) -> None:
        super(CNNActor, self).__init__()

        # Convolutional layers
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
            nn.Linear(256, action_size)
        )


    def forward(self, x):
            if len(x.shape) < 4:
                x = x.unsqueeze(0)

            if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(2, 0, 1)
            x = self.conv_layers(x)
            #x = x.view(-1, 66560)  # Flatten the output
            x = x.reshape(x.shape[0], -1)
            x = self.fc_layers(x)
            x = nn.Softmax(dim=-1)(x)
            return x