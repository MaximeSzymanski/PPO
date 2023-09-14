import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN model, used before both the actor and critic networks
    """

    def __init__(self, input_channel, output_dim):

        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, output_dim),
            nn.ReLU(),
        )




    def forward(self, x):
        x = self.convolutions(x)

        return x