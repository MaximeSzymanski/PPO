import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
class CNN(nn.Module):

    def __init__(self,channels, action_size: int = 1, hidden_size=None) -> None:
        super(CNN, self).__init__()

        # Convolutional layers
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )




    def forward(self, x):
            x = x / 255.0
            # if dimension is 4, remove the first dimensionpr
            # plot the 4 images in the batch

            """for i in range(4):
                plt.subplot(1, 4, i+1)
                plt.imshow(x[i])
                plt.axis('off')
            plt.show()"""
            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            # plot
            """if len(x.shape) == 4:
                x = x.permute(1, 0, 2, 3)"""
            batch_size = x.shape[0]
            """if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(2, 0, 1)"""
            x = self.conv_layers(x)
            x = x.reshape(batch_size, -1)
            return x