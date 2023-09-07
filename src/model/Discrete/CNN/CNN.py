import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
class CNN(nn.Module):

    def __init__(self,channels, action_size: int = 1, hidden_size=None) -> None:
        super(CNN, self).__init__()
        self.epoch = 0

        # Convolutional layers
        # We now describe the exact architecture used for all seven Atari games. The input to the neural
        # network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
        # filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
        # hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
        # final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fullyconnected linear layer with a single output for each valid action


        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),

        )

        self.weight = self.conv_layers[0].weight


    def forward(self, x):
            # normalize
            # save the image in the folder corresponding to the current epoch
            # create the folder if it does not exist

            if len(x.shape) < 4:
                x = x.unsqueeze(0)
            batch_size = x.shape[0]
            """number_of_images = x.shape[1]
            for u in range(batch_size):
                for i in range(number_of_images):
                    print("i ",i)
                    print("u ",u)

                    if not os.path.exists('images/' + str(self.epoch)):
                        os.makedirs('images/' + str(self.epoch))
                    plt.imsave('images/' + str(self.epoch) + '/' + str(i) + '.png', x[u][i])
            self.epoch += 1"""
            x = x / 255.0
            # if dimension is 4, remove the first dimensionpr
            # plot the 4 images in the batch
            """for i in range(4):
                plt.subplot(1, 4, i+1)
                plt.imshow(x[i])
                plt.axis('off')
            plt.show()"""

            # plot
            """if len(x.shape) == 4:
                x = x.permute(1, 0, 2, 3)"""
            """if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(2, 0, 1)"""
            x = self.conv_layers(x)
            x = x.reshape(batch_size, -1)
            # print the weights of the first layer
            weight = self.conv_layers[0].weight

            # check if the weight has changed
            if not torch.equal(weight, self.weight):
                print("weight CHANGED")

            self.weight = weight

            return x