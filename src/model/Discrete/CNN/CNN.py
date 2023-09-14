import torch
import torch.nn as nn

class CNN_layer(nn.Module):
    """
    CNN model, used before both the actor and critic networks
    """

    def __init__(self, input_channel, output_dim):
        super(CNN_layer, self).__init__()
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
        self.init_weights()





    def forward(self, x):
        x = self.convolutions(x)

        return x

    # ortogonal initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)