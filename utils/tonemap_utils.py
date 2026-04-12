import torch
import torch.nn as nn

    
    
class ToneMapper_illumination(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=16):
        super(ToneMapper_illumination, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, in_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        return self.net(input)