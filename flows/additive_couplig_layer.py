import torch
import torch.nn as nn

from .check import check_tensors

class AdditiveCouplingLayer(nn.Module):
    def __init__(self, channels, channels_h):
        super(AdditiveCouplingLayer, self).__init__()
        self.channels = channels
        d = channels // 2
        conv1 = nn.Conv2d(d, channels_h, 3, 1, 1)
        conv2 = nn.Conv2d(channels_h, channels_h, 1, 1, 0)
        conv3 = nn.Conv2d(channels_h, channels - d, 3, 1, 1)
        def init_normal(m):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.05)
            nn.init.constant_(m.bias.data, 0.0)
        def init_zero(m):
            nn.init.constant_(m.weight.data, 0.0)
            nn.init.constant_(m.bias.data, 0.0)
        conv1.apply(init_normal)
        conv2.apply(init_normal)
        conv3.apply(init_zero)
        self.nn = nn.Sequential(conv1,
                                nn.ReLU(True),
                                conv2,
                                nn.ReLU(True),
                                conv3)

    def split(self, x):
        d = self.channels // 2
        x1, x2 = torch.split(x, [d, self.channels - d], 1)
        return x1, x2

    def concat(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        return x

    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        if not inverse:
            x1, x2 = self.split(inputs)
            y1 = x1
            t = self.nn(x1)
            y2 = x2 + t
            y = self.concat(y1, y2)
            check_tensors({'y': y}, str(self.__class__) + ': forward')
            return y, torch.zeros(batch_size, device=inputs.device)
        else:
            y1, y2 = self.split(inputs)
            x1 = y1
            t = self.nn(x1)
            x2 = y2 - t
            x = self.concat(x1, x2)
            check_tensors({'x': x}, str(self.__class__) + ': inverse')
            return x, torch.zeros(batch_size, device=inputs.device)
