import torch
import torch.nn as nn

from .check import check_tensors

class AffineCouplingLayer(nn.Module):
    def __init__(self, channels, channels_h):
        super(AffineCouplingLayer, self).__init__()
        self.channels = channels
        d = channels // 2
        conv1 = nn.Conv2d(d, channels_h, 3, 1, 1)
        conv2 = nn.Conv2d(channels_h, channels_h, 1, 1, 0)
        conv3 = nn.Conv2d(channels_h, (channels - d) * 2, 3, 1, 1)
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
        self.log_scale = nn.Parameter(torch.zeros(channels, 1, 1))

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
            log_s, t = torch.chunk(self.nn(x1) * self.log_scale.exp(), 2, 1)
            #s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2) + 1.0 # numerically stable ver
            log_s = s.log()
            y2 = x2 * s + t
            y = self.concat(y1, y2)
            LDJ = log_s.view(batch_size, -1).sum(-1)
            check_tensors({'y': y, 'LDJ': LDJ}, str(self.__class__) + ': forward')
            return y, LDJ
        else:
            y1, y2 = self.split(inputs)
            x1 = y1
            log_s, t = torch.chunk(self.nn(x1) * self.log_scale.exp(), 2, 1)
            #s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2) + 1.0 # numerically stable ver
            log_s = s.log()
            x2 = (y2 - t) / s
            x = self.concat(x1, x2)
            LDJ = -log_s.view(batch_size, -1).sum(-1)
            check_tensors({'x': x, 'LDJ': LDJ}, str(self.__class__) + ': inverse')
            return x, LDJ
