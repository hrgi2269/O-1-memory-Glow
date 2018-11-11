import torch
import torch.nn as nn

from .check import check_tensors

class ActNorm(nn.Module):
    def __init__(self, channels):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.mean = nn.Parameter(torch.empty(channels, 1, 1))
        self.log_std = nn.Parameter(torch.empty(channels, 1, 1))
        self.initialized = False

    def forward(self, inputs, inverse=False):
        if not self.initialized:
            inputs_view = inputs.transpose(0, 1).contiguous().view(self.channels, -1)
            mean = inputs_view.mean(-1).view(-1, 1, 1)
            std = inputs_view.std(-1).view(-1, 1, 1)
            std = std.clamp(min=1e-16) # avoid nan
            self.mean.data.copy_(mean)
            self.log_std.data.copy_(std.log())
            self.initialized = True

        batch_size = inputs.size(0)
        pixels = inputs.size(-1) * inputs.size(-2)
        if not inverse:
            y = (inputs - self.mean) * torch.exp(-self.log_std)
            LDJ = -self.log_std.sum().repeat(batch_size) * pixels
            check_tensors({'y': y, 'LDJ': LDJ}, str(self.__class__) + ': forward')
            return y, LDJ
        else:
            x = inputs * torch.exp(self.log_std) + self.mean
            LDJ = self.log_std.sum().repeat(batch_size) * pixels
            check_tensors({'x': x, 'LDJ': LDJ}, str(self.__class__) + ': inverse')
            return x, LDJ
