import torch
import torch.nn as nn

from .check import check_tensors

# Unlearnable version of InversibleConv1x1
class RandomRotation(nn.Module):
    def __init__(self, channels):
        super(RandomRotation, self).__init__()
        self.register_buffer('w', torch.qr(torch.randn(channels, channels))[0])

    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        pixels = inputs.size(-1) * inputs.size(-2)
        if not inverse:
            y = nn.functional.conv2d(inputs, self.w.unsqueeze(-1).unsqueeze(-1))
            check_tensors({'y': y}, str(self.__class__) + ': forward')
            return y, torch.zeros(batch_size, device=inputs.device)
        else:
            inv_w = self.w.t()
            x = nn.functional.conv2d(inputs, inv_w.unsqueeze(-1).unsqueeze(-1))
            check_tensors({'x': x}, str(self.__class__) + ': inverse')
            return x, torch.zeros(batch_size, device=inputs.device)
