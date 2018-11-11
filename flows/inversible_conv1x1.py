import torch
import torch.nn as nn

from .check import check_tensors

class InversibleConv1x1(nn.Module):
    def __init__(self, channels):
        super(InversibleConv1x1, self).__init__()
        self.w = nn.Parameter(torch.qr(torch.randn(channels, channels))[0])

    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        pixels = inputs.size(-1) * inputs.size(-2)
        if not inverse:
            y = nn.functional.conv2d(inputs, self.w.unsqueeze(-1).unsqueeze(-1))
            abs_det = torch.det(self.w).abs()
            LDJ = abs_det.log().repeat(batch_size) * pixels
            check_tensors({'y': y, 'LDJ': LDJ}, str(self.__class__) + ': forward')
            return y, LDJ
        else:
            inv_w = torch.inverse(self.w)
            x = nn.functional.conv2d(inputs, inv_w.unsqueeze(-1).unsqueeze(-1))
            abs_det = torch.det(inv_w).abs()
            LDJ = abs_det.log().repeat(batch_size) * pixels
            check_tensors({'x': x, 'LDJ': LDJ}, str(self.__class__) + ': inverse')
            return x, LDJ
