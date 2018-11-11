import torch
import torch.nn as nn

class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, inputs, inverse=False):
        if not inverse:
            y1, y2 = torch.chunk(inputs, 2, 1)
            LDJ = torch.zeros(inputs.size(0), inputs.device)
            return (y1, y2), LDJ
        else:
