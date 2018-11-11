import torch
import torch.nn as nn

class Shuffle(nn.Module):
    def __init__(self, channels):
        super(Shuffle, self).__init__()
        self.perm = torch.randperm(channels)
        _, self.inv_perm = torch.sort(self.perm)

    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        if not inverse:
            return inputs[:, self.perm], torch.zeros(batch_size, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(batch_size, device=inputs.device)
