import torch
import torch.nn as nn

class Flip(nn.Module):
  def __init__(self):
    super(Flip, self).__init__()

  def forward(self, inputs, inverse=False):
    batch_size = inputs.size(0)
    return inputs.flip([1]), torch.zeros(batch_size, device=inputs.device)
