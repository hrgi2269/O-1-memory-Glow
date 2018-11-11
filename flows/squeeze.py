import torch
import torch.nn as nn

class Squeeze(nn.Module):
  def __init__(self):
    super(Squeeze, self).__init__()

  def forward(self, inputs, inverse=False):
    batch_size, c, h, w = inputs.size()
    if not inverse:
      x_view = inputs.contiguous().view(batch_size, c, h // 2, 2, w // 2, 2)
      y = x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(batch_size, c * 4, h // 2, w // 2)
      return y, torch.zeros(batch_size, device=inputs.device)
    else:
      y_view = inputs.contiguous().view(batch_size, c // 4, 2, 2, h, w)
      x = y_view.permute(0, 1, 4, 2, 5, 3).contiguous().view(batch_size, c // 4, h * 2, w * 2)
      return x, torch.zeros(batch_size, device=inputs.device)
