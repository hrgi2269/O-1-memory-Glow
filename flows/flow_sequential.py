import torch
import torch.nn as nn

class FlowSequential(nn.Sequential):
    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        sum_LDJ = torch.zeros(batch_size, device=inputs.device)
        if not inverse:
            for m in self._modules.values():
                inputs, LDJ = m(inputs, inverse=False)
                sum_LDJ += LDJ
            return inputs, sum_LDJ
        else:
            for m in reversed(self._modules.values()):
                inputs, LDJ = m(inputs, inverse=True)
                sum_LDJ += LDJ
            return inputs, sum_LDJ
