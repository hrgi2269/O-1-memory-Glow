import torch
import torch.nn as nn
import flows

class Glow(nn.Module):
    def __init__(self, input_size, channels_h, K, L, save_memory=False):
        super(Glow, self).__init__()
        self.L = L
        self.save_memory = save_memory
        self.output_sizes = []
        blocks = []
        c, h, w = input_size
        for l in range(L):
            block = [flows.Squeeze()]
            c *= 4; h //= 2; w //= 2 # squeeze
            for _ in range(K):
                norm_layer = flows.ActNorm(c)
                if save_memory:
                    perm_layer = flows.RandomRotation(c) # easily inversible ver
                else:
                    perm_layer = flows.InversibleConv1x1(c)
                coupling_layer = flows.AffineCouplingLayer(c, channels_h)
                block += [norm_layer, perm_layer, coupling_layer]
            blocks.append(flows.FlowSequential(*block))
            self.output_sizes.append((c, h, w))
            c //= 2 # split
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs, inverse=False):
        batch_size = inputs.size(0)
        if not inverse:
            h = inputs
            sum_LDJ = 0
            xs = []
            for l in range(self.L):
                if self.save_memory:
                    h, LDJ = flows.rev_sequential(self.blocks[l], h, inverse=False)
                else:
                    h, LDJ = self.blocks[l](h, inverse=False)
                sum_LDJ += LDJ
                if l < self.L - 1:
                    x, h = torch.chunk(h, 2, 1)
                else:
                    x = h
                xs.append(x.view(batch_size, -1))
            x = torch.cat(xs, -1)
            return x, sum_LDJ
        else:
            sections = [inputs.size(-1) // (2 ** (l + 1)) for l in range(self.L)]
            sections[-1] *= 2
            xs = torch.split(inputs, sections, -1)
            h = xs[-1]
            sum_LDJ = 0
            for l in reversed(range(self.L)):
                h = h.view(batch_size, *self.output_sizes[l])
                if self.save_memory:
                    h, LDJ = flows.rev_sequential(self.blocks[l], h, inverse=True)
                else:
                    h, LDJ = self.blocks[l](h, inverse=True)
                sum_LDJ += LDJ
                if l > 0:
                    h = torch.cat([xs[l - 1], h.view(batch_size, -1)], -1)
            y = h
            return y, sum_LDJ
    
    def log_prob(self, y):
        x, LDJ = self.forward(y, inverse=False)
        log_2pi = 0.79817986835
        log_p_x = -0.5 * (x.pow(2) + log_2pi).sum(-1) # x ~ N(0, I)
        log_p_y = log_p_x + LDJ
        return log_p_y

    def sample(self, n, device, temperature=1.0):
        size = self.output_sizes[0][0] * self.output_sizes[0][1] * self.output_sizes[0][2]
        x = torch.randn(n, size, device=device) * temperature # sample from the reduced-temperature distribution
        y, LDJ = self.forward(x, inverse=True)
        return y
