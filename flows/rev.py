import torch
import torch.nn as nn
from .flow_sequential import FlowSequential

class RevForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _, flow_sequential, x):
        ctx.flow_sequential = flow_sequential
        with torch.no_grad():
            y, LDJ = flow_sequential(x, inverse=False)
        ctx.save_for_backward(y)
        return y, LDJ

    @staticmethod
    def backward(ctx, grad_y, grad_LDJ):
        y, = ctx.saved_tensors
        for f in reversed(list(ctx.flow_sequential.children())):
            with torch.no_grad():
                x, _ = f(y, inverse=True)
            x.requires_grad_()
            with torch.enable_grad():
                y, LDJ = f(x, inverse=False)

            if LDJ.requires_grad:
                LDJ.backward(grad_LDJ, retain_graph=True)
            y.backward(grad_y)
            y = x
            grad_y = x.grad
        return None, None, grad_y

class RevInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _, flow_sequential, y):
        ctx.flow_sequential = flow_sequential
        with torch.no_grad():
            x, LDJ = flow_sequential(y, inverse=True)
        ctx.save_for_backward(x)
        return x, LDJ

    @staticmethod
    def backward(ctx, grad_x, grad_LDJ):
        x, = ctx.saved_tensors
        for f in list(ctx.flow_sequential.children()):
            with torch.no_grad():
                y, _ = f(x, inverse=False)
            y.requires_grad_()
            with torch.enable_grad():
                x, LDJ = f(y, inverse=True)

            if LDJ.requires_grad:
                LDJ.backward(grad_LDJ, retain_graph=True)
            x.backward(grad_x)
            x = y
            grad_x = y.grad
        return None, None, grad_x

def rev_sequential(flow_sequential, inputs, inverse=False):
    assert isinstance(flow_sequential, FlowSequential)
    # To avoid RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    dummy = torch.empty(0, requires_grad=True)
    if not inverse:
        return RevForward.apply(dummy, flow_sequential, inputs)
    else:
        return RevInverse.apply(dummy, flow_sequential, inputs)
