"""Gradient Reversal Layer (GRL).

Used by the synergy-exclusivity objective: per-modality discriminators try to
predict the synergistic factor ``S`` from a single modality, while the synergy
encoder is trained adversarially to make ``S`` *unpredictable* from any single
modality (i.e. truly emergent information). The sign flip on the backward pass
implements this min-max with a single forward graph.
"""
import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse and scale the gradient
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity on the forward pass, negated-and-scaled gradient on backward."""
    return _GradReverse.apply(x, lambd)
