"""Tail intensity ranking loss.

Encourages the hyperbolic geodesic radius to be monotone in the *magnitude* of
the ground-truth sentiment: samples with stronger sentiment must sit further
from the origin. This directly targets the long-tail extreme-sentiment classes
(e.g. Highly-Negative / Highly-Positive) where Euclidean heads tend to fail.
"""
import torch


def tail_intensity_rank_loss(radius: torch.Tensor, y_true: torch.Tensor,
                             margin: float = 0.1,
                             max_pairs: int = 4096) -> torch.Tensor:
    """Margin ranking over pairs where |y_i| < |y_j|.

    Args:
        radius: (B,) predicted geodesic radii (>= 0).
        y_true: (B,) ground-truth sentiment scores.
        margin: required radius gap delta.
        max_pairs: cap on the number of sampled pairs for efficiency.
    """
    mag = y_true.abs()
    B = radius.size(0)
    if B < 2:
        return radius.new_zeros(())

    # all ordered pairs (i, j) with mag_i < mag_j
    mag_i = mag.unsqueeze(1)        # (B, 1)
    mag_j = mag.unsqueeze(0)        # (1, B)
    valid = (mag_i < mag_j)        # (B, B), True where |y_i| < |y_j|
    idx_i, idx_j = valid.nonzero(as_tuple=True)
    if idx_i.numel() == 0:
        return radius.new_zeros(())

    if idx_i.numel() > max_pairs:
        sel = torch.randperm(idx_i.numel(), device=radius.device)[:max_pairs]
        idx_i, idx_j = idx_i[sel], idx_j[sel]

    r_i = radius[idx_i]
    r_j = radius[idx_j]
    # want r_j >= r_i + margin  ->  hinge on (r_i - r_j + margin)
    return torch.clamp(r_i - r_j + margin, min=0.0).mean()
