"""Disentanglement objectives for the three-factor split.

These regularisers keep the Consensus / Specific / Conflict factors in their
intended roles and supervise the conflict-driven polarity-reversal gate.
"""
from typing import Dict

import torch
import torch.nn.functional as F


def redundancy_infonce(C: Dict[str, torch.Tensor], tau: float = 0.1) -> torch.Tensor:
    """Pull the per-modality redundant factors of the *same* sample together
    and push different samples apart (InfoNCE over all modality pairs).
    """
    mods = list(C.keys())
    if len(mods) < 2:
        return C[mods[0]].new_zeros(())
    feats = {m: F.normalize(C[m], dim=-1) for m in mods}
    loss = 0.0
    n_pairs = 0
    for i in range(len(mods)):
        for j in range(len(mods)):
            if i == j:
                continue
            a, b = feats[mods[i]], feats[mods[j]]      # (B, d)
            logits = a @ b.t() / tau                    # (B, B)
            target = torch.arange(a.size(0), device=a.device)
            loss = loss + F.cross_entropy(logits, target)
            n_pairs += 1
    return loss / max(n_pairs, 1)


def unique_orthogonality(C: Dict[str, torch.Tensor],
                         U: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Make each Unique factor orthogonal to its Redundant factor and to the
    other modalities' Unique factors.
    """
    mods = list(U.keys())
    loss = 0.0
    # unique vs its own redundant
    for m in mods:
        c = F.normalize(C[m], dim=-1)
        u = F.normalize(U[m], dim=-1)
        loss = loss + (c * u).sum(dim=-1).pow(2).mean()
    # unique vs unique (cross modality)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            ui = F.normalize(U[mods[i]], dim=-1)
            uj = F.normalize(U[mods[j]], dim=-1)
            loss = loss + (ui * uj).sum(dim=-1).abs().mean()
    return loss


def conflict_directionality(reversal: torch.Tensor,
                            surface_pred: torch.Tensor,
                            y: torch.Tensor,
                            tau: float = 0.1) -> torch.Tensor:
    """Supervise the polarity-reversal gate ``rho`` (``reversal``).

    A sample *needs* reversal when the surface (consensus+specific) prediction
    disagrees in sign with the ground-truth sentiment -- the canonical sarcasm
    case where the lexical polarity is the opposite of the true polarity. We
    build that target from labels and train ``rho`` to detect it with a BCE,
    restricted to confident samples (``|y| > tau``) where the sign is meaningful.

        r* = 1[ sign(surface_pred) != sign(y) ]   on  |y| > tau

    This turns conflict into a *directed* signal: the gate learns *when* to flip,
    while the main regression loss (after reversal) learns the magnitude.
    """
    confident = y.abs() > tau                       # (B,) bool
    if confident.sum() == 0:
        return reversal.new_zeros(())
    disagree = (torch.sign(surface_pred) != torch.sign(y)).float()
    target = disagree[confident]
    pred = reversal[confident].clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(pred, target)


def reconstruction_loss(proj_pool: Dict[str, torch.Tensor],
                        recon: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Ensure the disentangled factors retain the input information."""
    loss = 0.0
    for m in proj_pool:
        loss = loss + F.mse_loss(recon[m], proj_pool[m].detach())
    return loss / max(len(proj_pool), 1)
