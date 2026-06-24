"""Disentanglement objectives for the three-factor split.

These regularisers keep the Redundant / Unique / Synergy factors in their
intended roles and prevent them from collapsing into copies of one another.
"""
from typing import Dict

import torch
import torch.nn.functional as F

from models.grl import grad_reverse


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


def synergy_exclusivity(S: torch.Tensor,
                        syn_preds: Dict[str, torch.Tensor],
                        grl_lambda: float = 1.0) -> torch.Tensor:
    """Adversarial objective making S unpredictable from any single modality.

    The discriminators (``syn_preds``) try to reconstruct ``S`` from one
    modality; the gradient-reversal on ``S`` trains the synergy encoder to
    defeat them, so only genuinely *emergent* information survives in ``S``.
    The discriminator inputs are detached inside the model, so this single MSE
    term updates the discriminators (minimise) and the synergy encoder
    (maximise, via the reversed gradient) simultaneously.
    """
    target = grad_reverse(S, grl_lambda)
    loss = 0.0
    for m, pred in syn_preds.items():
        loss = loss + F.mse_loss(pred, target)
    return loss / max(len(syn_preds), 1)


def reconstruction_loss(proj_pool: Dict[str, torch.Tensor],
                        recon: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Ensure the disentangled factors retain the input information."""
    loss = 0.0
    for m in proj_pool:
        loss = loss + F.mse_loss(recon[m], proj_pool[m].detach())
    return loss / max(len(proj_pool), 1)
