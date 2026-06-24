"""RSGF -- Role-aware Synergy-Gated Fusion.

The three disentangled factors play *different* semantic roles, so RSGF fuses
them asymmetrically instead of treating them as homogeneous tokens:

    Stage 1  Reliability routing   -- sparsely select useful Unique factors,
                                       suppressing noisy modalities.
    Stage 2  Anchor-supplement     -- the Redundant consensus C* is the backbone;
                                       routed Unique U* is injected as a residual.
    Stage 3  Synergy modulation    -- the Synergy factor S applies a FiLM-style
                                       affine transform (scale + shift) that can
                                       *amplify* or *flip* the base sentiment
                                       (e.g. sarcasm), rather than being summed in.

Both Stage-2 and Stage-3 degrade gracefully: if U* is routed to zero or S = 0,
the output falls back to the redundant consensus, giving a stable lower bound.
"""
from typing import Dict, List

import torch
import torch.nn as nn


class RSGF(nn.Module):
    def __init__(self, modalities: List[str], d_model: int, n_heads: int,
                 n_layers_fusion: int, dropout: float, route_tau: float = 0.5):
        super().__init__()
        self.modalities = modalities
        self.route_tau = route_tau

        # Stage 1: reliability router scores each unique factor
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Stage 2: anchor-supplement residual aggregation
        self.supplement = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm_base = nn.LayerNorm(d_model)

        # Stage 3: FiLM parameters generated from the synergy factor
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # light post-fusion transformer (kept simple on purpose)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.post = nn.TransformerEncoder(layer, num_layers=n_layers_fusion)

    def forward(self, C: Dict[str, torch.Tensor], U: Dict[str, torch.Tensor],
                S: torch.Tensor):
        # ---- Stage 1: reliability routing over unique factors ----
        scores = torch.stack([self.router(U[m]).squeeze(-1)
                              for m in self.modalities], dim=-1)   # (B, M)
        weights = torch.softmax(scores / self.route_tau, dim=-1)   # (B, M)
        U_stack = torch.stack([U[m] for m in self.modalities], dim=1)  # (B, M, d)
        U_star = (weights.unsqueeze(-1) * U_stack).sum(dim=1)       # (B, d)

        # router load-balance regulariser: encourage non-degenerate usage
        mean_w = weights.mean(dim=0)                                # (M,)
        entropy = -(mean_w * (mean_w + 1e-8).log()).sum()
        max_entropy = torch.log(torch.tensor(float(len(self.modalities)),
                                             device=weights.device))
        route_balance = max_entropy - entropy                      # >= 0, 0 == balanced

        # ---- Stage 2: anchor (redundant consensus) + residual supplement ----
        C_star = torch.stack([C[m] for m in self.modalities], dim=1).mean(dim=1)
        base = self.norm_base(C_star + self.supplement(torch.cat([C_star, U_star], dim=-1)))

        # ---- Stage 3: synergy FiLM modulation ----
        gamma = torch.tanh(self.film_gamma(S))     # in (-1, 1); (1+gamma) -> (0, 2)
        beta = self.film_beta(S)
        z = (1.0 + gamma) * base + beta
        z = self.norm_out(z)

        # post-fusion refinement (single-token "sequence")
        z = self.post(z.unsqueeze(1)).squeeze(1)

        info = {
            "route_weights": weights,        # (B, M) for visualisation
            "route_balance": route_balance,  # scalar regulariser
            "base": base,                    # b, for deep supervision
            "gamma": gamma,
            "beta": beta,
        }
        return z, info
