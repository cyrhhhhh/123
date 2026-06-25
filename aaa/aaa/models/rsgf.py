"""CPRF -- Conflict-gated Polarity-Reversal Fusion.

The three disentangled factors play *different* semantic roles, so CPRF fuses
them asymmetrically instead of treating them as homogeneous tokens:

    Stage 1  Reliability routing   -- sparsely select useful Specific factors,
                                       suppressing noisy modalities.
    Stage 2  Anchor-supplement     -- the Consensus C* is the backbone; routed
                                       Specific U* is injected as a residual.
                                       This yields the *surface* representation
                                       ``base`` whose sign is the surface
                                       sentiment polarity.
    Stage 3  Conflict reversal     -- the Conflict factor D produces a reversal
                                       gate rho in (0, 1). The intensity
                                       representation ``z`` is left untouched
                                       (sarcasm keeps its strength), while rho
                                       is exported so the head can *flip* the
                                       polarity:  rho=0 keep, rho=1 reverse,
                                       rho=0.5 neutralise.

Unlike conflict-aware methods that can only *down-weight* disagreement, the
reversal gate can change the sign of the prediction -- the defining mechanism of
FLIP. If D carries no conflict the gate stays near 0 and the model falls back to
the surface (consensus) sentiment, giving a stable lower bound.
"""
from typing import Dict, List

import torch
import torch.nn as nn


class CPRF(nn.Module):
    def __init__(self, modalities: List[str], d_model: int, n_heads: int,
                 n_layers_fusion: int, dropout: float, route_tau: float = 0.5):
        super().__init__()
        self.modalities = modalities
        self.route_tau = route_tau

        # Stage 1: reliability router scores each specific factor
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

        # Stage 3: conflict-driven polarity-reversal gate generated from D
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.norm_out = nn.LayerNorm(d_model)

        # light post-fusion transformer (kept simple on purpose)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.post = nn.TransformerEncoder(layer, num_layers=n_layers_fusion)

    def forward(self, C: Dict[str, torch.Tensor], U: Dict[str, torch.Tensor],
                D: torch.Tensor):
        # ---- Stage 1: reliability routing over specific factors ----
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

        # ---- Stage 2: anchor (consensus) + residual supplement -> surface base ----
        C_star = torch.stack([C[m] for m in self.modalities], dim=1).mean(dim=1)
        base = self.norm_base(C_star + self.supplement(torch.cat([C_star, U_star], dim=-1)))

        # ---- Stage 3: conflict-gated polarity reversal ----
        # The intensity representation is the (refined) surface base; conflict
        # does NOT change the strength, only the sign downstream.
        z = self.norm_out(base)
        z = self.post(z.unsqueeze(1)).squeeze(1)
        reversal = torch.sigmoid(self.gate(D)).squeeze(-1)         # rho in (0, 1); (B,)

        info = {
            "route_weights": weights,        # (B, M) for visualisation
            "route_balance": route_balance,  # scalar regulariser
            "base": base,                    # surface representation (deep supervision)
            "reversal": reversal,            # rho, polarity-reversal gate
        }
        return z, info
