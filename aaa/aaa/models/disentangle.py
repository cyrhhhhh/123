"""Three-factor disentanglement (the core of FLIP).

We factorise multimodal sentiment information along a *consistency* axis rather
than the usual "shared / specific" sharing-degree axis:

    * Consensus  C_m : the agreement sentiment that all modalities corroborate
                       (a shared-weight encoder extracts the cross-modal
                       consensus).
    * Specific   U_m : sentiment cues carried by a single modality.
    * Conflict   D   : the *directed disagreement* between modalities -- the
                       signal behind sarcasm / irony, where the surface (e.g.
                       lexical) sentiment is contradicted by the others.

Unlike prior conflict-aware methods that *suppress* disagreement, FLIP keeps the
conflict factor as a first-class signal and uses it downstream to drive a
polarity reversal (see ``models/rsgf.py``).
"""
from typing import Dict, List

import torch
import torch.nn as nn


def _transformer(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
        dropout=dropout, batch_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def masked_mean(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Mean pool over the sequence dimension, honouring an optional pad mask.

    Args:
        x:    (B, T, D)
        mask: (B, T) with 1 for valid tokens, 0 for padding. If ``None`` all
              tokens are treated as valid.
    """
    if mask is None:
        return x.mean(dim=1)
    mask = mask.float().unsqueeze(-1)              # (B, T, 1)
    summed = (x * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp_min(1.0)
    return summed / count


class ThreeFactorDisentangler(nn.Module):
    def __init__(self, feat_dims: Dict[str, int], d_model: int, n_heads: int,
                 n_layers_shared: int, n_layers_unique: int, n_layers_conflict: int,
                 dropout: float):
        super().__init__()
        self.modalities: List[str] = list(feat_dims.keys())   # e.g. ["L","V","A"]

        # per-modality input projection to the common space
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(feat_dims[m], d_model), nn.Dropout(dropout))
            for m in self.modalities
        })
        self.pos = PositionalEncoding(d_model)

        # consensus encoder: ONE module reused across modalities
        self.shared_encoder = _transformer(d_model, n_heads, n_layers_shared, dropout)
        # specific encoders: one independent module per modality
        self.unique_encoders = nn.ModuleDict({
            m: _transformer(d_model, n_heads, n_layers_unique, dropout)
            for m in self.modalities
        })
        # conflict encoder: a symmetric cross-modal transformer over all tokens
        # that captures the *disagreement* between modalities
        self.conflict_encoder = _transformer(d_model, n_heads, n_layers_conflict, dropout)
        self.modality_emb = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for m in self.modalities
        })

        # decoders for the reconstruction regulariser (C_m ⊕ U_m -> pooled input)
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                             nn.Linear(d_model, d_model))
            for m in self.modalities
        })

    def forward(self, feats: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor] = None):
        masks = masks or {m: None for m in self.modalities}

        proj_seq, proj_pool = {}, {}
        for m in self.modalities:
            h = self.pos(self.proj[m](feats[m]))          # (B, T_m, d)
            proj_seq[m] = h
            proj_pool[m] = masked_mean(h, masks.get(m))

        # ---- consensus factor: shared-weight encoder ----
        C = {m: masked_mean(self.shared_encoder(proj_seq[m]), masks.get(m))
             for m in self.modalities}

        # ---- specific factors: independent encoders ----
        U = {m: masked_mean(self.unique_encoders[m](proj_seq[m]), masks.get(m))
             for m in self.modalities}

        # ---- conflict factor: symmetric cross-modal transformer ----
        tokens, cfl_mask = [], []
        for m in self.modalities:
            tokens.append(proj_seq[m] + self.modality_emb[m])
            mk = masks.get(m)
            if mk is None:
                mk = torch.ones(proj_seq[m].shape[:2], device=proj_seq[m].device)
            cfl_mask.append(mk)
        joint = torch.cat(tokens, dim=1)                  # (B, sum T, d)
        joint_mask = torch.cat(cfl_mask, dim=1)
        key_padding = (joint_mask == 0)                   # True where padded
        D = masked_mean(self.conflict_encoder(joint, src_key_padding_mask=key_padding),
                        joint_mask)

        out = {
            "C": C, "U": U, "D": D,
            "proj_pool": proj_pool,
        }
        return out

    # -- helper used by the reconstruction loss --
    def reconstruct(self, C: Dict[str, torch.Tensor], U: Dict[str, torch.Tensor]):
        return {m: self.decoders[m](torch.cat([C[m], U[m]], dim=-1))
                for m in self.modalities}
