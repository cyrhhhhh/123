"""TRIDENT -- TRI-factor Disentanglement with hyperbolic iNTensity.

End-to-end pipeline:

    raw features ─► (optional BERT for text)
                 ─► ThreeFactorDisentangler ─► {C_m, U_m, S}
                 ─► RSGF (role-aware fusion)  ─► z
                 ─► HyperbolicIntensityHead   ─► y_hat

Auxiliary linear heads on the redundant consensus, routed unique factor,
synergy factor and fused base provide deep supervision and stabilise training.
"""
from typing import Dict

import torch
import torch.nn as nn

from config import TridentConfig
from models.disentangle import ThreeFactorDisentangler
from models.rsgf import RSGF
from models.hyperbolic import HyperbolicIntensityHead


class TRIDENT(nn.Module):
    def __init__(self, cfg: TridentConfig):
        super().__init__()
        self.cfg = cfg
        self.modalities = list(cfg.feat_dims.keys())     # e.g. ["L", "V", "A"]

        # optional text encoder (BERT) -- only built when requested & available
        self.bert = None
        feat_dims = dict(cfg.feat_dims)
        if cfg.use_bert:
            try:
                from transformers import BertModel
                self.bert = BertModel.from_pretrained(cfg.bert_name)
                feat_dims["L"] = self.bert.config.hidden_size   # 768
            except Exception as e:   # pragma: no cover - offline / no transformers
                print(f"[TRIDENT] BERT unavailable ({e}); "
                      f"using precomputed text features.")
                self.bert = None

        self.disentangler = ThreeFactorDisentangler(
            feat_dims=feat_dims, d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers_shared=cfg.n_layers_shared, n_layers_unique=cfg.n_layers_unique,
            n_layers_syn=cfg.n_layers_syn, dropout=cfg.dropout,
        )
        self.rsgf = RSGF(
            modalities=self.modalities, d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_layers_fusion=cfg.n_layers_fusion, dropout=cfg.dropout,
        )
        self.head = HyperbolicIntensityHead(
            in_dim=cfg.d_model, curvature=cfg.curvature,
            learn_curvature=cfg.learn_curvature,
        )

        # auxiliary regression heads for deep supervision
        self.aux_heads = nn.ModuleDict({
            "C": nn.Linear(cfg.d_model, 1),
            "U": nn.Linear(cfg.d_model, 1),
            "S": nn.Linear(cfg.d_model, 1),
            "base": nn.Linear(cfg.d_model, 1),
        })

    def hyperbolic_parameters(self):
        """Params that should use the (larger) hyperbolic learning rate."""
        return list(self.head.parameters())

    def base_parameters(self):
        hyper = set(id(p) for p in self.hyperbolic_parameters())
        return [p for p in self.parameters() if id(p) not in hyper]

    def _encode_text(self, batch: Dict):
        """Return the text feature sequence and its pad mask."""
        if self.bert is not None and "text_bert" in batch:
            tb = batch["text_bert"]                       # (B, 3, L)
            input_ids = tb[:, 0, :].long()
            attn = tb[:, 1, :].long()
            token_type = tb[:, 2, :].long()
            out = self.bert(input_ids=input_ids, attention_mask=attn,
                            token_type_ids=token_type)
            return out.last_hidden_state, attn.float()
        # precomputed text features
        feat = batch["L"]
        mask = batch.get("L_mask")
        return feat, mask

    def forward(self, batch: Dict):
        feats: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        text_feat, text_mask = self._encode_text(batch)
        feats["L"], masks["L"] = text_feat, text_mask
        for m in self.modalities:
            if m == "L":
                continue
            feats[m] = batch[m]
            masks[m] = batch.get(f"{m}_mask")

        dis = self.disentangler(feats, masks)
        C, U, S, proj_pool = dis["C"], dis["U"], dis["S"], dis["proj_pool"]

        z, info = self.rsgf(C, U, S)
        y_hat, radius, point = self.head(z)

        # tensors needed by the loss functions
        recon = self.disentangler.reconstruct(C, U)
        syn_preds = self.disentangler.discriminate_synergy(
            {m: proj_pool[m].detach() for m in self.modalities})

        # deep-supervision auxiliary predictions
        C_star = torch.stack([C[m] for m in self.modalities], dim=1).mean(dim=1)
        U_star = (info["route_weights"].unsqueeze(-1)
                  * torch.stack([U[m] for m in self.modalities], dim=1)).sum(dim=1)
        aux = {
            "C": self.aux_heads["C"](C_star).squeeze(-1),
            "U": self.aux_heads["U"](U_star).squeeze(-1),
            "S": self.aux_heads["S"](S).squeeze(-1),
            "base": self.aux_heads["base"](info["base"]).squeeze(-1),
        }

        return {
            "y_hat": y_hat,
            "radius": radius,
            "point": point,
            "C": C, "U": U, "S": S,
            "proj_pool": proj_pool,
            "recon": recon,
            "syn_preds": syn_preds,
            "route_weights": info["route_weights"],
            "route_balance": info["route_balance"],
            "aux": aux,
        }
