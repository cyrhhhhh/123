"""
Compact DMRL for multimodal sentiment analysis.

Three core innovations:
1. Sparse Polar Evidence Discovery.
2. Signed Evidence Transport.
3. Counterfactual Routing.
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

TRANSFORMERS_MAP = {"bert": (BertTokenizer, BertModel), "roberta": (RobertaTokenizer, RobertaModel)}
ORDINAL_ANCHORS = (-3., -2., -1., 0., 1., 2., 3.)
MODS = ("text", "audio", "video")
ROLES = ("consensus", "conflict", "residual")
PAIRS = (("text", "audio"), ("text", "video"), ("audio", "video"))


def build_sequence_mask(x, mask=None):
    m = mask.bool() if mask is not None else (x.abs().sum(-1).gt(0) if x.dim() == 3 else x.gt(0))
    if m.dim() == 2:
        empty = ~m.any(1)
        if empty.any():
            m = m.clone()
            m[empty, 0] = True
    return m


def masked_mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1e-6)


def energy_pool(x):
    w = x.abs().sum(-1, keepdim=True)
    return (x * (w / w.sum(1, keepdim=True).clamp_min(1e-6))).sum(1)


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers="bert", pretrained="bert-base-uncased"):
        super().__init__()
        tokenizer_cls, model_cls = TRANSFORMERS_MAP[transformers]
        self.tokenizer = tokenizer_cls.from_pretrained(pretrained)
        self.model = model_cls.from_pretrained(pretrained)
        self.use_finetune = use_finetune
        self.transformers = transformers

    def forward(self, text):
        ids, attn, seg = text[:, 0, :].long(), text[:, 1, :].long(), text[:, 2, :].long()
        kwargs = {"input_ids": ids, "attention_mask": attn}
        if self.transformers == "bert":
            kwargs["token_type_ids"] = seg
        if self.use_finetune:
            return self.model(**kwargs)[0]
        with torch.no_grad():
            return self.model(**kwargs)[0]


class MultiScaleTemporalStem(nn.Module):
    def __init__(self, d_in, d_model, kernels=(1, 3, 5), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d_in, d_model, k, padding=k // 2, bias=False), nn.GELU(), nn.Dropout(dropout))
            for k in kernels
        ])
        self.out = nn.Linear(d_model * len(kernels), d_model)
        self.res = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        xt = x.transpose(1, 2)
        h = self.out(torch.cat([branch(xt).transpose(1, 2) for branch in self.branches], -1))
        return self.norm(self.drop(h) + self.res(x))


class MaskedTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_layers=2, n_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout, "gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.mix = nn.Parameter(torch.full((n_layers,), 0.5))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        pad = ~mask if mask is not None else None
        for i, layer in enumerate(self.layers):
            h = layer(x, src_key_padding_mask=pad)
            x = x + torch.sigmoid(self.mix[i]) * (h - x)
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
        x = self.norm(x)
        return x * mask.unsqueeze(-1).to(x.dtype) if mask is not None else x


class SparsePolarEvidence(nn.Module):
    def __init__(self, d_model, k_slots=4, dropout=0.1):
        super().__init__()
        self.k_slots = k_slots
        self.slots = nn.Parameter(torch.randn(1, k_slots, d_model) * 0.02)
        self.polar_anchors = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.ctx = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attr = nn.Linear(d_model, 4)
        self.anchor_proj = nn.Linear(d_model * 2, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        b, t, d = x.shape
        base = self.slots.expand(b, -1, -1)
        ctx = self.ctx(masked_mean(x, mask)).unsqueeze(1).expand(-1, self.k_slots, -1)
        query = self.norm(base + self.gate(torch.cat([base, ctx], -1)) * ctx)
        score = torch.matmul(query, self.norm(x).transpose(1, 2)) / (d ** 0.5)
        score = score.masked_fill(~mask.unsqueeze(1), -1e4)
        attn = F.softmax(score, -1) * mask.unsqueeze(1).to(x.dtype)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
        ev = self.drop(torch.matmul(attn, self.value(x)))
        anchor_score = torch.matmul(
            F.normalize(ev, dim=-1, eps=1e-6),
            F.normalize(self.polar_anchors.expand(b, -1, -1), dim=-1, eps=1e-6).transpose(1, 2),
        )
        anchor_prob = F.softmax(anchor_score, -1)
        anchor_ctx = torch.matmul(anchor_prob, self.polar_anchors.expand(b, -1, -1))
        ev = self.norm(ev + self.anchor_proj(torch.cat([ev, anchor_ctx], -1)))
        raw_pol, raw_strength, raw_unc, raw_pos = self.attr(ev).unbind(-1)
        grid = torch.linspace(0, 1, t, device=x.device).view(1, 1, t)
        slot_sim = torch.matmul(F.normalize(ev, dim=-1, eps=1e-6), F.normalize(ev, dim=-1, eps=1e-6).transpose(1, 2))
        return {"evidence": ev, "polarity": raw_pol.tanh(), "strength": F.softplus(raw_strength), "uncertainty": F.softplus(raw_unc), "tau": 0.5 * (attn * grid).sum(-1) + 0.5 * raw_pos.sigmoid(), "attn": attn, "slot_sim": slot_sim, "anchor_prob": anchor_prob}

class SignedEvidenceTransport(nn.Module):
    def __init__(self, d_model, match_threshold=0.12, sinkhorn_iters=5):
        super().__init__()
        self.match_threshold = match_threshold
        self.sinkhorn_iters = sinkhorn_iters
        self.role_proj = nn.Linear(d_model, d_model)
        self.role_type = nn.Parameter(torch.randn(1, len(ROLES), d_model) * 0.02)
        self.role_mixer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def sinkhorn(self, logits):
        z = logits
        for _ in range(self.sinkhorn_iters):
            z = z - torch.logsumexp(z, -1, keepdim=True)
            z = z - torch.logsumexp(z, -2, keepdim=True)
        p = z.exp()
        return p / p.sum((-1, -2), keepdim=True).clamp_min(1e-6)

    def transport_plan(self, a, b):
        ea = F.normalize(a["evidence"], dim=-1, eps=1e-6)
        eb = F.normalize(b["evidence"], dim=-1, eps=1e-6)
        sim = torch.matmul(ea, eb.transpose(1, 2))
        cost = 1.0 * (1 - sim)
        cost = cost + 0.5 * (a["tau"].unsqueeze(-1) - b["tau"].unsqueeze(-2)).abs()
        cost = cost + 0.5 * (a["polarity"].unsqueeze(-1) - b["polarity"].unsqueeze(-2)).abs()
        cost = cost + 0.2 * (a["uncertainty"].unsqueeze(-1) + b["uncertainty"].unsqueeze(-2))
        return self.sinkhorn(-cost), cost, sim

    def split_roles(self, a, b, trans, sim):
        paired = torch.matmul(trans, b["evidence"])
        paired_pol = torch.matmul(trans, b["polarity"].unsqueeze(-1)).squeeze(-1)
        align = trans.sum(-1)
        avg_sim = (trans * sim).sum(-1) / align.clamp_min(1e-6)
        gap = (a["polarity"] - paired_pol).abs()
        consensus = ((gap < 0.5) & (avg_sim > self.match_threshold)).float().unsqueeze(-1)
        conflict = ((gap >= 0.5) & (avg_sim > self.match_threshold)).float().unsqueeze(-1)
        residual = (1 - align.gt(1.0 / max(1, trans.size(-1))).float()).unsqueeze(-1)
        return consensus * 0.5 * (a["evidence"] + paired), conflict * (a["evidence"] - paired), residual * a["evidence"]

    def forward(self, ev):
        banks = {role: [] for role in ROLES}
        aux = {}
        for x, y in PAIRS:
            trans, cost, sim = self.transport_plan(ev[x], ev[y])
            c, f, r = self.split_roles(ev[x], ev[y], trans, sim)
            reliability = 0.5 * (
                ev[x]["strength"] / (ev[x]["uncertainty"] + 1.0)
                + torch.matmul(trans, (ev[y]["strength"] / (ev[y]["uncertainty"] + 1.0)).unsqueeze(-1)).squeeze(-1)
            ).unsqueeze(-1)
            c, f, r = c * reliability, f * reliability, r * reliability
            banks["consensus"].append(c)
            banks["conflict"].append(f)
            banks["residual"].append(r)
            suffix = f"{x[0]}{y[0]}"
            aux[f"transport_{suffix}"] = trans
            aux[f"cost_{suffix}"] = cost
        banks["residual"].extend([ev[m]["evidence"] for m in MODS])
        roles = {}
        for role_idx, role in enumerate(ROLES):
            bank = self.role_proj(torch.cat(banks[role], 1))
            role_token = self.role_type[:, role_idx:role_idx + 1, :]
            roles[role] = self.role_mixer(energy_pool(bank + role_token))
            aux[f"{role}_bank"] = bank
        return roles, aux


class CounterfactualRouter(nn.Module):
    def __init__(self, d_model, dropout=0.1, tau=0.5):
        super().__init__()
        self.tau = tau
        self.mod_router = self.router_net(d_model, dropout)
        self.role_router = self.router_net(d_model, dropout)
        self.mod_unc = nn.Linear(d_model * 3, 1)
        self.role_unc = nn.Linear(d_model * 3, 1)

    def router_net(self, d_model, dropout):
        return nn.Sequential(nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3))

    def forward(self, pooled, roles):
        mod_in = torch.cat([pooled[m] for m in MODS], -1)
        role_in = torch.cat([roles[r] for r in ROLES], -1)
        mod_temp = F.softplus(self.mod_unc(mod_in)) + 0.5
        role_temp = F.softplus(self.role_unc(role_in)) + 0.5
        mw = F.softmax(self.mod_router(mod_in) / mod_temp, -1)
        rw = F.softmax(self.role_router(role_in) / role_temp, -1)
        return mw, rw

    def targets(self, factual, preds):
        delta = torch.cat([(factual - pred).abs() for pred in preds], -1)
        return F.softmax(delta / self.tau, -1), delta


class OrdinalHead(nn.Module):
    def __init__(self, d_model, anchors=ORDINAL_ANCHORS, dropout=0.1):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors).float())
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, len(anchors)))

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, -1)
        return {"ordinal_logits": logits, "ordinal_probs": probs, "output_logit": (probs * self.anchors.unsqueeze(0)).sum(-1, keepdim=True)}


class DMRLLoss(nn.Module):
    def __init__(self, anchors=ORDINAL_ANCHORS, w_ord=0.2, w_sparse=0.01, w_transport=0.02, w_cf=0.05, w_bin=0.2):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(anchors).float())
        self.l1 = nn.L1Loss()
        self.w_ord, self.w_sparse, self.w_transport, self.w_cf, self.w_bin = w_ord, w_sparse, w_transport, w_cf, w_bin

    def ordinal_loss(self, logits, labels):
        target = torch.exp(-((labels - self.anchors.view(1, -1).to(labels.device)) ** 2) / 2.0)
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
        return -(target * F.log_softmax(logits, -1)).sum(-1).mean()

    def sparse_loss(self, out):
        terms = []
        for m in MODS:
            attn = out[f"{m}_attn"].clamp_min(1e-8)
            slot_sim = out[f"{m}_slot_sim"]
            entropy = -(attn * attn.log()).sum(-1).mean()
            eye = torch.eye(slot_sim.size(-1), device=slot_sim.device).unsqueeze(0)
            diversity = ((slot_sim - eye) ** 2 * (1 - eye)).mean()
            terms += [out[f"{m}_strength"].mean(), out[f"{m}_uncertainty"].mean(), 0.1 * entropy, 0.1 * diversity]
        return sum(terms) / len(terms)

    def forward(self, out, labels):
        main = self.l1(out["output_logit"], labels)
        ord_loss = self.ordinal_loss(out["ordinal_logits"], labels)
        sparse = self.sparse_loss(out)
        transport = (out["cost_ta"].mean() + out["cost_tv"].mean() + out["cost_av"].mean()) / 3.0
        cf = 0.5 * (F.kl_div((out["modality_weights"] + 1e-8).log(), out["modality_cf_target"], reduction="batchmean") + F.kl_div((out["role_weights"] + 1e-8).log(), out["role_cf_target"], reduction="batchmean"))
        binary_target = labels.ge(0).long().view(-1)
        binary = F.cross_entropy(out["binary_logits"], binary_target)
        total = main + self.w_ord * ord_loss + self.w_sparse * sparse + self.w_transport * transport + self.w_cf * cf + self.w_bin * binary
        return {"main_loss": main, "ordinal_loss": ord_loss, "sparse_loss": sparse, "transport_loss": transport, "counterfactual_loss": cf, "binary_loss": binary, "total_loss": total}

class DMRL(nn.Module):
    def __init__(self, args):
        super().__init__()
        _, d_a, d_v = args.feature_dims
        d_model, dropout = args.d_model, args.dropout
        n_layers = getattr(args, "n_layers", 2)
        k_slots = getattr(args, "k_slots", 4)
        n_heads = getattr(args, "n_heads", 4)
        ff_mult = getattr(args, "ff_mult", 4)
        self.use_bert = getattr(args, "use_bert", True)
        if self.use_bert:
            self.bert = BertTextEncoder(
                getattr(args, "use_finetune", True),
                getattr(args, "transformers", "bert"),
                getattr(args, "pretrained", "bert-base-uncased"),
            )
        dims = {"text": 768, "audio": d_a, "video": d_v}
        self.stems = nn.ModuleDict({m: MultiScaleTemporalStem(dims[m], d_model, dropout=dropout) for m in MODS})
        self.encoders = nn.ModuleDict({m: MaskedTransformerEncoder(d_model, n_layers, n_heads, ff_mult, dropout) for m in MODS})
        self.evidence = nn.ModuleDict({m: SparsePolarEvidence(d_model, k_slots, dropout) for m in MODS})
        self.mod_proj = nn.ModuleDict({m: nn.Linear(d_model, d_model) for m in MODS})
        self.transport = SignedEvidenceTransport(
            d_model,
            match_threshold=getattr(args, "match_threshold", 0.12),
            sinkhorn_iters=getattr(args, "sinkhorn_iters", 5),
        )
        self.router = CounterfactualRouter(d_model, dropout)
        self.mod_fuse = nn.Linear(d_model * 3, d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = OrdinalHead(d_model, dropout=dropout)
        self.binary_head = nn.Linear(d_model, 2)
        self.loss_module = DMRLLoss(
            w_ord=getattr(args, "w_ord", 0.2),
            w_sparse=getattr(args, "w_sparse", 0.01),
            w_transport=getattr(args, "w_transport", 0.02),
            w_cf=getattr(args, "w_cf", 0.05),
            w_bin=getattr(args, "w_bin", 0.2),
        )

    def encode(self, text, audio, video, masks):
        raw = {"text": self.bert(text) if self.use_bert else text, "audio": audio, "video": video}
        return {m: self.encoders[m](self.stems[m](raw[m]), masks[m]) for m in MODS}

    def pool_evidence(self, ev):
        pooled = {}
        for m in MODS:
            mask = torch.ones(ev[m]["evidence"].shape[:2], dtype=torch.bool, device=ev[m]["evidence"].device)
            pooled[m] = self.mod_proj[m](masked_mean(ev[m]["evidence"], mask))
        return pooled

    def compose_modality(self, pooled, weights):
        stack = torch.stack([pooled[m] for m in MODS], 1)
        weighted = (weights.unsqueeze(-1) * stack).reshape(stack.size(0), -1)
        return self.final_norm(self.mod_fuse(weighted))

    def compose_role(self, roles, weights):
        stack = torch.stack([roles[r] for r in ROLES], 1)
        return self.final_norm((weights.unsqueeze(-1) * stack).sum(1))

    def predict(self, pooled, roles, mw, rw):
        z_mod = self.compose_modality(pooled, mw)
        z_role = self.compose_role(roles, rw)
        fused = self.final_norm(z_mod + z_role)
        out = self.head(fused)
        out["binary_logits"] = self.binary_head(fused)
        return out

    def counterfactual(self, pooled, roles, mw, rw):
        factual = self.predict(pooled, roles, mw, rw)["output_logit"]
        z_pooled = {m: torch.zeros_like(pooled[m]) for m in MODS}
        z_roles = {r: torch.zeros_like(roles[r]) for r in ROLES}
        cf_mod = tuple(
            self.predict({**pooled, m: z_pooled[m]}, roles, mw, rw)["output_logit"]
            for m in MODS
        )
        cf_role = tuple(
            self.predict(pooled, {**roles, r: z_roles[r]}, mw, rw)["output_logit"]
            for r in ROLES
        )
        return factual, cf_mod, cf_role

    def forward(self, text, audio, video, text_mask=None, audio_mask=None, video_mask=None, labels=None):
        masks = {
            "text": build_sequence_mask(text[:, 1, :], text_mask),
            "audio": build_sequence_mask(audio, audio_mask),
            "video": build_sequence_mask(video, video_mask),
        }
        seq = self.encode(text, audio, video, masks)
        ev = {m: self.evidence[m](seq[m], masks[m]) for m in MODS}
        roles, aux = self.transport(ev)
        pooled = self.pool_evidence(ev)
        mw, rw = self.router(pooled, roles)
        main = self.predict(pooled, roles, mw, rw)
        factual, cf_mod, cf_role = self.counterfactual(pooled, roles, mw, rw)
        mod_target, mod_delta = self.router.targets(factual, cf_mod)
        role_target, role_delta = self.router.targets(factual, cf_role)

        out = {
            **main,
            "modality_weights": mw,
            "role_weights": rw,
            "modality_cf_target": mod_target.detach(),
            "role_cf_target": role_target.detach(),
            "modality_delta": mod_delta.detach(),
            "role_delta": role_delta.detach(),
        }
        for m in MODS:
            for key, value in ev[m].items():
                out[f"{m}_{key}"] = value
        for r in ROLES:
            out[f"{r}_repr"] = roles[r]
        out.update(aux)
        if labels is not None:
            out["losses"] = self.loss_module(out, labels)
        return out
