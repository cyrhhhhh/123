"""Training / evaluation entry point for FLIP (class name kept as TRIDENT).

Usage:
    python train.py                       # train on MOSI (config defaults)
    python train.py --dataset synthetic   # smoke test without the real dataset
    python train.py --epochs 40 --seed 1111

Total objective:
    L = L1(y_hat, y)
        + a_con * L_consensus  + a_spec * L_specific
        + w_conflict * L_conflict + a_rec * L_reconstruction
        + w_rank * L_rank
        + w_aux  * sum(L1(aux_k, y))
        + w_route_balance * route_balance
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import TridentConfig
from data.data_loader import build_dataloaders
from models import TRIDENT
from losses import (redundancy_infonce, unique_orthogonality, conflict_directionality,
                    reconstruction_loss, tail_intensity_rank_loss)
from utils import eval_regression, dict_to_str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(cfg, out, y, l1):
    main = l1(out["y_hat"], y)

    l_con = redundancy_infonce(out["C"], tau=cfg.infonce_tau)
    l_spec = unique_orthogonality(out["C"], out["U"])
    l_conflict = conflict_directionality(out["reversal_gate"], out["surface_pred"],
                                         y, tau=cfg.conflict_tau)
    l_rec = reconstruction_loss(out["proj_pool"], out["recon"])
    l_dis = (cfg.a_con * l_con + cfg.a_spec * l_spec
             + cfg.w_conflict * l_conflict + cfg.a_rec * l_rec)

    l_rank = tail_intensity_rank_loss(out["radius"], y, margin=cfg.rank_margin)
    l_aux = sum(l1(out["aux"][k], y) for k in out["aux"]) / len(out["aux"])
    l_route = out["route_balance"]

    total = (main + l_dis + cfg.w_rank * l_rank
             + cfg.w_aux * l_aux + cfg.w_route_balance * l_route)
    parts = {"total": total.item(), "main": main.item(), "dis": l_dis.item(),
             "conflict": l_conflict.item(), "rank": l_rank.item(), "aux": l_aux.item()}
    return total, parts


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        preds.append(out["y_hat"].cpu().numpy())
        gts.append(batch["label"].cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return eval_regression(preds, gts)


def train_once(cfg: TridentConfig, seed: int):
    set_seed(seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = TRIDENT(cfg).to(device)

    optim = torch.optim.AdamW([
        {"params": model.base_parameters(), "lr": cfg.lr},
        {"params": model.hyperbolic_parameters(), "lr": cfg.lr_hyper},
    ], weight_decay=cfg.weight_decay)
    l1 = nn.L1Loss()

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt = os.path.join(cfg.save_dir, f"trident_{cfg.dataset}_{seed}.pth")
    best_mae, best_metrics, wait = float("inf"), None, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            y = batch["label"]
            out = model(batch)
            loss, parts = compute_loss(cfg, out, y, l1)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            running += parts["total"]

        val = evaluate(model, val_loader, device)
        print(f"[seed {seed}] epoch {epoch:02d} | "
              f"train_loss {running / len(train_loader):.4f} | val {dict_to_str(val)}")

        if val["MAE"] < best_mae:
            best_mae = val["MAE"]
            best_metrics = evaluate(model, test_loader, device)
            torch.save(model.state_dict(), ckpt)
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print(f"[seed {seed}] early stopping at epoch {epoch}")
                break

    print(f"[seed {seed}] best test: {dict_to_str(best_metrics)}")
    return best_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mosi")
    p.add_argument("--data_path", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="run a single seed")
    p.add_argument("--no_bert", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TridentConfig()
    cfg.dataset = args.dataset
    if args.data_path:
        cfg.data_path = args.data_path
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_bert:
        cfg.use_bert = False
    if args.cpu:
        cfg.device = "cpu"

    seeds = (args.seed,) if args.seed is not None else cfg.seeds
    all_metrics = []
    for seed in seeds:
        all_metrics.append(train_once(cfg, seed))

    if len(all_metrics) > 1:
        print("\n===== averaged over seeds =====")
        keys = all_metrics[0].keys()
        for k in keys:
            vals = [m[k] for m in all_metrics]
            print(f"{k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
