"""Evaluation metrics, MMSA-style for CMU-MOSI / MOSEI."""
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _clip_round(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.round(x), lo, hi)


def eval_regression(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Standard MMSA regression metrics for sentiment in [-3, 3].

    Returns MAE, Corr, Acc-7, Acc-5, and (non-zero) Acc-2 / F1.
    """
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_pred, y_true)[0, 1])

    # 7-class and 5-class accuracy (rounded, clipped)
    a7 = float(accuracy_score(_clip_round(y_true, -3, 3), _clip_round(y_pred, -3, 3)))
    a5 = float(accuracy_score(_clip_round(y_true, -2, 2), _clip_round(y_pred, -2, 2)))

    # non-zero binary accuracy / F1 (drop neutral samples, MMSA convention)
    nz = y_true != 0
    if nz.sum() > 0:
        bt = y_true[nz] > 0
        bp = y_pred[nz] > 0
        acc2 = float(accuracy_score(bt, bp))
        f1 = float(f1_score(bt, bp, average="weighted"))
    else:
        acc2 = f1 = 0.0

    return {
        "MAE": mae, "Corr": corr,
        "Acc_7": a7, "Acc_5": a5,
        "Acc_2": acc2, "F1": f1,
    }


def dict_to_str(d: Dict[str, float]) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in d.items())
