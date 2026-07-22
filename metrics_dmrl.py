"""MMSA/DLF-style evaluation metrics for MOSI and MOSEI regression."""

from typing import Dict

import numpy as np
import torch


__all__ = ["MetricsTop", "dict_to_str"]


def dict_to_str(src_dict: Dict[str, float]) -> str:
    return " ".join(f"{key}: {value:.4f}" for key, value in src_dict.items())


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 without requiring scikit-learn."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    weighted_sum = 0.0
    total_support = 0.0

    for cls in classes:
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        tp = float(np.sum(true_cls & pred_cls))
        fp = float(np.sum(~true_cls & pred_cls))
        fn = float(np.sum(true_cls & ~pred_cls))
        support = float(np.sum(true_cls))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

        weighted_sum += support * f1
        total_support += support

    return weighted_sum / (total_support + 1e-8)


def _multiclass_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """DLF/MMSA multiclass accuracy after rounding continuous scores."""
    return float(np.mean(np.round(y_pred) == np.round(y_true)))


class MetricsTop:
    """Metric selector with the same public API used by DLF.

    Usage:
        metrics = MetricsTop("regression").getMetics("mosi")
        result = metrics(pred_tensor, label_tensor)
    """

    def __init__(self, train_mode: str = "regression"):
        if train_mode != "regression":
            raise ValueError("DMRL currently supports regression metrics only.")
        self.metrics_dict = {
            "MOSI": self._eval_mosi_regression,
            "MOSEI": self._eval_mosei_regression,
            "SYNTHETIC": self._eval_mosi_regression,
        }

    def _eval_mosei_regression(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        exclude_zero: bool = False,
    ) -> Dict[str, float]:
        pred = y_pred.reshape(-1).detach().cpu().numpy().astype(np.float64)
        true = y_true.reshape(-1).detach().cpu().numpy().astype(np.float64)

        if exclude_zero:
            keep = true != 0
            pred = pred[keep]
            true = true[keep]

        if true.size == 0:
            return {
                "MAE": 0.0,
                "Corr": 0.0,
                "Acc_7": 0.0,
                "Acc_5": 0.0,
                "Acc_3": 0.0,
                "Acc_2": 0.0,
                "F1_score": 0.0,
                "Has0_Acc_2": 0.0,
                "Has0_F1_score": 0.0,
            }

        pred_a7 = np.clip(pred, -3.0, 3.0)
        true_a7 = np.clip(true, -3.0, 3.0)
        pred_a5 = np.clip(pred, -2.0, 2.0)
        true_a5 = np.clip(true, -2.0, 2.0)
        pred_a3 = np.clip(pred, -1.0, 1.0)
        true_a3 = np.clip(true, -1.0, 1.0)

        mae = float(np.mean(np.abs(pred - true)))
        if np.std(pred) < 1e-8 or np.std(true) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(pred, true)[0, 1])

        # Standard MMSA non-zero binary metrics: negative (<0) vs positive (>0).
        nonzero = true != 0
        if nonzero.any():
            nonzero_true = true[nonzero] > 0
            nonzero_pred = pred[nonzero] > 0
            acc2 = float(np.mean(nonzero_pred == nonzero_true))
            f1 = float(_weighted_f1(nonzero_true, nonzero_pred))
        else:
            acc2, f1 = 0.0, 0.0

        # Binary metrics including neutral samples: negative (<0) vs non-negative (>=0).
        has0_true = true >= 0
        has0_pred = pred >= 0
        has0_acc2 = float(np.mean(has0_pred == has0_true))
        has0_f1 = float(_weighted_f1(has0_true, has0_pred))

        return {
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),
            "Acc_7": round(_multiclass_acc(pred_a7, true_a7), 4),
            "Acc_5": round(_multiclass_acc(pred_a5, true_a5), 4),
            "Acc_3": round(_multiclass_acc(pred_a3, true_a3), 4),
            "Acc_2": round(acc2, 4),
            "F1_score": round(f1, 4),
            "Has0_Acc_2": round(has0_acc2, 4),
            "Has0_F1_score": round(has0_f1, 4),
        }

    def _eval_mosi_regression(self, y_pred, y_true):
        return self._eval_mosei_regression(y_pred, y_true)

    def getMetics(self, dataset_name: str):
        """Keep DLF's original misspelled method name for API compatibility."""
        name = dataset_name.upper()
        if name not in self.metrics_dict:
            raise KeyError(f"Unsupported dataset for metrics: {dataset_name}")
        return self.metrics_dict[name]
