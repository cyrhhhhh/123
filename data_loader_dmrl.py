import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MMDataset(Dataset):
    """DLF-style MMSA dataset wrapper for DMRL."""

    def __init__(self, args: Dict, mode: str = "train"):
        self.args = args
        self.mode = mode

        if args["dataset_name"] == "synthetic":
            data = self._make_synthetic_split(args, mode)
            self.use_bert = False
        else:
            with open(args["featurePath"], "rb") as f:
                raw = pickle.load(f)
            if "valid" not in raw and "dev" in raw:
                raw["valid"] = raw["dev"]
            data = raw[mode]
            self.use_bert = bool(args.get("use_bert", True) and ("text_bert" in data))

        self.vision = np.asarray(data["vision"], dtype=np.float32)
        self.audio = np.asarray(data["audio"], dtype=np.float32)

        # Cleanup numeric artifacts often seen in MMSA audio/vision features.
        self.audio[self.audio == -np.inf] = 0.0
        self.audio[np.isnan(self.audio)] = 0.0
        self.vision[self.vision == -np.inf] = 0.0
        self.vision[np.isnan(self.vision)] = 0.0

        if self.use_bert:
            self.text = np.asarray(data["text_bert"], dtype=np.float32)  # (N, 3, L)
        else:
            self.text = np.asarray(data["text"], dtype=np.float32)       # (N, T, D)

        if "regression_labels" in data:
            labels = data["regression_labels"]
        else:
            labels = data["labels"]
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)

        # Update inferred dimensions for model construction (DLF-style).
        d_v = int(self.vision.shape[-1])
        d_a = int(self.audio.shape[-1])
        d_t = 768 if self.use_bert else int(self.text.shape[-1])
        args["feature_dims"] = [d_t, d_a, d_v]
        args["effective_use_bert"] = self.use_bert

    def _make_synthetic_split(self, args: Dict, mode: str):
        n_map = {
            "train": int(args.get("synthetic_train", 256)),
            "valid": int(args.get("synthetic_valid", 64)),
            "test": int(args.get("synthetic_test", 64)),
        }
        seq = int(args.get("synthetic_seq_len", 50))
        n = n_map[mode]

        seed_map = {"train": 0, "valid": 1, "test": 2}
        rng = np.random.default_rng(seed_map[mode])

        vision = rng.standard_normal((n, seq, 20)).astype(np.float32)
        audio = rng.standard_normal((n, seq, 5)).astype(np.float32)
        text = rng.standard_normal((n, seq, 768)).astype(np.float32)
        signal = (0.5 * text.mean((1, 2)) + 0.3 * audio.mean((1, 2)) + 0.2 * vision.mean((1, 2)))
        labels = np.clip(signal * 2.0, -3, 3).astype(np.float32)

        return {
            "text": text,
            "audio": audio,
            "vision": vision,
            "regression_labels": labels,
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "labels": {"M": torch.tensor([self.labels[index]], dtype=torch.float32)},
        }


def MMDataLoader(args: Dict, num_workers: int = 0):
    datasets = {
        "train": MMDataset(args, mode="train"),
        "valid": MMDataset(args, mode="valid"),
        "test": MMDataset(args, mode="test"),
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=int(args["batch_size"]),
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
        for split in ("train", "valid", "test")
    }
    return loaders
