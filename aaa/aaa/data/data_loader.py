"""Data loading for CMU-MOSI / MOSEI in the MMSA processed format.

The MMSA ``*.pkl`` files are dictionaries with ``train`` / ``valid`` / ``test``
splits, each holding:

    text       : (N, L, 768)  precomputed BERT features   (or)
    text_bert  : (N, 3, L)    [input_ids, attention_mask, token_type_ids]
    vision     : (N, L, Dv)
    audio      : (N, L, Da)
    regression_labels / labels : (N,)

Feature dimensions are inferred here and written back into ``cfg.feat_dims`` so
the model builds itself to match the data. A synthetic fallback dataset lets the
whole pipeline run (smoke test) without the real pickle.
"""
import pickle
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TridentConfig


class MMSADataset(Dataset):
    def __init__(self, split: Dict, use_bert: bool):
        self.use_bert = use_bert and ("text_bert" in split)
        self.vision = np.asarray(split["vision"], dtype=np.float32)
        self.audio = np.asarray(split["audio"], dtype=np.float32)

        # clean degenerate values that COVAREP / OpenFace can produce
        # (-inf / NaN), following the MMSA / DLF data pipeline
        self.audio[self.audio == -np.inf] = 0.0
        self.audio[np.isnan(self.audio)] = 0.0
        self.vision[self.vision == -np.inf] = 0.0
        self.vision[np.isnan(self.vision)] = 0.0

        if self.use_bert:
            self.text_bert = np.asarray(split["text_bert"], dtype=np.float32)
            self.text = None
        else:
            self.text = np.asarray(split["text"], dtype=np.float32)
            self.text_bert = None

        if "regression_labels" in split:
            labels = split["regression_labels"]
        else:
            labels = split["labels"]
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "V": torch.from_numpy(self.vision[idx]),
            "A": torch.from_numpy(self.audio[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.use_bert:
            item["text_bert"] = torch.from_numpy(self.text_bert[idx])
        else:
            item["L"] = torch.from_numpy(self.text[idx])
        return item


def _make_synthetic(n: int, seq: int = 50) -> Dict:
    """Random tensors with a learnable signal, for smoke testing."""
    rng = np.random.default_rng(0)
    vision = rng.standard_normal((n, seq, 20)).astype(np.float32)
    audio = rng.standard_normal((n, seq, 5)).astype(np.float32)
    text = rng.standard_normal((n, seq, 768)).astype(np.float32)
    # label correlated with a projection of the features (so training can move)
    signal = (vision.mean((1, 2)) + audio.mean((1, 2)) + text.mean((1, 2)))
    labels = np.clip(signal * 1.5, -3, 3).astype(np.float32)
    return {"vision": vision, "audio": audio, "text": text,
            "regression_labels": labels}


def _infer_feat_dims(split: Dict, use_bert: bool) -> Dict[str, int]:
    dv = int(np.asarray(split["vision"]).shape[-1])
    da = int(np.asarray(split["audio"]).shape[-1])
    if use_bert and "text_bert" in split:
        dl = 768   # BERT hidden size; the model overrides this when BERT is built
    else:
        dl = int(np.asarray(split["text"]).shape[-1])
    return {"L": dl, "V": dv, "A": da}


def _collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "label":
            out["label"] = torch.stack([b["label"] for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def build_dataloaders(cfg: TridentConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.dataset == "synthetic":
        data = {"train": _make_synthetic(256),
                "valid": _make_synthetic(64),
                "test": _make_synthetic(64)}
        use_bert = False
    else:
        with open(cfg.data_path, "rb") as f:
            data = pickle.load(f)
        # MMSA uses "valid"; some dumps use "dev"
        if "valid" not in data and "dev" in data:
            data["valid"] = data["dev"]
        use_bert = cfg.use_bert and ("text_bert" in data["train"])

    cfg.feat_dims = _infer_feat_dims(data["train"], use_bert)
    cfg.use_bert = use_bert

    loaders = []
    for split_name, shuffle in [("train", True), ("valid", False), ("test", False)]:
        ds = MMSADataset(data[split_name], use_bert)
        loaders.append(DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, collate_fn=_collate, drop_last=shuffle,
        ))
    return tuple(loaders)
