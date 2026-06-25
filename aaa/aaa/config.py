"""Global configuration for the FLIP model.

FLIP = conFLict-Induced Polarity reversal for multimodal sentiment analysis.

The representation is disentangled into three role-specific factors --
Consensus (C), modality-Specific (U), and directed Conflict (D) -- and the
conflict factor drives a *polarity-reversal* gate: when modalities disagree
(e.g. sarcasm), the surface sentiment sign is flipped while its intensity is
preserved.

All hyper-parameters are collected here so experiments are reproducible and
easy to sweep. Feature dimensions for each modality are inferred automatically
from the data at runtime (see ``data/data_loader.py``) and injected into the
model, so you normally only need to touch the values below.
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TridentConfig:
    # ----- data -----
    dataset: str = "mosi"                # "mosi" | "mosei" | "synthetic"
    data_path: str = "./dataset/MOSI/aligned_50.pkl"
    use_bert: bool = True                # encode text from `text_bert` with BERT
    bert_name: str = "bert-base-uncased"
    batch_size: int = 16
    num_workers: int = 0

    # ----- model dims (feature dims are inferred at runtime) -----
    d_model: int = 128                   # common hidden size after projection
    n_heads: int = 4
    n_layers_shared: int = 1             # layers in the shared (consensus) encoder
    n_layers_unique: int = 1             # layers in each modality-specific encoder
    n_layers_conflict: int = 2           # layers in the cross-modal conflict transformer
    n_layers_fusion: int = 1             # layers inside the fusion transformer
    dropout: float = 0.3
    curvature: float = 1.0               # initial Lorentz curvature c (>0, learnable)
    learn_curvature: bool = True

    # ----- loss weights -----
    w_rank: float = 0.2                  # tail intensity ranking loss
    w_aux: float = 0.1                   # deep supervision on factors / fusion
    # disentanglement loss weights
    a_con: float = 0.1                   # consensus alignment (InfoNCE)
    a_spec: float = 0.1                  # specific orthogonality
    a_rec: float = 0.1                   # reconstruction
    w_conflict: float = 0.2              # conflict-directionality (reversal-gate supervision)
    w_route_balance: float = 0.01        # reliability-router load-balance regulariser
    conflict_tau: float = 0.1            # |y| threshold to count a sample as confident
    rank_margin: float = 0.1             # margin delta in ranking loss
    infonce_tau: float = 0.1             # temperature for consensus alignment

    # ----- optimisation -----
    epochs: int = 40
    lr: float = 1e-4                     # base lr for Euclidean params
    lr_hyper: float = 1e-3               # lr for hyperbolic / curvature params
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10                   # early-stopping patience
    seeds: tuple = (1111, 1112, 1113, 1114, 1115)

    # ----- misc -----
    device: str = "cuda"                 # falls back to cpu automatically
    save_dir: str = "./pt"
    log_dir: str = "./log"

    # filled in at runtime: {"L": dim, "V": dim, "A": dim}
    feat_dims: Dict[str, int] = field(default_factory=dict)
