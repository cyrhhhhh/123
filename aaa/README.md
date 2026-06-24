# TRIDENT

**TRI-factor Disentanglement with hyperbolic iNTensity** — a novel multimodal
sentiment analysis model for CMU-MOSI / MOSEI.

TRIDENT replaces the conventional *shared / specific* two-factor decomposition
with a **three-factor** decomposition grounded in Partial Information
Decomposition (Williams & Beer, 2010), fuses the factors with a purpose-built
**role-aware** module, and reads out sentiment in **hyperbolic** space to handle
the long-tail of extreme-sentiment samples.

## Key ideas

1. **Three-factor disentanglement** (`models/disentangle.py`)
   - **Redundant** `C_m`: a weight-shared encoder extracts the cross-modal
     consensus from every modality.
   - **Unique** `U_m`: an independent encoder per modality captures
     modality-private information.
   - **Synergy** `S`: a symmetric cross-modal transformer captures information
     that only emerges from the *joint* of all modalities. An adversarial
     gradient-reversal objective makes `S` unpredictable from any single
     modality, so it stays genuinely emergent.

2. **RSGF — Role-aware Synergy-Gated Fusion** (`models/rsgf.py`)
   The three factors are fused *asymmetrically* by their semantic role:
   - **Stage 1 Reliability routing** sparsely weights the Unique factors,
     suppressing noisy modalities (with a load-balance regulariser).
   - **Stage 2 Anchor-supplement** uses the Redundant consensus as the backbone
     and injects routed Unique information as a residual.
   - **Stage 3 Synergy modulation** lets `S` apply a FiLM-style affine transform
     (scale + shift) that can amplify or flip the base sentiment (e.g. sarcasm).

3. **Hyperbolic Intensity Head** (`models/hyperbolic.py`)
   Sentiment is read out as a *radial* quantity in the Lorentz model:
   intensity = geodesic radius, polarity = learned direction. A **tail
   intensity ranking loss** (`losses/rank_loss.py`) forces stronger sentiment
   to sit further from the origin, targeting the long-tail extreme classes.

> No LFA / external fusion blocks are used — TRIDENT is a self-contained
> architecture.

## Project layout

```
config.py                 # all hyper-parameters (TridentConfig)
train.py                  # training / evaluation entry point
models/
  disentangle.py          # three-factor disentanglement
  rsgf.py                 # role-aware synergy-gated fusion
  hyperbolic.py           # hyperbolic intensity head
  grl.py                  # gradient reversal layer
  trident.py              # full model assembly
losses/
  disentangle_losses.py   # redundancy / unique / synergy / reconstruction
  rank_loss.py            # tail intensity ranking
data/data_loader.py       # MMSA pkl loader (+ synthetic fallback)
utils/metrics.py          # MMSA-style regression metrics
```

## Install

```bash
pip install -r requirements.txt
```

## Data

Download the MMSA-processed CMU-MOSI `aligned_50.pkl` and place it at
`./dataset/MOSI/aligned_50.pkl` (or pass `--data_path`). Feature dimensions are
inferred automatically at runtime.

## Run

```bash
# smoke test (no dataset required, runs on CPU in seconds)
python train.py --dataset synthetic --epochs 3 --cpu

# full training on MOSI (multi-seed, averaged)
python train.py --dataset mosi --data_path ./dataset/MOSI/aligned_50.pkl

# single seed
python train.py --dataset mosi --seed 1111
```

## Metrics

Reports MMSA-standard `MAE`, `Corr`, `Acc-7`, `Acc-5`, and non-zero
`Acc-2` / `F1`.
