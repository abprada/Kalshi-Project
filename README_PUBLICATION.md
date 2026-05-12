# Kalshi Prediction Markets: Mixture of Experts

This repository contains a publication-facing version of the Kalshi short-horizon price movement pipeline. The main executable entry point is [`main.py`](main.py), which trains and evaluates a Mixture of Experts model over 5, 15, 30, and 60 minute horizons.

## Project Overview

The task is to classify whether a Kalshi contract's `yes_price` moves down, stays flat, or moves up over a forward window. The final model combines six base experts:

- LightGBM
- LSTM
- Mamba
- Moirai
- FT-Transformer
- Conformer-Tiny Time-Series

Each expert produces a 3-class probability vector. A learned PyTorch gating network assigns per-trade weights to the experts and returns the final prediction.

## Artifacts

Large data and model artifacts are hosted outside Git:

[Google Drive artifact folder](https://drive.google.com/drive/folders/1bwPVZxxEhNfPodRlXFBZV-RKXiuSnlkq?usp=sharing)

Expected artifact layout:

```text
project/
  data/
    moe_data.parquet
    models/
      moe_gate_5m.pt
      moe_gate_15m.pt
      moe_gate_30m.pt
      moe_gate_60m.pt
  moe_cache/
  notebooks/
  requirements.txt
```

The artifact bundle is larger than 10 GB. Git tracks the publication code and documentation, while Drive stores data, checkpoints, and caches.

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run from a local artifact folder:

```bash
PROJECT_ROOT=/path/to/project python main.py
```

Run from the repository root if `data/`, `moe_cache/`, and `requirements.txt` are already present there:

```bash
python main.py
```

To let the script install dependencies from the artifact folder automatically, set:

```bash
INSTALL_REQUIREMENTS=1 PROJECT_ROOT=/path/to/project python main.py
```

## Repository Files

- [`main.py`](main.py): publication script generated from the MoE notebook and cleaned for standard Python execution.
- [`requirements.txt`](requirements.txt): minimal Python dependency list.
- [`notebooks/`](notebooks/): research notebooks used during model development and analysis.
- [`README.md`](README.md): original project README with full development context.

## Reproducibility Notes

The script expects `data/moe_data.parquet` and saved MoE checkpoints under `data/models/`. Training uses chronological train, validation, and test splits. Evaluation reports balanced accuracy, macro F1, confusion matrices, gate weight dynamics, and permutation feature importance.
