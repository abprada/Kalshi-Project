# Kalshi Project

## 1. Data Preprocessing

Data sourced from [prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis).
Run `notebooks/data.ipynb` to produce the following in `data/kalshi/processed/`:

### `all_markets.parquet` (~7.6M rows)

One row per market contract, a snapshot of each contract's final state.
- `ticker` — unique market identifier
- `yes_bid`, `yes_ask`, `no_bid`, `no_ask` — bid/ask prices in cents
- `yes_spread`, `no_spread` — ask − bid for each side
- `last_price` — last traded price
- `open_time`, `close_time` — market open/close timestamps
- `status`, `volume`

### `all_trades.parquet` (~72M rows)

One row per trade execution, the full tick-by-tick history of every transaction on Kalshi.
- `ticker` — which market contract was traded
- `yes_price` — execution price in cents (0–100); `no_price` is dropped as it is always `100 − yes_price`
- `count` — number of contracts in the fill
- `taker_side` — `"yes"` or `"no"`
- `created_time` — trade timestamp
- `volume` — notional size in dollars (`count × yes_price / 100`)

### `trades_with_labels.parquet`

Extends the filtered trade history with two EDA-derived columns used downstream by the model:
- `return_15min` — absolute 15-minute backward return in cents (`yes_price − yes_price_15m_ago`)
- `y` — large-move label: `+1` for trades above the 75th-percentile return, `-1` for trades below the 25th percentile, `0` otherwise

Built after two filters: (1) trades before the first date with >10M aggregate daily volume are removed, and (2) trades where the base price 15 minutes ago was below 3 cents are dropped to avoid extreme-valuation noise.

## 2. Milestone 3: Baseline Jump Detection Model

### Overview

A 3-block 1D CNN (`JumpCNN`) that predicts whether a 1-minute bar sits at the endpoint of a large forward price move. One model is trained per prediction horizon H ∈ {5, 15, 30, 60} minutes, with labels and thresholds computed separately for each horizon from the training-set return distribution.

### Architecture

**Input:** $(B, F, 32)$ — batch size $B$, $F = 6$ features, lookback of 32 bars.

| Block | Layers | Output shape |
|---|---|---|
| 1 | `Conv1d(F→64, k=3, p=1)` → BN → ReLU × 2, `MaxPool1d(2)`, `Dropout(0.2)` | $(B, 64, 16)$ |
| 2 | `Conv1d(64→128, k=3, p=1)` → BN → ReLU × 2, `MaxPool1d(2)`, `Dropout(0.2)` | $(B, 128, 8)$ |
| 3 | `Conv1d(128→256, k=3, p=1)` → BN → ReLU, `AdaptiveAvgPool1d(1)` | $(B, 256, 1)$ |
| Head | `Linear(256→1)` | $(B, 1)$ |

### Pipeline

1. **Load** `trades_with_labels.parquet`, keep only raw columns (`ticker`, `created_time`, `yes_price`, `volume`, `count`).
2. **Filter** tickers with fewer than `min_trades_per_ticker = 50` trades.
3. **Aggregate** into 1-minute bars per ticker: `close` (last price), `volume` (sum), `n_trades` (sum).
4. **Build features** — all causal (no lookahead):

| Feature | Description |
|---|---|
| `close_norm` | `close / 100` |
| `ret_1` | 1-bar % return |
| `ret_5` | 5-bar % return |
| `log_volume` | `log(1 + volume)` |
| `hour_sin` / `hour_cos` | Cyclical hour-of-day encoding |

5. **Compute forward returns** using real timestamps for each horizon H, with a tolerance window of `max(2, H × 0.5)` minutes. Bars without a future observation within the tolerance are assigned `NaN`.
6. **Split chronologically** 70 / 15 / 15 (train / val / test) on global bar timestamps.
7. **Binarise labels**: thresholds are the 10th and 90th percentiles of the training-set forward return distribution; the same fixed thresholds are applied to val and test. A bar is labeled `1` (jump) if its forward return exceeds the upper threshold or falls below the lower threshold.
8. **Save** the processed dataset to `bars_clean.parquet` (skip if the file already exists).
9. **Train** one `JumpCNN` per horizon with `BCEWithLogitsLoss` + `pos_weight` for class imbalance. Windows spanning more than 90 real minutes are excluded. Checkpoints are saved to `checkpoints/jump_cnn_{H}m.pt`.
10. **Threshold search** on the validation set: find the lowest decision threshold that achieves at least 60% precision (`MIN_PRECISION_TARGET = 0.60`).
11. **Evaluate** on the test set: AUC-PR, AUC-ROC, accuracy, balanced accuracy, F1, MCC, and a confusion matrix.

### Interpretability

After training, the notebook computes:
- **Gradient saliency maps** averaged over the 200 most confident correct jump and no-jump predictions.
- **Feature ablation**: each feature is zeroed out one at a time to measure its contribution to the predicted jump probability.

Key finding: `close_norm` dominates — ablating it collapses predictions in ~99% of cases. Recent momentum features (`ret_1`, `ret_5`) contribute very little, suggesting the CNN is primarily learning a price-level heuristic rather than temporal patterns.

### Caching

The bar construction (steps 1–4) is checkpointed to `bars_clean.parquet`. Subsequent runs load this file and skip straight to training. Delete the file to force a full rebuild.

### Notes

- Windows longer than 90 real minutes are rejected so sparse tickers with large inactivity gaps don't produce misleading inputs.
- The flat-price pattern in illiquid markets inflates AUC: the CNN learns that any tick after a long quiet period is a jump. These windows should be filtered or discounted in the next milestone.

---

## Milestone 4 (MS4): Mixture of Experts — instructions for course staff

This milestone is delivered as a **main Colab notebook** plus **large artifacts on Google Drive** (parquet tables, trained checkpoints, caches). The notebook is self-documenting (table of contents, folder layout, and a **TF setup** cell).

**Teaching staff — Drive is optional for review.** You can read the submitted notebook (method, code, and saved outputs) without copying anything from Drive. **Only download or copy the Drive `project` folder if you want to re-run cells**; the bundle is **larger than 10 GB**.

### Shared Drive folder (data + models + notebook)

**Folder:** [project — Google Drive](https://drive.google.com/drive/folders/1bwPVZxxEhNfPodRlXFBZV-RKXiuSnlkq?usp=sharing)

Expected top-level contents: `data/`, `models/` (under `data/` as in the notebook), `moe_cache/`, `notebooks/`, and at the project root `requirements.txt` (and the main notebook under `notebooks/`).

### What course staff should do

1. **Default (read-only):** Open `notebooks/cs1090b_ms4_main_group52.ipynb` from the submission (e.g. GitHub preview, downloaded `.ipynb`, or Colab upload of the file alone). No Drive access required to see what we did.
2. **Only if re-running:** Open the [Drive folder](https://drive.google.com/drive/folders/1bwPVZxxEhNfPodRlXFBZV-RKXiuSnlkq?usp=sharing) and copy the entire **`project`** tree into **your own** Google Drive (for example **Organize → Add shortcut to Drive** or duplicate into `MyDrive`). Then open the notebook in Colab from that tree.
3. **Run the TF setup cell** near the top of the notebook (after the table of contents). It mounts Drive (on Colab), sets `PROJECT_ROOT`, and installs from `requirements.txt` if that file sits next to `data/` and `notebooks/`.
4. If your copy of `project` is **not** at `MyDrive/CS1090B/project`, **edit `PROJECT_ROOT`** in that setup cell to the path of the folder that contains both `data/` and `notebooks/`.
5. Run the rest **top to bottom** (for example **Runtime → Run all**). **Do not** run the other experimental notebooks in `notebooks/` unless you intend to regenerate intermediates; the main notebook states this in its disclaimer.

### GitHub vs. Drive

- **Drive** holds everything needed to **execute** MS4 (data, checkpoints, caches, notebook, `requirements.txt`).
- **GitHub** (this repo or a small spin-off) is useful for **version history and code review** of the notebook and `requirements.txt`. It is **not** required to reproduce runs if staff use Drive only.
