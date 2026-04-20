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
- `yes_price`, `no_price` — execution prices (always sum to 100)
- `count` — number of contracts in the fill
- `taker_side` — `"yes"` or `"no"`
- `created_time` — trade timestamp

### `trade_timeseries_1min.parquet`

1-minute resampled time series per ticker, derived from `all_trades`:
- `ticker` — contract ticker
- `last_yes_price` — last trade price in each minute
- `vwap_yes` — volume-weighted average price
- `n_trades` — number of trades in the minute

### `trades_with_labels.parquet`

Extends `all_trades` with two EDA-derived columns used downstream by the Milestone 3 baseline:
- `return_15min` — 15-minute backward return in cents at each trade
- `y` — large-move label, set to `+1` for trades at the top of the return distribution, `-1` for the bottom, `0` otherwise

Built in the same notebook after filtering out trades from the illiquid early period (days before the first date with >10M aggregate daily volume).

## 2. Milestone 3: Baseline Jump Detection Model

### Overview

A 1D CNN with a sector embedding (`SimpleJumpCNN`) that predicts whether a 1-minute bar sits at the endpoint of a large 15-minute price move. Trained on features derived from 1-minute bars of trade data, with labels inherited from the EDA (`y = ±1` → `y_jump = 1`).

### Pipeline

1. **Load** `trades_with_labels.parquet`, capped by `memory.recent_days` and `memory.max_tickers` to fit in Colab RAM.
2. **Resample** trades into 1-minute bars per ticker.
3. **Build features**: `close_norm`, `ret_1`, `log_volume`, `hour_sin`, `hour_cos`, `eda_ret_15m`. Forward-fill close prices across empty minutes so illiquid gaps don't produce NaN channels.
4. **Assemble windows**: each bar becomes the endpoint of a causal lookback window of length `lookback_bars = 32`.
5. **Split** chronologically (70/30) on unique bar timestamps.
6. **Train** the CNN with BCE loss, early stopping on validation AUC-PR, gradient clipping, and NaN-loss guards.
7. **Evaluate** on AUC-PR, AUC-ROC, Brier score, with a per-sector breakdown.

### Caching

The bar construction (steps 1–3) is the slowest part of the run, so it's cached to `data/kalshi/processed/bars_cache_{recent_days}d_{max_tickers}tkr.parquet` after the first build. Subsequent runs load the cache and jump straight to training. Pass `use_cache=False` to `train_baseline_milestone` to force a rebuild, or delete the cache file manually.

The trained model checkpoint is saved to `jump_baseline_cnn.pt`.

### Results

The baseline produces strong signal on liquid sectors (crypto price contracts, major sports markets) and is effectively unusable on sparser sectors where 32-minute continuous windows rarely form. See `notebooks/data.ipynb` for the full per-sector table.

## Notes

- Non-traded contracts typically have a price of 0. These need to be filtered out without removing contracts that legitimately settled at 0.
- The `ffill` on close prices is important: without it, illiquid minutes produce NaN `close_norm`, which propagates through the CNN and corrupts predictions.