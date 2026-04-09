# Kalshi Project

## 1. Data Preprocessing

Data sourced from [prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis).

Run `notebooks/data.ipynb` to produce the following in `data/kalshi/processed/`:

### `all_markets.parquet` (~7.6M rows)
One row per market contract — a snapshot of each contract's final state.
- `ticker` — unique market identifier
- `yes_bid`, `yes_ask`, `no_bid`, `no_ask` — bid/ask prices in cents
- `yes_spread`, `no_spread` — ask − bid for each side
- `last_price` — last traded price
- `open_time`, `close_time` — market open/close timestamps
- `status`, `volume`

### `all_trades.parquet` (~72M rows)
One row per trade execution — the full tick-by-tick history of every transaction on Kalshi.
- `ticker` — which market contract was traded
- `yes_price`, `no_price` — execution prices (always sum to 100)
- `count` — number of contracts in the fill
- `taker_side` — `"yes"` or `"no"`
- `created_time` — trade timestamp

### `trade_timeseries_1min.parquet`
1-minute resampled time series per ticker, derived from `all_trades`:
- `ticker` — Contract ticker
- `last_yes_price` — last trade price in each minute
- `vwap_yes` — volume-weighted average price
- `n_trades` — number of trades in the minute

# Notes Andres
- Non-traded contracts typically a price of 0, have to remove those without removing settled contracts at 0