# US Market Trading Bot

An automated trading system for US equities built on **Interactive Brokers** (IBKR) via `ib_insync`. Runs three independent strategies concurrently, with a full event-driven backtester, trade database, and Optuna-powered hyperparameter optimisation.

---

## Strategies

| Strategy | Symbol(s) | Logic |
|---|---|---|
| **CLF Grid** | CLF (Cleveland-Cliffs) | Geometric grid — buy orders placed below anchor price at 1.5% intervals; each fill places a take-profit sell at `fill × GRID_RATIO`. 30% safety switch halts new orders on large price swings. |
| **AMZN Reversion** | AMZN | Mean-reversion — enters long when price breaks below the lower Bollinger Band (20-period) **and** RSI(14) < 25. Exits at the 20-period SMA or 3% take-profit. |
| **SmallCap Arb** | IREN / WULF | Statistical arbitrage — trades the z-score of the IREN/WULF price ratio. Enters when \|z\| > 2.0, exits when \|z\| < 0.5. Dollar-neutral sizing with 1% market-impact cap. |

---

## Project Structure

```
├── main.py                     # Entry point — runs all strategies live
│
├── core/
│   ├── connection.py           # IBConnection — Gateway/TWS lifecycle
│   ├── backtester.py           # SimulatedIB + BacktestEngine (event-driven)
│   └── database.py             # SQLite / PostgreSQL via SQLAlchemy
│
├── strategies/
│   ├── base.py                 # BaseStrategy — abstract, shared order helpers
│   ├── clf_grid.py             # CLF geometric grid
│   ├── amzn_reversion.py       # AMZN BB + RSI mean-reversion
│   └── smallcap_arb.py         # IREN/WULF z-score pairs trade
│
├── utils/
│   ├── data_fetcher.py         # DataFetcher + IndicatorUtils, parquet cache
│   └── dashboard_stats.py      # Sharpe, Sortino, Calmar, VaR
│
├── scripts/
│   ├── download_cache.py       # Download historical data from IBKR → parquet
│   └── optimize.py             # Optuna hyperparameter study
│
├── data/
│   └── cache/                  # Parquet files (gitignored)
│
├── logs/                       # Rotating log files (gitignored)
├── Dockerfile
└── requirements.txt
```

---

## Prerequisites

- Python 3.11+
- **IBKR Gateway** or **TWS** running and accepting API connections
  - Paper trading port: `7497`
  - Live trading port: `7496`
- An active IBKR market data subscription covering AMZN, CLF, IREN, WULF

---

## Installation

```bash
git clone https://github.com/bxjxis/US-market-trading-bot.git
cd US-market-trading-bot

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed):

```env
IB_HOST=127.0.0.1
IB_PORT=7497          # 7497 paper | 7496 live | 4001 Gateway paper | 4002 Gateway live
IB_CLIENT_ID=1
```

---

## Workflow

### 1 — Download historical data

Populates `data/cache/` with Parquet files used by the backtester and optimiser.
Requires a live IBKR connection.

```bash
python scripts/download_cache.py                        # default: 20 days, 5-min bars
python scripts/download_cache.py --duration "30 D"      # longer history
python scripts/download_cache.py --bar-size "1 min"     # finer resolution
```

### 2 — Validate the pipeline

Runs a single backtest with default parameters to confirm everything is wired up correctly.

```bash
python scripts/optimize.py --dry-run
```

### 3 — Optimise hyperparameters

Runs an Optuna study that maximises the **Calmar Ratio** subject to a 30-day 99% VaR constraint of **< $15,000** on the full $180,000 portfolio.

```bash
python scripts/optimize.py --trials 100

# Persistent study (resumable, supports parallel workers)
python scripts/optimize.py --trials 500 --jobs 4 --storage sqlite:///data/optuna.db
```

After the study completes, three interactive HTML charts are saved to `data/`:

| File | Chart |
|---|---|
| `optuna_history.html` | Objective value across all trials |
| `optuna_importances.html` | Parameter importance (fANOVA) — which knobs matter most |
| `optuna_contour.html` | 2-D contour of grid ratio vs AMZN RSI vs arb z-score entry |

Open any file directly in a browser — no server required. Requires `pip install plotly`.

**Parameters tuned:**

| Strategy | Parameter | Range |
|---|---|---|
| CLF Grid | `GRID_RATIO` | 1.010 – 1.030 |
| CLF Grid | `NUM_BUY_LEVELS` | 6 – 15 |
| CLF Grid | `ACCOUNT_SIZE` | $36k – $108k |
| AMZN | `BB_PERIOD` | 15 – 30 |
| AMZN | `RSI_ENTRY` | 20 – 30 |
| AMZN | `TAKE_PROFIT` | 2% – 5% |
| SmallCap Arb | `ZSCORE_ENTRY` | 1.5 – 3.0 |
| SmallCap Arb | `ZSCORE_EXIT` | 0.3 – 1.0 |
| SmallCap Arb | `HISTORY_DURATION` | 10 D / 20 D / 30 D |

### 4 — Run live trading

```bash
python main.py
```

All three strategies start concurrently. Press `Ctrl-C` for a clean shutdown.

---

## Backtester Design

The backtester uses the **exact same strategy class files** as live trading — no duplicated logic.

`SimulatedIB` is a drop-in replacement for `ib_insync.IB`. It intercepts:
- `placeOrder` → queues a simulated limit order
- `reqHistoricalDataAsync` → serves bars from parquet
- `reqMktData` → returns a `SimulatedTicker` seeded from parquet data

**Order fill model:**
- `BUY` limit at price `P`: fills when `bar.low ≤ P` at `min(P, bar.open) × (1 + 0.1%)`
- `SELL` limit at price `P`: fills when `bar.high ≥ P` at `max(P, bar.open) × (1 − 0.1%)`

**Cost model:** `max($0.35, qty × $0.0035)` commission per order + 0.1% slippage.

---

## Database

Trade fills and equity snapshots are logged to SQLite by default (`data/trading.db`).
Switch to PostgreSQL by setting `DATABASE_URL` in `.env`:

```env
DATABASE_URL=postgresql://user:pass@host:5432/trading
```

**Schema:**

| Table | Contents |
|---|---|
| `backtest_runs` | One row per run — params, start/end time, final metrics |
| `trades` | Every filled order — symbol, action, qty, price, commission, slippage |
| `equity_snapshots` | Periodic NAV snapshots for drawdown / VaR computation |

---

## Performance Metrics

`utils/dashboard_stats.py` provides:

| Metric | Description |
|---|---|
| Sharpe Ratio | Annualised, daily returns |
| Sortino Ratio | Downside-deviation weighted |
| Max Drawdown | Peak-to-trough, as a fraction |
| Calmar Ratio | CAGR / Max Drawdown |
| 30d 99% VaR | Historical VaR in dollars on the $180k portfolio |

---

## Configuration Reference

All strategy parameters can be overridden at instantiation via a `params` dict — used by the backtester and optimiser without modifying strategy source files.

```python
from strategies.clf_grid import CLFGridStrategy

# Custom parameters for a backtest run
s = CLFGridStrategy(ib, params={
    "GRID_RATIO":     1.02,
    "NUM_BUY_LEVELS": 8,
    "ACCOUNT_SIZE":   72_000,
})
```

**CLF Grid defaults:**

| Parameter | Default | Description |
|---|---|---|
| `GRID_RATIO` | `1.015` | Geometric step between grid levels (1.5%) |
| `SAFETY_PCT` | `0.30` | Halt new orders if price deviates > 30% from anchor |
| `ACCOUNT_SIZE` | `1_000` | USD allocated to this strategy |
| `NUM_BUY_LEVELS` | `10` | Number of buy levels below anchor |

**AMZN Reversion defaults:**

| Parameter | Default | Description |
|---|---|---|
| `BB_PERIOD` | `20` | Bollinger Band SMA period |
| `BB_STD` | `2.0` | Standard deviation multiplier |
| `RSI_PERIOD` | `14` | RSI lookback (Wilder smoothing) |
| `RSI_ENTRY` | `25.0` | RSI threshold for oversold entry |
| `TAKE_PROFIT` | `0.03` | Exit on 3% unrealised gain |

**SmallCap Arb defaults:**

| Parameter | Default | Description |
|---|---|---|
| `ZSCORE_ENTRY` | `2.0` | Open position when \|z\| exceeds this |
| `ZSCORE_EXIT` | `0.5` | Close position when \|z\| falls below this |
| `HISTORY_DURATION` | `"20 D"` | Lookback for computing pair statistics |
| `MARKET_IMPACT_PCT` | `0.01` | Max order size as fraction of avg 5-min volume |

---

## Risk Controls

- **CLF Safety Switch** — no new grid orders when price deviates > 30% from the anchor; resets automatically when price returns to the band.
- **SmallCap Market Impact Cap** — each leg capped at 1% of average 5-minute volume.
- **Dollar-Neutral Sizing** — IREN and WULF notional values are matched on every entry.
- **VaR Constraint** — optimiser objective penalises parameter sets where the 30-day 99% VaR exceeds $15,000 on the $180k portfolio.
- **DAY orders** for SmallCap Arb — legs expire at session end; no unhedged overnight exposure.

---

## Known Limitations

### Gateway reconnection

`IBConnection` retries the initial connection up to 3 times with a 5-second back-off. However, it does **not** automatically reconnect if the Gateway drops during live trading.

IBKR Gateway restarts every **Sunday ~11:45 PM ET** for weekly maintenance, which will disconnect any running bot. The current workaround is a cron job or process manager (e.g. `supervisord`, `systemd`, or Docker `--restart=unless-stopped`) to relaunch `main.py` after the outage window. Full in-process reconnection logic is on the roadmap.

### CLF Grid — price below all levels

If CLF falls below every grid level (i.e. more than `SAFETY_PCT = 30%` from the anchor), no new buy orders are submitted by the safety switch and all existing GTC buy orders remain live. There is **no automatic stop-loss** — the strategy holds its filled positions and waits for a recovery. This is a deliberate choice for a grid strategy (mean-reversion assumption), but it means maximum drawdown is bounded only by the account size allocated to CLF. Set `ACCOUNT_SIZE` conservatively.

### Slippage model for small-cap stocks

The backtester uses a flat **0.1% slippage** on every fill. For liquid large-caps (AMZN) this is a reasonable assumption. For IREN and WULF — which can have wide bid/ask spreads and low 5-minute volumes — actual slippage during high-volatility periods may be 0.3%–1.0% or more. Backtest results for the SmallCap Arb strategy should be treated as an **upper bound** on real-world performance. When interpreting optimiser output, favour parameter sets with a comfortable margin over the VaR limit rather than the boundary-hugging optimum.

---

## Deployment (Docker)

The `Dockerfile` packages the Python bot only. IBKR Gateway must run separately (natively or via a dedicated container) and be reachable at the configured host/port.

```bash
# Build
docker build -t trading-bot .

# Run — Gateway on the same host, logs and data persisted outside the container
docker run -d \
  --name trading-bot \
  --restart unless-stopped \
  --network host \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  trading-bot

# Follow logs
docker logs -f trading-bot
```

`--restart unless-stopped` means Docker automatically relaunches the bot after the weekly Gateway restart (once Gateway is back up and accepting connections).

---

## Logging

`main.py` writes to both stdout and a rotating file log:

```
logs/trading_bot.log        ← current file  (up to 10 MB)
logs/trading_bot.log.1      ← previous
...
logs/trading_bot.log.7      ← oldest kept
```

Each strategy emits structured `INFO` lines for every bar evaluation, order placement, fill, and safety-switch state change — making post-session debugging straightforward without needing to query the database.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ib_insync` | IBKR Gateway / TWS API client |
| `pandas` / `numpy` | Data manipulation and indicators |
| `pyarrow` | Parquet cache read/write |
| `sqlalchemy` | Database ORM (SQLite / PostgreSQL) |
| `optuna` | Bayesian hyperparameter optimisation |
| `python-dotenv` | `.env` file loading |
| `plotly` *(optional)* | Interactive HTML charts for optimisation results |
