"""
scripts/grid_screener.py
------------------------
Screen US equities for grid trading suitability using IBKR's server-side
scanner and 20-day 5-minute historical data.

Workflow
--------
1. Connect to IBKR Gateway / TWS
2. Run a server-side scanner to discover low-price, high-volume US stocks
3. Fetch 20-day 5-min OHLCV via DataFetcher (pacing-compliant, parquet-cached)
4. Score each symbol on range, ATR, volume, and commission efficiency
5. Filter on ADX < 25, RSI 35-65, ATR% 0.5-4%, range 1.2-2.5×, Hurst < 0.55
6. Write ranked results to  data/grid_screener_YYYYMMDD.csv

Usage
-----
    python scripts/grid_screener.py
    python scripts/grid_screener.py --max-results 30 --max-price 30
    python scripts/grid_screener.py --scan-code MOST_ACTIVE --account-size 72000

Pre-requisites
--------------
* IBKR Gateway or TWS must be running and accepting API connections.
  Default: 127.0.0.1:7497 (paper trading port).
  Override with IB_HOST / IB_PORT / IB_CLIENT_ID env vars or a .env file.
"""

import argparse
import asyncio
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Make project root importable when run as a script ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import ScannerSubscription, TagValue, util as ib_util

from core.connection import IBConnection
from utils.data_fetcher import DataFetcher

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("grid_screener")
_log.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_BARS            = 200     # minimum bars for a symbol to be scored
SCANNER_WAIT_SECS   = 3.0    # seconds to wait for ScanDataList to populate
COMMISSION_PER_SIDE = 0.0035  # $0.0035 / share
MIN_COMMISSION      = 0.35   # minimum commission per side ($)
BARS_PER_DAY        = 78     # 9:30–16:00 at 5-min resolution


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicator helpers (pure pandas/numpy, no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Wilder's Average Directional Index (ADX).
    Returns the latest ADX value as a float, or nan if not computable.

    ADX < 20 : ranging market   — ideal for grid
    ADX < 25 : weak/no trend    — acceptable
    ADX > 30 : established trend — dangerous for grid
    """
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    # Wilder's smoothing: alpha = 1 / period
    alpha = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di   = 100.0 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_s
    minus_di  = 100.0 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_s

    di_sum  = plus_di + minus_di
    dx      = (plus_di - minus_di).abs() / di_sum.where(di_sum != 0) * 100.0
    adx     = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI using Wilder's exponential smoothing. Returns latest value [0,100]."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.where(avg_loss != 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def _compute_hurst(close: pd.Series, min_lag: int = 20, max_lag: int = 300) -> float:
    """
    Hurst exponent via Rescaled Range (R/S) analysis on 5-min close prices.

    H < 0.50 : mean-reverting  — ideal for grid trading
    H ≈ 0.50 : random walk     — neutral
    H > 0.55 : trending        — dangerous for grid (single-side fill risk)

    Uses log-spaced lags between min_lag and max_lag bars.
    Returns nan if there are fewer than 4 × min_lag bars.
    """
    prices = close.dropna().values
    n = len(prices)
    if n < min_lag * 4:
        return float("nan")

    actual_max = min(max_lag, n // 4)
    if actual_max <= min_lag:
        return float("nan")

    lags = np.unique(
        np.logspace(np.log10(min_lag), np.log10(actual_max), 20).astype(int)
    )
    rs_vals = []
    for lag in lags:
        chunks = [prices[i : i + lag] for i in range(0, n - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            r = np.diff(np.log(np.maximum(chunk, 1e-12)))
            if len(r) < 2:
                continue
            dev = np.cumsum(r - r.mean())
            rs  = (dev.max() - dev.min()) / (r.std() + 1e-12)
            rs_chunk.append(rs)
        if rs_chunk:
            rs_vals.append((lag, float(np.mean(rs_chunk))))

    if len(rs_vals) < 3:
        return float("nan")

    lags_a, rs_a = zip(*rs_vals)
    H = float(np.polyfit(np.log(lags_a), np.log(rs_a), 1)[0])
    return float(np.clip(H, 0.1, 0.9))


# ═════════════════════════════════════════════════════════════════════════════
# IBKR scanner
# ═════════════════════════════════════════════════════════════════════════════

async def run_scanner(
    ib,
    scan_code: str,
    min_price: float,
    max_price: float,
    max_results: int,
    min_market_cap_m: int = 1_000,
) -> list:
    """
    Run an IBKR server-side scanner subscription, wait for results, cancel it,
    and return a list of Contract objects.

    reqScannerSubscription() is synchronous — it returns a live ScanDataList
    immediately and populates it asynchronously via events.  We sleep briefly
    to let the event loop receive the incoming scan items.
    """
    sub = ScannerSubscription(
        instrument   = "STK",
        locationCode = "STK.US.MAJOR",
        scanCode     = scan_code,
        numberOfRows = max_results,
    )
    filter_options = [
        TagValue("priceAbove",        str(min_price)),
        TagValue("priceBelow",        str(max_price)),
        TagValue("volumeAbove",       "100000"),                   # min daily volume: 100k shares
        TagValue("marketCapAbove1e6", str(min_market_cap_m)),      # min market cap (millions)
    ]
    _log.info(
        "Scanner: code=%s  price=[%.1f, %.1f]  maxRows=%d",
        scan_code, min_price, max_price, max_results,
    )

    scan_data = ib.reqScannerSubscription(sub, scannerSubscriptionFilterOptions=filter_options)
    await asyncio.sleep(SCANNER_WAIT_SECS)
    ib.cancelScannerSubscription(scan_data)

    contracts = [sd.contractDetails.contract for sd in scan_data]
    _log.info("Scanner returned %d contracts.", len(contracts))
    return contracts


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute grid-screening metrics from a 5-minute OHLCV DataFrame.

    Returns None if the DataFrame has fewer than MIN_BARS rows or if the
    latest close is non-positive.

    Expected columns (from DataFetcher): date, open, high, low, close, volume
    """
    if len(df) < MIN_BARS:
        return None

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    price = float(close.iloc[-1])
    if price <= 0:
        return None

    # 20-day high / low (all available bars used as proxy)
    h20 = float(high.max())
    l20 = float(low.min())
    range_ratio = (h20 / l20) if l20 > 0 else float("nan")

    # ATR% — 14-bar Average True Range as % of latest close
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_last = tr.rolling(14).mean().iloc[-1]
    atr_pct  = float(atr_last / price * 100) if pd.notna(atr_last) else float("nan")

    # Average 5-min volume (shares) and dollar volume
    avg_5min_vol     = float(volume.mean())
    avg_5min_vol_usd = avg_5min_vol * price   # dollar-based; price-agnostic threshold

    # Last-day volume in USD (last 78 bars × close)
    daily_vol_usd = float(volume.tail(BARS_PER_DAY).sum() * price)

    # ADX(14) — trend strength; <25 = ranging, >30 = trending
    adx_14 = _compute_adx(high, low, close)

    # RSI(14) — current momentum; 35–65 = neutral zone
    rsi_current = _compute_rsi(close)

    # Hurst exponent — mean-reversion quality; <0.55 = suitable for grid
    hurst = _compute_hurst(close)

    return {
        "price":         price,
        "20d_high":      h20,
        "20d_low":       l20,
        "range_ratio":   range_ratio,
        "atr_pct":       atr_pct,
        "avg_5min_vol":      avg_5min_vol,
        "avg_5min_vol_usd":  avg_5min_vol_usd,
        "daily_vol_usd":     daily_vol_usd,
        "adx_14":        adx_14,
        "rsi_current":   rsi_current,
        "hurst":         hurst,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Grid parameter suggestions
# ═════════════════════════════════════════════════════════════════════════════

def suggest_grid_params(metrics: dict, account_size: float) -> dict:
    """
    Map ATR% → GRID_RATIO, derive NUM_BUY_LEVELS and commission breakeven qty.
    """
    atr_pct     = metrics["atr_pct"]
    price       = metrics["price"]
    range_ratio = metrics["range_ratio"]

    # GRID_RATIO from ATR%
    if atr_pct < 1.0:
        grid_ratio = 1.010
    elif atr_pct < 2.0:
        grid_ratio = 1.015
    elif atr_pct < 3.0:
        grid_ratio = 1.020
    else:
        grid_ratio = 1.025

    # NUM_BUY_LEVELS: spread the range across grid steps, clamped [6, 15]
    step = grid_ratio - 1.0
    raw_levels = math.floor(range_ratio / step * 0.5) if step > 0 else 6
    num_levels = max(6, min(15, raw_levels))

    # Commission breakeven quantity
    # Gross profit per round-trip = price × (grid_ratio - 1) × qty
    # Round-trip flat-floor commission = 2 × $0.35 = $0.70
    # Breakeven qty = ceil(0.70 / (price × step))
    gross_per_share = price * step
    if gross_per_share > 0:
        comm_be_qty = math.ceil(2.0 * MIN_COMMISSION / gross_per_share)
    else:
        comm_be_qty = 9_999  # degenerate — mark as unviable

    return {
        "grid_ratio_suggested":    grid_ratio,
        "num_levels_suggested":    num_levels,
        "commission_breakeven_qty": comm_be_qty,
        "_account_size":           account_size,  # internal, not written to CSV
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════════

def score_candidate(metrics: dict, params: dict, hurst_max: float = 0.55) -> Tuple[float, bool]:
    """
    Return (grid_score 0-100, passes_filter bool).

    Score breakdown:
        30 pts  range_ratio  — tent peak at 1.8×
        30 pts  ATR%         — tent peak at 2.0%
        20 pts  avg 5-min vol — log-scale, saturates at 50 000 shares
        20 pts  commission efficiency — gross/comm ratio, saturates at 5×
    """
    range_ratio      = metrics["range_ratio"]
    atr_pct          = metrics["atr_pct"]
    avg_5min_vol     = metrics["avg_5min_vol"]
    avg_5min_vol_usd = metrics["avg_5min_vol_usd"]
    adx_14           = metrics["adx_14"]
    rsi_current      = metrics["rsi_current"]
    hurst            = metrics["hurst"]
    price            = metrics["price"]
    grid_ratio   = params["grid_ratio_suggested"]
    num_levels   = params["num_levels_suggested"]
    be_qty       = params["commission_breakeven_qty"]
    account_size = params["_account_size"]

    # Guard: NaN or non-positive inputs
    if not (math.isfinite(range_ratio) and math.isfinite(atr_pct)
            and range_ratio > 0 and atr_pct >= 0):
        return 0.0, False

    # ── 1. Range ratio score (30 pts, tent peak at 1.8×) ─────────────────────
    RANGE_LOW, RANGE_PEAK, RANGE_HIGH = 1.0, 1.8, 3.5
    if range_ratio <= RANGE_LOW:
        range_score = 0.0
    elif range_ratio <= RANGE_PEAK:
        range_score = 30.0 * (range_ratio - RANGE_LOW) / (RANGE_PEAK - RANGE_LOW)
    elif range_ratio <= RANGE_HIGH:
        range_score = 30.0 * (RANGE_HIGH - range_ratio) / (RANGE_HIGH - RANGE_PEAK)
    else:
        range_score = 0.0

    # ── 2. ATR% score (30 pts, tent peak at 2.0%) ────────────────────────────
    ATR_LOW, ATR_PEAK, ATR_HIGH = 0.0, 2.0, 5.0
    if atr_pct <= ATR_LOW:
        atr_score = 0.0
    elif atr_pct <= ATR_PEAK:
        atr_score = 30.0 * (atr_pct - ATR_LOW) / (ATR_PEAK - ATR_LOW)
    elif atr_pct <= ATR_HIGH:
        atr_score = 30.0 * (ATR_HIGH - atr_pct) / (ATR_HIGH - ATR_PEAK)
    else:
        atr_score = 0.0

    # ── 3. Volume score (20 pts, log-scale on $ volume, saturates at $500k/bar)
    # Using dollar volume removes bias against high-price stocks.
    VOL_SAT_USD = 500_000.0
    if avg_5min_vol_usd <= 0:
        vol_score = 0.0
    else:
        raw = 20.0 * math.log10(max(1.0, avg_5min_vol_usd)) / math.log10(VOL_SAT_USD)
        vol_score = min(20.0, max(0.0, raw))

    # ── 4. Commission efficiency score (20 pts, saturates at 5×) ─────────────
    alloc_per_level = account_size / num_levels if num_levels > 0 else 0.0
    qty = math.floor(alloc_per_level / price) if price > 0 else 0
    if qty < 1:
        comm_score = 0.0
    else:
        gross      = price * (grid_ratio - 1.0) * qty
        commission = 2.0 * max(MIN_COMMISSION, qty * COMMISSION_PER_SIDE)
        efficiency = gross / commission if commission > 0 else 0.0
        comm_score = min(20.0, 20.0 * efficiency / 5.0)

    grid_score = range_score + atr_score + vol_score + comm_score

    # ── Filter criteria ───────────────────────────────────────────────────────
    adx_ok   = math.isfinite(adx_14) and adx_14 < 25.0
    rsi_ok   = math.isfinite(rsi_current) and 35.0 <= rsi_current <= 65.0
    hurst_ok = math.isfinite(hurst) and hurst < hurst_max
    passes = (
        1.2 <= range_ratio <= 2.5
        and 0.5 <= atr_pct <= 4.0
        and avg_5min_vol_usd >= 50_000   # $50k/bar ≈ min viable liquidity regardless of price
        and qty > 0
        and be_qty <= qty
        and adx_ok
        and rsi_ok
        and hurst_ok
    )

    return round(grid_score, 2), passes


# ═════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_output_row(
    symbol: str,
    metrics: dict,
    params: dict,
    grid_score: float,
    passes: bool,
) -> dict:
    return {
        "symbol":                    symbol,
        "price":                     round(metrics["price"], 4),
        "20d_high":                  round(metrics["20d_high"], 4),
        "20d_low":                   round(metrics["20d_low"], 4),
        "range_ratio":               round(metrics["range_ratio"], 4),
        "atr_pct":                   round(metrics["atr_pct"], 4),
        "avg_5min_vol":              round(metrics["avg_5min_vol"], 1),
        "avg_5min_vol_usd":          round(metrics["avg_5min_vol_usd"], 0),
        "daily_vol_usd":             round(metrics["daily_vol_usd"], 0),
        "grid_ratio_suggested":      params["grid_ratio_suggested"],
        "num_levels_suggested":      params["num_levels_suggested"],
        "commission_breakeven_qty":  params["commission_breakeven_qty"],
        "adx_14":                    round(metrics["adx_14"], 2),
        "rsi_current":               round(metrics["rsi_current"], 2),
        "hurst":                     round(metrics["hurst"], 3) if math.isfinite(metrics["hurst"]) else float("nan"),
        "grid_score":                grid_score,
        "passes_filter":             passes,
    }


def save_results(rows: List[dict], output_path: Optional[str]) -> Path:
    if output_path is None:
        date_tag = datetime.today().strftime("%Y%m%d")
        out = Path("data") / f"grid_screener_{date_tag}.csv"
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("grid_score", ascending=False)
    df.to_csv(out, index=False)
    return out


def print_summary(rows: List[dict], csv_path: Path) -> None:
    total   = len(rows)
    passing = sum(1 for r in rows if r["passes_filter"])
    top5    = sorted(rows, key=lambda r: r["grid_score"], reverse=True)[:5]

    print(f"\nScanned: {total}  |  Passing filter: {passing}")
    print(f"\n{'':2}{'Symbol':<8} {'Score':>7} {'Price':>7} {'Range':>7} "
          f"{'ATR%':>6} {'ADX':>6} {'RSI':>6} {'Hurst':>7} {'GRatio':>8} {'Levels':>7}")
    print(f"{'':2}{'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7}")
    for r in top5:
        mark = "* " if r["passes_filter"] else "  "
        hurst_str = f"{r['hurst']:>7.3f}" if math.isfinite(r.get("hurst", float("nan"))) else f"{'n/a':>7}"
        print(
            f"{mark}{r['symbol']:<8} {r['grid_score']:>7.1f} "
            f"{r['price']:>7.2f} {r['range_ratio']:>7.3f} "
            f"{r['atr_pct']:>6.2f} {r['adx_14']:>6.1f} {r['rsi_current']:>6.1f} "
            f"{hurst_str} "
            f"{r['grid_ratio_suggested']:>8.3f} {r['num_levels_suggested']:>7}"
        )
    print(f"\n* = passes all filter criteria (range 1.2-2.5, ATR 0.5-4%, ADX<25, RSI 35-65, Hurst<0.55, vol>$50k/bar)")
    print(f"\nResults saved to: {csv_path.resolve()}")


# ═════════════════════════════════════════════════════════════════════════════
# Async body
# ═════════════════════════════════════════════════════════════════════════════

async def _screen(ib, args: argparse.Namespace) -> None:
    account = ib.managedAccounts()[0]
    _log.info("Connected | account=%s", account)

    # ── 1. Scanner ────────────────────────────────────────────────────────────
    raw_contracts = await run_scanner(
        ib,
        scan_code        = args.scan_code,
        min_price        = args.min_price,
        max_price        = args.max_price,
        max_results      = args.max_results,
        min_market_cap_m = args.min_market_cap,
    )

    if not raw_contracts:
        print("Scanner returned no results. Check scan code and price range.")
        return

    # ── 2. Qualify ────────────────────────────────────────────────────────────
    print(f"Qualifying {len(raw_contracts)} contracts …")
    qualified = await ib.qualifyContractsAsync(*raw_contracts)
    _log.info("Qualified %d / %d contracts.", len(qualified), len(raw_contracts))

    if not qualified:
        print("No contracts could be qualified. Exiting.")
        return

    # ── 3. Fetch historical data ──────────────────────────────────────────────
    print(f"Fetching {args.duration} {args.bar_size} bars for {len(qualified)} symbols …")
    fetcher  = DataFetcher(ib)
    data_map: Dict[str, pd.DataFrame] = await fetcher.fetch_multiple_stocks(
        contracts    = qualified,
        duration     = args.duration,
        bar_size     = args.bar_size,
        use_cache    = True,
        pacing_delay = 2.0,
    )

    # ── 4. Score ──────────────────────────────────────────────────────────────
    rows = []
    for contract in qualified:
        symbol = contract.symbol
        df     = data_map.get(symbol)

        if df is None:
            _log.warning("%s: no data returned — skipped.", symbol)
            continue

        metrics = compute_metrics(df)
        if metrics is None:
            _log.warning(
                "%s: only %d bars (<%d required) — skipped.",
                symbol, len(df), MIN_BARS,
            )
            continue

        params = suggest_grid_params(metrics, args.account_size)
        grid_score, passes = score_candidate(metrics, params, args.hurst_max)
        rows.append(build_output_row(symbol, metrics, params, grid_score, passes))

    # ── 5. Output ─────────────────────────────────────────────────────────────
    if not rows:
        print("No candidates survived the minimum-bar filter.")
        return

    csv_path = save_results(rows, args.output)
    print_summary(rows, csv_path)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen US stocks for grid trading suitability via IBKR scanner."
    )
    parser.add_argument(
        "--scan-code", default="MOST_ACTIVE", dest="scan_code",
        help=(
            "IBKR scanner code (default: MOST_ACTIVE). "
            "Other useful codes: HOT_BY_PRICE_RANGE, HOT_BY_VOLUME, "
            "TOP_PERC_GAIN, TOP_PRICE_RANGE, MOST_ACTIVE_USD."
        ),
    )
    parser.add_argument(
        "--min-price", type=float, default=5.0, dest="min_price",
        help="Minimum stock price in USD (default: 5.0).",
    )
    parser.add_argument(
        "--max-price", type=float, default=50.0, dest="max_price",
        help="Maximum stock price in USD (default: 50.0).",
    )
    parser.add_argument(
        "--max-results", type=int, default=50, dest="max_results",
        help="Maximum candidates requested from the scanner (default: 50).",
    )
    parser.add_argument(
        "--account-size", type=float, default=36_000.0, dest="account_size",
        help="Capital allocated per grid position, used for parameter suggestion (default: 36000).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output CSV path (default: data/grid_screener_YYYYMMDD.csv).",
    )
    parser.add_argument(
        "--duration", default="20 D",
        help="Historical data duration string (default: '20 D').",
    )
    parser.add_argument(
        "--bar-size", default="5 mins", dest="bar_size",
        help="Historical bar size string (default: '5 mins').",
    )
    parser.add_argument(
        "--min-market-cap", type=int, default=1_000, dest="min_market_cap",
        metavar="MILLIONS",
        help="Minimum market cap in millions USD for IBKR scanner pre-filter "
             "(default: 1000 = $1B).  Use 10000 for $10B (Gemini recommendation).",
    )
    parser.add_argument(
        "--hurst-max", type=float, default=0.55, dest="hurst_max",
        help="Maximum Hurst exponent to pass filter (default: 0.55). "
             "H<0.5=mean-reverting, H=0.5=random walk, H>0.5=trending.",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point  (mirrors download_cache.py exactly)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    print("Connecting to IBKR Gateway …")
    conn = IBConnection()
    ib   = conn.connect()

    try:
        ib_util.run(_screen(ib, args))
    finally:
        conn.disconnect()
        _log.info("Disconnected.")
