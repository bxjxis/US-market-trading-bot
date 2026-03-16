"""
scripts/backtest_earnings.py
----------------------------
3-layer institutional earnings momentum strategy — A/B comparison backtest.

Signal architecture
-------------------
  Layer 1 (Trend)     : close > MA(200)
  Layer 2 (Catalyst)  : earnings day AND RVOL > 2.5 AND gap_up > 1.5 * ATR(14)
  Layer 3 (Patience)  : D+1..D+3 closes all above earnings-day midpoint ((H+L)/2)
                        Entry at close of D+3

Exit rules (dual stop)
----------------------
  Hard stop : close falls below earnings-day midpoint at any point during hold
  Time exit : sell at open the day after max_hold_days (default: 20)

A/B comparison
--------------
  Treatment : all 3 layers (earnings filter active)
  Control   : Layer 1 + Layer 2 volume/gap (no earnings calendar — any gap day)

Usage
-----
    # Quick test — 2 symbols, 1 year
    python scripts/backtest_earnings.py --symbols AAPL NVDA --start-date 2023-01-01

    # Full NDX100 subset
    python scripts/backtest_earnings.py

    # Without MA200 filter (pure momentum)
    python scripts/backtest_earnings.py --no-ma-filter
"""

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dashboard_stats import (
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# Suppress yfinance chatter
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

_DEFAULT_SYMBOLS = [
    "SMCI", "CRWD", "MELI", "COIN", "DKNG",
    "RBLX", "TTD",  "DUOL", "AFRM", "UPST",
    "SOFI", "CELH", "DDOG", "NET",  "HOOD",
]
_SLIPPAGE = 0.001   # 0.1% — conservative fill assumption


# ---- Data structures ---------------------------------------------------------

@dataclass
class Signal:
    symbol:         str
    earnings_date:  date
    earnings_mid:   float   # (high + low) / 2 on earnings day
    entry_date:     date
    entry_price:    float   # close of entry day

@dataclass
class Trade:
    symbol:       str
    entry_date:   date
    entry_price:  float
    stop_price:   float    # earnings midpoint
    exit_date:    date
    exit_price:   float
    pnl_pct:      float    # (exit - entry) / entry
    n_days_held:  int
    exit_reason:  str      # "stop" | "time"


# ---- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Earnings momentum backtest with A/B comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols",       nargs="+", default=_DEFAULT_SYMBOLS,
                   help="Ticker symbols to include in the universe.")
    p.add_argument("--start-date",    default="2022-01-01", dest="start_date",
                   help="Backtest start date (YYYY-MM-DD).")
    p.add_argument("--end-date",      default=str(date.today()), dest="end_date",
                   help="Backtest end date (YYYY-MM-DD).")
    p.add_argument("--rvol-thr",      type=float, default=2.5, dest="rvol_thr",
                   help="Min relative volume on signal day.")
    p.add_argument("--gap-atr-mult",  type=float, default=1.5, dest="gap_atr_mult",
                   help="Min gap (open - prev_close) as multiple of ATR(14).")
    p.add_argument("--wait-days",     type=int,   default=3,   dest="wait_days",
                   help="Trading days to observe post-earnings before entry.")
    p.add_argument("--max-hold",      type=int,   default=20,  dest="max_hold",
                   help="Maximum trading days to hold a position.")
    p.add_argument("--account-size",  type=float, default=10_000.0, dest="account_size",
                   help="USD allocated per position (equal-weight).")
    p.add_argument("--no-ma-filter",  action="store_true", dest="no_ma_filter",
                   help="Disable the MA(200) Layer 1 trend filter.")
    p.add_argument("--bb-filter",     action="store_true", dest="bb_filter",
                   help="Require Bollinger Bands to be expanding on signal day.")
    return p.parse_args()


# ---- Data fetching -----------------------------------------------------------

def fetch_daily_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download daily OHLCV from yfinance. Returns None on failure."""
    try:
        import yfinance as yf
        # Add buffer before start for indicator warmup (200+ trading days)
        start_dt = pd.Timestamp(start) - pd.Timedelta(days=320)
        df = yf.download(
            symbol, start=start_dt.strftime("%Y-%m-%d"), end=end,
            interval="1d", progress=False, auto_adjust=True,
        )
        if df.empty:
            return None
        # yfinance returns MultiIndex columns when multiple symbols — flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date", "adj close": "close"})
        # Ensure standard columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                return None
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        _log.warning("fetch_daily_data(%s): %s", symbol, e)
        return None


def fetch_earnings_dates(symbol: str) -> Set[date]:
    """Return set of historical earnings dates for a symbol using yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        ed = ticker.earnings_dates
        if ed is None or (hasattr(ed, "empty") and ed.empty):
            return set()
        return set(pd.DatetimeIndex(ed.index).normalize().date)
    except Exception as e:
        _log.warning("fetch_earnings_dates(%s): %s", symbol, e)
        return set()


# ---- Indicators --------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA200, ATR14, avg_vol_20d, RVOL, BB_width columns to df."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # MA(200)
    df["ma200"] = close.rolling(200, min_periods=200).mean()

    # Wilder ATR(14): True Range -> EWM with alpha = 1/14
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

    # 20-day average volume (shift 1 to avoid look-ahead bias)
    df["avg_vol_20d"] = df["volume"].rolling(20, min_periods=20).mean().shift(1)
    df["rvol"]        = df["volume"] / df["avg_vol_20d"]

    # Bollinger Band width (20, 2 std) — for --bb-filter
    sma20  = close.rolling(20, min_periods=20).mean()
    std20  = close.rolling(20, min_periods=20).std(ddof=0)
    df["bb_width"]    = (4 * std20) / sma20   # (upper - lower) / mid = 4*std/sma

    return df


# ---- Signal generation -------------------------------------------------------

def generate_signals(
    df: pd.DataFrame,
    symbol: str,
    earnings_dates: Set[date],
    args: argparse.Namespace,
    use_earnings_filter: bool = True,
) -> List[Signal]:
    """
    Scan df for entry signals.

    use_earnings_filter=True  → Treatment group (earnings catalyst required)
    use_earnings_filter=False → Control group (any qualifying gap-up day)
    """
    signals: List[Signal] = []
    n = len(df)
    dates_arr  = df["date"].dt.date.values
    close_arr  = df["close"].values
    open_arr   = df["open"].values
    high_arr   = df["high"].values
    low_arr    = df["low"].values
    ma200_arr  = df["ma200"].values
    atr_arr    = df["atr14"].values
    rvol_arr   = df["rvol"].values
    bb_arr     = df["bb_width"].values

    # Min start index: need 200 bars for MA200 + wait_days + max_hold lookahead
    min_i = 200
    max_i = n - args.wait_days - 1   # leave room for wait-days + at least 1 day of holding

    # Track occupied dates per symbol to avoid duplicate entries
    occupied_entry_dates: Set[date] = set()

    for i in range(min_i, max_i):
        d = dates_arr[i]

        # Earnings filter — yfinance stores AMC report date; gap appears next day.
        # Check both same-day (BMO reports) and previous day (AMC reports).
        if use_earnings_filter:
            prev_d = dates_arr[i - 1] if i > 0 else None
            if d not in earnings_dates and prev_d not in earnings_dates:
                continue

        # Layer 1: trend
        if not args.no_ma_filter:
            ma = ma200_arr[i]
            if math.isnan(ma) or close_arr[i] < ma:
                continue

        # Layer 2: RVOL
        rv = rvol_arr[i]
        if math.isnan(rv) or rv < args.rvol_thr:
            continue

        # Layer 2: gap up in dollar terms vs ATR
        atr = atr_arr[i]
        if math.isnan(atr) or atr <= 0:
            continue
        gap = open_arr[i] - close_arr[i - 1]
        if gap < args.gap_atr_mult * atr:
            continue

        # Layer 2 optional: BB expanding
        if args.bb_filter:
            bb_w = bb_arr[i]
            # Expanding = today's width > yesterday's width
            bb_prev = bb_arr[i - 1]
            if math.isnan(bb_w) or math.isnan(bb_prev) or bb_w <= bb_prev:
                continue

        # Layer 3: wait_days observation window
        earnings_mid = (high_arr[i] + low_arr[i]) / 2.0
        valid = True
        for j in range(1, args.wait_days + 1):
            if i + j >= n:
                valid = False
                break
            if close_arr[i + j] < earnings_mid:
                valid = False
                break
        if not valid:
            continue

        # Entry on close of day D + wait_days
        entry_idx   = i + args.wait_days
        entry_date  = dates_arr[entry_idx]
        entry_price = close_arr[entry_idx]

        if entry_date in occupied_entry_dates:
            continue   # skip if already entered on this day for this symbol

        occupied_entry_dates.add(entry_date)
        signals.append(Signal(
            symbol=symbol,
            earnings_date=d,
            earnings_mid=earnings_mid,
            entry_date=entry_date,
            entry_price=entry_price,
        ))

    return signals


# ---- Trade simulation --------------------------------------------------------

def simulate_trades(
    df: pd.DataFrame,
    signals: List[Signal],
    max_hold: int,
) -> List[Trade]:
    """Simulate exits for each signal using daily OHLCV data."""
    trades: List[Trade] = []
    n = len(df)

    # Build fast lookup: date -> row index
    date_to_idx: Dict[date, int] = {
        row["date"].date(): idx for idx, row in df.iterrows()
    }

    for sig in signals:
        entry_idx = date_to_idx.get(sig.entry_date)
        if entry_idx is None:
            continue

        entry_price = sig.entry_price * (1 + _SLIPPAGE)   # fill slippage
        stop_price  = sig.earnings_mid
        exit_date   = sig.entry_date
        exit_price  = entry_price
        exit_reason = "time"
        n_days      = 0

        for k in range(1, max_hold + 2):
            hold_idx = entry_idx + k
            if hold_idx >= n:
                # End of data — exit at last known close
                exit_price  = df["close"].iloc[-1] * (1 - _SLIPPAGE)
                exit_date   = df["date"].iloc[-1].date()
                n_days      = k
                exit_reason = "time"
                break

            row_close = df["close"].iloc[hold_idx]
            row_date  = df["date"].iloc[hold_idx].date()

            # Hard stop: close below earnings midpoint
            if row_close < stop_price:
                # Exit at next open (approximated as current open with slippage)
                exit_open = df["open"].iloc[hold_idx]
                exit_price  = min(exit_open, stop_price) * (1 - _SLIPPAGE)
                exit_date   = row_date
                n_days      = k
                exit_reason = "stop"
                break

            # Time exit after max_hold days
            if k == max_hold:
                exit_open  = df["open"].iloc[hold_idx]
                exit_price = exit_open * (1 - _SLIPPAGE)
                exit_date  = row_date
                n_days     = k
                exit_reason = "time"
                break

        pnl_pct = (exit_price - entry_price) / entry_price

        trades.append(Trade(
            symbol=sig.symbol,
            entry_date=sig.entry_date,
            entry_price=entry_price,
            stop_price=stop_price,
            exit_date=exit_date,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            n_days_held=n_days,
            exit_reason=exit_reason,
        ))

    return trades


# ---- Performance metrics -----------------------------------------------------

def build_equity_curve(trades: List[Trade], account_size: float) -> pd.Series:
    """
    Build a daily equity curve using equal-weight position sizing.
    Each trade uses account_size dollars; P&L is absolute.
    Positions may overlap; curve is cumulative sum of P&L indexed by exit date.
    """
    if not trades:
        return pd.Series(dtype=float)
    records = [(t.exit_date, t.pnl_pct * account_size) for t in trades]
    df = pd.DataFrame(records, columns=["date", "pnl"])
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date")["pnl"].sum()
    daily = daily.sort_index()
    # Base equity + cumulative P&L
    equity = account_size + daily.cumsum()
    return equity


def compute_performance(trades: List[Trade], account_size: float) -> dict:
    """Return performance metrics dict."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
            "sharpe": float("nan"), "sortino": float("nan"),
            "max_dd": 0.0, "avg_hold_days": 0.0,
            "stop_rate": 0.0, "total_pnl_pct": 0.0,
        }

    pnl_pcts    = [t.pnl_pct for t in trades]
    wins        = sum(1 for p in pnl_pcts if p > 0)
    stops       = sum(1 for t in trades if t.exit_reason == "stop")
    avg_hold    = float(np.mean([t.n_days_held for t in trades]))
    total_pnl   = float(np.sum(pnl_pcts))

    # Build equity for drawdown / Sharpe
    equity = build_equity_curve(trades, account_size)
    if len(equity) >= 2:
        dr      = equity.pct_change().dropna()
        sharpe  = compute_sharpe(dr)
        sortino = compute_sortino(dr)
        mdd     = compute_max_drawdown(equity) * 100
    else:
        sharpe = sortino = float("nan")
        mdd = 0.0

    return {
        "n_trades":      len(trades),
        "win_rate":      wins / len(trades) * 100,
        "avg_pnl_pct":   float(np.mean(pnl_pcts)) * 100,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_dd":        mdd,
        "avg_hold_days": avg_hold,
        "stop_rate":     stops / len(trades) * 100,
        "total_pnl_pct": total_pnl * 100,
    }


# ---- Reporting ---------------------------------------------------------------

def _fmt(v, spec=".2f", nan_str="-") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return nan_str
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return format(v, spec)


def print_ab_report(
    treatment: dict,
    control:   dict,
    args:      argparse.Namespace,
    n_symbols: int,
) -> None:
    W1, W2 = 30, 22

    print("\nEarnings Momentum Strategy -- A/B Comparison")
    print(f"  Universe: {' '.join(args.symbols)} ({n_symbols} symbols)")
    print(
        f"  Period  : {args.start_date} -> {args.end_date}  |  "
        f"RVOL thr: {args.rvol_thr}x  |  "
        f"Gap mult: {args.gap_atr_mult}x ATR  |  "
        f"Wait: {args.wait_days}d  |  Max hold: {args.max_hold}d"
    )
    ma_note = " (MA200 disabled)" if args.no_ma_filter else ""
    bb_note = " +BB-filter" if args.bb_filter else ""
    print(f"  Filters : Layer1{ma_note}  Layer2  Layer3{bb_note}")
    print("=" * 62)

    row = lambda label, t_val, c_val: print(
        f"  {label:<{W1}}: {t_val:>{W2}}   {c_val:>{W2}}"
    )
    header = f"  {'Metric':<{W1}}  {'Treatment (w/ Earnings)':>{W2}}   {'Control (w/o Earnings)':>{W2}}"
    print(header)
    print("-" * 62)

    row("Trades",             str(treatment["n_trades"]),    str(control["n_trades"]))
    row("Win Rate",           f"{treatment['win_rate']:.1f}%",   f"{control['win_rate']:.1f}%")
    row("Avg P&L per Trade",  f"{treatment['avg_pnl_pct']:+.2f}%", f"{control['avg_pnl_pct']:+.2f}%")
    row("Total P&L",          f"{treatment['total_pnl_pct']:+.1f}%", f"{control['total_pnl_pct']:+.1f}%")
    row("Sharpe Ratio",       _fmt(treatment["sharpe"]),     _fmt(control["sharpe"]))
    row("Sortino Ratio",      _fmt(treatment["sortino"]),    _fmt(control["sortino"]))
    row("Max Drawdown",       f"{treatment['max_dd']:.1f}%", f"{control['max_dd']:.1f}%")
    row("Avg Hold Days",      f"{treatment['avg_hold_days']:.1f}",   f"{control['avg_hold_days']:.1f}")
    row("Stop-out Rate",      f"{treatment['stop_rate']:.1f}%",  f"{control['stop_rate']:.1f}%")

    print("=" * 62)

    # Earnings filter impact
    t_s = treatment["sharpe"]
    c_s = control["sharpe"]
    t_w = treatment["win_rate"]
    c_w = control["win_rate"]
    t_p = treatment["avg_pnl_pct"]
    c_p = control["avg_pnl_pct"]

    if not (math.isnan(t_s) or math.isnan(c_s)):
        d_sharpe = t_s - c_s
        d_win    = t_w - c_w
        d_pnl    = t_p - c_p
        print(
            f"  Earnings filter impact: "
            f"Sharpe {d_sharpe:+.2f}  |  "
            f"Win rate {d_win:+.1f}pp  |  "
            f"Avg P&L {d_pnl:+.2f}pp"
        )
    print()


def print_per_symbol_table(
    all_treatment: List[Trade],
    all_control:   List[Trade],
) -> None:
    """Per-symbol trade count and win rate."""
    symbols = sorted(set(t.symbol for t in all_treatment + all_control))
    if not symbols:
        return
    print("Per-Symbol Summary (Treatment):")
    print(f"  {'Symbol':<8}  {'Trades':>6}  {'Win%':>6}  {'AvgP&L%':>8}  {'AvgHold':>8}")
    print("  " + "-" * 40)
    for sym in symbols:
        ts = [t for t in all_treatment if t.symbol == sym]
        if not ts:
            continue
        wins = sum(1 for t in ts if t.pnl_pct > 0)
        avg_p = float(np.mean([t.pnl_pct for t in ts])) * 100
        avg_h = float(np.mean([t.n_days_held for t in ts]))
        print(f"  {sym:<8}  {len(ts):>6}  {wins/len(ts)*100:>5.1f}%  {avg_p:>+7.2f}%  {avg_h:>7.1f}d")
    print()


# ---- Main --------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    all_treatment: List[Trade] = []
    all_control:   List[Trade] = []
    symbols_ok: List[str] = []

    print(f"\nFetching data for {len(args.symbols)} symbols ...")

    for sym in args.symbols:
        print(f"  {sym} ...", end="", flush=True)

        df = fetch_daily_data(sym, args.start_date, args.end_date)
        if df is None or len(df) < 220:
            print(f" skipped (insufficient data)")
            continue

        df = compute_indicators(df)

        # Filter to backtest date range (after indicator warmup)
        start_dt = pd.Timestamp(args.start_date)
        df_bt = df[df["date"] >= start_dt].reset_index(drop=True)
        if len(df_bt) < args.wait_days + args.max_hold + 5:
            print(f" skipped (too few bars in range)")
            continue

        earnings = fetch_earnings_dates(sym)
        n_earn   = len([d for d in earnings if pd.Timestamp(args.start_date).date() <= d <= pd.Timestamp(args.end_date).date()])
        print(f" {len(df_bt)} bars, {n_earn} earnings dates in range")

        # Use full df (with pre-range warmup) for indicator accuracy, but only
        # generate signals within the requested date range.
        t_sigs = generate_signals(df, sym, earnings, args, use_earnings_filter=True)
        c_sigs = generate_signals(df, sym, earnings, args, use_earnings_filter=False)

        # Filter signals to backtest range
        t_sigs = [s for s in t_sigs if s.entry_date >= pd.Timestamp(args.start_date).date()]
        c_sigs = [s for s in c_sigs if s.entry_date >= pd.Timestamp(args.start_date).date()]

        t_trades = simulate_trades(df, t_sigs, args.max_hold)
        c_trades = simulate_trades(df, c_sigs, args.max_hold)

        all_treatment.extend(t_trades)
        all_control.extend(c_trades)
        symbols_ok.append(sym)

    if not symbols_ok:
        print("\nNo symbols had sufficient data. Exiting.")
        sys.exit(1)

    treatment_perf = compute_performance(all_treatment, args.account_size)
    control_perf   = compute_performance(all_control,   args.account_size)

    print_ab_report(treatment_perf, control_perf, args, len(symbols_ok))
    print_per_symbol_table(all_treatment, all_control)


if __name__ == "__main__":
    main()
