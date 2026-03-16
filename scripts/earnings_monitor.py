"""
scripts/earnings_monitor.py
----------------------------
Real-time earnings signal monitor.

Checks each symbol and reports which stage of the 3-layer signal it is in:

  PRE-EARNINGS  : MA200 passing + earnings due in next --days-ahead days
  WATCHING      : Earnings reported, Layer 2 qualified, waiting for Layer 3
                  (D+1 .. D+wait_days must all close above earnings midpoint)
  ENTRY SIGNAL  : All 3 layers satisfied -> enter at today's close  ***
  IN TRADE      : Past entry date, position open, monitoring stop
  CLOSED        : Stop hit or time exit already reached
  BELOW-MA200   : Trend filter failing (Layer 1)
  OK            : No active signal

Usage
-----
    python scripts/earnings_monitor.py
    python scripts/earnings_monitor.py --symbols SMCI CRWD COIN DKNG
    python scripts/earnings_monitor.py --rvol-thr 1.3 --gap-atr-mult 0.8 --no-ma-filter
"""

import argparse
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_earnings import (
    _DEFAULT_SYMBOLS,
    compute_indicators,
    fetch_daily_data,
    fetch_earnings_dates,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

_STATUS_RANK = {
    "ENTRY SIGNAL": 0,
    "WATCHING":     1,
    "IN TRADE":     2,
    "PRE-EARNINGS": 3,
    "CLOSED":       4,
    "BELOW-MA200":  5,
    "OK":           6,
    "SKIPPED":      7,
}


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check which symbols are generating earnings signals today.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols",      nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--rvol-thr",     type=float, default=1.5,  dest="rvol_thr",
                   help="Min RVOL on signal day.")
    p.add_argument("--gap-atr-mult", type=float, default=1.0,  dest="gap_atr_mult",
                   help="Min gap / ATR(14) on signal day.")
    p.add_argument("--wait-days",    type=int,   default=3,    dest="wait_days",
                   help="Observation days post-earnings before entry.")
    p.add_argument("--max-hold",     type=int,   default=20,   dest="max_hold",
                   help="Max holding days.")
    p.add_argument("--days-ahead",   type=int,   default=7,    dest="days_ahead",
                   help="Calendar days ahead to flag upcoming earnings.")
    p.add_argument("--no-ma-filter", action="store_true", dest="no_ma_filter")
    return p.parse_args()


# ---------------------------------------------------------------------------

def _check_symbol(sym: str, args, today: date):
    """
    Returns (status, detail) string pair.
    """
    # Download ~5y to guarantee 200-bar warmup
    df = fetch_daily_data(sym, "2020-01-01", str(today))
    if df is None or len(df) < 220:
        return "SKIPPED", "insufficient data"

    df       = compute_indicators(df)
    earnings = fetch_earnings_dates(sym)
    n        = len(df)

    last     = df.iloc[-1]
    close    = float(last["close"])
    ma200    = float(last["ma200"])
    last_dt  = last["date"].date()

    # --- Layer 1 ---
    if not args.no_ma_filter:
        if math.isnan(ma200) or close < ma200:
            return "BELOW-MA200", f"close={close:.2f}  MA200={ma200:.2f}"

    # --- Scan recent bars for a qualifying Layer-2 event ---
    # Look back enough to cover: wait_days + max_hold days worth of bars
    lookback = args.wait_days + args.max_hold + 3
    scan_start = max(1, n - lookback)

    for earn_i in range(n - 1, scan_start - 1, -1):
        d_earn  = df["date"].iloc[earn_i].date()
        d_prev  = df["date"].iloc[earn_i - 1].date() if earn_i > 0 else None

        # Check if this bar is an earnings gap day (same or next day after report)
        is_earn_day = (d_earn in earnings) or (d_prev is not None and d_prev in earnings)
        if not is_earn_day:
            continue

        # Layer 2: RVOL
        rvol = float(df["rvol"].iloc[earn_i])
        if math.isnan(rvol) or rvol < args.rvol_thr:
            continue

        # Layer 2: gap up
        atr = float(df["atr14"].iloc[earn_i])
        if math.isnan(atr) or atr <= 0:
            continue
        gap = float(df["open"].iloc[earn_i]) - float(df["close"].iloc[earn_i - 1])
        if gap < args.gap_atr_mult * atr:
            continue

        # Layer 2 passed — compute midpoint
        h_earn    = float(df["high"].iloc[earn_i])
        l_earn    = float(df["low"].iloc[earn_i])
        mid_price = (h_earn + l_earn) / 2.0

        days_since = n - 1 - earn_i   # bars elapsed after earn_i

        if days_since < args.wait_days:
            # Still in observation window — check holds so far
            broken = any(
                float(df["close"].iloc[earn_i + j]) < mid_price
                for j in range(1, days_since + 1)
            )
            if broken:
                continue  # midpoint violated — this event voided

            days_left = args.wait_days - days_since
            detail = (
                f"earn={d_earn}  mid={mid_price:.2f}  "
                f"RVOL={rvol:.1f}x  gap={gap:+.2f}  "
                f"D+{days_since}/{args.wait_days} ({days_left} day(s) to confirm)"
            )
            return "WATCHING", detail

        elif days_since == args.wait_days:
            # Today is D+wait_days — check all observation closes
            broken = any(
                float(df["close"].iloc[earn_i + j]) < mid_price
                for j in range(1, args.wait_days + 1)
            )
            if broken:
                continue
            entry_close = float(df["close"].iloc[-1])
            detail = (
                f"earn={d_earn}  mid={mid_price:.2f}  "
                f"RVOL={rvol:.1f}x  gap={gap:+.2f}  "
                f"enter at today's close ~{entry_close:.2f}"
            )
            return "ENTRY SIGNAL", detail

        else:
            # Position should be open or already closed
            entry_idx   = earn_i + args.wait_days
            if entry_idx >= n:
                continue
            entry_price = float(df["close"].iloc[entry_idx])
            days_in     = days_since - args.wait_days

            # Check Layer-3 observation validity first
            obs_broken = any(
                float(df["close"].iloc[earn_i + j]) < mid_price
                for j in range(1, args.wait_days + 1)
                if earn_i + j < n
            )
            if obs_broken:
                continue  # signal was never valid

            # Check stop during hold
            stop_hit = False
            for k in range(1, min(days_in + 1, args.max_hold + 1)):
                hold_i = entry_idx + k
                if hold_i >= n:
                    break
                if float(df["close"].iloc[hold_i]) < mid_price:
                    stop_hit = True
                    break

            if stop_hit or days_in >= args.max_hold:
                reason = "stop hit" if stop_hit else f"time exit (D+{days_in})"
                detail = (
                    f"earn={d_earn}  entry={entry_price:.2f}  "
                    f"mid={mid_price:.2f}  {reason}"
                )
                return "CLOSED", detail
            else:
                current = float(df["close"].iloc[-1])
                unreal  = current / entry_price - 1
                detail = (
                    f"earn={d_earn}  entry={entry_price:.2f}  "
                    f"stop={mid_price:.2f}  D+{days_in}/{args.max_hold}  "
                    f"now={current:.2f} ({unreal:+.1%})"
                )
                return "IN TRADE", detail

    # --- Upcoming earnings? ---
    upcoming = sorted([
        d for d in earnings
        if today < d <= today + timedelta(days=args.days_ahead)
    ])
    if upcoming:
        days_to = (upcoming[0] - today).days
        return "PRE-EARNINGS", f"next earnings: {upcoming[0]} (in {days_to}d)  close={close:.2f}"

    return "OK", f"close={close:.2f}  MA200={ma200:.2f}"


# ---------------------------------------------------------------------------

def main() -> None:
    args  = parse_args()
    today = date.today()

    ma_note = "" if args.no_ma_filter else " +MA200"
    print(f"\nEarnings Signal Monitor -- {today}")
    print(f"  Universe : {' '.join(args.symbols)} ({len(args.symbols)} symbols)")
    print(
        f"  Filters  : RVOL>{args.rvol_thr}x  Gap>{args.gap_atr_mult}x ATR  "
        f"Wait={args.wait_days}d  MaxHold={args.max_hold}d{ma_note}"
    )
    print(f"  Upcoming : earnings within {args.days_ahead} calendar days flagged as PRE-EARNINGS")
    print()

    rows = []
    for sym in args.symbols:
        print(f"  {sym} ...", end="", flush=True)
        status, detail = _check_symbol(sym, args, today)
        rows.append((sym, status, detail))
        print(f" {status}")

    # Sort by urgency
    rows.sort(key=lambda r: (_STATUS_RANK.get(r[1], 9), r[0]))

    print(f"\n{'='*80}")
    print(f"  {'Symbol':<8}  {'Status':<16}  Detail")
    print(f"  {'-'*76}")
    for sym, status, detail in rows:
        tag = ""
        if status == "ENTRY SIGNAL":
            tag = "  *** ENTER TODAY ***"
        elif status == "IN TRADE":
            tag = "  (monitor stop)"
        print(f"  {sym:<8}  {status:<16}  {detail}{tag}")

    # Summary counts
    active = [r for r in rows if r[1] in ("ENTRY SIGNAL", "WATCHING", "IN TRADE")]
    if active:
        print(f"\n  Active signals: {len(active)}")
        for sym, status, _ in active:
            print(f"    {sym}: {status}")
    print(f"{'='*80}")
    print()


if __name__ == "__main__":
    main()
