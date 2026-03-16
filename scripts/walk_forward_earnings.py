"""
scripts/walk_forward_earnings.py
---------------------------------
Walk-forward validation for the earnings momentum strategy.

For each fold:
  IS  : optimise rvol_thr / gap_atr_mult / wait_days / max_hold with Optuna
  OOS : evaluate frozen IS-best params on the immediately following window

Robustness: IS->OOS Sharpe ratio  >0.70=[OK]  0.50-0.70=[!]  <0.50=[X]

Usage
-----
    # Smoke test — no Optuna, 1-2 folds expected on 4y data
    python scripts/walk_forward_earnings.py --no-optimize

    # Full run: 18-month IS, 6-month OOS, 3 folds from 4y data
    python scripts/walk_forward_earnings.py --trials 80

    # Tighter windows — more folds but noisier signals
    python scripts/walk_forward_earnings.py --train-months 12 --test-months 6 --step-months 6
"""

import argparse
import copy
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest_earnings import (
    _DEFAULT_SYMBOLS,
    compute_indicators,
    compute_performance,
    fetch_daily_data,
    fetch_earnings_dates,
    generate_signals,
    simulate_trades,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx:    int
    is_start:    pd.Timestamp
    is_end:      pd.Timestamp
    oos_end:     pd.Timestamp
    is_params:   Dict
    is_sharpe:   float
    oos_sharpe:  float
    oos_n_trades: int
    oos_win_rate: float
    oos_avg_pnl:  float


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation for earnings momentum strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols",       nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--start-date",    default="2022-01-01", dest="start_date")
    p.add_argument("--end-date",      default=str(date.today()), dest="end_date")
    p.add_argument("--train-months",  type=int, default=18, dest="train_months",
                   help="Length of IS (training) window in calendar months.")
    p.add_argument("--test-months",   type=int, default=6,  dest="test_months",
                   help="Length of OOS (test) window in calendar months.")
    p.add_argument("--step-months",   type=int, default=6,  dest="step_months",
                   help="How far to roll the window each fold.")
    p.add_argument("--trials",        type=int, default=80,
                   help="Optuna trials per IS fold.")
    p.add_argument("--min-trades",    type=int, default=5,  dest="min_trades",
                   help="Min IS treatment trades; objective = -inf below this.")
    p.add_argument("--no-optimize",   action="store_true", dest="no_optimize",
                   help="Skip Optuna; use fixed CLI params for all folds.")
    p.add_argument("--no-ma-filter",  action="store_true", dest="no_ma_filter")
    p.add_argument("--bb-filter",     action="store_true", dest="bb_filter")
    p.add_argument("--seed",          type=int, default=42)
    # Fixed param overrides used with --no-optimize or as Optuna fallback
    p.add_argument("--rvol-thr",     type=float, default=1.5,  dest="rvol_thr")
    p.add_argument("--gap-atr-mult", type=float, default=1.0,  dest="gap_atr_mult")
    p.add_argument("--wait-days",    type=int,   default=3,    dest="wait_days")
    p.add_argument("--max-hold",     type=int,   default=20,   dest="max_hold")
    p.add_argument("--account-size", type=float, default=10_000, dest="account_size")
    return p.parse_args()


# ---------------------------------------------------------------------------

def _eval_window(
    preloaded,
    ws: pd.Timestamp,
    we: pd.Timestamp,
    fold_args,
) -> Optional[Dict]:
    """Run treatment backtest on [ws, we). Returns perf dict or None."""
    trades = []
    for sym, df, earnings in preloaded:
        sigs = generate_signals(df, sym, earnings, fold_args, use_earnings_filter=True)
        sigs = [s for s in sigs if ws.date() <= s.entry_date < we.date()]
        trades.extend(simulate_trades(df, sigs, fold_args.max_hold))
    if not trades:
        return None
    perf = compute_performance(trades, fold_args.account_size)
    perf["n_trades"] = len(trades)
    return perf


def _optimise_is(
    preloaded,
    ws: pd.Timestamp,
    we: pd.Timestamp,
    args,
    fold_idx: int,
) -> Dict:
    """Optimise on IS window. Returns best_params dict (without is_sharpe key)."""
    def objective(trial):
        local = copy.copy(args)
        local.rvol_thr     = trial.suggest_float("rvol_thr",     1.0, 3.0, step=0.1)
        local.gap_atr_mult = trial.suggest_float("gap_atr_mult", 0.5, 2.5, step=0.1)
        local.wait_days    = trial.suggest_int(  "wait_days",    1, 5)
        local.max_hold     = trial.suggest_int(  "max_hold",     10, 30, step=5)

        trades = []
        for sym, df, earnings in preloaded:
            sigs = generate_signals(df, sym, earnings, local, use_earnings_filter=True)
            sigs = [s for s in sigs if ws.date() <= s.entry_date < we.date()]
            trades.extend(simulate_trades(df, sigs, local.max_hold))

        if len(trades) < args.min_trades:
            return float("-inf")
        perf = compute_performance(trades, args.account_size)
        s = perf["sharpe"]
        return s if not math.isnan(s) else float("-inf")

    sampler = optuna.samplers.TPESampler(seed=args.seed + fold_idx)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    if study.best_value == float("-inf"):
        _log.warning("Fold %d IS: all trials returned -inf; using CLI defaults.", fold_idx)
        return {
            "rvol_thr":     args.rvol_thr,
            "gap_atr_mult": args.gap_atr_mult,
            "wait_days":    args.wait_days,
            "max_hold":     args.max_hold,
            "_is_sharpe":   float("nan"),
        }

    return {**study.best_params, "_is_sharpe": study.best_value}


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    start_ts = pd.Timestamp(args.start_date)
    end_ts   = pd.Timestamp(args.end_date)

    mode_str = "no-optimize (fixed params)" if args.no_optimize else f"Optuna {args.trials} trials/fold"
    print(f"\nWalk-Forward Validation: Earnings Momentum")
    print(f"  Universe  : {' '.join(args.symbols)} ({len(args.symbols)} symbols)")
    print(f"  Period    : {args.start_date} -> {args.end_date}")
    print(f"  Windows   : {args.train_months}m IS / {args.test_months}m OOS / {args.step_months}m step")
    print(f"  Mode      : {mode_str}  |  Min trades: {args.min_trades}")
    print()

    # --- Load data ---
    print("Loading data ...")
    preloaded = []
    for sym in args.symbols:
        print(f"  {sym} ...", end="", flush=True)
        df = fetch_daily_data(sym, args.start_date, args.end_date)
        if df is None or len(df) < 220:
            print(" skipped"); continue
        df = compute_indicators(df)
        earnings = fetch_earnings_dates(sym)
        n_earn = len([
            d for d in earnings
            if start_ts.date() <= d <= end_ts.date()
        ])
        print(f" {len(df)} bars, {n_earn} earnings dates")
        preloaded.append((sym, df, earnings))

    if not preloaded:
        print("No data. Exiting."); sys.exit(1)

    # --- Build fold windows ---
    fold_start = start_ts
    windows    = []
    while True:
        is_end  = fold_start + pd.DateOffset(months=args.train_months)
        oos_end = is_end    + pd.DateOffset(months=args.test_months)
        if oos_end > end_ts + pd.Timedelta(days=1):
            break
        windows.append((fold_start, is_end, oos_end))
        fold_start += pd.DateOffset(months=args.step_months)

    if not windows:
        avail = (end_ts - start_ts).days // 30
        need  = args.train_months + args.test_months
        print(f"\n[!] Not enough data for 1 fold: need {need}m, available ~{avail}m.")
        print(f"    Extend --start-date further back, or reduce --train-months/--test-months.")
        sys.exit(1)

    print(f"\n  Total folds: {len(windows)}")

    # --- Run folds ---
    results: List[FoldResult] = []

    for i, (is_s, is_e, oos_e) in enumerate(windows):
        print(f"\nFold {i+1}/{len(windows)}")
        print(f"  IS : {is_s.date()} -> {is_e.date()}")
        print(f"  OOS: {is_e.date()} -> {oos_e.date()}")

        if args.no_optimize:
            best_p = {
                "rvol_thr":     args.rvol_thr,
                "gap_atr_mult": args.gap_atr_mult,
                "wait_days":    args.wait_days,
                "max_hold":     args.max_hold,
            }
            fold_args_is = copy.copy(args)
            for k, v in best_p.items():
                setattr(fold_args_is, k, v)
            is_perf   = _eval_window(preloaded, is_s, is_e, fold_args_is) or {}
            is_sharpe = is_perf.get("sharpe", float("nan"))
            n_is      = is_perf.get("n_trades", 0)
            print(f"  IS eval: {n_is} trades  Sharpe={is_sharpe:.2f}" if not math.isnan(is_sharpe) else f"  IS eval: {n_is} trades  Sharpe=n/a")
        else:
            print(f"  Optimising IS ({args.trials} trials) ...", end="", flush=True)
            raw_p     = _optimise_is(preloaded, is_s, is_e, args, fold_idx=i)
            is_sharpe = raw_p.pop("_is_sharpe", float("nan"))
            best_p    = raw_p
            is_str = f"{is_sharpe:.2f}" if not math.isnan(is_sharpe) else "n/a"
            print(f" IS Sharpe={is_str}  rvol={best_p['rvol_thr']:.1f}  gap={best_p['gap_atr_mult']:.1f}  wait={best_p['wait_days']}  hold={best_p['max_hold']}")

        # OOS evaluation
        fold_args_oos = copy.copy(args)
        for k, v in best_p.items():
            setattr(fold_args_oos, k, v)

        oos_perf = _eval_window(preloaded, is_e, oos_e, fold_args_oos)
        if oos_perf is None:
            print(f"  OOS: 0 trades — fold skipped")
            continue

        oos_sharpe = oos_perf.get("sharpe", float("nan"))
        oos_str    = f"{oos_sharpe:.2f}" if not math.isnan(oos_sharpe) else "n/a"
        print(
            f"  OOS: {oos_perf['n_trades']} trades  "
            f"Win={oos_perf['win_rate']:.1f}%  "
            f"AvgP&L={oos_perf['avg_pnl_pct']:+.2f}%  "
            f"Sharpe={oos_str}"
        )

        results.append(FoldResult(
            fold_idx=i + 1,
            is_start=is_s, is_end=is_e, oos_end=oos_e,
            is_params=best_p,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            oos_n_trades=oos_perf["n_trades"],
            oos_win_rate=oos_perf["win_rate"],
            oos_avg_pnl=oos_perf["avg_pnl_pct"],
        ))

    if not results:
        print("\nNo completed folds (all had 0 OOS trades). Exiting.")
        sys.exit(1)

    # --- Summary ---
    profitable  = [r for r in results if r.oos_avg_pnl > 0]
    is_sharpes  = [r.is_sharpe  for r in results if not math.isnan(r.is_sharpe)]
    oos_sharpes = [r.oos_sharpe for r in results if not math.isnan(r.oos_sharpe)]

    print(f"\n{'='*66}")
    print("Summary")
    print(f"  Completed folds       : {len(results)}")
    print(f"  OOS profitable folds  : {len(profitable)}/{len(results)}")
    avg_oos_trades = float(np.mean([r.oos_n_trades for r in results]))
    if is_sharpes and oos_sharpes:
        avg_is  = float(np.mean(is_sharpes))
        avg_oos = float(np.mean(oos_sharpes))
        ratio   = avg_oos / avg_is if avg_is != 0 else float("nan")
        is_str  = f"{avg_is:.2f}"  if not math.isnan(avg_is)  else "n/a"
        oos_str = f"{avg_oos:.2f}" if not math.isnan(avg_oos) else "n/a"
        print(f"  Avg IS  Sharpe        : {is_str}")
        print(f"  Avg OOS Sharpe        : {oos_str}")
        # Sharpe ratio is unreliable when OOS sample < 5 trades/fold
        if avg_oos_trades < 5:
            print(f"  IS->OOS ratio         : n/a  [!] OOS sample too small ({avg_oos_trades:.1f} trades/fold avg) -- use P&L below")
        elif not math.isnan(ratio):
            rob = "[OK] robust" if ratio > 0.70 else ("[!] review" if ratio > 0.50 else "[X] overfitting")
            print(f"  IS->OOS ratio         : {ratio:.2f}  {rob}")
    avg_oos_pnl = float(np.mean([r.oos_avg_pnl for r in results]))
    print(f"  Avg OOS P&L/trade     : {avg_oos_pnl:+.2f}%")

    if profitable:
        print(f"\nParam Stability (OOS-profitable folds):")
        spec_map = {"rvol_thr": ".1f", "gap_atr_mult": ".1f", "wait_days": "d", "max_hold": "d"}
        for k, spec in spec_map.items():
            vals = [r.is_params[k] for r in profitable]
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            if spec == "d":
                print(f"  {k:<16}: mean={int(round(mean_v))}  +-{int(round(std_v))}")
            else:
                print(f"  {k:<16}: mean={mean_v:{spec}}  +-{std_v:{spec}}")

        print(f"\nRecommended live params (median of OOS-profitable folds):")
        for k, spec in spec_map.items():
            vals = [r.is_params[k] for r in profitable]
            med  = float(np.median(vals))
            if spec == "d":
                print(f"  {k:<16} = {int(round(med))}")
            else:
                print(f"  {k:<16} = {med:{spec}}")

    print(f"{'='*66}")


if __name__ == "__main__":
    main()
