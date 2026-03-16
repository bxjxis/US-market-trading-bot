"""
scripts/optimize_earnings.py
-----------------------------
Optuna parameter search for the earnings momentum strategy.

Search space
------------
  rvol_thr      [1.0, 3.0 step 0.1]   min relative volume on signal day
  gap_atr_mult  [0.5, 2.5 step 0.1]   min gap / ATR(14)
  wait_days     [1, 5]                 observation days before entry
  max_hold      [10, 30 step 5]        max holding days

Objective: Sharpe of treatment group.  Returns -inf if n_trades < --min-trades.

Usage
-----
    python scripts/optimize_earnings.py
    python scripts/optimize_earnings.py --trials 200 --min-trades 15
    python scripts/optimize_earnings.py --no-ma-filter --start-date 2022-01-01
"""

import argparse
import copy
import logging
import math
import sys
from datetime import date
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtest_earnings import (
    _DEFAULT_SYMBOLS,
    compute_indicators,
    compute_performance,
    fetch_daily_data,
    fetch_earnings_dates,
    generate_signals,
    print_ab_report,
    print_per_symbol_table,
    simulate_trades,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna optimisation for earnings momentum strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbols",      nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--start-date",   default="2022-01-01", dest="start_date")
    p.add_argument("--end-date",     default=str(date.today()), dest="end_date")
    p.add_argument("--trials",       type=int,   default=150)
    p.add_argument("--min-trades",   type=int,   default=10,  dest="min_trades",
                   help="Min treatment trades required; else objective = -inf.")
    p.add_argument("--no-ma-filter", action="store_true", dest="no_ma_filter")
    p.add_argument("--bb-filter",    action="store_true", dest="bb_filter")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------

def _run_treatment(preloaded, fold_args, start_ts: pd.Timestamp):
    """Return list[Trade] for treatment group with given fold_args."""
    trades = []
    for sym, df, earnings in preloaded:
        sigs = generate_signals(df, sym, earnings, fold_args, use_earnings_filter=True)
        sigs = [s for s in sigs if s.entry_date >= start_ts.date()]
        trades.extend(simulate_trades(df, sigs, fold_args.max_hold))
    return trades


def _make_objective(preloaded, start_ts, base_args, min_trades):
    def objective(trial):
        local = copy.copy(base_args)
        local.rvol_thr     = trial.suggest_float("rvol_thr",     1.0, 3.0, step=0.1)
        local.gap_atr_mult = trial.suggest_float("gap_atr_mult", 0.5, 2.5, step=0.1)
        local.wait_days    = trial.suggest_int(  "wait_days",    1, 5)
        local.max_hold     = trial.suggest_int(  "max_hold",     10, 30, step=5)

        trades = _run_treatment(preloaded, local, start_ts)
        if len(trades) < min_trades:
            return float("-inf")
        perf = compute_performance(trades, 10_000)
        sharpe = perf["sharpe"]
        return sharpe if not math.isnan(sharpe) else float("-inf")
    return objective


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    start_ts = pd.Timestamp(args.start_date)

    print(f"\nEarnings Momentum -- Parameter Optimisation")
    print(f"  Universe : {' '.join(args.symbols)} ({len(args.symbols)} symbols)")
    print(f"  Period   : {args.start_date} -> {args.end_date}")
    print(f"  Trials   : {args.trials}  |  Min trades: {args.min_trades}")
    print()

    # Pre-load all symbol data once (shared across all Optuna trials)
    print("Loading data ...")
    preloaded = []
    for sym in args.symbols:
        print(f"  {sym} ...", end="", flush=True)
        df = fetch_daily_data(sym, args.start_date, args.end_date)
        if df is None or len(df) < 220:
            print(" skipped (insufficient data)")
            continue
        df = compute_indicators(df)
        earnings = fetch_earnings_dates(sym)
        n_earn = len([
            d for d in earnings
            if start_ts.date() <= d <= pd.Timestamp(args.end_date).date()
        ])
        print(f" {len(df)} bars, {n_earn} earnings dates in range")
        preloaded.append((sym, df, earnings))

    if not preloaded:
        print("No data loaded. Exiting.")
        sys.exit(1)

    # Run Optuna
    print(f"\nRunning {args.trials} trials ...")
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _make_objective(preloaded, start_ts, args, args.min_trades),
        n_trials=args.trials,
        show_progress_bar=False,
    )

    best   = study.best_params
    best_v = study.best_value

    print(f"\n{'='*54}")
    print(f"Best Sharpe  : {best_v:.3f}")
    print(f"Best Params  :")
    print(f"  rvol_thr     = {best['rvol_thr']:.1f}")
    print(f"  gap_atr_mult = {best['gap_atr_mult']:.1f}")
    print(f"  wait_days    = {best['wait_days']}")
    print(f"  max_hold     = {best['max_hold']}")
    print(f"{'='*54}")
    print()

    # Re-evaluate best params — full A/B report
    best_args = copy.copy(args)
    best_args.rvol_thr     = best["rvol_thr"]
    best_args.gap_atr_mult = best["gap_atr_mult"]
    best_args.wait_days    = best["wait_days"]
    best_args.max_hold     = best["max_hold"]

    all_t, all_c = [], []
    for sym, df, earnings in preloaded:
        t_sigs = generate_signals(df, sym, earnings, best_args, use_earnings_filter=True)
        c_sigs = generate_signals(df, sym, earnings, best_args, use_earnings_filter=False)
        t_sigs = [s for s in t_sigs if s.entry_date >= start_ts.date()]
        c_sigs = [s for s in c_sigs if s.entry_date >= start_ts.date()]
        all_t.extend(simulate_trades(df, t_sigs, best_args.max_hold))
        all_c.extend(simulate_trades(df, c_sigs, best_args.max_hold))

    print_ab_report(
        compute_performance(all_t, 10_000),
        compute_performance(all_c, 10_000),
        best_args,
        len(preloaded),
    )
    print_per_symbol_table(all_t, all_c)

    # Convenience: print CLI command with best params
    sym_str = " ".join(args.symbols)
    ma_flag = " --no-ma-filter" if args.no_ma_filter else ""
    print(f"Reproduce with:")
    print(
        f"  python scripts/backtest_earnings.py "
        f"--symbols {sym_str}"
        f" --start-date {args.start_date}"
        f" --rvol-thr {best['rvol_thr']:.1f}"
        f" --gap-atr-mult {best['gap_atr_mult']:.1f}"
        f" --wait-days {best['wait_days']}"
        f" --max-hold {best['max_hold']}"
        f"{ma_flag}"
    )
    print()


if __name__ == "__main__":
    main()
