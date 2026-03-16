"""
scripts/walk_forward.py
-----------------------
Walk-forward validation for the CLF grid strategy.

Splits historical data into (train, test) window pairs, runs Optuna
hyperparameter optimisation on each training window (IS), then evaluates
the best params on the immediately following test window (OOS) without
re-optimising. Reports IS vs OOS Sharpe / Calmar / P&L per fold, parameter
stability across folds, and recommends live params.

Usage
-----
    # Smoke test — no Optuna, uses CLI defaults (1 fold on 90 D data)
    python scripts/walk_forward.py --symbol AMZN --no-optimize

    # Download 180 D for 4+ folds, then run
    python scripts/download_cache.py --symbols AMZN NVTS --duration "180 D"
    python scripts/walk_forward.py --symbol AMZN --adx-tiered --atr-adaptive --trials 50

    # Parallel (fold-level, ~N× faster)
    python scripts/walk_forward.py --symbol AMZN --adx-tiered --atr-adaptive \\
        --trials 50 --jobs 4
"""

import argparse
import logging
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtester import BacktestConfig, BacktestEngine
from strategies.clf_grid import CLFGridStrategy
from utils.dashboard_stats import (
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var_dollars,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_VAR_LIMIT_USD    = 15_000.0
_VAR_PENALTY_RATE = 10.0       # Calmar penalty per $1k VaR overage
_CALMAR_CAP       = 100.0      # prevent zero-DD trials dominating TPE sampler


# --- Data structures ----------------------------------------------------------

@dataclass
class FoldWindow:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp


@dataclass
class FoldResult:
    fold_idx:    int
    window:      FoldWindow
    is_params:   Dict
    is_metrics:  Dict
    oos_metrics: Dict


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation for the CLF grid strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Walk-forward windows
    p.add_argument("--symbol",          default="AMZN",
                   help="Ticker to validate (must have data in data/cache/).")
    p.add_argument("--train-days",      type=int,   default=60,   dest="train_days",
                   help="Calendar days in each training (IS) window.")
    p.add_argument("--test-days",       type=int,   default=30,   dest="test_days",
                   help="Calendar days in each test (OOS) window.")
    p.add_argument("--step-days",       type=int,   default=30,   dest="step_days",
                   help="Calendar days to advance the fold start between folds.")
    p.add_argument("--expanding",       action="store_true",
                   help="Expanding window: train always starts at data start.")
    # Optimisation
    p.add_argument("--trials",          type=int,   default=50,
                   help="Optuna trials per fold.")
    p.add_argument("--no-optimize",     action="store_true", dest="no_optimize",
                   help="Skip Optuna; evaluate CLI params on both IS and OOS windows.")
    p.add_argument("--study-db",        default=None, dest="study_db",
                   help="SQLite URL to persist Optuna studies (e.g. sqlite:///data/wf.db).")
    p.add_argument("--jobs",            type=int,   default=1,
                   help="Fold-level parallelism workers. -1 = os.cpu_count().")
    p.add_argument("--min-trades",      type=int,   default=5, dest="min_trades",
                   help="Min trades required per window; below this objective returns -inf.")
    # Backtest config
    p.add_argument("--warmup",          type=int,   default=100,
                   help="Bars prepended before each window to seed indicators.")
    p.add_argument("--account-size",    type=float, default=90_000.0,  dest="account_size")
    p.add_argument("--portfolio-value", type=float, default=180_000.0, dest="portfolio_value")
    p.add_argument("--safety-pct",      type=float, default=0.30,      dest="safety_pct")
    p.add_argument("--window-dd-cap",   type=float, default=0.0,       dest="window_dd_cap",
                   help="Halt new fills when drawdown exceeds this fraction (0 = disabled).")
    p.add_argument("--risk-free",       type=float, default=5.1,       dest="risk_free",
                   help="Annual risk-free rate %% for alpha calculation.")
    # Default grid params (used as base / when --no-optimize)
    p.add_argument("--grid-ratio",      type=float, default=1.015, dest="grid_ratio")
    p.add_argument("--levels",          type=int,   default=20)
    # ADX flags
    p.add_argument("--adx-tiered",           action="store_true",       dest="adx_tiered")
    p.add_argument("--adx-halt",             type=float, default=0.0,   dest="adx_halt")
    p.add_argument("--adx-resume",           type=float, default=25.0,  dest="adx_resume")
    p.add_argument("--adx-halt-thr",         type=float, default=45.0,  dest="adx_halt_thr")
    p.add_argument("--adx-stable-thr",       type=float, default=25.0,  dest="adx_stable_thr")
    p.add_argument("--adx-trend-thr",        type=float, default=35.0,  dest="adx_trend_thr")
    p.add_argument("--adx-stable-ratio",     type=float, default=1.012, dest="adx_stable_ratio")
    p.add_argument("--adx-stable-qty-scale", type=float, default=0.5,   dest="adx_stable_qty_scale")
    p.add_argument("--adx-ultra-thr",        type=float, default=20.0,  dest="adx_ultra_thr")
    p.add_argument("--adx-ultra-qty-scale",  type=float, default=2.0,   dest="adx_ultra_qty_scale")
    p.add_argument("--adx-trend-ratio",      type=float, default=1.015, dest="adx_trend_ratio")
    p.add_argument("--adx-trend-qty-scale",  type=float, default=0.25,  dest="adx_trend_qty_scale")
    # ATR flags
    p.add_argument("--atr-adaptive",         action="store_true",       dest="atr_adaptive")
    p.add_argument("--atr-grid-coverage",    type=float, default=0.5,   dest="atr_grid_coverage")
    p.add_argument("--atr-sizing",           action="store_true",       dest="atr_sizing")
    p.add_argument("--atr-risk-pct",         type=float, default=0.5,   dest="atr_risk_pct")
    # Risk
    p.add_argument("--max-open-loss",        type=float, default=0.0,   dest="max_open_loss")
    return p.parse_args()


# --- Param dict ---------------------------------------------------------------

def _build_base_params(args: argparse.Namespace) -> dict:
    """Build the full CLFGridStrategy param dict from CLI args."""
    return {
        "SYMBOL":               args.symbol,
        "GRID_RATIO":           args.grid_ratio,
        "NUM_BUY_LEVELS":       args.levels,
        "ACCOUNT_SIZE":         args.account_size,
        "SAFETY_PCT":           args.safety_pct,
        "ADX_HALT":             args.adx_halt,
        "ADX_RESUME":           args.adx_resume,
        "ADX_TIERED":           args.adx_tiered,
        "ADX_STABLE_THR":       args.adx_stable_thr,
        "ADX_TREND_THR":        args.adx_trend_thr,
        "ADX_HALT_THR":         args.adx_halt_thr,
        "ADX_STABLE_RATIO":     args.adx_stable_ratio,
        "ADX_STABLE_QTY_SCALE": args.adx_stable_qty_scale,
        "ADX_ULTRA_THR":        args.adx_ultra_thr,
        "ADX_ULTRA_QTY_SCALE":  args.adx_ultra_qty_scale,
        "ADX_TREND_RATIO":      args.adx_trend_ratio,
        "ADX_TREND_QTY_SCALE":  args.adx_trend_qty_scale,
        "MAX_OPEN_LOSS":        args.max_open_loss,
        "ATR_SIZING":           args.atr_sizing,
        "ATR_RISK_PCT":         args.atr_risk_pct,
        "ATR_ADAPTIVE":         args.atr_adaptive,
        "ATR_GRID_COVERAGE":    args.atr_grid_coverage,
        "MIN_EVAL_SECS":        0,
    }


# --- Data loading -------------------------------------------------------------

def _load_data(symbol: str) -> pd.DataFrame:
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache" / symbol
    files = sorted(cache_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No parquet files for '{symbol}' in {cache_dir}.\n"
            f"Run: python scripts/download_cache.py --symbols {symbol} --duration '180 D'"
        )
    df = pd.read_parquet(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --- Window construction ------------------------------------------------------

def _build_wf_windows(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    expanding: bool,
) -> List[FoldWindow]:
    """
    Build (train, test) window pairs.

    Rolling (default): train window slides forward by step_days each fold.
    Expanding        : train always starts at data_start, grows each fold.
    """
    data_start = df["date"].min().normalize()
    data_end   = df["date"].max().normalize()
    fold_start = data_start
    windows: List[FoldWindow] = []

    while True:
        train_s = data_start if expanding else fold_start
        train_e = fold_start + pd.Timedelta(days=train_days)
        test_s  = train_e
        test_e  = test_s + pd.Timedelta(days=test_days)

        if test_e > data_end + pd.Timedelta(days=1):
            break

        windows.append(FoldWindow(
            train_start=train_s,
            train_end=train_e,
            test_start=test_s,
            test_end=test_e,
        ))
        fold_start += pd.Timedelta(days=step_days)

    return windows


# --- Metrics ------------------------------------------------------------------

def _metrics_from_equity(
    eq_slice: pd.Series,
    initial_equity: float,
    portfolio_value: float,
) -> dict:
    """Compute Sharpe/Sortino/Calmar/MaxDD/VaR/P&L from a bar-resolution equity Series."""
    if eq_slice.empty or len(eq_slice) < 2:
        return {
            "sharpe": float("nan"), "sortino": float("nan"),
            "calmar": float("nan"), "max_dd_pct": 0.0,
            "pnl": 0.0, "pnl_pct": 0.0, "var_usd": 0.0,
            "ann_return_pct": float("nan"), "n_trading_days": 0,
        }
    eq_daily = eq_slice.resample("1D").last().dropna()
    dr = eq_daily.pct_change().dropna()

    sharpe  = compute_sharpe(dr)
    sortino = compute_sortino(dr)
    calmar  = compute_calmar(eq_daily)
    mdd     = compute_max_drawdown(eq_slice)   # full resolution for intraday troughs
    var_usd = compute_var_dollars(dr, portfolio_value)
    pnl     = float(eq_slice.iloc[-1]) - initial_equity
    pnl_pct = pnl / initial_equity * 100 if initial_equity > 0 else 0.0

    n_days = max(len(eq_daily), 1)
    ann_return_pct = (
        ((1.0 + pnl / initial_equity) ** (252.0 / n_days) - 1.0) * 100
        if initial_equity > 0 and n_days > 1
        else 0.0
    )

    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd_pct": mdd * 100, "pnl": pnl, "pnl_pct": pnl_pct,
        "var_usd": var_usd, "ann_return_pct": ann_return_pct,
        "n_trading_days": n_days,
    }


# --- Single-window backtest ---------------------------------------------------

def _run_window(
    df: pd.DataFrame,
    symbol: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    params: dict,
    warmup: int,
    account_size: float,
    portfolio_value: float,
    drawdown_cap: float = 0.0,
) -> Optional[dict]:
    """Run BacktestEngine on [ts_start, ts_end). Returns metrics dict or None."""
    mask   = (df["date"] >= ts_start) & (df["date"] < ts_end)
    win_df = df[mask].reset_index(drop=True)
    if len(win_df) < 50:
        return None

    pre_df  = df[df["date"] < ts_start].tail(warmup)
    full_df = pd.concat([pre_df, win_df], ignore_index=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        sym_dir = Path(tmpdir) / symbol
        sym_dir.mkdir()
        full_df.to_parquet(sym_dir / "window.parquet")

        cfg = BacktestConfig(
            initial_capital=account_size,
            portfolio_value=portfolio_value,
            warmup_bars=len(pre_df),
            data_dir=Path(tmpdir),
            drawdown_cap=drawdown_cap,
        )
        result = BacktestEngine(cfg).run(
            strategies=[CLFGridStrategy],
            symbols=[symbol],
            strategy_params={"CLFGridStrategy": params},
        )

    if result.equity_curve.empty:
        return None

    eq = result.equity_curve.set_index("timestamp")["equity"]
    eq.index = pd.to_datetime(eq.index)
    m = _metrics_from_equity(eq, account_size, portfolio_value)
    m["n_trades"] = len(result.trade_log)
    return m


# --- Optuna objective factory -------------------------------------------------

def _make_objective(
    df: pd.DataFrame,
    symbol: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    base_params: dict,
    warmup: int,
    account_size: float,
    portfolio_value: float,
    drawdown_cap: float,
    adx_tiered: bool,
    atr_adaptive: bool,
    min_trades: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = dict(base_params)
        params["GRID_RATIO"]     = trial.suggest_float("grid_ratio",  1.003, 1.025, step=0.001)
        params["NUM_BUY_LEVELS"] = trial.suggest_int("num_levels", 5, 25)

        if adx_tiered:
            params["ADX_TIERED"]        = True
            params["ADX_STABLE_THR"]    = trial.suggest_int("adx_stable_thr", 18, 30)
            params["ADX_TREND_THR"]     = trial.suggest_int("adx_trend_thr",  28, 42)
            params["ADX_STABLE_RATIO"]  = trial.suggest_float(
                "adx_stable_ratio", 1.005, 1.020, step=0.001
            )

        if atr_adaptive:
            params["ATR_ADAPTIVE"]      = True
            params["ATR_GRID_COVERAGE"] = trial.suggest_float(
                "atr_grid_coverage", 0.25, 0.80, step=0.05
            )

        m = _run_window(
            df, symbol, ts_start, ts_end,
            params, warmup, account_size, portfolio_value, drawdown_cap,
        )
        if m is None or m["n_trades"] < min_trades:
            return float("-inf")

        calmar = m["calmar"]
        if math.isnan(calmar) or math.isinf(calmar):
            calmar = 0.0
        calmar = min(calmar, _CALMAR_CAP)

        var_penalty = max(0.0, m["var_usd"] - _VAR_LIMIT_USD) / 1_000.0 * _VAR_PENALTY_RATE
        score = calmar - var_penalty

        trial.set_user_attr("sharpe",   m["sharpe"])
        trial.set_user_attr("calmar",   calmar)
        trial.set_user_attr("var_usd",  m["var_usd"])
        trial.set_user_attr("pnl_pct",  m["pnl_pct"])
        trial.set_user_attr("n_trades", m["n_trades"])
        return score

    return objective


# --- Single fold --------------------------------------------------------------

def _run_fold(
    fold_idx: int,
    fold: FoldWindow,
    df: pd.DataFrame,
    symbol: str,
    base_params: dict,
    args: argparse.Namespace,
) -> Optional[FoldResult]:
    """Optimise on IS window, evaluate on OOS window. Returns FoldResult or None."""
    print(
        f"  Fold {fold_idx:>2}  IS: {fold.train_start.date()} -> {fold.train_end.date()}"
        f"   OOS: {fold.test_start.date()} -> {fold.test_end.date()} ...",
        flush=True,
    )

    if args.no_optimize:
        best_params = dict(base_params)
    else:
        study_name = f"wf_{symbol}_fold{fold_idx}"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + fold_idx),
            storage=args.study_db,
            load_if_exists=True,
        )
        objective = _make_objective(
            df, symbol,
            fold.train_start, fold.train_end,
            base_params, args.warmup,
            args.account_size, args.portfolio_value,
            args.window_dd_cap, args.adx_tiered, args.atr_adaptive,
            args.min_trades,
        )
        study.optimize(
            objective,
            n_trials=args.trials,
            show_progress_bar=False,
            catch=(Exception,),
        )

        if study.best_trial is None or study.best_value == float("-inf"):
            print(f"    -> no valid IS trials, fold skipped.")
            return None

        best_params = dict(base_params)
        best_params["GRID_RATIO"]     = study.best_params["grid_ratio"]
        best_params["NUM_BUY_LEVELS"] = study.best_params["num_levels"]
        if args.adx_tiered:
            best_params["ADX_TIERED"]       = True
            best_params["ADX_STABLE_THR"]   = study.best_params["adx_stable_thr"]
            best_params["ADX_TREND_THR"]    = study.best_params["adx_trend_thr"]
            best_params["ADX_STABLE_RATIO"] = study.best_params["adx_stable_ratio"]
        if args.atr_adaptive:
            best_params["ATR_ADAPTIVE"]      = True
            best_params["ATR_GRID_COVERAGE"] = study.best_params["atr_grid_coverage"]

    # IS evaluation with best params
    is_m = _run_window(
        df, symbol, fold.train_start, fold.train_end,
        best_params, args.warmup, args.account_size, args.portfolio_value,
        args.window_dd_cap,
    )
    # OOS evaluation — NO re-optimisation
    oos_m = _run_window(
        df, symbol, fold.test_start, fold.test_end,
        best_params, args.warmup, args.account_size, args.portfolio_value,
        args.window_dd_cap,
    )

    if is_m is None or oos_m is None:
        print(f"    -> insufficient bars, fold skipped.")
        return None

    return FoldResult(
        fold_idx=fold_idx,
        window=fold,
        is_params=best_params,
        is_metrics=is_m,
        oos_metrics=oos_m,
    )


# --- Reporting ----------------------------------------------------------------

def _fmt(v, spec=".2f", nan_str="-") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return nan_str
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return format(v, spec)


def _print_header(args: argparse.Namespace, df: pd.DataFrame, n_folds: int) -> None:
    mode = "expanding" if args.expanding else "rolling"
    opt  = "none (--no-optimize)" if args.no_optimize else f"{args.trials} trials/fold"
    print(f"\nWalk-Forward Validation: {args.symbol}")
    print(
        f"  Mode: {mode}  |  Train: {args.train_days}d  |  Test: {args.test_days}d  "
        f"|  Step: {args.step_days}d  |  Optuna: {opt}"
    )
    print(
        f"  Data: {df['date'].min().date()} -> {df['date'].max().date()}  "
        f"|  Folds: {n_folds}  |  Min trades/window: {args.min_trades}"
    )
    print("=" * 92)


def _print_fold_table(results: List[FoldResult]) -> None:
    hdr = (
        f"  {'Fold':>4}  {'Train Period':>23}  {'Test Period':>23}  "
        f"{'IS Sharpe':>9}  {'OOS Sharpe':>10}  {'OOS Calmar':>10}  "
        f"{'OOS P&L%':>8}  {'Trades':>6}"
    )
    print(hdr)
    print("-" * 92)
    for r in results:
        w = r.window
        print(
            f"  {r.fold_idx:>4}  "
            f"  {str(w.train_start.date()):>11}->{str(w.train_end.date()):<11}  "
            f"  {str(w.test_start.date()):>11}->{str(w.test_end.date()):<11}  "
            f"  {_fmt(r.is_metrics.get('sharpe')):>9}  "
            f"  {_fmt(r.oos_metrics.get('sharpe')):>9}  "
            f"  {_fmt(r.oos_metrics.get('calmar')):>9}  "
            f"  {_fmt(r.oos_metrics.get('pnl_pct'), '+.2f'):>8}  "
            f"  {r.oos_metrics.get('n_trades', 0):>5}"
        )
    print("=" * 92)


def _print_params_table(results: List[FoldResult], args: argparse.Namespace) -> None:
    if args.no_optimize:
        return
    print("\nBest IS Params per Fold:")
    cols = [("GRID_RATIO", ".3f"), ("NUM_LEVELS", "d")]
    if args.adx_tiered:
        cols += [("ADX_STABLE_THR", ".0f"), ("ADX_TREND_THR", ".0f"), ("ADX_STABLE_RATIO", ".3f")]
    if args.atr_adaptive:
        cols += [("ATR_GRID_COVERAGE", ".2f")]

    header = f"  {'Fold':>4}  " + "  ".join(f"{c:>16}" for c, _ in cols)
    print(header)
    print("-" * len(header))
    key_map = {
        "GRID_RATIO": "GRID_RATIO", "NUM_LEVELS": "NUM_BUY_LEVELS",
        "ADX_STABLE_THR": "ADX_STABLE_THR", "ADX_TREND_THR": "ADX_TREND_THR",
        "ADX_STABLE_RATIO": "ADX_STABLE_RATIO", "ATR_GRID_COVERAGE": "ATR_GRID_COVERAGE",
    }
    for r in results:
        vals = [_fmt(r.is_params.get(key_map[c], None), spec) for c, spec in cols]
        print(f"  {r.fold_idx:>4}  " + "  ".join(f"{v:>16}" for v in vals))


def _print_summary(results: List[FoldResult], args: argparse.Namespace) -> None:
    def _safe_mean(vals):
        clean = [v for v in vals if v is not None and not math.isnan(v) and not math.isinf(v)]
        return float(np.mean(clean)) if clean else float("nan")

    is_sharpes  = [r.is_metrics.get("sharpe",  float("nan")) for r in results]
    oos_sharpes = [r.oos_metrics.get("sharpe", float("nan")) for r in results]
    is_calmar   = [r.is_metrics.get("calmar",  float("nan")) for r in results]
    oos_calmar  = [r.oos_metrics.get("calmar", float("nan")) for r in results]
    oos_pnls    = [r.oos_metrics.get("pnl",    0.0)          for r in results]
    oos_pnl_pct = [r.oos_metrics.get("pnl_pct", 0.0)        for r in results]
    oos_vars    = [r.oos_metrics.get("var_usd", 0.0)         for r in results]
    oos_ann     = [r.oos_metrics.get("ann_return_pct", 0.0)  for r in results]

    avg_is_s  = _safe_mean(is_sharpes)
    avg_oos_s = _safe_mean(oos_sharpes)
    avg_is_c  = _safe_mean(is_calmar)
    avg_oos_c = _safe_mean(oos_calmar)
    avg_pnl   = _safe_mean(oos_pnl_pct)
    avg_var   = _safe_mean(oos_vars)
    avg_ann   = _safe_mean(oos_ann)
    oos_wins  = sum(1 for p in oos_pnls if p > 0)

    # IS->OOS robustness ratio
    if not math.isnan(avg_is_s) and avg_is_s != 0:
        ratio = avg_oos_s / abs(avg_is_s)
        if ratio > 0.70:
            robustness = f"{ratio:.2f}  robust [OK] (>0.70)"
        elif ratio > 0.50:
            robustness = f"{ratio:.2f}  moderate decay [!] (0.50-0.70)"
        else:
            robustness = f"{ratio:.2f}  significant overfitting [X] (<0.50)"
    else:
        robustness = "N/A"

    var_flag = "  *** EXCEEDS $15,000 LIMIT ***" if avg_var > _VAR_LIMIT_USD else ""

    # Alpha vs risk-free
    rf    = args.risk_free / 100.0
    alpha = avg_ann / 100.0 - rf
    excess_usd = alpha * args.account_size

    print("\nSummary:")
    print(f"  Avg IS  Sharpe : {_fmt(avg_is_s)}    "
          f"OOS Sharpe : {_fmt(avg_oos_s)}    "
          f"IS->OOS ratio: {robustness}")
    print(f"  Avg IS  Calmar : {_fmt(avg_is_c)}    "
          f"OOS Calmar : {_fmt(avg_oos_c)}")
    print(f"  Avg OOS P&L%   : {avg_pnl:+.2f}%    "
          f"OOS Win rate: {oos_wins}/{len(results)} folds profitable")
    print(f"  30d 99% VaR (OOS avg): ${avg_var:,.0f}{var_flag}")
    print(f"  Alpha vs {args.risk_free:.1f}% risk-free: {alpha*100:+.2f}%/yr  "
          f"(est. ${excess_usd:+,.0f}/yr on ${args.account_size:,.0f})")


def _print_param_stability(results: List[FoldResult], args: argparse.Namespace) -> None:
    if args.no_optimize:
        return
    profitable = [r for r in results if r.oos_metrics.get("pnl", 0) > 0] or results

    print(f"\nParam Stability ({len(profitable)} OOS-profitable folds):")
    keys = [("GRID_RATIO", ".3f"), ("NUM_BUY_LEVELS", "d")]
    if args.adx_tiered:
        keys += [
            ("ADX_STABLE_THR",  ".0f"),
            ("ADX_TREND_THR",   ".0f"),
            ("ADX_STABLE_RATIO", ".3f"),
        ]
    if args.atr_adaptive:
        keys += [("ATR_GRID_COVERAGE", ".2f")]

    for k, spec in keys:
        vals = [r.is_params[k] for r in profitable if k in r.is_params]
        if not vals:
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        # "d" spec requires int; convert mean/std to nearest int for display
        if spec == "d":
            mean_v_fmt = int(round(mean_v))
            std_v_fmt  = int(round(std_v))
            min_v_fmt  = int(min(vals))
            max_v_fmt  = int(max(vals))
            print(
                f"  {k:<22}: mean={mean_v_fmt:{spec}}  +-{std_v_fmt:{spec}}"
                f"  (range {min_v_fmt:{spec}}-{max_v_fmt:{spec}})"
            )
        else:
            print(
                f"  {k:<22}: mean={mean_v:{spec}}  +-{std_v:{spec}}"
                f"  (range {min(vals):{spec}}-{max(vals):{spec}})"
            )


def _print_recommended_params(results: List[FoldResult], args: argparse.Namespace) -> None:
    if args.no_optimize:
        return
    profitable = [r for r in results if r.oos_metrics.get("pnl", 0) > 0] or results

    rec_gr  = float(np.median([r.is_params["GRID_RATIO"]     for r in profitable]))
    rec_lvl = int(round(float(np.median([r.is_params["NUM_BUY_LEVELS"] for r in profitable]))))

    print("\nRecommended live params (median of OOS-profitable folds):")
    print(f"  GRID_RATIO     = {rec_gr:.3f}")
    print(f"  NUM_BUY_LEVELS = {rec_lvl}")
    if args.adx_tiered:
        sthr = [r.is_params.get("ADX_STABLE_THR",  args.adx_stable_thr)  for r in profitable]
        tthr = [r.is_params.get("ADX_TREND_THR",   args.adx_trend_thr)   for r in profitable]
        srat = [r.is_params.get("ADX_STABLE_RATIO", args.adx_stable_ratio) for r in profitable]
        print(f"  ADX_STABLE_THR  = {float(np.median(sthr)):.0f}")
        print(f"  ADX_TREND_THR   = {float(np.median(tthr)):.0f}")
        print(f"  ADX_STABLE_RATIO= {float(np.median(srat)):.3f}")
    if args.atr_adaptive:
        cov = [r.is_params.get("ATR_GRID_COVERAGE", args.atr_grid_coverage) for r in profitable]
        print(f"  ATR_GRID_COVERAGE= {float(np.median(cov)):.2f}")
    print()


# --- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    n_jobs = os.cpu_count() if args.jobs == -1 else args.jobs

    # Silence strategy / backtester logs
    for name in ("CLFGridStrategy", "DataFetcher", "BacktestEngine", "SimulatedIB"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache" / args.symbol
    if not cache_dir.exists() or not list(cache_dir.glob("*.parquet")):
        print(
            f"ERROR: No cached data for '{args.symbol}'.\n"
            f"Download with:\n"
            f"  python scripts/download_cache.py --symbols {args.symbol} --duration '180 D'"
        )
        sys.exit(1)

    df = _load_data(args.symbol)
    total_cal_days = (df["date"].max() - df["date"].min()).days
    print(
        f"\nLoaded {args.symbol}: {len(df):,} bars  "
        f"({df['date'].min().date()} -> {df['date'].max().date()},  "
        f"{total_cal_days} calendar days)"
    )

    windows = _build_wf_windows(
        df, args.train_days, args.test_days, args.step_days, args.expanding
    )

    if not windows:
        print(
            f"ERROR: No folds fit within {total_cal_days} calendar days.\n"
            f"Need at least {args.train_days + args.test_days} calendar days.\n"
            f"Download more data:\n"
            f"  python scripts/download_cache.py --symbols {args.symbol} --duration '180 D'"
        )
        sys.exit(1)

    if len(windows) == 1:
        print(
            f"WARNING: Only 1 fold found ({total_cal_days} calendar days of data).\n"
            f"  Walk-forward with a single fold is not statistically meaningful.\n"
            f"  Recommend: python scripts/download_cache.py "
            f"--symbols {args.symbol} --duration '180 D'  (gives 4+ folds)"
        )

    base_params = _build_base_params(args)
    _print_header(args, df, len(windows))
    print()

    if n_jobs == 1:
        raw = []
        for i, fold in enumerate(windows, 1):
            raw.append(_run_fold(i, fold, df, args.symbol, base_params, args))
    else:
        print(f"  [parallel: {n_jobs} workers]", flush=True)
        raw = [None] * len(windows)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {
                pool.submit(_run_fold, i + 1, fold, df, args.symbol, base_params, args): i
                for i, fold in enumerate(windows)
            }
            for fut in as_completed(futs):
                raw[futs[fut]] = fut.result()

    results = [r for r in raw if r is not None]
    print()

    if not results:
        print("All folds were skipped (insufficient bars or no valid Optuna trials).")
        sys.exit(1)

    _print_fold_table(results)
    _print_params_table(results, args)
    _print_summary(results, args)
    _print_param_stability(results, args)
    _print_recommended_params(results, args)


if __name__ == "__main__":
    main()
