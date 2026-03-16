"""
scripts/drift_analysis.py
--------------------------
Rolling-window backtest comparison to detect strategy decay.

Slices the cached parquet dataset into consecutive N-day windows,
runs the full CLFGridStrategy on each window, and prints a side-by-side
performance table with Sharpe, Sortino, Calmar, MaxDD, and trade count.

Three evaluation modes
----------------------
default (independent windows)
    Each window is a fresh backtest.  Fast; isolates each period cleanly.

--carry-positions
    Runs ONE continuous backtest over all data, then slices metrics at window
    boundaries.  Open positions and unrealised P&L from window N carry into
    window N+1 — the same way a live grid would behave.

--slippage-sweep
    Runs the full analysis at slippage = 0.05 %, 0.10 %, 0.20 % and compares
    stability scores.  Reveals whether the strategy is sensitive to execution
    quality (bid-ask spread, IBKR latency, partial fills).

Usage
-----
    # Default 30-day windows on AMZN
    python scripts/drift_analysis.py --symbol AMZN

    # Carry-over mode — more realistic for live deployment
    python scripts/drift_analysis.py --symbol NVTS --carry-positions

    # Slippage sensitivity
    python scripts/drift_analysis.py --symbol AMZN --slippage-sweep

    # Tiered ADX with carry-over
    python scripts/drift_analysis.py --symbol AMZN --adx-tiered --carry-positions --window 20
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.backtester as _bt_module          # needed for slippage patching
from core.backtester import BacktestConfig, BacktestEngine
from strategies.clf_grid import CLFGridStrategy
from utils.dashboard_stats import (
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var_dollars,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_SWEEP_SLIPPAGES = [0.0005, 0.001, 0.002]   # 0.05 %, 0.10 %, 0.20 %


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling-window grid strategy drift analysis.")
    p.add_argument("--symbol",        default="AMZN",
                   help="Ticker symbol (must have data in data/cache/).")
    p.add_argument("--window",        type=int, default=30,
                   help="Calendar-day width of each window (default: 30).")
    p.add_argument("--step",          type=int, default=0,
                   help="Calendar-day step between windows (0 = non-overlapping).")
    p.add_argument("--warmup",        type=int, default=100,
                   help="Bars used to seed indicators before replay (default: 100).")
    p.add_argument("--carry-positions", action="store_true", dest="carry_positions",
                   help="Run one continuous backtest; carry open positions across window boundaries.")
    p.add_argument("--slippage-sweep",  action="store_true", dest="slippage_sweep",
                   help="Re-run analysis at slippage 0.05 %%, 0.10 %%, 0.20 %% and compare stability.")
    # Grid
    p.add_argument("--grid-ratio",    type=float, default=1.005, dest="grid_ratio")
    p.add_argument("--levels",        type=int,   default=20)
    p.add_argument("--account-size",  type=float, default=90_000.0, dest="account_size")
    p.add_argument("--safety-pct",    type=float, default=0.30,     dest="safety_pct")
    p.add_argument("--portfolio-value", type=float, default=180_000.0, dest="portfolio_value")
    # ADX
    p.add_argument("--adx-tiered",           action="store_true", dest="adx_tiered")
    p.add_argument("--adx-stable-thr",        type=float, default=25.0,  dest="adx_stable_thr")
    p.add_argument("--adx-trend-thr",         type=float, default=35.0,  dest="adx_trend_thr")
    p.add_argument("--adx-halt-thr",          type=float, default=45.0,  dest="adx_halt_thr")
    p.add_argument("--adx-stable-ratio",      type=float, default=1.012, dest="adx_stable_ratio")
    p.add_argument("--adx-stable-qty-scale",  type=float, default=0.5,   dest="adx_stable_qty_scale")
    p.add_argument("--adx-ultra-thr",         type=float, default=20.0,  dest="adx_ultra_thr")
    p.add_argument("--adx-ultra-qty-scale",   type=float, default=2.0,   dest="adx_ultra_qty_scale")
    p.add_argument("--adx-trend-ratio",       type=float, default=1.015, dest="adx_trend_ratio")
    p.add_argument("--adx-trend-qty-scale",   type=float, default=0.25,  dest="adx_trend_qty_scale")
    p.add_argument("--adx-halt",              type=float, default=0.0,   dest="adx_halt")
    p.add_argument("--adx-resume",            type=float, default=25.0,  dest="adx_resume")
    # Risk
    p.add_argument("--max-open-loss",    type=float, default=0.0,   dest="max_open_loss")
    p.add_argument("--atr-sizing",       action="store_true",       dest="atr_sizing")
    p.add_argument("--atr-risk-pct",     type=float, default=0.5,   dest="atr_risk_pct")
    p.add_argument("--atr-adaptive",     action="store_true",       dest="atr_adaptive")
    p.add_argument("--atr-grid-coverage",type=float, default=0.5,   dest="atr_grid_coverage")
    p.add_argument("--window-dd-cap",    type=float, default=0.0,   dest="window_dd_cap",
                   help="Halt new fills when DD > this fraction per window (e.g. 0.015). 0 = disabled.")
    p.add_argument("--risk-free",        type=float, default=5.1,   dest="risk_free",
                   help="Annual risk-free rate %% for capital efficiency calculation (default: 5.1).")
    return p.parse_args()


def _build_params(args: argparse.Namespace) -> dict:
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


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_full_data(symbol: str) -> pd.DataFrame:
    cache_dir = Path("data/cache") / symbol
    files = sorted(cache_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No parquet files for '{symbol}' in {cache_dir.resolve()}")
    df = pd.read_parquet(files[-1])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_windows(df: pd.DataFrame, window_days: int, step_days: int) -> list:
    first = df["date"].min().normalize()
    last  = df["date"].max().normalize()
    windows, t = [], first
    while t + pd.Timedelta(days=window_days) <= last + pd.Timedelta(days=1):
        windows.append((t, t + pd.Timedelta(days=window_days)))
        t += pd.Timedelta(days=step_days)
    return windows


# ── Metrics helper ─────────────────────────────────────────────────────────────

def _metrics_from_equity(
    eq_slice: pd.Series,
    initial_equity: float,
    portfolio_value: float,
) -> dict:
    """
    Compute all per-window stats from a bar-resolution equity slice.

    Resamples to daily for Sharpe/Sortino/Calmar (annualisation requires
    consistent periodicity); uses full-resolution equity for MaxDD so
    intraday troughs are captured.
    """
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
    mdd     = compute_max_drawdown(eq_slice)          # full resolution
    var_usd = compute_var_dollars(dr, portfolio_value)
    pnl     = float(eq_slice.iloc[-1]) - initial_equity
    pnl_pct = pnl / initial_equity * 100 if initial_equity > 0 else 0.0

    # Annualized return — compound, using actual trading days in this window
    n_days = max(len(eq_daily), 1)
    if initial_equity > 0 and n_days > 1:
        ann_return_pct = ((1.0 + pnl / initial_equity) ** (252.0 / n_days) - 1.0) * 100
    else:
        ann_return_pct = 0.0

    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd_pct": mdd * 100, "pnl": pnl, "pnl_pct": pnl_pct,
        "var_usd": var_usd, "ann_return_pct": ann_return_pct, "n_trading_days": n_days,
    }


# ── Mode A: independent windows ────────────────────────────────────────────────

def run_independent(
    df: pd.DataFrame,
    symbol: str,
    windows: list,
    params: dict,
    warmup: int,
    account_size: float,
    portfolio_value: float,
    drawdown_cap: float = 0.0,
) -> list:
    rows = []
    for i, (ws, we) in enumerate(windows, 1):
        print(f"  [{i:>2}/{len(windows)}]  {ws.date()} → {we.date()} ...", end="", flush=True)

        mask_win = (df["date"] >= ws) & (df["date"] < we)
        win_df   = df[mask_win].reset_index(drop=True)
        if len(win_df) < max(warmup, 50):
            print(" skipped (too few bars)")
            continue

        pre_df   = df[df["date"] < ws].tail(warmup)
        full_df  = pd.concat([pre_df, win_df], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            sym_dir = Path(tmpdir) / symbol
            sym_dir.mkdir()
            full_df.to_parquet(sym_dir / "window.parquet")

            cfg    = BacktestConfig(
                initial_capital = account_size,
                portfolio_value = portfolio_value,
                warmup_bars     = len(pre_df),
                data_dir        = Path(tmpdir),
                drawdown_cap    = drawdown_cap,
            )
            result = BacktestEngine(cfg).run(
                strategies      = [CLFGridStrategy],
                symbols         = [symbol],
                strategy_params = {"CLFGridStrategy": params},
            )

        trades = len(result.trade_log)

        if result.equity_curve.empty:
            m = {"sharpe": float("nan"), "sortino": float("nan"), "calmar": float("nan"),
                 "max_dd_pct": 0.0, "pnl": 0.0, "pnl_pct": 0.0, "var_usd": 0.0,
                 "ann_return_pct": float("nan"), "n_trading_days": 0}
        else:
            eq = result.equity_curve.set_index("timestamp")["equity"]
            eq.index = pd.to_datetime(eq.index)
            m = _metrics_from_equity(eq, account_size, portfolio_value)

        _print_window_progress(m, trades)
        rows.append({"start": ws.date(), "end": we.date(),
                     "bars": len(win_df), "trades": trades, **m})
    return rows


# ── Mode B: carry-over (single continuous run, slice by window) ────────────────

def run_carry(
    df: pd.DataFrame,
    symbol: str,
    windows: list,
    params: dict,
    warmup: int,
    account_size: float,
    portfolio_value: float,
    drawdown_cap: float = 0.0,
) -> list:
    """
    Run one continuous backtest over the full dataset.  Open positions and
    unrealised P&L carry across window boundaries naturally.
    """
    print("  Running single continuous backtest (carry-positions mode) ...", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        sym_dir = Path(tmpdir) / symbol
        sym_dir.mkdir()
        df.to_parquet(sym_dir / "full.parquet")

        cfg    = BacktestConfig(
            initial_capital = account_size,
            portfolio_value = portfolio_value,
            warmup_bars     = warmup,
            data_dir        = Path(tmpdir),
            drawdown_cap    = drawdown_cap,
        )
        result = BacktestEngine(cfg).run(
            strategies      = [CLFGridStrategy],
            symbols         = [symbol],
            strategy_params = {"CLFGridStrategy": params},
        )

    if result.equity_curve.empty:
        print("  No trades in full run.")
        return []

    ec = result.equity_curve.copy()
    ec["timestamp"] = pd.to_datetime(ec["timestamp"])
    ec = ec.set_index("timestamp")["equity"].sort_index()

    rows = []
    for ws, we in windows:
        mask = (ec.index >= ws) & (ec.index < we)
        eq_slice = ec[mask]
        if eq_slice.empty:
            continue

        bars = int(((df["date"] >= ws) & (df["date"] < we)).sum())

        # Initial equity = last point before this window (= carry-over)
        pre_eq = ec[ec.index < ws]
        initial = float(pre_eq.iloc[-1]) if not pre_eq.empty else account_size

        m = _metrics_from_equity(eq_slice, initial, portfolio_value)

        # Trade count is approximate: equity step-changes signal fills
        # (trade_log has no timestamps, so we use equity variance as proxy)
        rows.append({
            "start": ws.date(), "end": we.date(),
            "bars": bars, "trades": None,  # not available per-window in carry mode
            **m,
        })

    # Print progress per window after the fact
    for i, r in enumerate(rows, 1):
        sharpe_str = f"Sharpe {r['sharpe']:.2f}" if not np.isnan(r["sharpe"]) else "no trades"
        carry_pnl  = f"{'+'if r['pnl']>=0 else ''}{r['pnl']:,.0f}"
        print(f"  [{i:>2}/{len(rows)}]  {r['start']} → {r['end']}  "
              f"P&L ${carry_pnl}  {sharpe_str}")

    return rows


def _print_window_progress(m: dict, trades: int) -> None:
    sharpe_str = f"Sharpe {m['sharpe']:.2f}" if not np.isnan(m["sharpe"]) else "no trades"
    print(f"  {trades:>3} trades  {sharpe_str}")


# ── Slippage sweep ─────────────────────────────────────────────────────────────

def run_slippage_sweep(
    df: pd.DataFrame,
    symbol: str,
    windows: list,
    params: dict,
    warmup: int,
    account_size: float,
    portfolio_value: float,
    carry_positions: bool,
    drawdown_cap: float = 0.0,
) -> list:
    """
    Run the full analysis at each slippage level in _SWEEP_SLIPPAGES.
    Returns a list of summary dicts, one per slippage value.
    """
    summaries = []
    for slip in _SWEEP_SLIPPAGES:
        label = f"{slip * 100:.2f}%"
        print(f"\n  ── Slippage {label} ─────────────────────────────────")
        with patch.object(_bt_module, "_SLIPPAGE_PCT", slip):
            if carry_positions:
                rows = run_carry(df, symbol, windows, params, warmup,
                                 account_size, portfolio_value, drawdown_cap)
            else:
                rows = run_independent(df, symbol, windows, params, warmup,
                                       account_size, portfolio_value, drawdown_cap)
        summaries.append({"slippage": slip, "rows": rows})
    return summaries


# ── Report helpers ─────────────────────────────────────────────────────────────

_W = 90   # total table width

def _stability(sharpes: list) -> tuple:
    """Return (avg, std, cv, score) for a list of Sharpe values."""
    if not sharpes:
        return float("nan"), float("nan"), float("nan"), 0.0
    arr = np.array(sharpes)
    avg = float(np.mean(arr))
    std = float(np.std(arr))
    cv  = std / abs(avg) if avg != 0 else float("inf")
    score = max(0.0, 1.0 - cv)
    return avg, std, cv, score


def _fmt_float(v, fmt=".2f", nan="  N/A") -> str:
    return f"{v:{fmt}}" if not np.isnan(v) and not np.isinf(v) else nan


def _stability_label(score: float) -> str:
    if score >= 0.6: return "stable"
    if score >= 0.3: return "moderate"
    return "unstable"


def print_report(rows: list, symbol: str, args: argparse.Namespace, mode: str = "") -> None:
    adx_mode = (
        "tiered" if args.adx_tiered
        else (f"halt>{args.adx_halt:.0f}" if args.adx_halt > 0 else "off")
    )
    carry_tag = " [carry-positions]" if args.carry_positions else ""
    print(f"\n{'═' * _W}")
    print(
        f"  Drift Analysis: {symbol}  |  {args.window}-day windows  |  "
        f"grid={args.grid_ratio}  ADX={adx_mode}{carry_tag}{mode}"
    )
    print(f"  Warmup: {args.warmup} bars  |  Account: ${args.account_size:,.0f}")
    print("═" * _W)

    hdr = (
        f"{'#':>3}  {'Date Range':>23}  {'Bars':>5}  {'Trd':>4}  "
        f"{'P&L $':>9}  {'P&L%':>6}  {'Sharpe':>7}  {'Sortino':>8}  "
        f"{'Calmar':>7}  {'MaxDD%':>7}"
    )
    print(hdr)
    print("─" * _W)

    sharpes, sortinos, calmars, pnls, trades_list = [], [], [], [], []

    for i, r in enumerate(rows, 1):
        trades_str = f"{r['trades']:>4}" if r["trades"] is not None else "  ─ "
        pnl_sign   = "+" if r["pnl"] >= 0 else ""
        sortino_v  = r["sortino"]
        sortino_s  = (
            "     ∞" if np.isinf(sortino_v)
            else f"{sortino_v:>8.2f}" if not np.isnan(sortino_v)
            else "    N/A"
        )
        print(
            f"{i:>3}  "
            f"{str(r['start'])} → {str(r['end']):>10}  "
            f"{r['bars']:>5}  "
            f"{trades_str}  "
            f"{pnl_sign}${r['pnl']:>8,.0f}  "
            f"{pnl_sign}{r['pnl_pct']:>5.2f}%  "
            f"{_fmt_float(r['sharpe'], '7.3f', '    N/A'):>7}  "
            f"{sortino_s}  "
            f"{_fmt_float(r['calmar'], '7.2f', '    N/A'):>7}  "
            f"{r['max_dd_pct']:>6.2f}%"
        )
        if not np.isnan(r["sharpe"]):
            sharpes.append(r["sharpe"])
        if not np.isnan(r["sortino"]) and not np.isinf(r["sortino"]):
            sortinos.append(r["sortino"])
        if not np.isnan(r["calmar"]) and not np.isinf(r["calmar"]):
            calmars.append(r["calmar"])
        pnls.append(r["pnl"])
        if r["trades"] is not None:
            trades_list.append(r["trades"])

    print("─" * _W)

    avg_pnl = np.mean(pnls) if pnls else float("nan")
    std_pnl = np.std(pnls)  if pnls else float("nan")
    avg_trd = np.mean(trades_list) if trades_list else float("nan")
    pnl_sign = "+" if avg_pnl >= 0 else ""

    avg_sh, std_sh, cv_sh, stab = _stability(sharpes)
    avg_so = np.mean(sortinos) if sortinos else float("nan")
    avg_ca = np.mean(calmars)  if calmars  else float("nan")

    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0.0

    trd_str = f"{avg_trd:>4.0f}" if not np.isnan(avg_trd) else "  ─ "
    print(
        f"{'AVG':>3}  {'':>23}  {'':>5}  {trd_str}  "
        f"{pnl_sign}${avg_pnl:>8,.0f}  {'':>6}   "
        f"{_fmt_float(avg_sh, '7.3f'):>7}  "
        f"{_fmt_float(avg_so, '8.2f'):>8}  "
        f"{_fmt_float(avg_ca, '7.2f'):>7}"
    )
    print(
        f"{'STD':>3}  {'':>23}  {'':>5}  {'':>4}  "
        f" ${std_pnl:>8,.0f}  {'':>6}   "
        f"{_fmt_float(std_sh, '7.3f'):>7}"
    )

    print("═" * _W)
    stab_lbl = _stability_label(stab)
    print(
        f"  Windows: {len(rows)}  |  Win rate: {win_rate:.0f}%  |  "
        f"Sharpe CV: {_fmt_float(cv_sh, '.2f')}  |  "
        f"Stability: {stab:.2f}  ({stab_lbl})"
    )
    print("═" * _W)

    if stab >= 0.6:
        print("  ✓  Consistent performance across market regimes.")
    elif stab >= 0.3:
        print("  ~  Moderate stability — performance varies by period.")
    else:
        print("  ✗  High period sensitivity — strategy may be overfit to recent data.")
        print("     Consider wider grid ratios or out-of-sample validation before deployment.")

    # ── Capital Efficiency ─────────────────────────────────────────────────────
    risk_free = getattr(args, "risk_free", 5.1)
    ann_returns = [
        r["ann_return_pct"] for r in rows
        if not np.isnan(r.get("ann_return_pct", float("nan")))
    ]
    if ann_returns:
        avg_ann      = float(np.mean(ann_returns))
        alpha        = avg_ann - risk_free
        excess_usd   = alpha / 100.0 * args.account_size
        above_rf_n   = sum(1 for x in ann_returns if x > risk_free)
        above_rf_pct = above_rf_n / len(ann_returns) * 100
        sign_ann     = "+" if avg_ann >= 0 else ""
        sign_alpha   = "+" if alpha >= 0 else ""
        sign_usd     = "+" if excess_usd >= 0 else ""
        print(f"\n  Capital Efficiency  (risk-free benchmark: {risk_free:.1f}%  |  deployed: ${args.account_size:,.0f})")
        print(f"  {'─' * 56}")
        print(f"    Avg annualized return : {sign_ann}{avg_ann:.2f}%")
        print(f"    Alpha vs. risk-free   : {sign_alpha}{alpha:.2f}%   (est. {sign_usd}${excess_usd:,.0f}/yr excess return)")
        print(f"    Windows > risk-free   : {above_rf_n}/{len(ann_returns)}  ({above_rf_pct:.0f}%)")
        if alpha < 0:
            print(f"    ✗  Strategy underperforms money-market fund — net opportunity cost ${-excess_usd:,.0f}/yr.")
        elif alpha < 2.0:
            print(f"    ~  Marginal alpha — weigh execution risk and monitoring overhead.")
        else:
            print(f"    ✓  Meaningful alpha above risk-free.")
    print()


def print_slippage_sweep(summaries: list, symbol: str, args: argparse.Namespace) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Slippage Sensitivity — {symbol}")
    print("═" * 60)
    print(f"  {'Slippage':>9}  {'Win%':>6}  {'Avg Sharpe':>10}  "
          f"{'Sharpe STD':>10}  {'Stability':>10}  {'Rating':>8}")
    print("─" * 60)

    for s in summaries:
        rows  = s["rows"]
        label = f"{s['slippage'] * 100:.2f}%"
        if not rows:
            print(f"  {label:>9}  (no results)")
            continue
        pnls    = [r["pnl"] for r in rows]
        sharpes = [r["sharpe"] for r in rows if not np.isnan(r["sharpe"])]
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100 if pnls else 0.0
        avg_sh, std_sh, cv_sh, stab = _stability(sharpes)
        is_base = abs(s["slippage"] - 0.001) < 1e-6
        base_tag = " ← base" if is_base else ""
        print(
            f"  {label:>9}  {win_rate:>5.0f}%  "
            f"{_fmt_float(avg_sh, '10.3f'):>10}  "
            f"{_fmt_float(std_sh, '10.3f'):>10}  "
            f"{stab:>10.2f}  "
            f"{_stability_label(stab):>8}"
            f"{base_tag}"
        )

    print("─" * 60)
    # Impact assessment
    base   = next((s for s in summaries if abs(s["slippage"] - 0.001) < 1e-6), None)
    high   = next((s for s in summaries if abs(s["slippage"] - 0.002) < 1e-6), None)
    if base and high and base["rows"] and high["rows"]:
        _, _, _, base_stab = _stability([r["sharpe"] for r in base["rows"]
                                         if not np.isnan(r["sharpe"])])
        _, _, _, high_stab = _stability([r["sharpe"] for r in high["rows"]
                                         if not np.isnan(r["sharpe"])])
        drop = base_stab - high_stab
        if drop > 0.3:
            print(
                f"\n  ✗  Stability drops {drop:.2f} pts from 0.10%→0.20% slippage.\n"
                f"     Strategy is execution-sensitive — IBKR latency/spread will hurt live P&L."
            )
        elif drop > 0.1:
            print(
                f"\n  ~  Moderate slippage sensitivity (drop {drop:.2f} pts).\n"
                f"     Use limit orders and avoid illiquid open/close periods."
            )
        else:
            print(
                f"\n  ✓  Low slippage sensitivity (drop {drop:.2f} pts).\n"
                f"     Strategy is robust to realistic execution costs."
            )
    print()

    # Print detailed window table for base slippage
    if base and base["rows"]:
        print(f"  Detailed table at base slippage (0.10 %):")
        print_report(base["rows"], symbol, args, mode="  [slippage=0.10%]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    params = _build_params(args)
    step   = args.step if args.step > 0 else args.window

    cache_dir = Path("data/cache") / args.symbol
    if not cache_dir.exists() or not list(cache_dir.glob("*.parquet")):
        print(
            f"ERROR: No cached data for '{args.symbol}'.\n"
            f"Download with: python scripts/download_cache.py --symbols {args.symbol}"
        )
        sys.exit(1)

    print(f"\nLoading {args.symbol} ...", end="", flush=True)
    df = load_full_data(args.symbol)
    print(
        f" {len(df):,} bars  "
        f"({df['date'].min().date()} → {df['date'].max().date()})"
    )

    windows = build_windows(df, args.window, step)
    if not windows:
        total_days = (df["date"].max() - df["date"].min()).days
        print(
            f"ERROR: No windows fit within the data range ({total_days} calendar days).\n"
            f"Try a smaller --window value."
        )
        sys.exit(1)

    mode_label = " [carry]" if args.carry_positions else ""
    print(f"Windows: {len(windows)}  ({args.window}-day, step={step}d){mode_label}\n")

    if args.slippage_sweep:
        summaries = run_slippage_sweep(
            df, args.symbol, windows, params, args.warmup,
            args.account_size, args.portfolio_value, args.carry_positions,
            args.window_dd_cap,
        )
        print_slippage_sweep(summaries, args.symbol, args)
    else:
        if args.carry_positions:
            rows = run_carry(
                df, args.symbol, windows, params, args.warmup,
                args.account_size, args.portfolio_value, args.window_dd_cap,
            )
        else:
            rows = run_independent(
                df, args.symbol, windows, params, args.warmup,
                args.account_size, args.portfolio_value, args.window_dd_cap,
            )
        if not rows:
            print("No windows produced results. Adjust --window or --warmup.")
            sys.exit(1)
        print_report(rows, args.symbol, args)


if __name__ == "__main__":
    main()
