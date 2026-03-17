"""
scripts/backtest_portfolio.py
------------------------------
Grid-only portfolio backtest (NVTS + TXG) vs SPY buy-and-hold.

Uses exact live parameters from main.py.
Total allocated capital: $90,000 ($45k per symbol).

Requires cached 5-min data (run once with IBKR connected):
    python scripts/download_cache.py --symbols NVTS TXG --duration "252 D"

Usage:
    python scripts/backtest_portfolio.py
    python scripts/backtest_portfolio.py --warmup 150
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.backtester import BacktestConfig, BacktestEngine
from strategies.clf_grid import CLFGridStrategy
from utils.dashboard_stats import (
    compute_sharpe, compute_sortino,
    compute_max_drawdown, compute_var_dollars,
)

logging.basicConfig(level=logging.WARNING)
for _n in ("CLFGridStrategy", "DataFetcher", "backtester"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)


# -- Live grid configuration (mirrors main.py exactly) -------------------------
PORTFOLIO_VALUE = 180_000.0   # full portfolio, used for VaR scaling
NVTS_ALLOC      =  45_000.0
TXG_ALLOC       =  45_000.0
GRID_ALLOC      = NVTS_ALLOC + TXG_ALLOC   # $90,000


# BacktestEngine keys strategy_params by class __name__; two CLFGridStrategy
# instances need distinct subclasses to receive different params.
class _NVTSGrid(CLFGridStrategy):
    pass


class _TXGGrid(CLFGridStrategy):
    pass


_GRID_PARAMS: dict = {
    "_NVTSGrid": {
        "SYMBOL":         "NVTS",
        "GRID_RATIO":     1.007,
        "NUM_BUY_LEVELS": 20,
        "ACCOUNT_SIZE":   NVTS_ALLOC,
        "ATR_ADAPTIVE":   True,
        "EVENT_GUARD":    True,
        "MIN_EVAL_SECS":  0.0,
    },
    "_TXGGrid": {
        "SYMBOL":         "TXG",
        "GRID_RATIO":     1.010,
        "NUM_BUY_LEVELS": 15,
        "ACCOUNT_SIZE":   TXG_ALLOC,
        "ATR_ADAPTIVE":   False,
        "EVENT_GUARD":    True,
        "MIN_EVAL_SECS":  0.0,
    },
}


# -- Helpers -------------------------------------------------------------------

def _run_grid(warmup: int) -> object:
    config = BacktestConfig(
        initial_capital = GRID_ALLOC,
        portfolio_value = PORTFOLIO_VALUE,
        warmup_bars     = warmup,
    )
    return BacktestEngine(config).run(
        strategies      = [_NVTSGrid, _TXGGrid],
        symbols         = ["NVTS", "TXG"],
        strategy_params = _GRID_PARAMS,
    )


def _to_daily(equity: pd.Series) -> pd.Series:
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    return eq.resample("1D").last().ffill().pct_change().dropna()


def _fetch_spy(start, end) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        raw = yf.download(
            "SPY",
            start=pd.Timestamp(start).date(),
            end=pd.Timestamp(end).date(),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        return close.pct_change().dropna()
    except Exception:
        return None


def _per_symbol_pnl(trade_log: pd.DataFrame) -> dict:
    """Realised cash P&L per symbol (BUY = outflow, SELL = inflow)."""
    if trade_log.empty:
        return {}
    out = {}
    for sym, grp in trade_log.groupby("symbol"):
        cash = 0.0
        for _, row in grp.iterrows():
            sign  = 1.0 if row["action"] == "SELL" else -1.0
            cash += sign * row["price"] * row["qty"]
            cash -= row["commission"] + row["slippage"]
        out[str(sym)] = cash
    return out


def _fp(v: float, sign: bool = True) -> str:
    return f"{v * 100:+.2f}%" if sign else f"{v * 100:.2f}%"


def _fd(v: float) -> str:
    return f"${v:+,.0f}"


# -- Report --------------------------------------------------------------------

def print_report(result, spy_daily: Optional[pd.Series]) -> None:
    eq      = result.equity_curve.set_index("timestamp")["equity"]
    dr      = _to_daily(eq)
    start   = eq.index[0]
    end     = eq.index[-1]
    n_days  = max((end - start).days, 1)
    final   = eq.iloc[-1]
    pnl     = final - GRID_ALLOC
    ret     = pnl / GRID_ALLOC
    ann_ret = (final / GRID_ALLOC) ** (365.0 / n_days) - 1
    sharpe  = compute_sharpe(dr)
    sortino = compute_sortino(dr)

    eq_daily = eq.resample("1D").last().ffill().dropna()
    max_dd   = compute_max_drawdown(eq_daily)
    calmar   = ann_ret / max_dd if max_dd > 0 else float("inf")
    var_usd  = compute_var_dollars(dr, portfolio_value=PORTFOLIO_VALUE)
    n_trades = len(result.trade_log)

    # Per-symbol realised P&L
    sym_pnl = _per_symbol_pnl(result.trade_log)
    sym_cnt = (result.trade_log.groupby("symbol").size().to_dict()
               if not result.trade_log.empty else {})

    # SPY metrics
    spy_m: Optional[dict] = None
    if spy_daily is not None:
        spy_eq  = GRID_ALLOC * (1 + spy_daily).cumprod()
        spy_ret = spy_eq.iloc[-1] / GRID_ALLOC - 1
        spy_nd  = max((spy_daily.index[-1] - spy_daily.index[0]).days, 1)
        spy_ann = (1 + spy_ret) ** (365.0 / spy_nd) - 1
        spy_dd  = compute_max_drawdown(spy_eq)
        spy_m   = {
            "pnl":    spy_eq.iloc[-1] - GRID_ALLOC,
            "ret":    spy_ret,
            "ann":    spy_ann,
            "sharpe": compute_sharpe(spy_daily),
            "max_dd": spy_dd,
        }

    W = 72

    print()
    print("=" * W)
    print("  GRID PORTFOLIO BACKTEST  vs  SPY")
    print(f"  Period    : {pd.Timestamp(start).date()} -> {pd.Timestamp(end).date()}"
          f"  ({n_days} calendar days)")
    print(f"  Capital   : ${GRID_ALLOC:,.0f}  (NVTS $45k + TXG $45k)")
    print("=" * W)

    # Per-strategy row
    print(f"\n  {'Symbol':<12} {'Alloc':>10}  {'Trades':>6}  {'Realised P&L':>14}  {'Return':>8}")
    print("  " + "-" * 58)
    for sym, alloc, label in [("NVTS", NVTS_ALLOC, "NVTS Grid"),
                               ("TXG",  TXG_ALLOC,  "TXG Grid")]:
        p = sym_pnl.get(sym, 0.0)
        t = sym_cnt.get(sym, 0)
        print(f"  {label:<12} ${alloc:>9,.0f}  {t:>6}  {_fd(p):>10}  {_fp(p / alloc):>8}")

    # Note: realised P&L excludes open grid positions (conservative)
    print(f"\n  * Realised P&L = closed round trips only.")
    print(f"    Portfolio equity below includes open positions marked-to-market.")

    # Main comparison table
    def _row(label, port_val, spy_val, fmt):
        sv = fmt(spy_val) if spy_val is not None else "N/A"
        print(f"\n  {label:<28}  {fmt(port_val):>12}  {sv:>12}")

    print(f"\n  {'Metric':<28}  {'Grid Portfolio':>14}  {'SPY':>12}")
    print("  " + "-" * 58)

    _row("Total P&L",          pnl,     spy_m["pnl"] if spy_m else None, _fd)
    _row("Total Return",       ret,     spy_m["ret"] if spy_m else None, _fp)
    _row("Annualised Return",  ann_ret, spy_m["ann"] if spy_m else None, _fp)
    _row("Sharpe Ratio",       sharpe,  spy_m["sharpe"] if spy_m else None,
         lambda v: f"{v:.3f}")
    _row("Max Drawdown",       max_dd,  spy_m["max_dd"] if spy_m else None,
         lambda v: _fp(v, sign=False))

    calmar_str = f"{calmar:.2f}" if calmar != float("inf") else "inf"
    print(f"\n  {'Sortino Ratio':<28}  {sortino:>14.3f}")
    print(f"  {'Calmar Ratio':<28}  {calmar_str:>14}")
    var_display = max(0.0, var_usd)
    var_tag = "[OK]" if var_usd < 15_000 else "[!!]"
    print(f"  {'30d 99% VaR':<28}  ${var_display:>12,.0f}  {var_tag}")
    print(f"  {'Total Trades':<28}  {n_trades:>14}")

    print("\n  " + "-" * 58)

    if spy_m is not None:
        alpha = ann_ret - spy_m["ann"]
        tag   = "[+] OUTPERFORMS SPY" if alpha > 0 else "[-] UNDERPERFORMS SPY"
        print(f"\n  Alpha vs SPY (annualised)   :  {_fp(alpha):>8}  {tag}")
    else:
        print("\n  SPY data unavailable.  pip install yfinance")

    ok = var_usd < 15_000
    print(f"  VaR $15k limit             :  "
          f"{'[OK] within limit' if ok else '[!!] EXCEEDS LIMIT'}"
          f"  (${var_display:,.0f})")

    print()
    print("=" * W)
    print()


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid portfolio backtest (NVTS + TXG) vs SPY."
    )
    p.add_argument(
        "--warmup", type=int, default=120,
        help="Warmup bars before replay (default: 120).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("\nGrid portfolio backtest: NVTS + TXG")
    print("Running... (this takes a few seconds)\n")

    try:
        result = _run_grid(args.warmup)
    except FileNotFoundError as exc:
        print(f"Data not found: {exc}\n")
        print("Download data first (requires IBKR Gateway / TWS):")
        print('  python scripts/download_cache.py --symbols NVTS TXG --duration "252 D"')
        sys.exit(1)

    if result.equity_curve.empty:
        print("No trades generated - verify cached data covers the full period.")
        sys.exit(1)

    eq      = result.equity_curve.set_index("timestamp")["equity"]
    spy_ret = _fetch_spy(eq.index[0], eq.index[-1])
    print_report(result, spy_ret)


if __name__ == "__main__":
    main()
