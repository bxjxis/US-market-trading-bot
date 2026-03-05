"""
utils/dashboard_stats.py
------------------------
Portfolio performance statistics — all pure functions, no I/O.

All functions accept a pandas Series or DataFrame and return a float (or
dict).  They are safe to call from both the backtester and live dashboard.

Functions
---------
compute_sharpe       — Annualised Sharpe Ratio
compute_sortino      — Annualised Sortino Ratio (downside deviation)
compute_max_drawdown — Peak-to-trough max drawdown (fraction)
compute_calmar       — CAGR / Max-Drawdown (Calmar Ratio)
compute_var_pct      — Historical 99% VaR as a fraction of portfolio
compute_var_dollars  — Historical 99% VaR in dollars
compute_all          — All stats in one call, returns a dict
print_summary        — Pretty-print stats to stdout / logger

Usage
-----
    from utils.dashboard_stats import compute_all, print_summary
    from core.database import Database

    db  = Database()
    eq  = db.get_equity(run_id).set_index("timestamp")["equity"]
    dr  = eq.resample("1D").last().pct_change().dropna()

    stats = compute_all(eq, dr, portfolio_value=180_000)
    print_summary(stats)
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_TRADING_DAYS_PER_YEAR  = 252
_INTRADAY_BARS_PER_YEAR = 252 * 78   # 5-min bars in a US equity trading year
_VaR_CONFIDENCE         = 0.99
_VaR_WINDOW_DAYS        = 30


# ═════════════════════════════════════════════════════════════════════════════
# Core metric functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(
    returns:          pd.Series,
    risk_free:        float = 0.0,
    periods_per_year: int   = _TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Sharpe Ratio.

    Parameters
    ----------
    returns          : Periodic return series (daily, bar-level, etc.).
    risk_free        : Annual risk-free rate (e.g. 0.05 for 5 %).
    periods_per_year : Number of return periods in one calendar year.
                       Use 252 for daily, 252*78 for 5-min bars.

    Returns
    -------
    float — Sharpe Ratio; 0.0 if returns are empty or have zero std.
    """
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or math.isnan(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def compute_sortino(
    returns:          pd.Series,
    risk_free:        float = 0.0,
    periods_per_year: int   = _TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualised Sortino Ratio (penalises only downside volatility).

    Returns
    -------
    float — Sortino Ratio; inf if positive mean and zero downside std.
    """
    if returns.empty:
        return 0.0
    excess   = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    down_std = downside.std(ddof=1)
    if down_std == 0 or math.isnan(down_std):
        return 0.0
    return float(excess.mean() / down_std * math.sqrt(periods_per_year))


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a positive fraction.

    Example: 0.15 means the portfolio fell 15 % from its peak.

    Returns
    -------
    float in [0, 1]; 0.0 if the equity series is empty or monotonically rising.
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd      = drawdown.min()
    return float(-mdd) if not math.isnan(mdd) else 0.0


def compute_calmar(
    equity:           pd.Series,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calmar Ratio = CAGR / Max-Drawdown.

    Uses the number of rows in ``equity`` as the holding period; assumes
    each row represents one trading period (day by default).

    Returns
    -------
    float — Calmar Ratio; inf if CAGR > 0 and MDD = 0; 0.0 on error.
    """
    if equity.empty or len(equity) < 2:
        return 0.0

    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        return 0.0

    cagr = (end / start) ** (1.0 / years) - 1.0
    mdd  = compute_max_drawdown(equity)

    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / mdd)


def compute_var_pct(
    returns:    pd.Series,
    confidence: float = _VaR_CONFIDENCE,
    window:     int   = _VaR_WINDOW_DAYS,
) -> float:
    """
    Historical Value-at-Risk as a positive fraction of portfolio value.

    Uses the most recent ``window`` observations of ``returns``.
    A result of 0.05 means "in the worst 1 % of days over the last
    30 days, the portfolio lost ≥ 5 % of its value."

    Returns
    -------
    float ≥ 0; 0.0 if insufficient data.
    """
    if len(returns) < 2:
        return 0.0
    tail    = returns.iloc[-window:] if len(returns) >= window else returns
    quantile = tail.quantile(1.0 - confidence)
    return float(max(-quantile, 0.0))


def compute_var_dollars(
    returns:         pd.Series,
    portfolio_value: float = 180_000.0,
    confidence:      float = _VaR_CONFIDENCE,
    window:          int   = _VaR_WINDOW_DAYS,
) -> float:
    """
    30-day 99% Historical VaR in dollars.

    Scales the fractional VaR by ``portfolio_value`` so it represents the
    dollar loss on the full $180k portfolio regardless of the test-account size.

    Returns
    -------
    float — expected loss in dollars on the worst 1 % of days.
    """
    return compute_var_pct(returns, confidence, window) * portfolio_value


def compute_cagr(
    equity:           pd.Series,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compound Annual Growth Rate.

    Returns
    -------
    float — e.g. 0.12 means 12 % CAGR; 0.0 on error.
    """
    if equity.empty or len(equity) < 2:
        return 0.0
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = len(equity) / periods_per_year
    return float((end / start) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def compute_win_rate(trade_log: pd.DataFrame) -> float:
    """
    Fraction of completed round-trips that ended in profit.

    Matches BUY fills with the next SELL fill per symbol to form round-trips.
    Returns 0.0 if no completed round-trips exist.
    """
    if trade_log.empty:
        return 0.0

    wins = 0
    total = 0
    for symbol in trade_log["symbol"].unique():
        sym_trades = trade_log[trade_log["symbol"] == symbol].reset_index(drop=True)
        buys  = sym_trades[sym_trades["action"] == "BUY"].reset_index(drop=True)
        sells = sym_trades[sym_trades["action"] == "SELL"].reset_index(drop=True)
        for i in range(min(len(buys), len(sells))):
            total += 1
            buy_price  = buys.iloc[i]["price"]
            sell_price = sells.iloc[i]["price"]
            comm       = buys.iloc[i]["commission"] + sells.iloc[i]["commission"]
            net_pnl    = (sell_price - buy_price) * buys.iloc[i]["qty"] - comm
            if net_pnl > 0:
                wins += 1

    return float(wins / total) if total > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate helper
# ═════════════════════════════════════════════════════════════════════════════

def compute_all(
    equity:          pd.Series,
    daily_returns:   pd.Series,
    portfolio_value: float                = 180_000.0,
    trade_log:       Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Compute all performance metrics in one call.

    Parameters
    ----------
    equity          : Equity curve (any periodicity).
    daily_returns   : Daily return series resampled from the equity curve.
    portfolio_value : Full portfolio size for VaR dollar scaling.
    trade_log       : Optional trades DataFrame for win-rate calculation.

    Returns
    -------
    dict with keys: sharpe, sortino, max_drawdown, calmar, cagr,
                    var_pct, var_dollars, win_rate, num_trades.
    """
    sharpe  = compute_sharpe(daily_returns)
    sortino = compute_sortino(daily_returns)
    mdd     = compute_max_drawdown(equity)
    calmar  = compute_calmar(equity)
    cagr    = compute_cagr(equity)
    var_pct = compute_var_pct(daily_returns)
    var_usd = compute_var_dollars(daily_returns, portfolio_value)
    win_rt  = compute_win_rate(trade_log) if trade_log is not None else 0.0
    n_trades = len(trade_log) if trade_log is not None else 0

    return {
        "sharpe_ratio":  sharpe,
        "sortino_ratio": sortino,
        "max_drawdown":  mdd,
        "calmar_ratio":  calmar,
        "cagr":          cagr,
        "var_pct":       var_pct,
        "var_dollars":   var_usd,
        "win_rate":      win_rt,
        "num_trades":    n_trades,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Pretty-print
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(
    stats:           Dict[str, float],
    portfolio_value: float = 180_000.0,
    logger:          Optional[logging.Logger] = None,
) -> None:
    """
    Print (or log) a formatted performance summary.

    Parameters
    ----------
    stats           : Output of compute_all().
    portfolio_value : Used to label the VaR dollar figure.
    logger          : If provided, emit via logger.info(); else print().
    """
    var_flag = (
        "  *** EXCEEDS $15,000 LIMIT ***"
        if stats.get("var_dollars", 0) > 15_000
        else ""
    )
    lines = [
        "─" * 56,
        "  Portfolio Performance Summary",
        "─" * 56,
        f"  Trades        : {stats.get('num_trades', 0)}",
        f"  Win Rate      : {stats.get('win_rate', 0) * 100:.1f}%",
        f"  CAGR          : {stats.get('cagr', 0) * 100:.2f}%",
        f"  Sharpe Ratio  : {stats.get('sharpe_ratio', 0):.3f}",
        f"  Sortino Ratio : {stats.get('sortino_ratio', 0):.3f}",
        f"  Max Drawdown  : {stats.get('max_drawdown', 0) * 100:.2f}%",
        f"  Calmar Ratio  : {stats.get('calmar_ratio', 0):.3f}",
        f"  30d 99% VaR   : "
        f"{stats.get('var_pct', 0) * 100:.2f}% "
        f"(${stats.get('var_dollars', 0):,.0f} on ${portfolio_value:,.0f})"
        f"{var_flag}",
        "─" * 56,
    ]
    output = "\n".join(lines)
    if logger:
        logger.info("\n%s", output)
    else:
        print(output)


# ═════════════════════════════════════════════════════════════════════════════
# Live dashboard helper — query DB and print in one call
# ═════════════════════════════════════════════════════════════════════════════

def live_summary(
    run_id:          str,
    db,                              # core.database.Database instance
    portfolio_value: float = 180_000.0,
) -> Dict[str, float]:
    """
    Fetch the latest equity data for *run_id* from the database and print
    a performance summary.

    Returns the stats dict so callers can act on the values.
    """
    equity_df = db.get_equity(run_id)
    trade_df  = db.get_trades(run_id)

    if equity_df.empty:
        _log.warning("live_summary | no equity data for run %s", run_id)
        return {}

    eq = equity_df.set_index("timestamp")["equity"]
    eq.index = pd.to_datetime(eq.index)
    daily_returns = eq.resample("1D").last().pct_change().dropna()

    stats = compute_all(eq, daily_returns, portfolio_value, trade_log=trade_df)
    print_summary(stats, portfolio_value)
    return stats
