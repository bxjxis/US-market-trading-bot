"""
utils/grid_screener.py
----------------------
Pure-computation module for scoring stocks as grid-trading candidates.

No I/O here — all functions accept DataFrames / Series and return plain
Python floats or DataFrames.  The CLI wrapper lives in
scripts/screen_grid_candidates.py.

Metrics computed
----------------
beta
    OLS slope of symbol daily log-returns vs SPY (252-day window).
    Range: 0 → ∞.  Target: 0.20 < beta < 1.00.

div_yield_pct
    Trailing-12-month cash dividends per share / current close × 100.
    Sourced externally (yfinance); NaN when unavailable.

atr_cov
    Coefficient of Variation of ATR-14 over the last 63 trading days
    (std / mean).  Lower = more stable oscillation = better for grids.
    Target: < 0.40.

atr_price_pct
    Mean ATR-14 / mean close × 100 over the last 63 trading days.
    This is the stock's average daily "oscillation budget" as a % of price.
    Must exceed (GRID_RATIO − 1) × 100 for grid levels to ever fill.
    Ideal: between 1× and 3× the grid step size.

adx_14
    Average Directional Index over the last 14 bars.
    Measures trend strength; lower ADX = more ranging market = safer for
    grid deployment.  Target: < 25.

avg_volume_k
    Mean daily share volume over the last 63 trading days, in thousands.
    Hard constraint: must be ≥ 500 (i.e. 500 K shares / day).
    Rationale: thinner markets mean grid limit orders move the price
    against you, turning theoretical slippage into realised slippage.
    A 500 K floor ensures the bot's orders stay well below 1 % of daily
    liquidity even with a generous account allocation.

last_price
    Most recent closing price (USD).
    Hard constraint: $5 ≤ price ≤ $500.
    • Below $5 → penny-stock territory; IBKR per-share commission
      ($0.0035/share) dominates the round-trip relative to the grid step.
    • Above $500 → position sizing becomes very coarse for smaller
      accounts (each grid level = large dollar jump per share).

Scoring
-------
Each symbol is ranked 1..N on every metric (lower rank = better for grid):
    rank_beta   ascending  (Beta closer to 0 → rank 1)
    rank_yield  descending (higher yield → rank 1; NaN goes last)
    rank_cov    ascending  (lower CoV → rank 1)
    rank_atr    by proximity to ideal band [1× step, 3× step]
    rank_adx    ascending  (lower ADX → rank 1)

avg_volume_k and last_price are pure hard-gate constraints and do NOT
contribute to rank scoring — once a stock passes the minimum threshold,
higher volume / different price do not make it a better grid candidate.

grid_score = sum of all rank columns (lower total = better grid candidate).

viable flag
    True when ALL hard constraints pass:
      • beta          < 1.0
      • atr_price_pct > (GRID_RATIO − 1) × 100   (grid can actually fill)
      • adx_14        < 25                         (not a trending stock)
      • avg_volume_k  ≥ 500                        (≥ 500 K shares / day)
      • last_price    ∈ [$5, $500]                 (commission-efficient range)
"""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# Individual metric functions
# ═════════════════════════════════════════════════════════════════════════════

def compute_beta(
    symbol_prices: pd.Series,
    bench_prices:  pd.Series,
) -> float:
    """
    252-day OLS Beta of symbol vs benchmark (typically SPY).

    Parameters
    ----------
    symbol_prices : Daily closing prices of the symbol.
    bench_prices  : Daily closing prices of the benchmark index.

    Returns
    -------
    float — Beta coefficient; nan if fewer than 30 overlapping observations.
    """
    sym  = np.log(symbol_prices / symbol_prices.shift(1)).dropna()
    bnch = np.log(bench_prices  / bench_prices.shift(1)).dropna()

    # Align on common dates
    aligned = pd.DataFrame({"sym": sym, "bench": bnch}).dropna()
    if len(aligned) < 30:
        return float("nan")

    cov_matrix = np.cov(aligned["sym"].values, aligned["bench"].values)
    var_bench  = cov_matrix[1, 1]
    if var_bench == 0 or math.isnan(var_bench):
        return float("nan")
    return float(cov_matrix[0, 1] / var_bench)


def _true_range(ohlcv: pd.DataFrame) -> pd.Series:
    """True Range series from a DataFrame with columns high, low, close."""
    high  = ohlcv["high"]
    low   = ohlcv["low"]
    prev  = ohlcv["close"].shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)


def _atr14(ohlcv: pd.DataFrame) -> pd.Series:
    """Wilder's ATR-14 (exponential moving average of True Range)."""
    return _true_range(ohlcv).ewm(alpha=1.0 / 14, adjust=False).mean()


def compute_atr_cov(ohlcv: pd.DataFrame, window: int = 63) -> float:
    """
    ATR Coefficient of Variation = std(ATR14) / mean(ATR14) over last
    `window` trading days.

    Lower values indicate a more stable oscillation rhythm, which is
    ideal for grid trading.  High CoV means the stock alternates between
    calm periods and violent spikes (earnings, sector rotation).

    Returns
    -------
    float ≥ 0; nan if the window has fewer than 14 data points.
    """
    atr  = _atr14(ohlcv).iloc[-window:]
    mean = float(atr.mean())
    std  = float(atr.std(ddof=1))
    if mean == 0 or math.isnan(mean):
        return float("nan")
    return std / mean


def compute_atr_price_pct(ohlcv: pd.DataFrame, window: int = 63) -> float:
    """
    Mean ATR-14 as a percentage of mean closing price, over the last
    `window` trading days.

    This is the stock's average daily "oscillation budget".  For a grid
    with step GRID_RATIO, the rule of thumb is:

        atr_price_pct > (GRID_RATIO − 1) × 100   →  grid levels can fill
        atr_price_pct > (GRID_RATIO − 1) × 300   →  too wide; one bar may
                                                      blow through 3+ levels

    Returns
    -------
    float ≥ 0; nan on empty input.
    """
    atr_tail   = _atr14(ohlcv).iloc[-window:]
    close_tail = ohlcv["close"].iloc[-window:]
    mean_close = float(close_tail.mean())
    if mean_close == 0 or math.isnan(mean_close):
        return float("nan")
    return float(atr_tail.mean() / mean_close * 100.0)


def compute_adx(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """
    Average Directional Index (ADX-14) — measures trend strength.

    ADX < 25  → ranging / sideways market  (safe for grid)
    ADX 25-50 → developing trend           (risky for grid)
    ADX > 50  → strong trend               (avoid)

    Returns
    -------
    float — last ADX value; nan if insufficient data.
    """
    high  = ohlcv["high"]
    low   = ohlcv["low"]
    alpha = 1.0 / period

    # Directional Movement
    up   = high.diff()
    down = -low.diff()
    dm_plus  = up.where((up > down) & (up > 0), 0.0)
    dm_minus = down.where((down > up) & (down > 0), 0.0)

    # Wilder EWM
    tr       = _true_range(ohlcv).ewm(alpha=alpha, adjust=False).mean()
    di_plus  = 100.0 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / tr
    di_minus = 100.0 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / tr

    dx_denom = (di_plus + di_minus).abs()
    dx = (100.0 * (di_plus - di_minus).abs() / dx_denom.replace(0, float("nan")))
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    last = adx.dropna()
    return float(last.iloc[-1]) if not last.empty else float("nan")


def compute_avg_volume_k(ohlcv: pd.DataFrame, window: int = 63) -> float:
    """
    Mean daily share volume over the last ``window`` trading days,
    expressed in thousands (divide by 1000 for readability).

    Hard constraint threshold: ≥ 500 (= 500 K shares / day).

    Returns
    -------
    float — mean volume in thousands; nan if volume column is missing or empty.
    """
    if "volume" not in ohlcv.columns:
        return float("nan")
    vol = ohlcv["volume"].iloc[-window:].dropna()
    if vol.empty:
        return float("nan")
    return float(vol.mean() / 1_000.0)


def compute_last_price(ohlcv: pd.DataFrame) -> float:
    """
    Most recent closing price from the OHLCV DataFrame.

    Hard constraint thresholds: $5 ≤ price ≤ $500.
      • < $5  → per-share commission dominates grid profit
      • > $500 → position sizing too coarse for small accounts

    Returns
    -------
    float — last close price in USD; nan if DataFrame is empty.
    """
    if ohlcv.empty or "close" not in ohlcv.columns:
        return float("nan")
    return float(ohlcv["close"].iloc[-1])


# ═════════════════════════════════════════════════════════════════════════════
# Aggregate scoring
# ═════════════════════════════════════════════════════════════════════════════

def score_candidates(
    records:          list,
    grid_ratio:       float = 1.015,
    min_volume_k:     float = 500.0,
    min_price:        float = 5.0,
    max_price:        float = 500.0,
) -> pd.DataFrame:
    """
    Given a list of per-symbol metric dicts, produce a ranked DataFrame.

    Parameters
    ----------
    records       : List of dicts with keys:
                      symbol, beta, div_yield_pct, atr_cov, atr_price_pct,
                      adx_14, avg_volume_k, last_price.
                    Missing or NaN values are handled gracefully.
    grid_ratio    : Geometric grid step (e.g. 1.015 = 1.5 % step).
                    Used to compute the ATR viability threshold.
    min_volume_k  : Minimum average daily volume in thousands (default 500 K).
    min_price     : Minimum close price in USD (default $5).
    max_price     : Maximum close price in USD (default $500).

    Returns
    -------
    DataFrame sorted by ``grid_score`` (ascending = best candidates first).
    Columns added: rank_beta, rank_yield, rank_cov, rank_atr, rank_adx,
                   grid_score, viable.
    """
    df = pd.DataFrame(records)

    step_pct      = (grid_ratio - 1.0) * 100.0   # e.g. 1.5 %
    ideal_atr_pct = step_pct * 2.0                # "sweet spot" centre

    # ── Per-metric ranks (NaN → placed last in ranking) ───────────────────
    df["rank_beta"] = df["beta"].rank(
        ascending=True, na_option="bottom"
    )
    df["rank_yield"] = df["div_yield_pct"].rank(
        ascending=False, na_option="bottom"
    )
    df["rank_cov"] = df["atr_cov"].rank(
        ascending=True, na_option="bottom"
    )
    # ATR/Price: rank by distance from ideal band centre
    df["_atr_dist"] = (df["atr_price_pct"] - ideal_atr_pct).abs()
    df["rank_atr"]  = df["_atr_dist"].rank(ascending=True, na_option="bottom")
    df["rank_adx"]  = df["adx_14"].rank(ascending=True, na_option="bottom")

    df["grid_score"] = (
        df["rank_beta"]
        + df["rank_yield"]
        + df["rank_cov"]
        + df["rank_atr"]
        + df["rank_adx"]
    )

    # ── Hard viability constraints (all five must pass) ────────────────────
    beta_ok  = df["beta"].lt(1.0).fillna(False)
    atr_ok   = df["atr_price_pct"].gt(step_pct).fillna(False)
    adx_ok   = df["adx_14"].lt(25.0).fillna(False)
    # Liquidity: ≥ min_volume_k thousand shares / day
    vol_ok   = df["avg_volume_k"].ge(min_volume_k).fillna(False)
    # Price in commission-efficient range [$min_price, $max_price]
    price_ok = (
        df["last_price"].ge(min_price).fillna(False)
        & df["last_price"].le(max_price).fillna(False)
    )

    df["viable"] = beta_ok & atr_ok & adx_ok & vol_ok & price_ok

    # ── Clean up internal column ───────────────────────────────────────────
    df.drop(columns=["_atr_dist"], inplace=True)

    return df.sort_values("grid_score").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Pretty-print
# ═════════════════════════════════════════════════════════════════════════════

def print_screen_results(
    df:           pd.DataFrame,
    grid_ratio:   float = 1.015,
    min_volume_k: float = 500.0,
    min_price:    float = 5.0,
    max_price:    float = 500.0,
    top_n:        int   = 20,
) -> None:
    """
    Print a formatted screening results table.

    Parameters
    ----------
    df           : Output of ``score_candidates()``.
    grid_ratio   : Used to label the viability threshold in the header.
    min_volume_k : Liquidity floor used to label the hard constraint.
    min_price    : Price floor used to label the hard constraint.
    max_price    : Price ceiling used to label the hard constraint.
    top_n        : Maximum number of rows to display.
    """
    step_pct = (grid_ratio - 1.0) * 100.0
    width    = 108

    header = (
        f"\n{'═' * width}\n"
        f"  GRID TRADING CANDIDATE SCREEN  "
        f"(step={step_pct:.1f}%  |  ATR>{step_pct:.1f}%  |  "
        f"Vol≥{min_volume_k:.0f}K/day  |  ${min_price:.0f}≤Price≤${max_price:.0f})\n"
        f"{'═' * width}\n"
        f"  {'#':<3}  {'Symbol':<7}  "
        f"{'Price':>7}  {'Vol(K)':>8}  "
        f"{'Beta':>6}  {'Div%':>6}  "
        f"{'ATR CoV':>8}  {'ATR/Px%':>8}  {'ADX':>6}  "
        f"{'Score':>6}  {'Viable':>7}\n"
        f"{'─' * width}"
    )
    print(header)

    def _fmt(val: float, fmt: str, na: str = "N/A") -> str:
        return format(val, fmt) if not math.isnan(val) else na

    display = df.head(top_n)
    for rank, row in enumerate(display.itertuples(), start=1):
        viable_str = "  YES ✓" if row.viable else "   NO ✗"
        print(
            f"  {rank:<3}  {row.symbol:<7}  "
            f"{_fmt(row.last_price,    '7.2f', '    N/A')}  "
            f"{_fmt(row.avg_volume_k,  '8,.0f', '     N/A')}  "
            f"{_fmt(row.beta,          '6.3f',  '   N/A')}  "
            f"{_fmt(row.div_yield_pct, '6.2f',  '   N/A')}  "
            f"{_fmt(row.atr_cov,       '8.3f',  '     N/A')}  "
            f"{_fmt(row.atr_price_pct, '8.3f',  '     N/A')}  "
            f"{_fmt(row.adx_14,        '6.1f',  '   N/A')}  "
            f"{row.grid_score:6.1f}  {viable_str}"
        )

    viable_syms = df[df["viable"]]["symbol"].tolist()
    print(f"{'─' * width}")
    if viable_syms:
        print(f"\n  Viable candidates ({len(viable_syms)}): {', '.join(viable_syms)}")
    else:
        print("\n  No symbols passed all hard constraints with current market data.")
    print(
        "\n  Hard constraints (ALL must pass for 'viable'):\n"
        f"    Beta     < 1.0           low systematic risk\n"
        f"    ATR/Px%  > {step_pct:.1f}%         daily oscillation covers the grid step\n"
        f"    ADX      < 25            ranging market, not trending\n"
        f"    Vol      ≥ {min_volume_k:,.0f}K shares/day  sufficient liquidity for limit orders\n"
        f"    Price    ${min_price:.0f} – ${max_price:.0f}         commission-efficient range\n"
        "\n  Soft ranking (lower score = better grid candidate overall):\n"
        "    ATR CoV  lower = more stable daily oscillation rhythm\n"
        "    Div%     higher = dividend buffer while holding stuck inventory\n"
    )
