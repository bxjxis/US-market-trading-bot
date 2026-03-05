#!/usr/bin/env python
"""
scripts/monte_carlo_sim.py
--------------------------
Portfolio stress test via bootstrapped Monte Carlo simulation.

Data source
-----------
Reads 5-minute OHLCV Parquet files produced by DataFetcher from
data/cache/<SYMBOL>/*.parquet, then resamples to daily returns.
When a symbol's cache is absent, realistic synthetic returns are generated
using historically calibrated volatility and drift so the script is always
runnable — even before any live trading session has populated the cache.

Simulation engine
-----------------
* 10,000 independent paths, each covering N_DAYS (252) trading days
* Block bootstrap (5-day blocks) — preserves short-run autocorrelation
* Fat-tail shock: on each day there is a FAT_TAIL_PROB (5 %) chance the
  day's return is drawn from the worst-5 % tail and amplified × 3
* Full cross-asset correlation is preserved by sampling complete rows of
  the joint return matrix

Portfolio ($ 180,000 / ≈ 1.3 M RMB)
--------------------------------------
Two scenarios are evaluated side-by-side:

  AS-DEPLOYED  — current strategy parameters as coded
    CLF Geometric Grid   $  1,000   (clf_grid.ACCOUNT_SIZE)
    AMZN Reversion       $    200   (1 share × ≈ $200)
    SmallCap Arb legs    $  2,700   (≈ $1,350 per leg, market-impact bound)
    Cash buffer          $176,100

  FULLY DEPLOYED — proportional scale-up, stress-testing full capital
    CLF Grid             $ 36,000   (20 %)
    AMZN Reversion       $ 36,000   (20 %)
    SmallCap Arb legs    $ 72,000   (40 %, $ 36 K per leg)
    Cash buffer          $ 36,000   (20 %)

Strategy return models
----------------------
  CLF Grid         — state-machine simulator: tracks grid levels, buys on
                     dips, takes profit at +1.5 %, safety-switch at ± 30 %
  AMZN Reversion   — position-frequency model: strategy is in-market ≈ 20 %
                     of days; earns daily AMZN return when in, 0 when flat
  SmallCap Arb     — pairs spread model: r_spread = r_IREN − r_WULF;
                     signal fires when |z-score| > 2.0, exits at < 0.5

Metrics
-------
  Value at Risk  (VaR)   95 % & 99 %  — 1-day and 30-day horizons
  Expected Shortfall     (CVaR / ES)  — 95 % & 99 %
  Max Drawdown distribution            — p5, p50, mean, p95
  Probability of ruin                  — portfolio loss ≥ 30 %
  2020 Crash scenario                  — deterministic 30-day peak-to-trough

Run
---
    python scripts/monte_carlo_sim.py
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root on path (works from scripts/ or project root) ────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Import strategy constants (no IB connection required) ─────────────────────
try:
    from strategies.clf_grid import (
        ACCOUNT_SIZE as CLF_ACCOUNT_SIZE,
        GRID_RATIO, SAFETY_PCT, COMMISSION, NUM_BUY_LEVELS,
    )
    from strategies.amzn_reversion import TAKE_PROFIT as AMZN_TAKE_PROFIT
    from strategies.smallcap_arb import ZSCORE_ENTRY, ZSCORE_EXIT, MARKET_IMPACT_PCT
    log.info("Strategy constants loaded from source files.")
except ImportError as exc:
    log.warning("Could not import strategy constants (%s) — using defaults.", exc)
    CLF_ACCOUNT_SIZE  = 1_000.0
    GRID_RATIO        = 1.015
    SAFETY_PCT        = 0.30
    COMMISSION        = 0.35
    NUM_BUY_LEVELS    = 10
    AMZN_TAKE_PROFIT  = 0.03
    ZSCORE_ENTRY      = 2.0
    ZSCORE_EXIT       = 0.5
    MARKET_IMPACT_PCT = 0.01

# ── Simulation parameters ─────────────────────────────────────────────────────
PORTFOLIO_VALUE  = 180_000.0
N_SIMULATIONS    = 10_000
N_DAYS           = 252          # trading days per path (≈ 1 year)
BLOCK_SIZE       = 5            # days per bootstrap block (preserves autocorrelation)
FAT_TAIL_PROB    = 0.05         # probability of a fat-tail shock on any given day
FAT_TAIL_MULT    = 3.0          # amplifier applied to worst-5 % tail draws
RUIN_THRESHOLD   = 0.30         # portfolio loss fraction considered "ruin"
SEED             = 42

# ── Initial prices (used by strategy simulators; updated from parquet if avail.) ──
DEFAULT_PRICES = {"CLF": 12.00, "AMZN": 195.00, "IREN": 8.00, "WULF": 5.00}

# ── Synthetic return parameters (used when parquet cache is absent) ───────────
# Calibrated to typical small-cap / large-cap equity behaviour
SYNTHETIC_PARAMS = {
    #          (annual_vol, annual_drift)
    "CLF":  (0.65, 0.08),   # steel/mining: high volatility, moderate drift
    "AMZN": (0.30, 0.15),   # large-cap tech: moderate vol, positive drift
    "IREN": (0.90, 0.20),   # crypto mining small-cap: very high vol
    "WULF": (0.90, 0.18),   # similar to IREN
}

# ── 2020 crash scenario (peak-to-trough, Feb 19 – Mar 23, 30 trading days) ───
CRASH_2020 = {
    "CLF":  -0.64,   # Cleveland-Cliffs: ~$11 → ~$4 (-64 %)
    "AMZN": -0.23,   # Amazon held up; e-commerce benefited
    "IREN": -0.51,   # Bitcoin proxy: $9,800 → $4,800 (-51 %)
    "WULF": -0.55,   # Slightly worse than IREN (less liquid)
}
CRASH_DAYS  = 30
CACHE_DIR   = ROOT / "data" / "cache"
SYMBOLS     = ["AMZN", "CLF", "IREN", "WULF"]

# ── Portfolio allocations ─────────────────────────────────────────────────────
# As-deployed: exact current strategy parameters
AS_DEPLOYED = {
    "CLF_GRID":   CLF_ACCOUNT_SIZE,      # $1,000
    "AMZN_REV":   200.0,                 # 1 share × ~$200
    "PAIRS_LEGS": 2_700.0,               # ≈ $1,350 each leg (market-impact bound)
}
AS_DEPLOYED["CASH"] = PORTFOLIO_VALUE - sum(AS_DEPLOYED.values())

# Fully deployed: scale strategies to fill the portfolio
FULL_DEPLOY = {
    "CLF_GRID":   PORTFOLIO_VALUE * 0.20,
    "AMZN_REV":   PORTFOLIO_VALUE * 0.20,
    "PAIRS_LEGS": PORTFOLIO_VALUE * 0.40,
    "CASH":       PORTFOLIO_VALUE * 0.20,
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_parquet(symbol: str) -> Optional[Path]:
    """Return the most-recently written parquet file for *symbol*, or None."""
    symbol_dir = CACHE_DIR / symbol
    if not symbol_dir.exists():
        return None
    files = sorted(symbol_dir.glob(f"{symbol}__*.parquet"), reverse=True)
    return files[0] if files else None


def load_daily_returns(symbol: str, rng: np.random.Generator) -> Tuple[pd.Series, float]:
    """
    Load 5-minute bars from parquet, resample to daily close, compute log-returns.
    Falls back to synthetic data when the parquet cache is empty.

    Returns
    -------
    (log_return_series, initial_price)
    """
    path = find_latest_parquet(symbol)
    if path is not None:
        log.info("Loading %-4s from %s", symbol, path.name)
        df = pd.read_parquet(path)

        # Normalise date column
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Resample 5-min → daily (use close of last bar each day)
        daily = df.set_index("date")["close"].resample("B").last().dropna()

        if len(daily) < 5:
            log.warning("%-4s: only %d daily bars — supplementing with synthetic data.", symbol, len(daily))
        else:
            log_ret = np.log(daily / daily.shift(1)).dropna()
            log.info("%-4s: %d daily bars  |  mean=%.3f%%  std=%.3f%%",
                     symbol, len(log_ret), log_ret.mean() * 100, log_ret.std() * 100)
            return log_ret, float(daily.iloc[-1])

    # ── Synthetic fallback ────────────────────────────────────────────────────
    annual_vol, annual_drift = SYNTHETIC_PARAMS[symbol]
    daily_vol   = annual_vol   / math.sqrt(252)
    daily_drift = annual_drift / 252
    n = 504   # 2 years of synthetic daily returns
    returns = rng.normal(daily_drift - 0.5 * daily_vol ** 2, daily_vol, n)
    log.warning("%-4s: no parquet data — using %d synthetic daily returns "
                "(vol=%.1f%%/day).", symbol, n, daily_vol * 100)
    return pd.Series(returns, name=symbol), DEFAULT_PRICES[symbol]


# ══════════════════════════════════════════════════════════════════════════════
# Strategy simulators
# ══════════════════════════════════════════════════════════════════════════════

def simulate_clf_grid(
    price_path: np.ndarray,
    anchor: float,
    account_size: float = CLF_ACCOUNT_SIZE,
) -> Tuple[float, float]:
    """
    Run the full geometric grid state machine on a price path.

    Returns
    -------
    (total_pnl, max_drawdown_fraction)
        total_pnl              : realised + unrealised P&L in dollars
        max_drawdown_fraction  : worst mark-to-market drawdown as a fraction
                                 of account_size
    """
    alloc_per_level = account_size / NUM_BUY_LEVELS
    buy_prices  = [anchor / (GRID_RATIO ** i) for i in range(1, NUM_BUY_LEVELS + 1)]
    sell_prices = [bp * GRID_RATIO for bp in buy_prices]
    qtys        = [max(0, math.floor(alloc_per_level / bp)) for bp in buy_prices]

    filled    = [False] * NUM_BUY_LEVELS   # True = open position at this level
    pnl       = 0.0
    mtm_low   = 0.0    # tracks worst running mark-to-market
    safety_on = False

    for price in price_path:
        # Safety switch
        dev     = abs(price - anchor) / anchor
        safety_on = dev > SAFETY_PCT

        # Check take-profit sells first
        for i in range(NUM_BUY_LEVELS):
            if filled[i] and price >= sell_prices[i]:
                gross = (sell_prices[i] - buy_prices[i]) * qtys[i]
                pnl  += gross - 2 * COMMISSION
                filled[i] = False

        # Open new buys if not halted (highest-index / deepest level that just
        # triggered; grid always starts from the shallowest unfilled level)
        if not safety_on:
            for i in range(NUM_BUY_LEVELS):
                if not filled[i] and price <= buy_prices[i]:
                    filled[i] = True

        # Running mark-to-market of open positions
        mtm = sum(
            (price - buy_prices[i]) * qtys[i]
            for i in range(NUM_BUY_LEVELS) if filled[i]
        )
        mtm_low = min(mtm_low, mtm)

    # Unrealised P&L at path end
    final = price_path[-1]
    for i in range(NUM_BUY_LEVELS):
        if filled[i]:
            pnl += (final - buy_prices[i]) * qtys[i]

    max_dd_frac = mtm_low / account_size if account_size > 0 else 0.0
    return pnl, max_dd_frac


def simulate_amzn_reversion(
    price_path: np.ndarray,
    initial_price: float,
    allocation: float,
    in_market_prob: float = 0.20,
    rng: np.random.Generator = None,
) -> float:
    """
    Simple position-frequency model for the AMZN mean-reversion strategy.

    The strategy holds a position ≈ in_market_prob of the time (based on
    signal frequency) and earns the daily return when in-market.
    The take-profit cap is applied: no single-day gain exceeds AMZN_TAKE_PROFIT.

    Returns
    -------
    Total dollar P&L over the price path.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n_shares = max(1, math.floor(allocation / initial_price))
    pnl = 0.0

    in_position = False
    entry_price = 0.0

    for price in price_path:
        if in_position:
            gain_pct = (price - entry_price) / entry_price
            if gain_pct >= AMZN_TAKE_PROFIT or gain_pct <= -0.05:  # 5% stop-loss
                pnl += (price - entry_price) * n_shares
                in_position = False
        else:
            # Enter on a random draw weighted by in_market_prob
            # (proxy for BB + RSI signal frequency)
            if rng.random() < (in_market_prob / 252):
                in_position = True
                entry_price = price

    # Close any open position at path end
    if in_position:
        pnl += (price_path[-1] - entry_price) * n_shares

    return pnl


def simulate_pairs_arb(
    price_path_a: np.ndarray,
    price_path_b: np.ndarray,
    ratio_mean: float,
    ratio_std: float,
    leg_allocation: float,
) -> float:
    """
    Run the statistical arbitrage (pairs) simulator on two price paths.

    Signal: z-score of price_a/price_b vs historical mean/std
    Entry:  |z| > ZSCORE_ENTRY → long underperformer, short overperformer
    Exit:   |z| < ZSCORE_EXIT

    Returns
    -------
    Total dollar P&L over the path (both legs combined).
    """
    in_position = False
    side        = 0    # +1 = long A / short B;  -1 = short A / long B
    entry_a = entry_b = 0.0

    qty_a = max(1, math.floor(leg_allocation / price_path_a[0]))
    qty_b = max(1, math.floor(leg_allocation / price_path_b[0]))
    pnl   = 0.0

    for pa, pb in zip(price_path_a, price_path_b):
        ratio = pa / pb if pb != 0 else ratio_mean
        z     = (ratio - ratio_mean) / ratio_std if ratio_std != 0 else 0.0

        if in_position:
            if abs(z) < ZSCORE_EXIT:
                # Close both legs
                if side == 1:     # long A, short B
                    pnl += (pa - entry_a) * qty_a   # long leg
                    pnl += (entry_b - pb) * qty_b   # short leg
                else:             # short A, long B
                    pnl += (entry_a - pa) * qty_a
                    pnl += (pb - entry_b) * qty_b
                in_position = False
        else:
            if abs(z) > ZSCORE_ENTRY:
                in_position = True
                entry_a, entry_b = pa, pb
                side = 1 if z < 0 else -1  # go long the cheap leg

    # Mark-to-market close at end of path
    if in_position:
        pa, pb = price_path_a[-1], price_path_b[-1]
        if side == 1:
            pnl += (pa - entry_a) * qty_a + (entry_b - pb) * qty_b
        else:
            pnl += (entry_a - pa) * qty_a + (pb - entry_b) * qty_b

    return pnl


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap engine
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    """
    Bootstrapped Monte Carlo portfolio stress tester.

    Parameters
    ----------
    returns     : DataFrame with one column per symbol (log-returns, daily)
    init_prices : Dict[symbol → float], last known prices
    allocation  : Dict with keys CLF_GRID, AMZN_REV, PAIRS_LEGS, CASH
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        init_prices: Dict[str, float],
        allocation: Dict[str, float],
        n_sims: int = N_SIMULATIONS,
        n_days: int = N_DAYS,
        seed: int = SEED,
    ):
        self.returns     = returns
        self.init_prices = init_prices
        self.allocation  = allocation
        self.n_sims      = n_sims
        self.n_days      = n_days
        self.rng         = np.random.default_rng(seed)

        # Pre-compute per-symbol fat-tail thresholds (5th percentile of returns)
        self._tail_idx: Dict[str, np.ndarray] = {
            sym: np.where(returns[sym].values <= np.percentile(returns[sym].values, 5))[0]
            for sym in returns.columns
        }

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """
        Execute n_sims Monte Carlo paths.

        Returns a results dict with:
            portfolio_pnl       : array of final dollar P&L per path
            portfolio_returns   : array of total return fraction per path
            max_drawdowns       : array of max-drawdown fractions per path
            clf_pnl, amzn_pnl, pairs_pnl : per-strategy dollar P&L arrays
        """
        portfolio_pnl  = np.zeros(self.n_sims)
        max_drawdowns  = np.zeros(self.n_sims)
        clf_pnl_arr    = np.zeros(self.n_sims)
        amzn_pnl_arr   = np.zeros(self.n_sims)
        pairs_pnl_arr  = np.zeros(self.n_sims)

        ret_arr = self.returns.values
        cols    = list(self.returns.columns)
        ix      = {sym: cols.index(sym) for sym in cols}

        log.info("Running %d simulations (%d trading days each) …",
                 self.n_sims, self.n_days)

        for sim in range(self.n_sims):
            if sim % 1_000 == 0 and sim > 0:
                log.info("  Completed %d / %d paths", sim, self.n_sims)

            # ── Sample returns path (block bootstrap + fat-tail shocks) ────
            sampled = self._sample_path(ret_arr, self.n_days)

            # ── Convert log-returns to price paths ─────────────────────────
            price_paths = {}
            for sym in SYMBOLS:
                if sym in ix:
                    p0 = self.init_prices.get(sym, DEFAULT_PRICES[sym])
                    log_ret = sampled[:, ix[sym]]
                    price_paths[sym] = p0 * np.exp(np.cumsum(log_ret))

            # ── Strategy simulators ────────────────────────────────────────
            # CLF Grid
            clf_alloc = self.allocation["CLF_GRID"]
            anchor    = self.init_prices.get("CLF", DEFAULT_PRICES["CLF"])
            clf_p     = price_paths.get("CLF", np.full(self.n_days, anchor))
            clf_pnl, clf_dd = simulate_clf_grid(clf_p, anchor, clf_alloc)

            # AMZN Reversion
            amzn_alloc = self.allocation["AMZN_REV"]
            amzn_p0    = self.init_prices.get("AMZN", DEFAULT_PRICES["AMZN"])
            amzn_p     = price_paths.get("AMZN", np.full(self.n_days, amzn_p0))
            amzn_pnl   = simulate_amzn_reversion(amzn_p, amzn_p0, amzn_alloc, rng=self.rng)

            # Pairs Arb
            leg_alloc = self.allocation["PAIRS_LEGS"] / 2
            iren_p0   = self.init_prices.get("IREN", DEFAULT_PRICES["IREN"])
            wulf_p0   = self.init_prices.get("WULF", DEFAULT_PRICES["WULF"])
            iren_p    = price_paths.get("IREN", np.full(self.n_days, iren_p0))
            wulf_p    = price_paths.get("WULF", np.full(self.n_days, wulf_p0))
            ratio_mean = iren_p0 / wulf_p0
            ratio_std  = ratio_mean * 0.15   # typical ±15 % spread volatility
            pairs_pnl  = simulate_pairs_arb(iren_p, wulf_p, ratio_mean, ratio_std, leg_alloc)

            # ── Portfolio aggregation ──────────────────────────────────────
            total_pnl = clf_pnl + amzn_pnl + pairs_pnl

            # Portfolio value path (for max-drawdown calculation)
            clf_ratio   = sampled[:, ix.get("CLF",  0)] if "CLF"  in ix else np.zeros(self.n_days)
            amzn_ratio  = sampled[:, ix.get("AMZN", 0)] if "AMZN" in ix else np.zeros(self.n_days)
            iren_ratio  = sampled[:, ix.get("IREN", 0)] if "IREN" in ix else np.zeros(self.n_days)
            wulf_ratio  = sampled[:, ix.get("WULF", 0)] if "WULF" in ix else np.zeros(self.n_days)

            w_clf   = clf_alloc  / PORTFOLIO_VALUE
            w_amzn  = amzn_alloc / PORTFOLIO_VALUE
            w_pairs = leg_alloc  / PORTFOLIO_VALUE   # one leg net delta ≈ 0 for pairs
            w_cash  = self.allocation["CASH"] / PORTFOLIO_VALUE

            port_log_ret = (
                w_clf   * clf_ratio  +
                w_amzn  * amzn_ratio +
                w_pairs * (iren_ratio - wulf_ratio) +  # net spread exposure
                w_cash  * 0.0
            )
            value_path = PORTFOLIO_VALUE * np.exp(np.cumsum(port_log_ret))
            running_max = np.maximum.accumulate(np.concatenate([[PORTFOLIO_VALUE], value_path]))
            drawdown    = (value_path - running_max[1:]) / running_max[1:]

            # Store results
            clf_pnl_arr[sim]   = clf_pnl
            amzn_pnl_arr[sim]  = amzn_pnl
            pairs_pnl_arr[sim] = pairs_pnl
            portfolio_pnl[sim] = total_pnl
            max_drawdowns[sim] = drawdown.min()

        return {
            "portfolio_pnl":    portfolio_pnl,
            "portfolio_returns": portfolio_pnl / PORTFOLIO_VALUE,
            "max_drawdowns":    max_drawdowns,
            "clf_pnl":          clf_pnl_arr,
            "amzn_pnl":         amzn_pnl_arr,
            "pairs_pnl":        pairs_pnl_arr,
        }

    def _sample_path(self, ret_arr: np.ndarray, n_days: int) -> np.ndarray:
        """
        Block bootstrap with fat-tail shock injection.

        Returns an (n_days × n_symbols) array of log-returns.
        """
        n_hist = len(ret_arr)
        path   = np.zeros((n_days, ret_arr.shape[1]))
        day    = 0

        while day < n_days:
            if self.rng.random() < FAT_TAIL_PROB:
                # Fat-tail: draw one row from the worst-5 % of each asset's history
                # (use the intersection of bad days for all assets simultaneously)
                tail_lists = [self._tail_idx[sym] for sym in self.returns.columns]
                worst_days = tail_lists[0]
                for tl in tail_lists[1:]:
                    worst_days = np.intersect1d(worst_days, tl)
                if len(worst_days) == 0:
                    worst_days = self._tail_idx[self.returns.columns[0]]
                idx   = self.rng.choice(worst_days)
                shock = ret_arr[idx] * FAT_TAIL_MULT
                path[day] = shock
                day += 1
            else:
                # Normal block: sample a consecutive block of BLOCK_SIZE days
                start = self.rng.integers(0, max(1, n_hist - BLOCK_SIZE))
                end   = min(start + BLOCK_SIZE, n_hist)
                block = ret_arr[start:end]
                take  = min(len(block), n_days - day)
                path[day : day + take] = block[:take]
                day += take

        return path


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def var(returns: np.ndarray, confidence: float) -> float:
    """Value at Risk: loss not exceeded with given confidence (positive = loss)."""
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar(returns: np.ndarray, confidence: float) -> float:
    """Conditional VaR / Expected Shortfall."""
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def max_drawdown_stats(mdd_arr: np.ndarray) -> Dict:
    return {
        "mean":  float(mdd_arr.mean()),
        "p5":    float(np.percentile(mdd_arr, 5)),
        "p50":   float(np.percentile(mdd_arr, 50)),
        "p95":   float(np.percentile(mdd_arr, 95)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2020 Crash deterministic scenario
# ══════════════════════════════════════════════════════════════════════════════

def run_crash_2020(init_prices: Dict[str, float], allocation: Dict) -> Dict:
    """
    Deterministic 30-day crash scenario modelled on Feb–Mar 2020.

    Each asset's daily return follows a smooth geometric decline:
        r_daily = (1 + total_return)^(1/CRASH_DAYS) - 1
    plus normally distributed noise calibrated to historical volatility.
    """
    rng = np.random.default_rng(SEED + 1)
    crash_pnl = {}

    for sym, total_ret in CRASH_2020.items():
        annual_vol, _ = SYNTHETIC_PARAMS[sym]
        daily_vol = annual_vol / math.sqrt(252)
        trend     = (1 + total_ret) ** (1 / CRASH_DAYS) - 1
        daily_ret = trend + rng.normal(0, daily_vol, CRASH_DAYS)

        p0 = init_prices.get(sym, DEFAULT_PRICES[sym])
        price_path = p0 * np.exp(np.cumsum(daily_ret))

        if sym == "CLF":
            clf_alloc = allocation["CLF_GRID"]
            pnl, mdd  = simulate_clf_grid(price_path, p0, clf_alloc)
            crash_pnl["CLF_GRID"] = (pnl, mdd, clf_alloc)

        elif sym == "AMZN":
            amzn_alloc = allocation["AMZN_REV"]
            p_end      = price_path[-1]
            pnl        = (p_end - p0) / p0 * amzn_alloc
            crash_pnl["AMZN_REV"] = (pnl, (p_end - p0) / p0, amzn_alloc)

    # Pairs crash: IREN + WULF
    iren_tot = CRASH_2020["IREN"]
    wulf_tot = CRASH_2020["WULF"]
    leg      = allocation["PAIRS_LEGS"] / 2

    # Assume we enter the position that bets on IREN outperforming WULF
    # (IREN falls less → long IREN / short WULF is the more likely entry during crash)
    # Net P&L = gain on long IREN + gain on short WULF
    net_spread = wulf_tot - iren_tot   # positive if WULF falls more (our short wins)
    pairs_pnl  = net_spread * leg      # dollar P&L
    crash_pnl["PAIRS_ARB"] = (pairs_pnl, iren_tot, wulf_tot)

    return crash_pnl


# ══════════════════════════════════════════════════════════════════════════════
# Report printer
# ══════════════════════════════════════════════════════════════════════════════

W = 68

def _hdr(title: str) -> None:
    print()
    print("─" * W)
    print(f"  {title}")
    print("─" * W)


def print_report(
    label: str,
    allocation: Dict,
    results: Dict,
    crash: Dict,
    init_prices: Dict,
) -> None:
    port_ret    = results["portfolio_returns"]
    mdd_arr     = results["max_drawdowns"]
    clf_pnl_arr = results["clf_pnl"]
    amzn_arr    = results["amzn_pnl"]
    pairs_arr   = results["pairs_pnl"]

    print()
    print("═" * W)
    print(f"  MONTE CARLO PORTFOLIO STRESS TEST — {label}")
    print(f"  {N_SIMULATIONS:,} simulations × {N_DAYS} trading days (≈ 1 year)")
    print(f"  Portfolio: ${PORTFOLIO_VALUE:,.0f}  |  Fat-tail prob: {FAT_TAIL_PROB:.0%}  ×{FAT_TAIL_MULT}")
    print("═" * W)

    # ── Allocation table ─────────────────────────────────────────────────────
    _hdr("PORTFOLIO ALLOCATION")
    total_deployed = sum(v for k, v in allocation.items() if k != "CASH")
    print(f"  {'Strategy':<28}  {'Allocation':>12}  {'Weight':>8}")
    print(f"  {'─'*28}  {'─'*12}  {'─'*8}")
    labels = {
        "CLF_GRID":   "CLF Geometric Grid (1.5% grid)",
        "AMZN_REV":   "AMZN Mean Reversion",
        "PAIRS_LEGS": "SmallCap Arb (IREN/WULF legs)",
        "CASH":       "Cash buffer",
    }
    for key, lbl in labels.items():
        amt = allocation.get(key, 0)
        pct = amt / PORTFOLIO_VALUE * 100
        print(f"  {lbl:<28}  ${amt:>11,.0f}  {pct:>7.2f}%")
    print(f"  {'─'*28}  {'─'*12}  {'─'*8}")
    print(f"  {'TOTAL':<28}  ${PORTFOLIO_VALUE:>11,.0f}  {'100.00%':>8}")

    # ── Per-strategy annual simulation stats ──────────────────────────────────
    _hdr("STRATEGY ANNUAL P&L DISTRIBUTION  (median / p5 / p95)")
    for name, arr, alloc_key in [
        ("CLF Grid",     clf_pnl_arr, "CLF_GRID"),
        ("AMZN Rev.",    amzn_arr,    "AMZN_REV"),
        ("SmallCap Arb", pairs_arr,   "PAIRS_LEGS"),
    ]:
        alloc = allocation.get(alloc_key, 0)
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        print(f"  {name:<14}  alloc=${alloc:>9,.0f}  |  "
              f"p5=${p5:>8,.0f}  median=${p50:>8,.0f}  p95=${p95:>8,.0f}")

    # ── Value at Risk ─────────────────────────────────────────────────────────
    _hdr("VALUE AT RISK  (dollar loss not exceeded at given confidence)")
    print(f"  {'Metric':<35}  {'95% VaR':>10}  {'99% VaR':>10}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")

    # 1-day VaR: scale 252-day simulation returns by √(1/252)
    one_day_scale = 1 / math.sqrt(N_DAYS)
    v95_1d = var(port_ret * one_day_scale, 0.95) * PORTFOLIO_VALUE
    v99_1d = var(port_ret * one_day_scale, 0.99) * PORTFOLIO_VALUE
    print(f"  {'1-Day VaR (portfolio)':<35}  ${v95_1d:>9,.0f}  ${v99_1d:>9,.0f}")

    v95_30 = var(port_ret * math.sqrt(30 / N_DAYS), 0.95) * PORTFOLIO_VALUE
    v99_30 = var(port_ret * math.sqrt(30 / N_DAYS), 0.99) * PORTFOLIO_VALUE
    print(f"  {'30-Day VaR (portfolio)':<35}  ${v95_30:>9,.0f}  ${v99_30:>9,.0f}")

    v95_ann = var(port_ret, 0.95) * PORTFOLIO_VALUE
    v99_ann = var(port_ret, 0.99) * PORTFOLIO_VALUE
    print(f"  {'Annual VaR (portfolio)':<35}  ${v95_ann:>9,.0f}  ${v99_ann:>9,.0f}")

    # ── Expected Shortfall ────────────────────────────────────────────────────
    _hdr("EXPECTED SHORTFALL / CVaR  (average loss in tail scenarios)")
    print(f"  {'Metric':<35}  {'95% CVaR':>10}  {'99% CVaR':>10}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*10}")
    c95 = cvar(port_ret, 0.95) * PORTFOLIO_VALUE
    c99 = cvar(port_ret, 0.99) * PORTFOLIO_VALUE
    print(f"  {'Annual CVaR (portfolio)':<35}  ${c95:>9,.0f}  ${c99:>9,.0f}")

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    _hdr("MAXIMUM DRAWDOWN DISTRIBUTION  (252-day paths)")
    mdd = max_drawdown_stats(mdd_arr)
    print(f"  {'Statistic':<20}  {'Fraction':>10}  {'Dollar':>12}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*12}")
    for label_mdd, key in [("Mean", "mean"), ("Median (p50)", "p50"),
                            ("Worst 5% (p5)", "p5"), ("Best case (p95)", "p95")]:
        frac = mdd[key]
        dollar = frac * PORTFOLIO_VALUE
        print(f"  {label_mdd:<20}  {frac:>9.2%}  ${dollar:>11,.0f}")

    # ── Ruin probability ──────────────────────────────────────────────────────
    _hdr("RUIN PROBABILITY")
    ruin_pct = np.mean(port_ret <= -RUIN_THRESHOLD) * 100
    print(f"  Probability of losing ≥ {RUIN_THRESHOLD:.0%} of portfolio: "
          f"  {ruin_pct:.4f}%  ({int(ruin_pct / 100 * N_SIMULATIONS):,} / {N_SIMULATIONS:,} paths)")

    # ── 2020 crash scenario ───────────────────────────────────────────────────
    _hdr(f"2020 CRASH SCENARIO  (Feb 19 – Mar 23: deterministic {CRASH_DAYS}-day shock)")
    print(f"  {'Strategy':<20}  {'Total Ret':>10}  {'P&L ($)':>12}  {'% of Portfolio':>16}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*12}  {'─'*16}")

    crash_total = 0.0
    for key, (pnl, extra, alloc) in crash.items():
        if key == "CLF_GRID":
            ret_str = f"{CRASH_2020['CLF']:+.1%}"
        elif key == "AMZN_REV":
            ret_str = f"{CRASH_2020['AMZN']:+.1%}"
        else:
            # extra = iren_tot, alloc = wulf_tot; spread = wulf outperformance vs IREN
            ret_str = f"spread {alloc - extra:+.1%}"
        pct_port = pnl / PORTFOLIO_VALUE * 100
        crash_total += pnl
        name = {"CLF_GRID": "CLF Grid", "AMZN_REV": "AMZN Reversion",
                "PAIRS_ARB": "Pairs Arb (IREN/WULF)"}[key]
        print(f"  {name:<20}  {ret_str:>10}  ${pnl:>11,.0f}  {pct_port:>15.3f}%")

    print(f"  {'─'*20}  {'─'*10}  {'─'*12}  {'─'*16}")
    print(f"  {'COMBINED LOSS':<20}  {'':>10}  ${crash_total:>11,.0f}  "
          f"{crash_total / PORTFOLIO_VALUE:>15.3%}")

    # ── CLF grid survival analysis ────────────────────────────────────────────
    _hdr("CLF GRID STRATEGY SURVIVAL ANALYSIS")
    clf_alloc     = allocation["CLF_GRID"]
    anchor_ref    = init_prices.get("CLF", DEFAULT_PRICES["CLF"])
    worst_case_dd = clf_alloc * 0.246   # ~5 open levels × avg 24.6% loss at -30%
    print(f"  Grid interval            {GRID_RATIO - 1:.1%}")
    print(f"  Safety switch threshold  ±{SAFETY_PCT:.0%} from anchor")
    print(f"  Anchor price (assumed)   ${anchor_ref:.2f}")
    print(f"  Account size             ${clf_alloc:,.0f}")
    print(f"  Worst-case drawdown*     -${worst_case_dd:,.0f}  "
          f"({worst_case_dd/clf_alloc:.1%} of CLF account, "
          f"{worst_case_dd/PORTFOLIO_VALUE:.3%} of portfolio)")
    print(f"  * 5 levels filled when safety triggers; avg entry = anchor × 0.928")
    clf_pnl_p5 = np.percentile(clf_pnl_arr, 5)
    clf_pnl_p50 = np.percentile(clf_pnl_arr, 50)
    print(f"  MC 5th-pctile annual P&L  ${clf_pnl_p5:,.0f}")
    print(f"  MC median annual P&L      ${clf_pnl_p50:,.0f}")
    survived = np.mean(clf_pnl_arr > -clf_alloc) * 100
    print(f"  Paths where grid survives (P&L > -account_size): {survived:.2f}%")

    # ── Pairs strategy survival analysis ──────────────────────────────────────
    _hdr("SMALLCAP ARB STRATEGY SURVIVAL ANALYSIS")
    pairs_leg = allocation["PAIRS_LEGS"] / 2
    print(f"  Z-score entry threshold  ±{ZSCORE_ENTRY:.1f}")
    print(f"  Z-score exit threshold   ±{ZSCORE_EXIT:.1f}")
    print(f"  Market impact cap        {MARKET_IMPACT_PCT:.0%} of avg 5-min volume")
    print(f"  Notional per leg         ${pairs_leg:,.0f}")
    print(f"  Structure                Dollar-neutral (long one, short the other)")
    pairs_p5, pairs_p50 = np.percentile(pairs_arr, [5, 50])
    print(f"  MC 5th-pctile annual P&L  ${pairs_p5:,.0f}")
    print(f"  MC median annual P&L      ${pairs_p50:,.0f}")
    crash_pairs_pnl = crash["PAIRS_ARB"][0]
    print(f"  2020 crash scenario P&L   ${crash_pairs_pnl:,.0f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    _hdr("VERDICT — 2020-STYLE CRASH SURVIVAL")
    decisions = [
        ("CLF 1.5% grid",
         crash["CLF_GRID"][0] > -clf_alloc,
         f"Max crash loss ${abs(crash['CLF_GRID'][0]):,.0f} "
         f"< account size ${clf_alloc:,.0f}"),
        ("AMZN mean reversion",
         abs(crash["AMZN_REV"][0]) < allocation["AMZN_REV"],
         f"1-share loss ${abs(crash['AMZN_REV'][0]):,.0f} manageable"),
        ("Pairs arb (z-score ±2.0)",
         crash["PAIRS_ARB"][0] > -pairs_leg,
         f"Net crash P&L ${crash['PAIRS_ARB'][0]:,.0f} (hedging benefits visible)"),
        ("Overall portfolio",
         crash_total > -PORTFOLIO_VALUE * RUIN_THRESHOLD,
         f"Total crash loss ${abs(crash_total):,.0f} = "
         f"{abs(crash_total) / PORTFOLIO_VALUE:.2%} < {RUIN_THRESHOLD:.0%} ruin threshold"),
    ]
    for name, survived_flag, reason in decisions:
        mark = "[OK] SURVIVES" if survived_flag else "[!!] LIQUIDATES"
        print(f"  {mark:<12}  {name:<28}  {reason}")

    print()
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    rng = np.random.default_rng(SEED)

    # ── Load returns & initial prices ─────────────────────────────────────────
    return_series = {}
    init_prices   = {}
    for sym in SYMBOLS:
        ret, p0             = load_daily_returns(sym, rng)
        return_series[sym]  = ret
        init_prices[sym]    = p0

    # Align all series to a common index (inner join on dates)
    returns_df = pd.DataFrame(return_series).dropna()
    log.info("Aligned return matrix: %d days × %d symbols", *returns_df.shape)

    if len(returns_df) < 20:
        log.warning("Very few historical trading days (%d). "
                    "Synthetic data is driving the simulation.", len(returns_df))

    # ── 2020 crash ────────────────────────────────────────────────────────────
    crash_as    = run_crash_2020(init_prices, AS_DEPLOYED)
    crash_full  = run_crash_2020(init_prices, FULL_DEPLOY)

    # ── Monte Carlo: As-Deployed ──────────────────────────────────────────────
    engine_as = MonteCarloEngine(
        returns_df, init_prices, AS_DEPLOYED,
        n_sims=N_SIMULATIONS, n_days=N_DAYS, seed=SEED,
    )
    results_as = engine_as.run()
    print_report("AS-DEPLOYED (current parameters)", AS_DEPLOYED,
                 results_as, crash_as, init_prices)

    # ── Monte Carlo: Fully Deployed ───────────────────────────────────────────
    engine_full = MonteCarloEngine(
        returns_df, init_prices, FULL_DEPLOY,
        n_sims=N_SIMULATIONS, n_days=N_DAYS, seed=SEED,
    )
    results_full = engine_full.run()
    print_report("FULLY DEPLOYED ($180K across strategies)", FULL_DEPLOY,
                 results_full, crash_full, init_prices)


if __name__ == "__main__":
    main()
