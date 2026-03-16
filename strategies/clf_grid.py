"""
strategies/clf_grid.py
----------------------
Geometric Grid strategy for CLF (Cleveland-Cliffs Inc.)

Grid mechanics
--------------
* All buy levels are placed below the anchor (current market price at startup).
* Formula: buy_level[i] = anchor / GRID_RATIO^i
* When a buy fills → a take-profit sell is placed at fill_price * GRID_RATIO
* When a sell fills → the realised P&L is logged (re-entry is opt-in, see _on_sell_fill)

Safety switch
-------------
If the market price deviates > SAFETY_PCT (30%) from the anchor in either
direction, no NEW orders are submitted. Existing GTC orders stay live but can
be cancelled manually if desired. The switch resets automatically when price
returns inside the safe band.

Commission analysis
-------------------
On startup, every grid level is evaluated for viability against the
approximate IBKR Tiered commission of $0.35 per side ($0.70 round-trip).

Params override
---------------
Pass a ``params`` dict to override any module-level constant at instantiation.
This allows the backtester and Optuna optimizer to sweep parameters without
modifying this file.  Example::

    CLFGridStrategy(ib, params={"GRID_RATIO": 1.02, "ACCOUNT_SIZE": 72_000})
"""

import asyncio
import collections
import math
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from typing import Dict, Optional, Set

import numpy as np

from ib_insync import IB, Stock, Ticker, Trade

from .base import BaseStrategy
from utils.data_fetcher import DataFetcher

# ── Module-level defaults (used when no params dict is supplied) ───────────────
GRID_RATIO: float       = 1.015       # 1.5% geometric step
SAFETY_PCT: float       = 0.30        # halt new orders if |price - anchor| / anchor > 30 %
COMMISSION: float       = 0.35        # approx IBKR Tiered, per side ($)
ACCOUNT_SIZE: float     = 1_000.0     # USD allocated to this strategy
NUM_BUY_LEVELS: int     = 10          # grid levels below anchor to pre-place
ADX_HALT: float         = 0.0         # legacy single-threshold (0 = disabled)
ADX_RESUME: float       = 25.0        # legacy: resume threshold
ADX_PERIOD: int         = 14          # Wilder's ADX lookback
# ── Tiered adaptive mode (replaces ADX_HALT when ADX_TIERED=True) ─────────────
ADX_TIERED: bool        = False       # enable three-tier adaptive mode
ADX_STABLE_THR: float   = 25.0       # ADX > this → stable mode
ADX_TREND_THR: float    = 35.0       # ADX > this → slope-sensitive zone
ADX_HALT_THR: float     = 45.0       # ADX > this → circuit breaker (always halt)
ADX_SLOPE_BARS: int     = 5          # bars over which to measure ADX slope
ADX_STABLE_RATIO: float = 1.012      # grid_ratio in stable mode (wider = safer)
ADX_STABLE_QTY_SCALE: float = 0.5   # position scale in stable mode
ADX_ULTRA_THR: float    = 20.0       # ADX below this → ultra-aggressive (pyramid) mode
ADX_ULTRA_QTY_SCALE: float = 2.0    # position scale in ultra mode (double size)
ADX_TREND_RATIO: float     = 1.015  # grid ratio in trending zone (ADX_TREND_THR..ADX_HALT_THR, rising slope)
ADX_TREND_QTY_SCALE: float = 0.25  # qty scale in trending zone (wide grid, tiny size)
# ── ATR-adaptive grid spacing (bidirectional, Guasoni 2012) ───────────────────
ATR_ADAPTIVE: bool        = False   # enable ATR-driven grid ratio adjustment
ATR_GRID_COVERAGE: float  = 0.5    # low-vol warmup: fraction of 1 ATR per grid step
ATR_WIDEN_MAX: float      = 1.8    # max widening: GRID_RATIO * this (e.g. 1.8 = up to 80% wider)
ATR_LONG_PERIOD: int      = 50     # bars to estimate long-term ATR baseline
# ── Event calendar guard (earnings / ex-dividend) ─────────────────────────────
EVENT_GUARD: bool         = False   # cancel pending buys on earnings / ex-div days
EVENT_REANCHOR: bool      = False   # rebuild grid from current price if price drifted >5% post-event
# ── Loss circuit breaker ───────────────────────────────────────────────────────
MAX_OPEN_LOSS: float    = 0.0        # cancel all buys when unrealized loss > this $ (0=disabled)
# ── ATR-based position sizing (replaces capital-allocation sizing) ─────────────
ATR_SIZING: bool        = False      # size qty by ATR risk rather than capital allocation
ATR_RISK_PCT: float     = 0.5        # % of (ACCOUNT_SIZE/NUM_BUY_LEVELS) to risk per 1 ATR move


@dataclass
class GridLevel:
    """Tracks the state of a single buy/sell pair in the grid."""
    index: int
    buy_price: float
    sell_price: float
    quantity: int
    buy_trade: Optional[Trade] = field(default=None, repr=False)
    sell_trade: Optional[Trade] = field(default=None, repr=False)


class CLFGridStrategy(BaseStrategy):
    """Geometric grid strategy on CLF with configurable interval and safety switch."""

    def __init__(self, ib: IB, account: str = "", params: dict = None):
        p = params or {}
        symbol = str(p.get("SYMBOL", "CLF"))
        super().__init__(ib, Stock(symbol, "SMART", "USD"), account)
        self._fetcher = DataFetcher(ib)
        self.anchor_price: Optional[float] = None
        self.levels: Dict[int, GridLevel] = {}
        self._ticker: Optional[Ticker] = None
        self._halted: bool = False
        self._trend_halted: bool = False

        # ── Configurable parameters (override via params dict) ─────────────────
        self.GRID_RATIO     = float(p.get("GRID_RATIO",     GRID_RATIO))
        self.SAFETY_PCT     = float(p.get("SAFETY_PCT",     SAFETY_PCT))
        self.COMMISSION     = float(p.get("COMMISSION",     COMMISSION))
        self.ACCOUNT_SIZE   = float(p.get("ACCOUNT_SIZE",   ACCOUNT_SIZE))
        self.NUM_BUY_LEVELS = int(p.get("NUM_BUY_LEVELS",   NUM_BUY_LEVELS))
        self.ADX_HALT            = float(p.get("ADX_HALT",            ADX_HALT))
        self.ADX_RESUME          = float(p.get("ADX_RESUME",          ADX_RESUME))
        self.ADX_PERIOD          = int(p.get("ADX_PERIOD",            ADX_PERIOD))
        self.ADX_TIERED          = bool(p.get("ADX_TIERED",           ADX_TIERED))
        self.ADX_STABLE_THR      = float(p.get("ADX_STABLE_THR",      ADX_STABLE_THR))
        self.ADX_TREND_THR       = float(p.get("ADX_TREND_THR",       ADX_TREND_THR))
        self.ADX_HALT_THR        = float(p.get("ADX_HALT_THR",        ADX_HALT_THR))
        self.ADX_SLOPE_BARS      = int(p.get("ADX_SLOPE_BARS",        ADX_SLOPE_BARS))
        self.ADX_STABLE_RATIO    = float(p.get("ADX_STABLE_RATIO",    ADX_STABLE_RATIO))
        self.ADX_STABLE_QTY_SCALE = float(p.get("ADX_STABLE_QTY_SCALE", ADX_STABLE_QTY_SCALE))
        self.ADX_ULTRA_THR       = float(p.get("ADX_ULTRA_THR",       ADX_ULTRA_THR))
        self.ADX_ULTRA_QTY_SCALE = float(p.get("ADX_ULTRA_QTY_SCALE", ADX_ULTRA_QTY_SCALE))
        self.ADX_TREND_RATIO     = float(p.get("ADX_TREND_RATIO",     ADX_TREND_RATIO))
        self.ADX_TREND_QTY_SCALE = float(p.get("ADX_TREND_QTY_SCALE", ADX_TREND_QTY_SCALE))
        self.MAX_OPEN_LOSS       = float(p.get("MAX_OPEN_LOSS",       MAX_OPEN_LOSS))
        self.ATR_SIZING          = bool(p.get("ATR_SIZING",           ATR_SIZING))
        self.ATR_RISK_PCT        = float(p.get("ATR_RISK_PCT",        ATR_RISK_PCT))
        self.ATR_ADAPTIVE        = bool(p.get("ATR_ADAPTIVE",         ATR_ADAPTIVE))
        self.ATR_GRID_COVERAGE   = float(p.get("ATR_GRID_COVERAGE",   ATR_GRID_COVERAGE))
        self.ATR_WIDEN_MAX       = float(p.get("ATR_WIDEN_MAX",       ATR_WIDEN_MAX))
        self.ATR_LONG_PERIOD     = int(p.get("ATR_LONG_PERIOD",        ATR_LONG_PERIOD))
        self.EVENT_GUARD         = bool(p.get("EVENT_GUARD",          EVENT_GUARD))
        self.EVENT_REANCHOR      = bool(p.get("EVENT_REANCHOR",       EVENT_REANCHOR))

        # ── Loss circuit breaker state ─────────────────────────────────────────
        self._open_qty: int          = 0       # net long shares across all levels
        self._open_cost_basis: float = 0.0     # total cost of open position
        self._loss_halted: bool      = False   # True after circuit breaker fires

        self._aggressive_ratio   = self.GRID_RATIO  # remember original ratio for mode restore
        self._adx_mode: str      = "aggressive"
        self._qty_scale: float   = 1.0
        self._adx_vals: collections.deque = collections.deque(maxlen=self.ADX_SLOPE_BARS + 1)

        # Rolling bar buffer for ADX computation: (high, low, close) per bar
        _buf_len = self.ADX_PERIOD * 3 + 1
        self._bar_buf: collections.deque = collections.deque(maxlen=_buf_len)

        # ATR long-term baseline buffer (for bidirectional Guasoni adaptation)
        self._atr_long_buf: collections.deque = collections.deque(maxlen=self.ATR_LONG_PERIOD)

        # Event calendar guard state
        self._event_dates: Set[_date]      = set()
        self._event_paused: bool           = False
        self._last_event_check: Optional[_date] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.qualify()

        # 1. Determine anchor price from a live market snapshot
        self.anchor_price = await self._fetcher.fetch_snapshot_price(self.contract)
        self.logger.info("Anchor price set to %.4f", self.anchor_price)

        # 2. Fetch event calendar (earnings / ex-div) before placing orders
        if self.EVENT_GUARD:
            await self._refresh_event_calendar()

        # 3. Log commission viability before placing anything
        self._log_commission_analysis()

        # 4. Build the grid and place initial buy orders
        self._build_grid()
        self._place_initial_buys()

        # 5. Subscribe to live ticks for the safety-switch monitor
        self._ticker = self._fetcher.subscribe_live(self.contract)
        self._ticker.updateEvent += self._on_ticker_update

        self.logger.info(
            "CLF Grid started | anchor=%.4f  levels=%d  account=$%.0f",
            self.anchor_price, len(self.levels), self.ACCOUNT_SIZE,
        )

    async def stop(self) -> None:
        if self._ticker is not None:
            self._ticker.updateEvent -= self._on_ticker_update
            self._fetcher.cancel_live(self.contract)
            self._ticker = None

        cancelled = 0
        for level in self.levels.values():
            for trade in (level.buy_trade, level.sell_trade):
                if trade and trade.isActive():
                    self._cancel_order(trade)
                    cancelled += 1

        self.logger.info("CLF Grid stopped | %d open orders cancelled.", cancelled)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        self.levels.clear()
        lower_bound = self.anchor_price * (1 - self.SAFETY_PCT)

        for i in range(1, self.NUM_BUY_LEVELS + 1):
            buy_price = self.anchor_price / (self.GRID_RATIO ** i)

            if buy_price < lower_bound:
                self.logger.warning(
                    "Level %d (%.4f) is below the %.0f%% safety floor (%.4f) — truncating grid.",
                    i, buy_price, self.SAFETY_PCT * 100, lower_bound,
                )
                break

            sell_price = buy_price * self.GRID_RATIO
            qty = self._calc_quantity(buy_price)

            self.levels[i] = GridLevel(
                index=i,
                buy_price=round(buy_price, 4),
                sell_price=round(sell_price, 4),
                quantity=qty,
            )

    def _place_initial_buys(self) -> None:
        for level in self.levels.values():
            if level.quantity < 1:
                self.logger.warning(
                    "Level %d: quantity rounds to 0 (price=%.4f, allocation=$%.0f) — skipped.",
                    level.index, level.buy_price, self.ACCOUNT_SIZE / self.NUM_BUY_LEVELS,
                )
                continue

            ref = f"CLF_BUY_L{level.index}"
            trade = self._place_limit_order(
                "BUY", level.quantity, level.buy_price, order_ref=ref
            )
            if trade:
                level.buy_trade = trade
                trade.fillEvent += self._make_buy_fill_handler(level)

    # ------------------------------------------------------------------
    # Fill handlers (closures keep level reference clean)
    # ------------------------------------------------------------------

    def _make_buy_fill_handler(self, level: GridLevel):
        """Return a fill callback bound to a specific grid level."""
        def _on_buy_fill(trade: Trade, fill) -> None:
            fill_price = fill.execution.price
            fill_qty   = int(fill.execution.shares)
            sell_price = round(fill_price * self.GRID_RATIO, 4)

            gross  = (sell_price - fill_price) * fill_qty
            net    = gross - 2 * self.COMMISSION
            viable = "PROFITABLE" if net > 0 else "UNDERWATER"

            self.logger.info(
                "BUY FILL L%d | %.4f x%d → sell @ %.4f | "
                "gross=$%.2f  net=$%.2f  [%s]",
                level.index, fill_price, fill_qty, sell_price,
                gross, net, viable,
            )

            # Track open position for loss circuit breaker
            self._open_qty        += fill_qty
            self._open_cost_basis += fill_price * fill_qty

            if self._halted:
                self.logger.warning(
                    "Safety switch active — take-profit sell for L%d NOT placed.",
                    level.index,
                )
                return

            ref = f"CLF_SELL_L{level.index}"
            sell_trade = self._place_limit_order(
                "SELL", fill_qty, sell_price, order_ref=ref
            )
            if sell_trade:
                level.sell_trade = sell_trade
                sell_trade.fillEvent += self._make_sell_fill_handler(level, fill_price, fill_qty)

        return _on_buy_fill

    def _make_sell_fill_handler(self, level: GridLevel, buy_price: float, qty: int):
        """Return a fill callback that logs realised P&L for a completed round-trip."""
        def _on_sell_fill(trade: Trade, fill) -> None:
            sell_price = fill.execution.price
            gross = (sell_price - buy_price) * qty
            net   = gross - 2 * self.COMMISSION

            self.logger.info(
                "SELL FILL L%d | %.4f x%d | buy=%.4f | "
                "gross=$%.2f  net=$%.2f  (round-trip complete)",
                level.index, sell_price, qty, buy_price, gross, net,
            )

            # Unwind position tracking
            self._open_qty        -= qty
            self._open_cost_basis -= buy_price * qty

            # ── Grid replenishment: re-enter the buy at this level after a sell ──
            if not self._halted and not self._trend_halted and not self._event_paused:
                ref    = f"CLF_BUY_L{level.index}_re"
                re_qty = max(1, math.floor(level.quantity * self._qty_scale))
                new_trade = self._place_limit_order(
                    "BUY", re_qty, level.buy_price, order_ref=ref
                )
                if new_trade:
                    level.buy_trade  = new_trade
                    new_trade.fillEvent += self._make_buy_fill_handler(level)

        return _on_sell_fill

    # ------------------------------------------------------------------
    # Safety switch
    # ------------------------------------------------------------------

    def _on_ticker_update(self, ticker: Ticker) -> None:
        price = ticker.marketPrice()
        if math.isnan(price) or price <= 0:
            return

        # Collect bar H/L/C for ADX (bar_high/bar_low set by SimulatedTicker;
        # live IB Ticker also exposes .high/.low as day H/L — good enough as proxy)
        high = getattr(ticker, "bar_high", None) or getattr(ticker, "high", None)
        low  = getattr(ticker, "bar_low",  None) or getattr(ticker, "low",  None)
        high = price if (not high or math.isnan(high) or high <= 0) else high
        low  = price if (not low  or math.isnan(low)  or low  <= 0) else low
        self._bar_buf.append((high, low, price))

        self._update_safety_switch(price)
        if self.MAX_OPEN_LOSS > 0:
            self._check_loss_circuit_breaker(price)
        if self.ADX_TIERED or self.ADX_HALT > 0:
            self._update_adx_filter()
        if self.ATR_ADAPTIVE:
            self._apply_atr_adaptive(price)
        if self.EVENT_GUARD:
            today = _date.today()
            # Refresh calendar once per day (schedule async without blocking tick handler)
            if self._last_event_check != today:
                self._last_event_check = today   # prevent repeated scheduling
                asyncio.ensure_future(self._refresh_event_calendar())
            self._check_event_guard(today, price)

    def _update_safety_switch(self, price: float) -> None:
        deviation = abs(price - self.anchor_price) / self.anchor_price

        if deviation > self.SAFETY_PCT and not self._halted:
            self._halted = True
            self.logger.warning(
                "SAFETY SWITCH ON  | price=%.4f  anchor=%.4f  deviation=%.1f%% > %.0f%%",
                price, self.anchor_price, deviation * 100, self.SAFETY_PCT * 100,
            )

        elif deviation <= self.SAFETY_PCT and self._halted:
            self._halted = False
            self.logger.info(
                "SAFETY SWITCH OFF | price=%.4f  deviation=%.1f%% back inside band.",
                price, deviation * 100,
            )

    def _check_loss_circuit_breaker(self, price: float) -> None:
        """Cancel all pending buys when unrealized loss exceeds MAX_OPEN_LOSS."""
        if self._loss_halted or self._open_qty <= 0:
            return
        unrealized = self._open_qty * price - self._open_cost_basis
        if unrealized < -self.MAX_OPEN_LOSS:
            self._loss_halted = True
            avg_entry = self._open_cost_basis / self._open_qty
            self.logger.warning(
                "LOSS CIRCUIT BREAKER | unrealized=$%.2f < -$%.0f "
                "| qty=%d avg_entry=%.4f price=%.4f — cancelling all pending buys.",
                unrealized, self.MAX_OPEN_LOSS, self._open_qty, avg_entry, price,
            )
            # Cancel all pending (unfilled) buy orders
            cancelled = 0
            for level in self.levels.values():
                if level.buy_trade and level.buy_trade.isActive():
                    self._cancel_order(level.buy_trade)
                    cancelled += 1
            self.logger.warning("LOSS CIRCUIT BREAKER | cancelled %d pending buy orders.", cancelled)

    def _update_adx_filter(self) -> None:
        """Update ADX-based entry filter (legacy threshold or tiered adaptive mode)."""
        min_bars = self.ADX_PERIOD * 2 + 1
        if len(self._bar_buf) < min_bars:
            return
        adx = self._adx_from_buf()
        if math.isnan(adx):
            return

        if self.ADX_TIERED:
            self._apply_tiered_mode(adx)
        else:
            # Legacy single-threshold behaviour
            if adx > self.ADX_HALT and not self._trend_halted:
                self._trend_halted = True
                self.logger.warning(
                    "ADX FILTER ON  | adx=%.1f > %.1f — pausing new buy entries.",
                    adx, self.ADX_HALT,
                )
            elif adx < self.ADX_RESUME and self._trend_halted:
                self._trend_halted = False
                self.logger.info(
                    "ADX FILTER OFF | adx=%.1f < %.1f — resuming buy entries.",
                    adx, self.ADX_RESUME,
                )

    def _apply_tiered_mode(self, adx: float) -> None:
        """Three-tier adaptive mode driven by ADX level and slope."""
        self._adx_vals.append(adx)

        # Slope = total change over ADX_SLOPE_BARS bars (positive = rising)
        vals  = list(self._adx_vals)
        slope = vals[-1] - vals[0] if len(vals) >= 2 else 0.0

        # Determine target mode
        if adx >= self.ADX_HALT_THR:
            new_mode = "halt"                                  # extreme circuit breaker
        elif adx >= self.ADX_TREND_THR:
            new_mode = "trending" if slope > 0 else "stable"  # rising trend → wide grid / small size
        elif adx >= self.ADX_STABLE_THR:
            new_mode = "stable"
        elif adx < self.ADX_ULTRA_THR:
            new_mode = "ultra"                                 # pure chop → pyramid
        else:
            new_mode = "aggressive"

        if new_mode == self._adx_mode:
            return

        self._adx_mode = new_mode
        if new_mode == "ultra":
            self.GRID_RATIO    = self._aggressive_ratio
            self._qty_scale    = self.ADX_ULTRA_QTY_SCALE
            self._trend_halted = False
            self.logger.info(
                "ADX MODE → ULTRA      | adx=%.1f slope=%+.2f | ratio=%.3f qty_scale=%.1f (pyramid)",
                adx, slope, self.GRID_RATIO, self._qty_scale,
            )
        elif new_mode == "aggressive":
            self.GRID_RATIO    = self._aggressive_ratio
            self._qty_scale    = 1.0
            self._trend_halted = False
            self.logger.info(
                "ADX MODE → AGGRESSIVE | adx=%.1f slope=%+.2f | ratio=%.3f qty_scale=1.0",
                adx, slope, self.GRID_RATIO,
            )
        elif new_mode == "stable":
            self.GRID_RATIO    = self.ADX_STABLE_RATIO
            self._qty_scale    = self.ADX_STABLE_QTY_SCALE
            self._trend_halted = False
            self.logger.info(
                "ADX MODE → STABLE     | adx=%.1f slope=%+.2f | ratio=%.3f qty_scale=%.1f",
                adx, slope, self.GRID_RATIO, self._qty_scale,
            )
        elif new_mode == "trending":
            self.GRID_RATIO    = self.ADX_TREND_RATIO
            self._qty_scale    = self.ADX_TREND_QTY_SCALE
            self._trend_halted = False
            self.logger.info(
                "ADX MODE → TRENDING   | adx=%.1f slope=%+.2f | ratio=%.3f qty_scale=%.2f (wide grid, small size)",
                adx, slope, self.GRID_RATIO, self._qty_scale,
            )
        else:  # halt — only ADX >= ADX_HALT_THR (extreme, e.g. 45)
            self._trend_halted = True
            self.logger.warning(
                "ADX MODE → HALT       | adx=%.1f slope=%+.2f — extreme trend, circuit breaker.",
                adx, slope,
            )

    def _apply_atr_adaptive(self, price: float) -> None:
        """
        Bidirectional ATR-adaptive grid spacing (Guasoni & Muhle-Karbe, 2012).

        Only active in "aggressive" and "ultra" ADX modes — when ADX has already
        set a deliberate ratio, that takes precedence.

        Behaviour
        ---------
        * Warmup (<20 long-term ATR samples): original unidirectional formula
            new_ratio = 1 + (ATR/price) * ATR_GRID_COVERAGE
            capped at _aggressive_ratio (tighten only)

        * Steady state (>=20 samples): Guasoni bidirectional
            vol_ratio = (current_ATR / long_term_ATR) ^ (2/3)
            new_ratio = _aggressive_ratio * vol_ratio
            - Low vol  (ATR < long avg): ratio tightens  → more fills, lower cost
            - High vol (ATR > long avg): ratio widens    → avoids whipsaw
            - Normal vol (ATR ≈ long avg): stays at baseline _aggressive_ratio
            capped at [1.001, _aggressive_ratio * ATR_WIDEN_MAX]

        Example (baseline=1.007, ATR_WIDEN_MAX=1.8):
            ATR = 0.5x long avg → vol_ratio=0.63 → new_ratio=1.004 (tighter)
            ATR = 1.0x long avg → vol_ratio=1.00 → new_ratio=1.007 (unchanged)
            ATR = 2.0x long avg → vol_ratio=1.59 → new_ratio=1.011 (wider)
            ATR = 3.0x long avg → vol_ratio=2.08 → new_ratio=1.013 (capped at 1.007*1.8=1.013)
        """
        if self._adx_mode not in ("aggressive", "ultra"):
            return
        if len(self._bar_buf) < self.ADX_PERIOD:
            return
        atr = self._atr_from_buf()
        if math.isnan(atr) or atr <= 0 or price <= 0:
            return

        self._atr_long_buf.append(atr)
        max_ratio = self._aggressive_ratio * self.ATR_WIDEN_MAX

        if len(self._atr_long_buf) >= 20:
            atr_long = float(np.mean(self._atr_long_buf))
            # Guasoni: only WIDEN when ATR is above its long-term baseline.
            # When ATR is normal or low, stay at _aggressive_ratio (baseline).
            # ADX tiered mode already handles the low-vol tightening side.
            if atr > atr_long:
                vol_ratio = (atr / atr_long) ** (2.0 / 3.0)
                new_ratio = min(self._aggressive_ratio * vol_ratio, max_ratio)
                new_ratio = max(self._aggressive_ratio, new_ratio)  # never go below baseline
            else:
                new_ratio = self._aggressive_ratio  # normal/low vol → keep baseline
        else:
            # Warmup: lightweight unidirectional tighten (original formula)
            new_ratio = 1.0 + (atr / price) * self.ATR_GRID_COVERAGE
            new_ratio = max(1.001, min(new_ratio, self._aggressive_ratio))

        if abs(new_ratio - self.GRID_RATIO) > 0.0001:
            direction    = "wider" if new_ratio > self.GRID_RATIO else "tighter"
            atr_long_str = f"{float(np.mean(self._atr_long_buf)):.4f}" if len(self._atr_long_buf) >= 2 else "n/a"
            self.logger.debug(
                "ATR ADAPTIVE | ATR=%.4f (%.3f%% price) long_ATR=%s "
                "-> grid %.4f -> %.4f (%s)",
                atr, atr / price * 100, atr_long_str,
                self.GRID_RATIO, new_ratio, direction,
            )
            self.GRID_RATIO = new_ratio

    # ------------------------------------------------------------------
    # Event calendar guard
    # ------------------------------------------------------------------

    async def _refresh_event_calendar(self) -> None:
        """
        Fetch upcoming earnings and ex-dividend dates (next 60 days) via yfinance.
        Stores both the report date AND the following trading day (for AMC reports).
        Called once at startup and once per trading day while the strategy is live.
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            self.logger.warning(
                "EVENT_GUARD: yfinance not installed — guard disabled. "
                "Run: pip install yfinance"
            )
            self.EVENT_GUARD = False
            return

        sym   = self.contract.symbol
        today = _date.today()
        horizon = today + timedelta(days=60)
        new_dates: Set[_date] = set()

        try:
            ticker = yf.Ticker(sym)

            # -- Earnings dates (historical + upcoming estimates) --
            ed = ticker.earnings_dates
            if ed is not None and not (hasattr(ed, "empty") and ed.empty):
                earn_raw = set(pd.DatetimeIndex(ed.index).normalize().date)
                for d in earn_raw:
                    if today <= d <= horizon:
                        new_dates.add(d)                      # BMO: same day
                        new_dates.add(d + timedelta(days=1))  # AMC: next day gap

            # -- Ex-dividend date from calendar --
            try:
                cal = ticker.calendar
                if isinstance(cal, dict):
                    ex_div = cal.get("Ex-Dividend Date")
                elif hasattr(cal, "loc"):                      # DataFrame format
                    ex_div = cal.loc["Ex-Dividend Date"].iloc[0] if "Ex-Dividend Date" in cal.index else None
                else:
                    ex_div = None
                if ex_div is not None:
                    ex_div_d = pd.Timestamp(ex_div).date()
                    if today <= ex_div_d <= horizon:
                        new_dates.add(ex_div_d)
            except Exception:
                pass  # calendar unavailable for many tickers

            self._event_dates = new_dates
            if new_dates:
                self.logger.info(
                    "EVENT GUARD | %s: %d protected dates in next 60d: %s",
                    sym, len(new_dates), sorted(new_dates),
                )
            else:
                self.logger.info("EVENT GUARD | %s: no events found in next 60d.", sym)

        except Exception as exc:
            self.logger.warning(
                "EVENT_GUARD | failed to fetch calendar for %s: %s", sym, exc
            )

    def _check_event_guard(self, today: _date, price: float) -> None:
        """
        Called on every tick update while EVENT_GUARD is active.

        If today is in _event_dates:
          - Set _event_paused = True
          - Cancel all pending (unfilled) buy orders
        Once the event day passes:
          - Clear _event_paused
          - If EVENT_REANCHOR=True and price drifted >5% from anchor: rebuild grid
          - Otherwise: restore cancelled buy orders
        """
        is_event_day = today in self._event_dates

        if is_event_day and not self._event_paused:
            self._event_paused = True
            cancelled = 0
            for level in self.levels.values():
                if level.buy_trade and level.buy_trade.isActive():
                    self._cancel_order(level.buy_trade)
                    cancelled += 1
            self.logger.warning(
                "EVENT GUARD ON  | %s %s — cancelled %d pending buy orders.",
                self.contract.symbol, today, cancelled,
            )

        elif not is_event_day and self._event_paused:
            self._event_paused = False
            deviation = abs(price - self.anchor_price) / self.anchor_price if self.anchor_price else 0.0
            self.logger.info(
                "EVENT GUARD OFF | %s %s — price=%.4f anchor=%.4f drift=%.1f%%",
                self.contract.symbol, today, price, self.anchor_price, deviation * 100,
            )
            if self.EVENT_REANCHOR and deviation > 0.05:
                # Price moved >5% — rebuild grid from current price
                self.logger.info(
                    "EVENT REANCHOR  | drift %.1f%% > 5%% — rebuilding grid from %.4f",
                    deviation * 100, price,
                )
                for level in self.levels.values():
                    for t in (level.buy_trade, level.sell_trade):
                        if t and t.isActive():
                            self._cancel_order(t)
                self.anchor_price = price
                self._build_grid()
                self._place_initial_buys()
            else:
                # Re-place buy orders on levels with no active order
                self._restore_cancelled_buys()

    def _restore_cancelled_buys(self) -> None:
        """Re-place buy orders for levels with no active buy or sell order."""
        placed = 0
        for level in self.levels.values():
            if level.quantity < 1:
                continue
            has_active = (
                (level.buy_trade  and level.buy_trade.isActive()) or
                (level.sell_trade and level.sell_trade.isActive())
            )
            if has_active:
                continue
            ref   = f"CLF_BUY_L{level.index}_rst"
            trade = self._place_limit_order("BUY", level.quantity, level.buy_price, order_ref=ref)
            if trade:
                level.buy_trade = trade
                trade.fillEvent += self._make_buy_fill_handler(level)
                placed += 1
        self.logger.info("EVENT GUARD | restored %d buy orders post-event.", placed)

    def _adx_from_buf(self) -> float:
        """Wilder's ADX computed from the rolling bar buffer. Returns nan if insufficient data."""
        buf   = list(self._bar_buf)
        high  = np.array([b[0] for b in buf], dtype=float)
        low   = np.array([b[1] for b in buf], dtype=float)
        close = np.array([b[2] for b in buf], dtype=float)
        n     = self.ADX_PERIOD

        prev_high  = high[:-1];   curr_high  = high[1:]
        prev_low   = low[:-1];    curr_low   = low[1:]
        prev_close = close[:-1]

        tr = np.maximum.reduce([
            curr_high - curr_low,
            np.abs(curr_high - prev_close),
            np.abs(curr_low  - prev_close),
        ])
        up_move   = curr_high - prev_high
        down_move = prev_low  - curr_low
        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's smoothing
        alpha = 1.0 / n
        def _ema(arr: np.ndarray) -> np.ndarray:
            out = np.empty_like(arr)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = out[i - 1] * (1 - alpha) + arr[i] * alpha
            return out

        atr_s    = _ema(tr)
        plus_di  = 100.0 * _ema(plus_dm)  / np.where(atr_s != 0, atr_s, np.nan)
        minus_di = 100.0 * _ema(minus_dm) / np.where(atr_s != 0, atr_s, np.nan)

        di_sum = plus_di + minus_di
        dx     = np.abs(plus_di - minus_di) / np.where(di_sum != 0, di_sum, np.nan) * 100.0
        adx    = _ema(np.nan_to_num(dx))
        val    = adx[-1]
        return float(val) if np.isfinite(val) else float("nan")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_quantity(self, price: float) -> int:
        """Shares per level.

        Capital-allocation mode (default):
            qty = floor(ACCOUNT_SIZE / NUM_BUY_LEVELS / price)

        ATR-risk mode (ATR_SIZING=True):
            risk_dollar = ACCOUNT_SIZE / NUM_BUY_LEVELS * ATR_RISK_PCT / 100
            qty = floor(risk_dollar / atr_dollar)
            This ensures a 1-ATR move against the position costs a fixed dollar amount.
        """
        if self.ATR_SIZING and len(self._bar_buf) >= self.ADX_PERIOD:
            atr = self._atr_from_buf()
            atr_dollar = atr  # ATR is already in price units
            if atr_dollar > 0:
                risk_per_level = (self.ACCOUNT_SIZE / self.NUM_BUY_LEVELS) * (self.ATR_RISK_PCT / 100)
                return max(1, math.floor(risk_per_level / atr_dollar))
        allocation = self.ACCOUNT_SIZE / self.NUM_BUY_LEVELS
        return max(0, math.floor(allocation / price))

    def _atr_from_buf(self) -> float:
        """14-bar Average True Range from the rolling bar buffer (price units)."""
        buf = list(self._bar_buf)
        if len(buf) < 2:
            return float("nan")
        high  = np.array([b[0] for b in buf], dtype=float)
        low   = np.array([b[1] for b in buf], dtype=float)
        close = np.array([b[2] for b in buf], dtype=float)
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1]),
        ])
        n = min(self.ADX_PERIOD, len(tr))
        return float(np.mean(tr[-n:]))

    def _log_commission_analysis(self) -> None:
        anchor = self.anchor_price
        self.logger.info(
            "── CLF Grid Commission Analysis | anchor=%.4f  account=$%.0f ──",
            anchor, self.ACCOUNT_SIZE,
        )
        self.logger.info(
            "  %-7s  %-10s  %-10s  %-5s  %-9s  %-9s  %s",
            "Level", "Buy ($)", "Sell ($)", "Qty", "Gross ($)", "Net ($)", "Viable?"
        )
        lower_bound = anchor * (1 - self.SAFETY_PCT)

        for i in range(1, self.NUM_BUY_LEVELS + 1):
            buy  = anchor / (self.GRID_RATIO ** i)
            if buy < lower_bound:
                break
            sell = buy * self.GRID_RATIO
            qty  = self._calc_quantity(buy)
            if qty < 1:
                self.logger.info(
                    "  Level %2d | buy=%.4f → qty=0 (allocation=$%.0f too small for this price)",
                    i, buy, self.ACCOUNT_SIZE / self.NUM_BUY_LEVELS,
                )
                continue
            gross = (sell - buy) * qty
            net   = gross - 2 * self.COMMISSION
            self.logger.info(
                "  %-7d  %-10.4f  %-10.4f  %-5d  %-9.2f  %-9.2f  %s",
                i, buy, sell, qty, gross, net,
                "YES" if net > 0 else "NO — too few shares for this commission",
            )
