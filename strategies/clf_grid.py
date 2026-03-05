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

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from ib_insync import IB, Stock, Ticker, Trade

from .base import BaseStrategy
from utils.data_fetcher import DataFetcher

# ── Module-level defaults (used when no params dict is supplied) ───────────────
GRID_RATIO: float       = 1.015       # 1.5% geometric step
SAFETY_PCT: float       = 0.30        # halt new orders if |price - anchor| / anchor > 30 %
COMMISSION: float       = 0.35        # approx IBKR Tiered, per side ($)
ACCOUNT_SIZE: float     = 1_000.0     # USD allocated to this strategy
NUM_BUY_LEVELS: int     = 10          # grid levels below anchor to pre-place


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
        super().__init__(ib, Stock("CLF", "SMART", "USD"), account)
        self._fetcher = DataFetcher(ib)
        self.anchor_price: Optional[float] = None
        self.levels: Dict[int, GridLevel] = {}
        self._ticker: Optional[Ticker] = None
        self._halted: bool = False

        # ── Configurable parameters (override via params dict) ─────────────────
        p = params or {}
        self.GRID_RATIO     = float(p.get("GRID_RATIO",     GRID_RATIO))
        self.SAFETY_PCT     = float(p.get("SAFETY_PCT",     SAFETY_PCT))
        self.COMMISSION     = float(p.get("COMMISSION",     COMMISSION))
        self.ACCOUNT_SIZE   = float(p.get("ACCOUNT_SIZE",   ACCOUNT_SIZE))
        self.NUM_BUY_LEVELS = int(p.get("NUM_BUY_LEVELS",   NUM_BUY_LEVELS))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.qualify()

        # 1. Determine anchor price from a live market snapshot
        self.anchor_price = await self._fetcher.fetch_snapshot_price(self.contract)
        self.logger.info("Anchor price set to %.4f", self.anchor_price)

        # 2. Log commission viability before placing anything
        self._log_commission_analysis()

        # 3. Build the grid and place initial buy orders
        self._build_grid()
        self._place_initial_buys()

        # 4. Subscribe to live ticks for the safety-switch monitor
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

            # ── Optional: re-enter the buy at this level after a sell fills ──
            # Uncomment the block below to enable grid replenishment.
            #
            # if not self._halted:
            #     ref = f"CLF_BUY_L{level.index}_re"
            #     new_trade = self._place_limit_order(
            #         "BUY", qty, level.buy_price, order_ref=ref
            #     )
            #     if new_trade:
            #         level.buy_trade = new_trade
            #         new_trade.fillEvent += self._make_buy_fill_handler(level)

        return _on_sell_fill

    # ------------------------------------------------------------------
    # Safety switch
    # ------------------------------------------------------------------

    def _on_ticker_update(self, ticker: Ticker) -> None:
        price = ticker.marketPrice()
        if math.isnan(price) or price <= 0:
            return
        self._update_safety_switch(price)

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_quantity(self, price: float) -> int:
        """Shares per level = floor(allocation_per_level / price)."""
        allocation = self.ACCOUNT_SIZE / self.NUM_BUY_LEVELS
        return max(0, math.floor(allocation / price))

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
