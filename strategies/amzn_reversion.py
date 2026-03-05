"""
strategies/amzn_reversion.py
-----------------------------
Mean-reversion strategy for AMZN (Amazon.com Inc.)

Entry conditions (ALL must be true on a completed bar):
  1. Close BREAKS BELOW the Lower Bollinger Band (BB_PERIOD SMA ± BB_STD·σ)
     — i.e., previous close was >= lower band, current close is < lower band
  2. RSI(RSI_PERIOD, Wilder smoothing) < RSI_ENTRY

Exit conditions (first to trigger wins):
  A. Close >= BB_PERIOD SMA  (price mean-reverted)
  B. Unrealised profit >= TAKE_PROFIT

Data source:
  reqHistoricalDataAsync with keepUpToDate=True delivers completed 5-min bars
  in real time via BarDataList.updateEvent — no separate subscription needed.

Params override
---------------
Pass a ``params`` dict to override any module-level constant at instantiation.
This allows the backtester and Optuna optimizer to sweep parameters without
modifying this file.  Example::

    AMZNReversionStrategy(ib, params={"RSI_ENTRY": 25, "BB_PERIOD": 20})
"""

from typing import Optional

from ib_insync import IB, BarDataList, Stock, Trade

from .base import BaseStrategy
from utils.data_fetcher import DataFetcher, IndicatorUtils

# ── Module-level defaults (used when no params dict is supplied) ───────────────
BB_PERIOD: int    = 20
BB_STD: float     = 2.0
RSI_PERIOD: int   = 14
RSI_ENTRY: float  = 25.0       # enter only when RSI is this oversold or worse
TAKE_PROFIT: float = 0.03      # 3 % gain → close position
BAR_SIZE: str     = "5 mins"
DURATION: str     = "3 D"      # enough history to seed both BB and RSI


class AMZNReversionStrategy(BaseStrategy):
    """
    Pure mean-reversion on AMZN using Bollinger Band breakout + RSI oversold filter.

    Position sizing: 1 share by default (suitable for high-price AMZN on a small account).
    Extend _calc_quantity() to implement risk-based sizing.
    """

    def __init__(self, ib: IB, account: str = "", params: dict = None):
        super().__init__(ib, Stock("AMZN", "SMART", "USD"), account)
        self._fetcher = DataFetcher(ib)
        self._bars: Optional[BarDataList] = None

        # Position state
        self._in_position: bool    = False
        self._entry_price: float   = 0.0
        self._position_qty: int    = 0
        self._pending_entry: bool  = False

        # ── Configurable parameters (override via params dict) ─────────────────
        p = params or {}
        self.BB_PERIOD   = int(p.get("BB_PERIOD",   BB_PERIOD))
        self.BB_STD      = float(p.get("BB_STD",    BB_STD))
        self.RSI_PERIOD  = int(p.get("RSI_PERIOD",  RSI_PERIOD))
        self.RSI_ENTRY   = float(p.get("RSI_ENTRY", RSI_ENTRY))
        self.TAKE_PROFIT = float(p.get("TAKE_PROFIT", TAKE_PROFIT))
        self.BAR_SIZE    = str(p.get("BAR_SIZE",    BAR_SIZE))
        self.DURATION    = str(p.get("DURATION",    DURATION))

        # Minimum bars before signals can fire (derived from indicator periods)
        self._min_bars = self.BB_PERIOD + self.RSI_PERIOD + 5

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.qualify()

        self._bars = await self._fetcher.fetch_historical(
            self.contract,
            duration=self.DURATION,
            bar_size=self.BAR_SIZE,
            keep_up_to_date=True,
        )
        self._bars.updateEvent += self._on_bar_update

        self.logger.info(
            "AMZN Reversion started | %d bars loaded | waiting for signals …",
            len(self._bars),
        )

    async def stop(self) -> None:
        if self._bars is not None:
            self._bars.updateEvent -= self._on_bar_update
            self._fetcher.cancel_historical(self._bars)
            self._bars = None

        self.logger.info("AMZN Reversion stopped.")

    # ------------------------------------------------------------------
    # Bar update — the main strategy loop
    # ------------------------------------------------------------------

    def _on_bar_update(self, bars: BarDataList, has_new_bar: bool) -> None:
        if not has_new_bar:
            return

        n = len(bars)
        if n < self._min_bars:
            self.logger.debug("Warming up … %d / %d bars.", n, self._min_bars)
            return

        df    = IndicatorUtils.bars_to_df(bars)
        close = df["close"]

        sma, upper_bb, lower_bb = IndicatorUtils.bollinger(
            close, period=self.BB_PERIOD, n_std=self.BB_STD
        )
        rsi = IndicatorUtils.rsi(close, period=self.RSI_PERIOD)

        # Latest completed bar values
        c0       = close.iloc[-1]
        c1       = close.iloc[-2]
        sma0     = sma.iloc[-1]
        lower0   = lower_bb.iloc[-1]
        lower1   = lower_bb.iloc[-2]
        rsi0     = rsi.iloc[-1]

        self.logger.info(
            "BAR  | close=%.4f  SMA=%.4f  BB_low=%.4f  RSI=%.2f  pos=%s",
            c0, sma0, lower0, rsi0, "YES" if self._in_position else "NO",
        )

        if self._in_position:
            self._check_exit(c0, sma0)
        elif not self._pending_entry:
            self._check_entry(c0, c1, lower0, lower1, rsi0)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        c0: float, c1: float,
        lower0: float, lower1: float,
        rsi0: float,
    ) -> None:
        broke_below  = (c1 >= lower1) and (c0 < lower0)
        rsi_oversold = rsi0 < self.RSI_ENTRY

        self.logger.info(
            "ENTRY CHECK | broke_below_BB=%s (prev=%.4f≥%.4f, now=%.4f<%.4f)  "
            "RSI_oversold=%s (%.2f<%.1f)",
            broke_below, c1, lower1, c0, lower0,
            rsi_oversold, rsi0, self.RSI_ENTRY,
        )

        if broke_below and rsi_oversold:
            qty = self._calc_quantity(c0)
            self.logger.info(
                "ENTRY SIGNAL | price=%.4f < BB_lower=%.4f  RSI=%.2f  qty=%d → BUY",
                c0, lower0, rsi0, qty,
            )
            trade = self._place_limit_order(
                "BUY", qty, round(c0, 2), order_ref="AMZN_ENTRY"
            )
            if trade:
                self._pending_entry = True
                trade.fillEvent += self._on_entry_fill

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit(self, c0: float, sma0: float) -> None:
        if not self._in_position or self._entry_price <= 0:
            return

        profit_pct  = (c0 - self._entry_price) / self._entry_price
        at_sma      = c0 >= sma0
        tp_hit      = profit_pct >= self.TAKE_PROFIT

        self.logger.info(
            "EXIT CHECK | price=%.4f  entry=%.4f  profit=%.2f%%  "
            "SMA=%.4f  at_sma=%s  tp_hit=%s",
            c0, self._entry_price, profit_pct * 100,
            sma0, at_sma, tp_hit,
        )

        if at_sma or tp_hit:
            reason = "SMA touch" if at_sma else f"{self.TAKE_PROFIT*100:.0f}% take-profit"
            self.logger.info("EXIT SIGNAL | %s | placing sell @ %.4f", reason, c0)
            trade = self._place_limit_order(
                "SELL", self._position_qty, round(c0, 2), order_ref="AMZN_EXIT"
            )
            if trade:
                trade.fillEvent += self._on_exit_fill

    # ------------------------------------------------------------------
    # Fill handlers
    # ------------------------------------------------------------------

    def _on_entry_fill(self, trade: Trade, fill) -> None:
        self._entry_price   = fill.execution.price
        self._position_qty  = int(fill.execution.shares)
        self._in_position   = True
        self._pending_entry = False
        self.logger.info(
            "ENTRY FILLED | %.4f x%d | monitoring for exit …",
            self._entry_price, self._position_qty,
        )

    def _on_exit_fill(self, trade: Trade, fill) -> None:
        exit_price = fill.execution.price
        pnl        = (exit_price - self._entry_price) * self._position_qty
        pnl_pct    = (exit_price - self._entry_price) / self._entry_price * 100

        self.logger.info(
            "EXIT FILLED  | %.4f x%d | entry=%.4f | P&L=$%.2f (%.2f%%)",
            exit_price, self._position_qty,
            self._entry_price, pnl, pnl_pct,
        )

        self._in_position  = False
        self._entry_price  = 0.0
        self._position_qty = 0

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_quantity(price: float) -> int:
        """
        Default: 1 share (AMZN trades at ~$200 +; adjust for your account size).
        Replace with risk-based sizing as needed, e.g.:
            return max(1, math.floor(ACCOUNT_RISK / (price * 0.05)))
        """
        return 1
