"""
strategies/smallcap_arb.py
--------------------------
Statistical Arbitrage (Pairs Trading) on two correlated small-cap stocks:
  Leg A — IREN  (Iris Energy, crypto-mining / HPC)
  Leg B — WULF  (TeraWulf, crypto-mining)

Trading hypothesis
------------------
IREN and WULF share nearly identical business drivers (Bitcoin hash-rate,
energy costs, and AI/HPC sentiment), making their price ratio mean-reverting.

Spread definition
-----------------
  ratio   = price_A / price_B
  z-score = (ratio − rolling_mean) / rolling_std

  where rolling_mean and rolling_std are computed from 20 days of
  5-minute bars fetched via DataFetcher.fetch_multiple_stocks().

Entry signal (one trade per direction at a time)
-------------------------------------------------
  z > +ZSCORE_ENTRY  →  ratio stretched high  →  SHORT IREN, LONG WULF
  z < −ZSCORE_ENTRY  →  ratio stretched low   →  LONG IREN, SHORT WULF

Exit signal
-----------
  |z| < ZSCORE_EXIT (default 0.5) → spread has mean-reverted → close both legs

Market impact guard
-------------------
Every order size is capped at MARKET_IMPACT_PCT (1 %) of the 5-minute
average volume for that leg.  The position is also dollar-neutral: the
notional of Leg A equals the notional of Leg B, so we derive the final
quantity for each leg from whichever constraint binds first.

Params override
---------------
Pass a ``params`` dict to override any module-level constant at instantiation.
This allows the backtester and Optuna optimizer to sweep parameters without
modifying this file.  Example::

    SmallCapArbStrategy(ib, params={"ZSCORE_ENTRY": 2.5, "HISTORY_DURATION": "30 D"})
"""

import math
import time
from typing import Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, LimitOrder, Stock, Ticker, Trade

from .base import BaseStrategy
from utils.data_fetcher import DataFetcher, IndicatorUtils

# ── Module-level defaults (used when no params dict is supplied) ───────────────
SYMBOL_A: str            = "IREN"
SYMBOL_B: str            = "WULF"

ZSCORE_ENTRY: float      = 2.0     # open a position when |z| exceeds this
ZSCORE_EXIT: float       = 0.5     # close the position when |z| falls below this

HISTORY_DURATION: str    = "20 D"  # lookback for computing pair statistics
BAR_SIZE: str            = "5 mins"

MARKET_IMPACT_PCT: float = 0.01    # max order size as fraction of avg 5-min volume
MIN_EVAL_SECS: float     = 5.0     # minimum seconds between signal evaluations


class SmallCapArbStrategy(BaseStrategy):
    """
    Dollar-neutral pairs trade on IREN / WULF using a Z-score spread signal.

    Inherits from BaseStrategy but overrides ``qualify()`` because this
    strategy manages two contracts simultaneously.

    Both legs use intraday DAY limit orders to avoid overnight exposure with
    only one side hedged.
    """

    def __init__(self, ib: IB, account: str = "", params: dict = None):
        self._contract_a = Stock(SYMBOL_A, "SMART", "USD")
        self._contract_b = Stock(SYMBOL_B, "SMART", "USD")
        super().__init__(ib, self._contract_a, account)

        self._fetcher = DataFetcher(ib)

        # ── Configurable parameters (override via params dict) ─────────────────
        p = params or {}
        self.ZSCORE_ENTRY       = float(p.get("ZSCORE_ENTRY",       ZSCORE_ENTRY))
        self.ZSCORE_EXIT        = float(p.get("ZSCORE_EXIT",        ZSCORE_EXIT))
        self.HISTORY_DURATION   = str(p.get("HISTORY_DURATION",     HISTORY_DURATION))
        self.BAR_SIZE           = str(p.get("BAR_SIZE",             BAR_SIZE))
        self.MARKET_IMPACT_PCT  = float(p.get("MARKET_IMPACT_PCT",  MARKET_IMPACT_PCT))
        # In backtest mode callers should pass MIN_EVAL_SECS=0 to disable throttle
        self._min_eval_secs     = float(p.get("MIN_EVAL_SECS",      MIN_EVAL_SECS))

        # ── Pair statistics (populated in start()) ──────────────────────
        self._ratio_mean: float = 0.0
        self._ratio_std: float  = 1.0
        self._avg_vol_a: float  = 0.0
        self._avg_vol_b: float  = 0.0

        # ── Live price cache ────────────────────────────────────────────
        self._price_a: float = 0.0
        self._price_b: float = 0.0
        self._last_eval_ts: float = 0.0

        # ── Market data subscriptions ───────────────────────────────────
        self._ticker_a: Optional[Ticker] = None
        self._ticker_b: Optional[Ticker] = None

        # ── Position state ──────────────────────────────────────────────
        self._in_position: bool    = False
        self._pending_entry: bool  = False
        self._position_side: str   = ""
        self._pos_qty_a: int       = 0
        self._pos_qty_b: int       = 0
        self._open_trades: list    = []

    # ------------------------------------------------------------------
    # Contract qualification (override: two contracts)
    # ------------------------------------------------------------------

    async def qualify(self) -> Contract:
        """Qualify both IREN and WULF in a single IBKR round-trip."""
        results = await self.ib.qualifyContractsAsync(
            self._contract_a, self._contract_b
        )
        if len(results) < 2:
            raise ValueError(
                f"Could not qualify both {SYMBOL_A} and {SYMBOL_B}. "
                "Check that both symbols exist and your data subscription covers them."
            )
        self._contract_a = results[0]
        self._contract_b = results[1]
        self.contract    = self._contract_a
        self._qualified  = True

        self.logger.info(
            "Qualified | %s conId=%s | %s conId=%s",
            self._contract_a.symbol, self._contract_a.conId,
            self._contract_b.symbol, self._contract_b.conId,
        )
        return self.contract

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self.qualify()

        self.logger.info(
            "Fetching %s of %s bars for %s and %s …",
            self.HISTORY_DURATION, self.BAR_SIZE, SYMBOL_A, SYMBOL_B,
        )
        data = await self._fetcher.fetch_multiple_stocks(
            contracts=[self._contract_a, self._contract_b],
            duration=self.HISTORY_DURATION,
            bar_size=self.BAR_SIZE,
        )

        missing = {SYMBOL_A, SYMBOL_B} - data.keys()
        if missing:
            raise RuntimeError(
                f"Historical data fetch failed for: {missing}. "
                "Check market hours and IB data subscriptions."
            )

        self._compute_pair_stats(data[SYMBOL_A], data[SYMBOL_B])

        self._ticker_a = self._fetcher.subscribe_live(self._contract_a)
        self._ticker_b = self._fetcher.subscribe_live(self._contract_b)
        self._ticker_a.updateEvent += self._on_tick_update
        self._ticker_b.updateEvent += self._on_tick_update

        self.logger.info(
            "SmallCap Arb started | %s/%s  mean=%.4f  std=%.4f  "
            "entry_z=±%.1f  exit_z=±%.1f",
            SYMBOL_A, SYMBOL_B,
            self._ratio_mean, self._ratio_std,
            self.ZSCORE_ENTRY, self.ZSCORE_EXIT,
        )

    async def stop(self) -> None:
        for ticker, contract in [
            (self._ticker_a, self._contract_a),
            (self._ticker_b, self._contract_b),
        ]:
            if ticker is not None:
                ticker.updateEvent -= self._on_tick_update
                self._fetcher.cancel_live(contract)
        self._ticker_a = self._ticker_b = None

        for trade in self._open_trades:
            if trade and trade.isActive():
                self._cancel_order(trade)
        self._open_trades.clear()

        self.logger.info("SmallCap Arb stopped.")

    # ------------------------------------------------------------------
    # Pair statistics
    # ------------------------------------------------------------------

    def _compute_pair_stats(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
    ) -> None:
        aligned = (
            df_a.set_index("date")[["close", "volume"]]
            .join(
                df_b.set_index("date")[["close", "volume"]],
                how="inner",
                lsuffix="_a",
                rsuffix="_b",
            )
            .dropna()
        )

        if len(aligned) < 2:
            raise RuntimeError(
                "Insufficient overlapping bars to compute pair statistics. "
                "Are both symbols trading on the same exchange / session?"
            )

        ratio = aligned["close_a"] / aligned["close_b"]

        self._ratio_mean = float(ratio.mean())
        self._ratio_std  = float(ratio.std(ddof=1))
        self._avg_vol_a  = float(aligned["volume_a"].mean())
        self._avg_vol_b  = float(aligned["volume_b"].mean())

        if self._ratio_std == 0:
            raise RuntimeError(
                "Price ratio has zero variance over the lookback period. "
                "The two assets may be identical — pairs trading requires spread."
            )

        self.logger.info(
            "Pair stats (%d aligned bars) | ratio_mean=%.4f  ratio_std=%.6f  "
            "avg_vol_%s=%.0f  avg_vol_%s=%.0f",
            len(aligned),
            self._ratio_mean, self._ratio_std,
            SYMBOL_A, self._avg_vol_a,
            SYMBOL_B, self._avg_vol_b,
        )

        cap_a = math.floor(self._avg_vol_a * self.MARKET_IMPACT_PCT)
        cap_b = math.floor(self._avg_vol_b * self.MARKET_IMPACT_PCT)
        self.logger.info(
            "Market impact cap (%.0f%% of avg vol) | %s ≤ %d shares | %s ≤ %d shares",
            self.MARKET_IMPACT_PCT * 100, SYMBOL_A, cap_a, SYMBOL_B, cap_b,
        )

    # ------------------------------------------------------------------
    # Tick update handler
    # ------------------------------------------------------------------

    def _on_tick_update(self, ticker: Ticker) -> None:
        """Cache the latest price for each leg; evaluate signal when both are live."""
        price = ticker.marketPrice()
        if math.isnan(price) or price <= 0:
            return

        if ticker.contract.symbol == SYMBOL_A:
            self._price_a = price
        else:
            self._price_b = price

        if self._price_a <= 0 or self._price_b <= 0:
            return

        now = time.monotonic()
        if self._min_eval_secs > 0 and (now - self._last_eval_ts) < self._min_eval_secs:
            return
        self._last_eval_ts = now

        self._evaluate_signal()

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _evaluate_signal(self) -> None:
        ratio = self._price_a / self._price_b
        z     = (ratio - self._ratio_mean) / self._ratio_std

        self.logger.info(
            "SIGNAL | %s=%.4f  %s=%.4f  ratio=%.4f  z=%.3f  pos=%s",
            SYMBOL_A, self._price_a,
            SYMBOL_B, self._price_b,
            ratio, z,
            self._position_side if self._in_position else "FLAT",
        )

        if self._in_position:
            if not self._pending_entry:
                self._check_exit(z)
        else:
            if not self._pending_entry:
                self._check_entry(z)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _check_entry(self, z: float) -> None:
        if abs(z) <= self.ZSCORE_ENTRY:
            return

        qty_a, qty_b = self._calc_pair_qty()
        if qty_a < 1 or qty_b < 1:
            self.logger.warning(
                "ENTRY BLOCKED | market impact cap too restrictive | "
                "qty_a=%d  qty_b=%d  (need ≥ 1 each)  z=%.3f",
                qty_a, qty_b, z,
            )
            return

        if z > self.ZSCORE_ENTRY:
            side = "SHORT_A_LONG_B"
            action_a, action_b = "SELL", "BUY"
        else:
            side = "LONG_A_SHORT_B"
            action_a, action_b = "BUY", "SELL"

        self.logger.info(
            "ENTRY SIGNAL | z=%.3f  side=%s | "
            "%s %s x%d @ %.4f | %s %s x%d @ %.4f",
            z, side,
            action_a, SYMBOL_A, qty_a, self._price_a,
            action_b, SYMBOL_B, qty_b, self._price_b,
        )

        trade_a = self._place_pair_order(
            self._contract_a, action_a, qty_a, self._price_a,
            ref=f"ARB_{action_a}_A",
        )
        trade_b = self._place_pair_order(
            self._contract_b, action_b, qty_b, self._price_b,
            ref=f"ARB_{action_b}_B",
        )

        if trade_a and trade_b:
            self._position_side = side
            self._pos_qty_a     = qty_a
            self._pos_qty_b     = qty_b
            self._pending_entry = True
            self._open_trades   = [trade_a, trade_b]

            trade_a.fillEvent += self._make_fill_handler(SYMBOL_A)
            trade_b.fillEvent += self._make_fill_handler(SYMBOL_B)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exit(self, z: float) -> None:
        if abs(z) >= self.ZSCORE_EXIT:
            return

        if self._position_side == "SHORT_A_LONG_B":
            exit_a, exit_b = "BUY", "SELL"
        else:
            exit_a, exit_b = "SELL", "BUY"

        self.logger.info(
            "EXIT SIGNAL | z=%.3f — spread mean-reverted | "
            "%s %s x%d @ %.4f | %s %s x%d @ %.4f",
            z,
            exit_a, SYMBOL_A, self._pos_qty_a, self._price_a,
            exit_b, SYMBOL_B, self._pos_qty_b, self._price_b,
        )

        for trade in self._open_trades:
            if trade and trade.isActive():
                self._cancel_order(trade)
        self._open_trades.clear()

        trade_a = self._place_pair_order(
            self._contract_a, exit_a, self._pos_qty_a, self._price_a,
            ref="ARB_EXIT_A",
        )
        trade_b = self._place_pair_order(
            self._contract_b, exit_b, self._pos_qty_b, self._price_b,
            ref="ARB_EXIT_B",
        )

        if trade_a and trade_b:
            self._open_trades = [trade_a, trade_b]
            self._pending_entry = True

            trade_a.fillEvent += self._make_exit_fill_handler(SYMBOL_A)
            trade_b.fillEvent += self._make_exit_fill_handler(SYMBOL_B)

        self._in_position    = False
        self._position_side  = ""

    # ------------------------------------------------------------------
    # Fill handlers
    # ------------------------------------------------------------------

    def _make_fill_handler(self, symbol: str):
        """Entry fill handler: marks the position live once both legs fill."""
        def _on_fill(trade: Trade, fill) -> None:
            self.logger.info(
                "ENTRY FILL | %s  %s x%g @ %.4f | orderId=%s",
                symbol, fill.execution.side,
                fill.execution.shares, fill.execution.price,
                trade.order.orderId,
            )
            self._in_position   = True
            self._pending_entry = False

        return _on_fill

    def _make_exit_fill_handler(self, symbol: str):
        """Exit fill handler: logs realised P&L when each leg closes."""
        def _on_fill(trade: Trade, fill) -> None:
            self.logger.info(
                "EXIT FILL  | %s  %s x%g @ %.4f | orderId=%s",
                symbol, fill.execution.side,
                fill.execution.shares, fill.execution.price,
                trade.order.orderId,
            )
            self._pending_entry = False

        return _on_fill

    # ------------------------------------------------------------------
    # Market impact & position sizing
    # ------------------------------------------------------------------

    def _calc_pair_qty(self) -> Tuple[int, int]:
        """
        Compute dollar-neutral, market-impact-limited share quantities.

        Step 1 — Market impact cap:
            max_qty_x = floor(avg_vol_x * MARKET_IMPACT_PCT)

        Step 2 — Dollar neutrality (anchor on the more restricted leg):
            notional = min(max_qty_a * price_a, max_qty_b * price_b)
            qty_a    = floor(notional / price_a)
            qty_b    = floor(notional / price_b)

        Returns
        -------
        (qty_a, qty_b) — shares for Leg A and Leg B respectively.
        """
        cap_a = math.floor(self._avg_vol_a * self.MARKET_IMPACT_PCT)
        cap_b = math.floor(self._avg_vol_b * self.MARKET_IMPACT_PCT)

        max_notional = min(cap_a * self._price_a, cap_b * self._price_b)

        qty_a = math.floor(max_notional / self._price_a) if self._price_a > 0 else 0
        qty_b = math.floor(max_notional / self._price_b) if self._price_b > 0 else 0

        self.logger.debug(
            "Sizing | cap_a=%d  cap_b=%d  max_notional=$%.2f  "
            "qty_a=%d ($%.0f)  qty_b=%d ($%.0f)",
            cap_a, cap_b, max_notional,
            qty_a, qty_a * self._price_a,
            qty_b, qty_b * self._price_b,
        )
        return qty_a, qty_b

    # ------------------------------------------------------------------
    # Order placement helper (bypasses BaseStrategy's single-contract method)
    # ------------------------------------------------------------------

    def _place_pair_order(
        self,
        contract: Contract,
        action: str,
        qty: int,
        price: float,
        ref: str = "",
    ) -> Optional[Trade]:
        """
        Place a DAY limit order for either leg of the pair.

        Uses DAY TIF (not GTC) to ensure both legs expire together at end of
        session and never leave an unhedged overnight position.
        """
        if not self._qualified:
            self.logger.error("Contracts not qualified — call qualify() first.")
            return None

        order = LimitOrder(
            action=action,
            totalQuantity=qty,
            lmtPrice=round(price, 2),
            tif="DAY",
            account=self.account,
            orderRef=ref,
        )
        trade = self.ib.placeOrder(contract, order)
        self.logger.info(
            "ORDER | %-4s %s x%d @ %.4f | ref=%-20s | orderId=%s",
            action, contract.symbol, qty, price, ref, trade.order.orderId,
        )
        trade.statusEvent += self._on_status
        return trade
