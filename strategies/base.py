"""
strategies/base.py
------------------
Abstract base class shared by all trading strategies.

Every strategy must implement:
    start()  — subscribe to market data, arm order logic
    stop()   — cancel subscriptions and open orders, release resources
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ib_insync import IB, Contract, LimitOrder, Trade


class BaseStrategy(ABC):
    """Common interface and shared utilities for all strategies."""

    def __init__(self, ib: IB, contract: Contract, account: str = ""):
        self.ib = ib
        self.contract = contract
        self.account = account
        self.logger = logging.getLogger(self.__class__.__name__)
        self._qualified = False

    # ------------------------------------------------------------------
    # Contract qualification
    # ------------------------------------------------------------------

    async def qualify(self) -> Contract:
        """
        Resolve full contract details via IBKR.
        Must be called once before start() or any order placement.
        """
        contracts = await self.ib.qualifyContractsAsync(self.contract)
        if not contracts:
            raise ValueError(f"Could not qualify contract: {self.contract}")
        self.contract = contracts[0]
        self._qualified = True
        # Attribute names differ across ib_insync versions; use getattr for safety.
        # In backtest mode, contracts are unqualified so some fields may be absent.
        primary_exch = (
            getattr(self.contract, "primaryExch", None)
            or getattr(self.contract, "primaryExchange", None)
            or getattr(self.contract, "exchange", "?")
        )
        self.logger.info(
            "Qualified | symbol=%s  conId=%s  primaryExch=%s",
            getattr(self.contract, "localSymbol", None) or self.contract.symbol,
            getattr(self.contract, "conId", "N/A"),
            primary_exch,
        )
        return self.contract

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _place_limit_order(
        self,
        action: str,          # 'BUY' | 'SELL'
        quantity: float,
        limit_price: float,
        order_ref: str = "",
    ) -> Optional[Trade]:
        """
        Submit a GTC limit order and return the Trade object.

        Attaches the base _on_fill / _on_status callbacks automatically;
        subclasses can add more callbacks after this call returns.
        """
        if not self._qualified:
            self.logger.error("Contract not qualified — call qualify() first.")
            return None

        order = LimitOrder(
            action=action,
            totalQuantity=quantity,
            lmtPrice=round(limit_price, 2),
            tif="GTC",
            account=self.account,
            orderRef=order_ref,
        )
        trade = self.ib.placeOrder(self.contract, order)
        self.logger.info(
            "ORDER | %-4s %s x%g @ %.4f | ref=%-30s | orderId=%s",
            action,
            self.contract.symbol,
            quantity,
            limit_price,
            order_ref,
            trade.order.orderId,
        )
        trade.fillEvent += self._on_fill
        trade.statusEvent += self._on_status
        return trade

    def _cancel_order(self, trade: Trade) -> None:
        self.ib.cancelOrder(trade.order)
        self.logger.info(
            "CANCEL | orderId=%s  ref=%s",
            trade.order.orderId,
            trade.order.orderRef,
        )

    # ------------------------------------------------------------------
    # Default event callbacks (override in subclass for custom behaviour)
    # ------------------------------------------------------------------

    def _on_fill(self, trade: Trade, fill) -> None:
        self.logger.info(
            "FILL   | %-4s %s x%g @ %.4f | orderId=%s",
            fill.execution.side,
            self.contract.symbol,
            fill.execution.shares,
            fill.execution.price,
            trade.order.orderId,
        )

    def _on_status(self, trade: Trade) -> None:
        self.logger.debug(
            "STATUS | orderId=%s  status=%-14s  remaining=%g",
            trade.order.orderId,
            trade.orderStatus.status,
            trade.orderStatus.remaining,
        )

    # ------------------------------------------------------------------
    # Context manager (convenience)
    # ------------------------------------------------------------------

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()

    # ------------------------------------------------------------------
    # Lifecycle (subclass responsibility)
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Subscribe to market data and arm the strategy logic."""

    @abstractmethod
    async def stop(self) -> None:
        """Cancel subscriptions and pending orders, release resources."""
