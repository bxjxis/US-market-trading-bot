"""
core/backtester.py
------------------
Event-driven backtesting engine that reuses the live strategy code verbatim.

Architecture
------------
SimulatedIB
    Drop-in replacement for ib_insync.IB.  Strategies are constructed with
    this object so all IB calls are intercepted and served from parquet data.

SimulatedTrade
    Mimics ib_insync.Trade: exposes fillEvent / statusEvent and order metadata.

SimulatedTicker
    Mimics ib_insync.Ticker: updateEvent + marketPrice().

SimulatedBarDataList
    Mimics ib_insync.BarDataList: a list subclass with updateEvent.

BacktestEngine
    Feeds bars chronologically, drives event dispatch, tracks equity.

Key design principles
---------------------
* "One version of truth" — the exact same strategy class files run in
  both live and backtest modes.  No strategy logic is duplicated here.
* Order fill simulation: BUY fills when bar.low <= limit; SELL fills when
  bar.high >= limit.  Fill price = min/max(limit, open) ± SLIPPAGE_PCT.
* Commissions: IBKR Tiered — max($0.35, qty × $0.0035) per order.
* asyncio.sleep and DataFetcher cache are patched to no-ops during startup
  so the backtester runs at full speed without disk I/O or real-time waits.

Usage
-----
    from core.backtester import BacktestEngine, BacktestConfig
    from strategies.clf_grid import CLFGridStrategy
    from strategies.amzn_reversion import AMZNReversionStrategy
    from strategies.smallcap_arb import SmallCapArbStrategy

    cfg = BacktestConfig(initial_capital=180_000)
    engine = BacktestEngine(cfg)
    result = engine.run(
        strategies=[CLFGridStrategy, AMZNReversionStrategy, SmallCapArbStrategy],
        symbols=["CLF", "AMZN", "IREN", "WULF"],
    )
    print(result.summary())
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pandas as pd
from eventkit import Event

from utils.data_fetcher import DataFetcher

# ── Commission / slippage constants ───────────────────────────────────────────
_COMMISSION_PER_SHARE = 0.0035   # IBKR Tiered
_COMMISSION_MIN       = 0.35     # per order
_SLIPPAGE_PCT         = 0.001    # 0.1 % per fill

_log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight data objects mirroring ib_insync internals
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _Bar:
    """Single OHLCV bar — matches ib_insync Bar attribute names."""
    date:   Any
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class _Execution:
    price:  float
    shares: float
    side:   str


@dataclass
class _Fill:
    execution: _Execution


@dataclass
class _OrderStatus:
    status:    str   = "Submitted"
    remaining: float = 0.0


@dataclass
class _Order:
    orderId:       int
    orderRef:      str
    action:        str
    totalQuantity: float
    lmtPrice:      float
    tif:           str
    account:       str = ""


# ═════════════════════════════════════════════════════════════════════════════
# SimulatedTrade
# ═════════════════════════════════════════════════════════════════════════════

class SimulatedTrade:
    """Mimics ib_insync.Trade with fillEvent / statusEvent / isActive()."""

    def __init__(self, order: _Order) -> None:
        self.order       = order
        self.orderStatus = _OrderStatus(
            status="Submitted", remaining=order.totalQuantity
        )
        self.fillEvent   = Event("fillEvent")
        self.statusEvent = Event("statusEvent")
        self._active     = True

    def isActive(self) -> bool:
        return self._active

    def _fill(self, exec_price: float) -> None:
        self._active               = False
        self.orderStatus.status    = "Filled"
        self.orderStatus.remaining = 0.0
        fill = _Fill(_Execution(
            price  = exec_price,
            shares = self.order.totalQuantity,
            side   = self.order.action,
        ))
        self.fillEvent.emit(self, fill)
        self.statusEvent.emit(self)

    def _cancel(self) -> None:
        self._active            = False
        self.orderStatus.status = "Cancelled"
        self.statusEvent.emit(self)


# ═════════════════════════════════════════════════════════════════════════════
# SimulatedBarDataList
# ═════════════════════════════════════════════════════════════════════════════

class SimulatedBarDataList(list):
    """Mimics ib_insync.BarDataList: a list with updateEvent."""

    def __init__(self, rows: list) -> None:
        super().__init__(rows)
        self.updateEvent = Event("updateEvent")


# ═════════════════════════════════════════════════════════════════════════════
# SimulatedTicker
# ═════════════════════════════════════════════════════════════════════════════

class SimulatedTicker:
    """Mimics ib_insync.Ticker: updateEvent + marketPrice()."""

    def __init__(self, contract, price: float = 0.0) -> None:
        self.contract    = contract
        self._price      = price
        self.bar_high    = price   # current bar high (used by ADX filter)
        self.bar_low     = price   # current bar low  (used by ADX filter)
        self.updateEvent = Event("updateEvent")

    def marketPrice(self) -> float:
        return self._price


# ═════════════════════════════════════════════════════════════════════════════
# SimulatedIB
# ═════════════════════════════════════════════════════════════════════════════

class SimulatedIB:
    """
    Drop-in replacement for ib_insync.IB.

    Strategies are instantiated with this object instead of a real IB
    connection.  All market-data, order, and contract calls are intercepted
    and served from the in-memory ``data`` dict (loaded from parquet).

    The backtester calls ``advance_bar(symbol, bar)`` on each replay step;
    this method checks order fills and fires both the BarDataList.updateEvent
    and the Ticker.updateEvent so that strategy callbacks execute normally.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        warmup_bars: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        data        : {symbol: OHLCV DataFrame} pre-loaded from parquet.
        warmup_bars : Rows returned as initial history by reqHistoricalDataAsync.
                      Strategy indicators are seeded from these rows before
                      any updateEvent fires.
        """
        self._data        = data
        self._warmup_bars = warmup_bars
        self._bar_lists:  Dict[str, SimulatedBarDataList] = {}
        self._tickers:    Dict[str, SimulatedTicker]      = {}
        # orderId → (SimulatedTrade, symbol)
        self._pending:    Dict[int, Tuple[SimulatedTrade, str]] = {}
        self._order_seq   = 0

        # Populated during replay — consumed by BacktestEngine
        self.trade_log: List[Dict] = []

    # ── Contract qualification ─────────────────────────────────────────────

    async def qualifyContractsAsync(self, *contracts):
        """No-op: contracts already have symbol/exchange info."""
        return list(contracts)

    def managedAccounts(self) -> List[str]:
        return ["DU_BACKTEST"]

    # ── Historical data ────────────────────────────────────────────────────

    async def reqHistoricalDataAsync(
        self,
        contract,
        endDateTime="",
        durationStr="",
        barSizeSetting="",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    ) -> SimulatedBarDataList:
        symbol = contract.symbol
        df     = self._data.get(symbol)
        if df is None or df.empty:
            raise RuntimeError(
                f"No backtest data available for '{symbol}'. "
                "Populate data/cache/ by running the live bot at least once."
            )

        warmup = df.iloc[: self._warmup_bars]
        rows   = [
            _Bar(r.date, r.open, r.high, r.low, r.close, r.volume)
            for r in warmup.itertuples(index=False)
        ]
        bar_list = SimulatedBarDataList(rows)
        self._bar_lists[symbol] = bar_list
        return bar_list

    def cancelHistoricalData(self, bars) -> None:
        pass  # no-op

    # ── Live market data ───────────────────────────────────────────────────

    def reqMktData(
        self,
        contract,
        tick_types: str = "",
        snapshot: bool = False,
        regulatory: bool = False,
    ) -> SimulatedTicker:
        symbol = contract.symbol
        df     = self._data.get(symbol)

        # Seed with the last warmup bar's close so strategies get a valid price
        seed_price = 0.0
        if df is not None and len(df) >= self._warmup_bars:
            seed_price = float(df.iloc[self._warmup_bars - 1]["close"])

        ticker = SimulatedTicker(contract, seed_price)
        self._tickers[symbol] = ticker
        return ticker

    def cancelMktData(self, contract) -> None:
        self._tickers.pop(contract.symbol, None)

    # ── Order management ──────────────────────────────────────────────────

    def placeOrder(self, contract, order) -> SimulatedTrade:
        self._order_seq += 1
        mock_order = _Order(
            orderId       = self._order_seq,
            orderRef      = getattr(order, "orderRef", ""),
            action        = order.action,
            totalQuantity = float(order.totalQuantity),
            lmtPrice      = float(order.lmtPrice),
            tif           = getattr(order, "tif", "GTC"),
            account       = getattr(order, "account", ""),
        )
        trade = SimulatedTrade(mock_order)
        self._pending[self._order_seq] = (trade, contract.symbol)
        return trade

    def cancelOrder(self, order) -> None:
        key = order.orderId
        if key in self._pending:
            trade, _ = self._pending.pop(key)
            trade._cancel()

    def cancel_all_pending(self) -> int:
        """Cancel every pending order. Returns the number cancelled."""
        count = len(self._pending)
        for key in list(self._pending.keys()):
            trade, _ = self._pending.pop(key)
            trade._cancel()
        return count

    # ── Simulation driver ─────────────────────────────────────────────────

    def advance_bar(self, symbol: str, bar: _Bar) -> None:
        """
        Advance the simulation by one bar for the given symbol.

        Execution order:
          1. Check pending limit orders against this bar's high/low (fills first).
          2. Append the bar to the BarDataList and fire updateEvent (has_new_bar=True).
          3. Update the Ticker price and fire its updateEvent (safety-switch etc.).

        New orders placed inside strategy callbacks in step 2/3 are NOT checked
        for fills until the *next* advance_bar call — correct simulation behaviour.
        """
        self._check_fills(symbol, bar)

        bar_list = self._bar_lists.get(symbol)
        if bar_list is not None:
            bar_list.append(bar)
            bar_list.updateEvent.emit(bar_list, True)

        ticker = self._tickers.get(symbol)
        if ticker is not None:
            ticker._price   = bar.close
            ticker.bar_high = bar.high
            ticker.bar_low  = bar.low
            ticker.updateEvent.emit(ticker)

    def _check_fills(self, symbol: str, bar: _Bar) -> None:
        """Check every pending order against the current bar's OHLC."""
        filled_keys = []

        for key, (trade, sym) in list(self._pending.items()):
            if sym != symbol or not trade.isActive():
                continue

            order = trade.order

            if order.action == "BUY":
                # BUY limit fills when price dips to or below the limit
                if bar.low <= order.lmtPrice:
                    # Gap-down open fills at open; otherwise fills at limit
                    fill_base  = min(order.lmtPrice, bar.open)
                    exec_price = fill_base * (1.0 + _SLIPPAGE_PCT)
                    commission = max(_COMMISSION_MIN,
                                     order.totalQuantity * _COMMISSION_PER_SHARE)
                    self._record_trade(trade, symbol, exec_price, commission)
                    trade._fill(exec_price)
                    filled_keys.append(key)

            elif order.action == "SELL":
                # SELL limit fills when price rises to or above the limit
                if bar.high >= order.lmtPrice:
                    fill_base  = max(order.lmtPrice, bar.open)
                    exec_price = fill_base * (1.0 - _SLIPPAGE_PCT)
                    commission = max(_COMMISSION_MIN,
                                     order.totalQuantity * _COMMISSION_PER_SHARE)
                    self._record_trade(trade, symbol, exec_price, commission)
                    trade._fill(exec_price)
                    filled_keys.append(key)

        for key in filled_keys:
            self._pending.pop(key, None)

    def _record_trade(
        self,
        trade: SimulatedTrade,
        symbol: str,
        price: float,
        commission: float,
    ) -> None:
        self.trade_log.append({
            "symbol":     symbol,
            "action":     trade.order.action,
            "qty":        trade.order.totalQuantity,
            "price":      price,
            "commission": commission,
            "slippage":   price * _SLIPPAGE_PCT * trade.order.totalQuantity,
            "order_ref":  trade.order.orderRef,
        })


# ═════════════════════════════════════════════════════════════════════════════
# BacktestConfig / BacktestResult
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Runtime parameters for BacktestEngine."""
    initial_capital: float = 1_000.0
    portfolio_value: float = 180_000.0   # used for VaR dollar scaling
    warmup_bars:     int   = 100         # bars fed as initial history (seeds indicators)
    data_dir:        Path  = field(default_factory=lambda: Path("data/cache"))
    drawdown_cap:    float = 0.0         # cancel all pending orders when DD > this fraction (0 = off)


class BacktestResult:
    """
    Holds all outputs of a completed backtest run.

    Attributes
    ----------
    trade_log    : DataFrame with columns [symbol, action, qty, price,
                   commission, slippage, order_ref].
    equity_curve : DataFrame with columns [timestamp, equity].
    config       : The BacktestConfig used for this run.
    """

    def __init__(
        self,
        trade_log:    List[Dict],
        equity_curve: List[Dict],
        config:       BacktestConfig,
    ) -> None:
        self.trade_log    = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
            columns=["symbol", "action", "qty", "price",
                     "commission", "slippage", "order_ref"]
        )
        self.equity_curve = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(
            columns=["timestamp", "equity"]
        )
        self.config = config

    @property
    def returns(self) -> pd.Series:
        """Bar-level returns derived from the equity curve."""
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        return (
            self.equity_curve
            .set_index("timestamp")["equity"]
            .pct_change()
            .dropna()
        )

    @property
    def daily_returns(self) -> pd.Series:
        """Daily returns resampled from the equity curve."""
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        eq = (
            self.equity_curve
            .set_index("timestamp")["equity"]
        )
        eq.index = pd.to_datetime(eq.index)
        return eq.resample("1D").last().pct_change().dropna()

    def summary(self) -> str:
        """Human-readable backtest summary."""
        from utils.dashboard_stats import (
            compute_sharpe, compute_sortino,
            compute_max_drawdown, compute_calmar, compute_var_dollars,
        )
        if self.equity_curve.empty:
            return "No equity data — no trades executed."

        eq       = self.equity_curve.set_index("timestamp")["equity"]
        dr       = self.daily_returns
        sharpe   = compute_sharpe(dr)
        sortino  = compute_sortino(dr)
        mdd      = compute_max_drawdown(eq)
        calmar   = compute_calmar(eq)
        var_usd  = compute_var_dollars(dr, self.config.portfolio_value)
        n_trades = len(self.trade_log)
        total_comm = (
            self.trade_log["commission"].sum() if not self.trade_log.empty else 0.0
        )
        final_eq = eq.iloc[-1]
        pnl      = final_eq - self.config.initial_capital

        lines = [
            "─" * 52,
            "  Backtest Result",
            "─" * 52,
            f"  Trades          : {n_trades}",
            f"  Total Commissions: ${total_comm:,.2f}",
            f"  Initial Capital : ${self.config.initial_capital:,.2f}",
            f"  Final Equity    : ${final_eq:,.2f}",
            f"  Net P&L         : ${pnl:,.2f}  ({pnl / self.config.initial_capital * 100:.2f}%)",
            f"  Sharpe Ratio    : {sharpe:.3f}",
            f"  Sortino Ratio   : {sortino:.3f}",
            f"  Max Drawdown    : {mdd * 100:.2f}%",
            f"  Calmar Ratio    : {calmar:.3f}",
            f"  30d 99% VaR     : ${var_usd:,.0f}  (on ${self.config.portfolio_value:,.0f} portfolio)",
            "─" * 52,
        ]
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# BacktestEngine
# ═════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Orchestrates a full event-driven backtest.

    Steps
    -----
    1. Load parquet files from data/cache/<SYMBOL>/*.parquet (most recent file).
    2. Create a SimulatedIB seeded with that data.
    3. Instantiate each strategy class with the SimulatedIB.
    4. Call strategy.start() with asyncio.sleep and DataFetcher caching patched
       to no-ops so startup is instant.
    5. Build a unified timeline from all replay bars (after the warmup window),
       sorted by timestamp.
    6. Replay bar-by-bar: check fills → fire bar events → track equity.
    7. Call strategy.stop() and return a BacktestResult.
    """

    def __init__(self, config: BacktestConfig = None) -> None:
        self.config = config or BacktestConfig()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        strategies:      List,
        symbols:         List[str],
        strategy_params: Dict[str, Dict] = None,
    ) -> BacktestResult:
        """
        Run a full backtest synchronously.

        Parameters
        ----------
        strategies      : List of strategy *classes* (not instances).
        symbols         : Symbols to load from the parquet cache.
        strategy_params : ``{StrategyClassName: {param: value}}`` overrides.

        Returns
        -------
        BacktestResult
        """
        return asyncio.run(
            self._async_run(strategies, symbols, strategy_params or {})
        )

    # ── Internal implementation ────────────────────────────────────────────

    def _load_parquet(self, symbol: str) -> pd.DataFrame:
        """Load the most-recent parquet snapshot for a symbol."""
        cache_dir = self.config.data_dir / symbol
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"No cache directory for '{symbol}' at {cache_dir}. "
                "Run the live bot at least once to populate data/cache/."
            )
        files = sorted(cache_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime)
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for '{symbol}' in {cache_dir}."
            )
        path = files[-1]  # most recent by modification time
        _log.info("Loading %-6s from %s", symbol, path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _load_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        return {sym: self._load_parquet(sym) for sym in symbols}

    async def _async_run(
        self,
        strategy_classes: List,
        symbols:          List[str],
        strategy_params:  Dict[str, Dict],
    ) -> BacktestResult:
        data   = self._load_all(symbols)
        sim_ib = SimulatedIB(data, warmup_bars=self.config.warmup_bars)

        # ── Strategy startup with I/O patches ─────────────────────────────
        # 1. asyncio.sleep → instant (avoids 3-second wait in fetch_snapshot_price)
        # 2. DataFetcher._load_cache → always miss (force data through SimulatedIB)
        # 3. DataFetcher._save_cache → no-op (don't write to disk during backtest)
        instances = []
        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as _,
            patch.object(DataFetcher, "_load_cache", return_value=None),
            patch.object(DataFetcher, "_save_cache", return_value=None),
        ):
            for cls in strategy_classes:
                params = strategy_params.get(cls.__name__, {})
                s = cls(sim_ib, account="DU_BACKTEST", params=params)
                await s.start()
                instances.append(s)

        # ── Build unified replay timeline ──────────────────────────────────
        frames = []
        for sym, df in data.items():
            replay = df.iloc[self.config.warmup_bars :].copy()
            replay["symbol"] = sym
            frames.append(replay)

        if not frames:
            raise RuntimeError("No replay data after the warmup period.")

        timeline = (
            pd.concat(frames, ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )

        # ── Replay loop ────────────────────────────────────────────────────
        cash:          float           = self.config.initial_capital
        holdings:      Dict[str, float] = {}   # symbol → net shares (+ long, − short)
        last_prices:   Dict[str, float] = {}
        equity_log:    List[Dict]      = []
        last_fill_idx: int             = 0
        peak_equity:   float           = self.config.initial_capital
        dd_cap_hit:    bool            = False

        for _, row in timeline.iterrows():
            symbol = row["symbol"]
            bar    = _Bar(
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

            # Always update price before any early-exit — keeps MtM accurate
            last_prices[symbol] = bar.close

            # If drawdown cap was hit, continue tracking mark-to-market but skip fills
            if dd_cap_hit:
                unrealised = sum(
                    qty * last_prices.get(s, 0.0) for s, qty in holdings.items()
                )
                equity_log.append({"timestamp": bar.date, "equity": cash + unrealised})
                continue

            # advance_bar: fills pending orders, fires bar/ticker events
            sim_ib.advance_bar(symbol, bar)

            # Process fills generated in this step
            new_fills = sim_ib.trade_log[last_fill_idx:]
            for t in new_fills:
                qty    = t["qty"]
                price  = t["price"]
                comm   = t["commission"]
                sym    = t["symbol"]
                sign   = 1.0 if t["action"] == "BUY" else -1.0

                # Cash flow: BUY costs money, SELL earns money
                cash -= sign * price * qty + comm
                holdings[sym] = holdings.get(sym, 0.0) + sign * qty

            last_fill_idx = len(sim_ib.trade_log)

            # Mark-to-market: equity = cash + unrealised position value
            unrealised = sum(
                qty * last_prices.get(s, 0.0) for s, qty in holdings.items()
            )
            current_eq = cash + unrealised
            equity_log.append({"timestamp": bar.date, "equity": current_eq})

            # Drawdown cap: halt new fills once peak-to-trough DD exceeds threshold
            if self.config.drawdown_cap > 0:
                if current_eq > peak_equity:
                    peak_equity = current_eq
                elif (peak_equity > 0
                      and (peak_equity - current_eq) / peak_equity > self.config.drawdown_cap):
                    n = sim_ib.cancel_all_pending()
                    dd_cap_hit = True
                    _log.warning(
                        "Drawdown cap %.1f%% hit (peak=%.2f current=%.2f) — %d pending orders cancelled.",
                        self.config.drawdown_cap * 100, peak_equity, current_eq, n,
                    )

        # ── Cleanup ────────────────────────────────────────────────────────
        for s in instances:
            try:
                await s.stop()
            except Exception:
                pass

        return BacktestResult(
            trade_log    = sim_ib.trade_log,
            equity_curve = equity_log,
            config       = self.config,
        )
