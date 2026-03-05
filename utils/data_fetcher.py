"""
utils/data_fetcher.py
---------------------
Centralised data acquisition and indicator computation.

Classes
-------
IndicatorUtils
    Stateless, vectorised technical indicators (BB, RSI) using pandas.
    All methods are static — use directly without instantiation.

DataFetcher
    Wraps IB market data APIs:
      fetch_historical()       — historical bars, auto-cached to parquet
      fetch_snapshot_price()   — current mid price (one-shot snapshot)
      subscribe_live()         — live tick subscription (returns Ticker)
      cancel_live()            — cancel a live tick subscription
      cancel_historical()      — cancel a keepUpToDate historical stream
      fetch_multiple_stocks()  — batch fetch with IBKR pacing compliance

Parquet cache
    Every live fetch writes a snapshot to  data/cache/<SYMBOL>/<key>.parquet.
    fetch_multiple_stocks() serves from cache when the file was written today.
    Cache files are named with the date so stale data is never silently used.
"""

import asyncio
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, BarDataList, Contract, Ticker

# ── Module-level defaults ──────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path("data/cache")

# IBKR allows ≤ 60 historical-data requests per 10-minute window.
# A 2-second gap between requests in fetch_multiple_stocks is conservative.
# Increase to 10+ seconds if you receive pacing-violation error code 162.
DEFAULT_PACING_DELAY: float = 2.0

_log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# IndicatorUtils
# ══════════════════════════════════════════════════════════════════════════════

class IndicatorUtils:
    """
    Stateless, vectorised technical indicator calculations using pandas.

    Usage (no instantiation needed)::

        sma, upper, lower = IndicatorUtils.bollinger(close)
        rsi_series        = IndicatorUtils.rsi(close)
        df                = IndicatorUtils.bars_to_df(bars)
    """

    @staticmethod
    def bars_to_df(bars: BarDataList) -> pd.DataFrame:
        """Convert an ib_insync BarDataList into a tidy OHLCV DataFrame."""
        return pd.DataFrame(
            [(b.date, b.open, b.high, b.low, b.close, b.volume) for b in bars],
            columns=["date", "open", "high", "low", "close", "volume"],
        )

    @staticmethod
    def bollinger(
        close: pd.Series,
        period: int = 20,
        n_std: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands using population standard deviation (ddof=0).

        Returns
        -------
        (sma, upper_band, lower_band) — all pd.Series aligned to *close*.
        """
        sma = close.rolling(period).mean()
        std = close.rolling(period).std(ddof=0)
        return sma, sma + n_std * std, sma - n_std * std

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Wilder's RSI via exponential smoothing (α = 1/period).

        Matches the standard RSI(14) displayed on TradingView and most
        institutional charting platforms.

        Returns
        -------
        pd.Series of RSI values in [0, 100].  First (period - 1) rows are NaN.
        """
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════════════════════
# DataFetcher
# ══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Central hub for all IB market data requests, with parquet caching.

    Parameters
    ----------
    ib        : A connected ib_insync IB instance.
    cache_dir : Root directory for parquet files (default: data/cache/).
    """

    def __init__(self, ib: IB, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.ib        = ib
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log      = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    async def fetch_historical(
        self,
        contract: Contract,
        duration: str = "3 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        keep_up_to_date: bool = False,
    ) -> BarDataList:
        """
        Fetch historical bars from IBKR and snapshot-cache the initial result.

        Even when *keep_up_to_date=True* the initial bars are written to parquet
        so they are available for offline analysis / Monte Carlo simulations.

        Parameters
        ----------
        contract        : A fully qualified IB Contract.
        duration        : IBKR duration string, e.g. ``"3 D"``, ``"1 M"``.
        bar_size        : IBKR bar-size string, e.g. ``"5 mins"``, ``"1 hour"``.
        what_to_show    : ``"TRADES"``, ``"MIDPOINT"``, ``"BID"``, ``"ASK"``, …
        use_rth         : Restrict to Regular Trading Hours.
        keep_up_to_date : If True the returned BarDataList streams new bars;
                          attach ``bars.updateEvent`` to process them.

        Returns
        -------
        BarDataList — live ib_insync object with data already populated.

        Raises
        ------
        RuntimeError if IBKR returns an empty response.
        """
        symbol = contract.symbol
        self._log.info(
            "fetch_historical | %s  dur=%s  bar=%s  rth=%s  live=%s",
            symbol, duration, bar_size, use_rth, keep_up_to_date,
        )

        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=keep_up_to_date,
        )

        if not bars:
            raise RuntimeError(
                f"No historical data returned for {symbol}. "
                "Verify contract qualification, market hours, and IB data subscription."
            )

        # Snapshot initial data to parquet for offline use
        df = IndicatorUtils.bars_to_df(bars)
        self._save_cache(symbol, bar_size, duration, what_to_show, df)
        self._log.info("fetch_historical | %s → %d bars loaded.", symbol, len(bars))
        return bars

    def cancel_historical(self, bars: BarDataList) -> None:
        """Cancel a keepUpToDate historical data stream."""
        self.ib.cancelHistoricalData(bars)
        self._log.info("Cancelled historical stream.")

    # ------------------------------------------------------------------
    # Live / snapshot price
    # ------------------------------------------------------------------

    async def fetch_snapshot_price(
        self,
        contract: Contract,
        wait_secs: float = 3.0,
    ) -> float:
        """
        Request a one-shot market data snapshot and return the current mid price.

        Parameters
        ----------
        contract  : A fully qualified IB Contract.
        wait_secs : Seconds to wait for the snapshot to populate (default 3).

        Returns
        -------
        float — current market price.

        Raises
        ------
        RuntimeError if no valid price is received within *wait_secs*.
        """
        symbol = contract.symbol
        self._log.info("fetch_snapshot_price | requesting %s …", symbol)
        ticker = self.ib.reqMktData(contract, "", True, False)   # snapshot=True
        await asyncio.sleep(wait_secs)
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        if math.isnan(price) or price <= 0:
            raise RuntimeError(
                f"Could not obtain a valid snapshot price for {symbol}. "
                "Is the market open and the IB data subscription active?"
            )

        self._log.info("fetch_snapshot_price | %s = %.4f", symbol, price)
        return price

    def subscribe_live(
        self,
        contract: Contract,
        tick_types: str = "",
        snapshot: bool = False,
    ) -> Ticker:
        """
        Subscribe to live streaming tick data.

        Parameters
        ----------
        contract   : A fully qualified IB Contract.
        tick_types : Comma-separated generic tick type IDs (e.g. ``"233,236"``).
        snapshot   : True for a one-shot snapshot (rarely used in live trading).

        Returns
        -------
        Ticker — attach ``ticker.updateEvent`` to process incoming ticks.
        """
        self._log.info("subscribe_live | %s", contract.symbol)
        return self.ib.reqMktData(contract, tick_types, snapshot, False)

    def cancel_live(self, contract: Contract) -> None:
        """Cancel a live tick subscription."""
        self.ib.cancelMktData(contract)
        self._log.info("cancel_live | %s", contract.symbol)

    # ------------------------------------------------------------------
    # Batch fetch with IBKR pacing compliance
    # ------------------------------------------------------------------

    async def fetch_multiple_stocks(
        self,
        contracts: List[Contract],
        duration: str = "1 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        pacing_delay: float = DEFAULT_PACING_DELAY,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for a list of contracts, honouring IBKR's
        pacing limits (≤ 60 historical requests per 10-minute window).

        Requests are executed **sequentially** with a configurable inter-request
        delay. Contracts already in today's parquet cache are served locally,
        so they do not count against the pacing limit.

        On pacing-violation error (code 162) the request is retried after a
        longer back-off delay.

        Parameters
        ----------
        contracts     : List of qualified IB Contracts.
        duration      : IBKR duration string applied to every contract.
        bar_size      : IBKR bar-size string applied to every contract.
        what_to_show  : Data type applied to every contract.
        use_rth       : Restrict to Regular Trading Hours.
        pacing_delay  : Seconds between live requests (≥ 2 recommended).
                        Raise to 10+ if you receive error code 162.
        use_cache     : Serve from today's parquet cache when available.
        max_retries   : Retry attempts per contract on transient errors.

        Returns
        -------
        Dict[symbol, pd.DataFrame] — one OHLCV DataFrame per contract.
        Failed contracts are omitted (an error is logged per contract).
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(contracts)

        for idx, contract in enumerate(contracts, start=1):
            symbol = contract.symbol
            self._log.info("fetch_multiple_stocks [%d/%d] | %s", idx, total, symbol)

            # ── Cache hit? ─────────────────────────────────────────────
            if use_cache:
                cached = self._load_cache(symbol, bar_size, duration, what_to_show)
                if cached is not None:
                    self._log.info("Cache HIT for %s — skipping live request.", symbol)
                    results[symbol] = cached
                    continue

            # ── Live fetch with retry on pacing / transient errors ─────
            for attempt in range(1, max_retries + 1):
                try:
                    bars = await self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow=what_to_show,
                        useRTH=use_rth,
                        formatDate=1,
                        keepUpToDate=False,
                    )

                    if not bars:
                        self._log.warning(
                            "Empty response for %s (attempt %d/%d).",
                            symbol, attempt, max_retries,
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(pacing_delay * 5)
                        continue

                    df = IndicatorUtils.bars_to_df(bars)
                    self._save_cache(symbol, bar_size, duration, what_to_show, df)
                    results[symbol] = df
                    self._log.info("Fetched %d bars for %s.", len(df), symbol)
                    break   # success — exit retry loop

                except Exception as exc:
                    err_str  = str(exc)
                    # IBKR pacing-violation error code is 162
                    is_pacing = "162" in err_str or "pacing" in err_str.lower()
                    back_off  = pacing_delay * 30 if is_pacing else pacing_delay * 3
                    self._log.warning(
                        "%s error for %s (attempt %d/%d): %s  → back-off %.0fs",
                        "Pacing" if is_pacing else "Transient",
                        symbol, attempt, max_retries, err_str, back_off,
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(back_off)
                    else:
                        self._log.error(
                            "Giving up on %s after %d attempts.", symbol, max_retries
                        )

            # ── Inter-request gap (skip after the last contract) ───────
            if idx < total:
                self._log.debug("Pacing delay %.1fs …", pacing_delay)
                await asyncio.sleep(pacing_delay)

        self._log.info(
            "fetch_multiple_stocks complete | %d/%d contracts succeeded.",
            len(results), total,
        )
        return results

    # ------------------------------------------------------------------
    # Parquet cache — internal helpers
    # ------------------------------------------------------------------

    def _cache_path(
        self,
        symbol: str,
        bar_size: str,
        duration: str,
        what_to_show: str,
    ) -> Path:
        """Return the parquet path for a given parameter combination + today's date."""
        safe_bar = bar_size.replace(" ", "_")
        safe_dur = duration.replace(" ", "_")
        date_tag = datetime.today().strftime("%Y%m%d")
        filename = f"{symbol}__{safe_bar}__{safe_dur}__{what_to_show}__{date_tag}.parquet"
        return self.cache_dir / symbol / filename

    def _save_cache(
        self,
        symbol: str,
        bar_size: str,
        duration: str,
        what_to_show: str,
        df: pd.DataFrame,
    ) -> None:
        path = self._cache_path(symbol, bar_size, duration, what_to_show)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self._log.info("Cached %d rows → %s", len(df), path)

    def _load_cache(
        self,
        symbol: str,
        bar_size: str,
        duration: str,
        what_to_show: str,
    ) -> Optional[pd.DataFrame]:
        """Return today's cached DataFrame, or None if the file is absent or stale."""
        path = self._cache_path(symbol, bar_size, duration, what_to_show)
        if not path.exists():
            return None
        mtime_date = datetime.fromtimestamp(path.stat().st_mtime).date()
        if mtime_date != datetime.today().date():
            self._log.debug("Cache stale for %s (written %s).", symbol, mtime_date)
            return None
        df = pd.read_parquet(path)
        self._log.debug("Loaded %d rows from cache: %s", len(df), path)
        return df
