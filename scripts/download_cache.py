"""
scripts/download_cache.py
--------------------------
One-shot script to download historical OHLCV data from IBKR and populate
the data/cache/ folder with Parquet files used by the backtester.

Run this before your first backtesting / optimisation session and whenever
you want to refresh the training data.

Usage
-----
    python scripts/download_cache.py

    # Custom duration or bar size
    python scripts/download_cache.py --duration "30 D" --bar-size "1 min"

Pre-requisites
--------------
* IBKR Gateway or TWS must be running and accepting API connections.
  Default: 127.0.0.1:7497  (paper trading port).
  Override with IB_HOST / IB_PORT / IB_CLIENT_ID env vars or a .env file.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Make project root importable when run as a script ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ib_insync import Stock, util as ib_util

from core.connection import IBConnection
from utils.data_fetcher import DataFetcher

# ── Logging: show INFO from this script, suppress ib_insync chatter ──────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("download_cache")
_log.setLevel(logging.INFO)

# ── Symbols to download ───────────────────────────────────────────────────────
CONTRACTS = [
    Stock("AMZN", "SMART", "USD"),
    Stock("CLF",  "SMART", "USD"),
    Stock("IREN", "SMART", "USD"),
    Stock("WULF", "SMART", "USD"),
]

# IBKR pacing: no more than 60 historical requests per 10-minute window.
# 3 seconds between requests is conservative and avoids error code 162.
PACING_DELAY_SECS = 3.0


# ─────────────────────────────────────────────────────────────────────────────

async def _fetch(ib, duration: str, bar_size: str, symbols: list) -> None:
    """
    Async body: qualifies contracts and downloads data.
    Receives an already-connected IB instance so there is no
    nested event-loop conflict with ib_insync's own loop.
    """
    account = ib.managedAccounts()[0]
    _log.info("Connected | account=%s", account)

    contracts = [Stock(s.upper(), "SMART", "USD") for s in symbols]
    print("Qualifying contracts …")
    qualified = await ib.qualifyContractsAsync(*contracts)
    if len(qualified) < len(contracts):
        missing = {c.symbol for c in contracts} - {c.symbol for c in qualified}
        print(f"  WARNING: could not qualify: {missing}  (skipping)")

    fetcher = DataFetcher(ib)
    total   = len(qualified)

    for i, contract in enumerate(qualified, start=1):
        symbol = contract.symbol
        print(f"[{i}/{total}] Downloading {symbol} ({duration}, {bar_size}) … ",
              end="", flush=True)

        try:
            bars = await fetcher.fetch_historical(
                contract,
                duration        = duration,
                bar_size        = bar_size,
                what_to_show    = "TRADES",
                use_rth         = True,
                keep_up_to_date = False,   # one-shot snapshot, no live stream
            )
            print(f"Done.  ({len(bars):,} bars)")

        except Exception as exc:
            print(f"FAILED — {exc}")
            _log.warning("Download failed for %s: %s", symbol, exc)

        # Respect IBKR pacing limit between requests
        if i < total:
            await asyncio.sleep(PACING_DELAY_SECS)

    print("\nAll downloads complete.")
    _log.info("Cache populated at: %s", Path("data/cache").resolve())


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download IBKR historical data into data/cache/ Parquet files."
    )
    parser.add_argument(
        "--duration", default="20 D",
        help="IBKR duration string (default: '20 D').  "
             "Examples: '5 D', '1 M', '3 M'.",
    )
    parser.add_argument(
        "--bar-size", default="5 mins",
        dest="bar_size",
        help="IBKR bar-size string (default: '5 mins').  "
             "Examples: '1 min', '15 mins', '1 hour'.",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        metavar="SYM",
        help="Symbols to download (default: AMZN CLF IREN WULF).  "
             "Example: --symbols NVTS CTMX",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Connect synchronously BEFORE starting the async event loop ────────────
    # ib_insync's IB.connect() is synchronous and internally calls
    # loop.run_until_complete().  Wrapping it in asyncio.run() creates a
    # nested-loop conflict.  The correct pattern is:
    #   1. connect() at the top level (no running loop yet)
    #   2. hand the connected IB to ib_insync's util.run() for async work
    print("Connecting to IBKR Gateway …")
    conn = IBConnection()
    ib   = conn.connect()

    try:
        symbols = args.symbols or [c.symbol for c in CONTRACTS]
        ib_util.run(_fetch(ib, duration=args.duration, bar_size=args.bar_size, symbols=symbols))
    finally:
        conn.disconnect()
        _log.info("Disconnected.")
