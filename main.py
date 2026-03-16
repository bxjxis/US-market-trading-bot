"""
main.py
-------
Entry point: connects to IBKR Gateway and runs all active strategies concurrently.

Usage:
    python main.py
"""

import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.connection import IBConnection
from strategies import AMZNReversionStrategy, CLFGridStrategy, SmallCapArbStrategy

# ── Logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)   # ensure directory exists before opening file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Rotate at 10 MB, keep 7 daily files — never fills the disk
        RotatingFileHandler(
            "logs/trading_bot.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=7,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ── Active grid configurations ─────────────────────────────────────────────────
# Each entry overrides CLFGridStrategy module-level defaults for that instance.
# Validated via 90-day backtest; see scripts/backtest_grid.py for reproduction.
_GRID_CONFIGS = [
    {
        "SYMBOL":         "NVTS",
        "GRID_RATIO":     1.007,
        "NUM_BUY_LEVELS": 20,
        "ACCOUNT_SIZE":   45_000,
        "ATR_ADAPTIVE":   True,
        "EVENT_GUARD":    True,
    },
    {
        "SYMBOL":         "TXG",
        "GRID_RATIO":     1.010,
        "NUM_BUY_LEVELS": 15,
        "ACCOUNT_SIZE":   45_000,
        "ATR_ADAPTIVE":   False,
        "EVENT_GUARD":    True,
    },
]


async def run_strategies(ib, account: str) -> None:
    strategies = [
        # Grid strategies — one instance per configured symbol
        *[CLFGridStrategy(ib, account=account, params=cfg) for cfg in _GRID_CONFIGS],
        # Other strategies
        AMZNReversionStrategy(ib, account=account),
        SmallCapArbStrategy(ib, account=account),
    ]

    # Start all strategies concurrently
    await asyncio.gather(*(s.start() for s in strategies))
    logger.info("All strategies started. Running event loop — press Ctrl-C to exit.")

    try:
        await asyncio.Event().wait()   # keep running until cancelled
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Stopping strategies …")
        await asyncio.gather(*(s.stop() for s in strategies))


async def main() -> None:
    logger.info("Starting trading bot …")

    conn = IBConnection()
    ib = conn.connect()
    account = ib.managedAccounts()[0]
    logger.info("Connected | account=%s", account)

    try:
        await run_strategies(ib, account)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        conn.disconnect()
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
