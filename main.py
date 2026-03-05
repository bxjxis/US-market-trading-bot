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

from core.connection import IBConnection
from strategies import AMZNReversionStrategy, CLFGridStrategy, SmallCapArbStrategy

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log"),
    ],
)
logger = logging.getLogger(__name__)


async def run_strategies(ib, account: str) -> None:
    strategies = [
        CLFGridStrategy(ib, account=account),
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
