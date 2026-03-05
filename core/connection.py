"""
core/connection.py
------------------
Manages the connection lifecycle to the Interactive Brokers Gateway / TWS.

Usage:
    from core.connection import IBConnection

    conn = IBConnection()
    ib = conn.connect()          # returns a live IB() instance
    conn.disconnect()
"""

import logging
import time
from typing import Optional

from ib_insync import IB
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)


class IBConnection:
    """Encapsulates connect / disconnect logic for ib_insync."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        client_id: int = None,
        readonly: bool = False,
        timeout: int = 10,
    ):
        """
        Parameters
        ----------
        host      : Gateway / TWS host. Defaults to IB_HOST env var or '127.0.0.1'.
        port      : Gateway / TWS port. Defaults to IB_PORT env var or 7497 (paper trading).
                    Use 7496 for live trading, 4001/4002 for IB Gateway paper/live.
        client_id : Unique client ID (0-999). Defaults to IB_CLIENT_ID env var or 1.
        readonly  : When True, no orders can be placed (safe for data-only sessions).
        timeout   : Seconds to wait for the connection handshake.
        """
        self.host = host or os.getenv("IB_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IB_PORT", 7497))
        self.client_id = int(client_id or os.getenv("IB_CLIENT_ID", 1))
        self.readonly = readonly
        self.timeout = timeout

        self._ib: Optional[IB] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def connect(self, max_retries: int = 3, retry_delay: float = 5.0) -> IB:
        """
        Connect to the IBKR Gateway / TWS with automatic retries.

        Returns
        -------
        IB
            A connected ib_insync IB instance ready for use.

        Raises
        ------
        ConnectionError
            If all retry attempts are exhausted.
        """
        if self._ib and self._ib.isConnected():
            logger.info("Already connected (clientId=%d).", self.client_id)
            return self._ib

        self._ib = IB()

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Connecting to IBKR Gateway at %s:%d (clientId=%d, attempt %d/%d) …",
                    self.host,
                    self.port,
                    self.client_id,
                    attempt,
                    max_retries,
                )
                self._ib.connect(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    readonly=self.readonly,
                    timeout=self.timeout,
                )
                logger.info(
                    "Connected. Server version: %s | Account: %s",
                    self._ib.client.serverVersion(),
                    self._ib.wrapper.accounts,
                )
                self._register_event_handlers()
                return self._ib

            except Exception as exc:
                logger.warning("Connection attempt %d failed: %s", attempt, exc)
                if attempt < max_retries:
                    logger.info("Retrying in %.1f seconds …", retry_delay)
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Could not connect to IBKR Gateway at {self.host}:{self.port} "
                        f"after {max_retries} attempts."
                    ) from exc

    def disconnect(self) -> None:
        """Gracefully disconnect from the Gateway / TWS."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR Gateway.")
        self._ib = None

    @property
    def ib(self) -> Optional[IB]:
        """Return the underlying IB instance, or None if not connected."""
        return self._ib

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> IB:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_event_handlers(self) -> None:
        """Attach logging callbacks to key IB events."""
        self._ib.errorEvent += self._on_error
        self._ib.disconnectedEvent += self._on_disconnect

    def _on_error(self, req_id: int, error_code: int, error_string: str, contract) -> None:
        # IB sends informational messages (code < 2000) alongside real errors.
        if error_code < 2000:
            logger.error(
                "IB error [reqId=%d, code=%d]: %s", req_id, error_code, error_string
            )
        else:
            logger.warning(
                "IB warning [reqId=%d, code=%d]: %s", req_id, error_code, error_string
            )

    def _on_disconnect(self) -> None:
        logger.warning("Lost connection to IBKR Gateway.")
