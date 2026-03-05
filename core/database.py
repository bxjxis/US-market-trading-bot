"""
core/database.py
----------------
Trade logging and equity-snapshot persistence.

Schema
------
backtest_runs   — one row per backtest / live session
trades          — every filled order (strategy, symbol, qty, price, fees)
equity_snapshots — periodic NAV snapshots for drawdown / VaR computation

Backends
--------
SQLite  (default) — no server required; ideal for local development.
PostgreSQL        — set DATABASE_URL env var, e.g.:
                    postgresql://user:pass@host:5432/trading

Usage
-----
    from core.database import Database

    db = Database()            # SQLite at data/trading.db
    run_id = db.new_run("CLFGridStrategy", params={"GRID_RATIO": 1.015})

    db.log_trade(run_id, symbol="CLF", action="BUY", qty=5,
                 price=12.34, commission=0.35, slippage=0.01,
                 order_ref="CLF_BUY_L1", strategy_id="CLFGridStrategy")

    db.log_equity(run_id, equity=1_023.50)
    db.close_run(run_id, final_equity=1_023.50, stats={...})

    # Query helpers
    trades = db.get_trades(run_id)
    equity = db.get_equity(run_id)
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import sqlalchemy as sa
    from sqlalchemy import text
    _HAS_SA = True
except ImportError:
    _HAS_SA = False

_log = logging.getLogger(__name__)

# ── DDL ───────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT    PRIMARY KEY,
    strategy        TEXT    NOT NULL,
    mode            TEXT    NOT NULL DEFAULT 'backtest',
    params_json     TEXT,
    start_ts        REAL    NOT NULL,
    end_ts          REAL,
    initial_capital REAL,
    final_equity    REAL,
    num_trades      INTEGER,
    sharpe_ratio    REAL,
    sortino_ratio   REAL,
    max_drawdown    REAL,
    calmar_ratio    REAL,
    var_99_30d      REAL
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    strategy_id TEXT,
    symbol      TEXT    NOT NULL,
    action      TEXT    NOT NULL,
    qty         REAL    NOT NULL,
    price       REAL    NOT NULL,
    commission  REAL    NOT NULL DEFAULT 0.0,
    slippage    REAL    NOT NULL DEFAULT 0.0,
    order_ref   TEXT,
    ts          REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    equity  REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_trades_run    ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_equity_run    ON equity_snapshots(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
"""


class Database:
    """
    Thin persistence layer over SQLite (default) or PostgreSQL.

    Designed to be used from both the live trading loop (log individual fills
    as they happen) and the backtester (bulk-insert an entire BacktestResult).
    """

    def __init__(self, url: str = None) -> None:
        """
        Parameters
        ----------
        url : SQLAlchemy connection URL.
              Defaults to DATABASE_URL env var, then
              ``sqlite:///data/trading.db``.
        """
        if not _HAS_SA:
            raise ImportError(
                "sqlalchemy is required for the database module. "
                "Run: pip install sqlalchemy"
            )

        resolved_url = (
            url
            or os.getenv("DATABASE_URL")
            or f"sqlite:///{Path('data/trading.db')}"
        )

        # Ensure the data directory exists for SQLite
        if resolved_url.startswith("sqlite:///"):
            db_path = Path(resolved_url[len("sqlite:///"):])
            db_path.parent.mkdir(parents=True, exist_ok=True)

        connect_args = {}
        if resolved_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = sa.create_engine(
            resolved_url,
            connect_args=connect_args,
            echo=False,
        )
        self._init_schema()
        _log.info("Database ready | %s", resolved_url)

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            for stmt in _SCHEMA_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))

    # ── Run lifecycle ──────────────────────────────────────────────────────

    def new_run(
        self,
        strategy:        str,
        params:          Dict[str, Any] = None,
        initial_capital: float          = 0.0,
        mode:            str            = "backtest",
    ) -> str:
        """
        Create a new run record and return its run_id (UUID string).
        """
        run_id = str(uuid.uuid4())
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO backtest_runs "
                    "(run_id, strategy, mode, params_json, start_ts, initial_capital) "
                    "VALUES (:run_id, :strategy, :mode, :params_json, :start_ts, :initial_capital)"
                ),
                {
                    "run_id":          run_id,
                    "strategy":        strategy,
                    "mode":            mode,
                    "params_json":     json.dumps(params or {}),
                    "start_ts":        time.time(),
                    "initial_capital": initial_capital,
                },
            )
        _log.debug("new_run | %s | strategy=%s", run_id, strategy)
        return run_id

    def close_run(
        self,
        run_id:       str,
        final_equity: float              = 0.0,
        stats:        Dict[str, float]   = None,
    ) -> None:
        """Update a run record with final metrics when it completes."""
        s = stats or {}
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE backtest_runs SET "
                    "end_ts=:end_ts, final_equity=:final_equity, "
                    "num_trades=:num_trades, sharpe_ratio=:sharpe, "
                    "sortino_ratio=:sortino, max_drawdown=:mdd, "
                    "calmar_ratio=:calmar, var_99_30d=:var "
                    "WHERE run_id=:run_id"
                ),
                {
                    "run_id":       run_id,
                    "end_ts":       time.time(),
                    "final_equity": final_equity,
                    "num_trades":   int(s.get("num_trades", 0)),
                    "sharpe":       s.get("sharpe_ratio",  None),
                    "sortino":      s.get("sortino_ratio", None),
                    "mdd":          s.get("max_drawdown",  None),
                    "calmar":       s.get("calmar_ratio",  None),
                    "var":          s.get("var_99_30d",    None),
                },
            )

    # ── Trade logging ──────────────────────────────────────────────────────

    def log_trade(
        self,
        run_id:      str,
        symbol:      str,
        action:      str,
        qty:         float,
        price:       float,
        commission:  float    = 0.0,
        slippage:    float    = 0.0,
        order_ref:   str      = "",
        strategy_id: str      = "",
        ts:          float    = None,
    ) -> None:
        """Insert a single fill record."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO trades "
                    "(run_id, strategy_id, symbol, action, qty, price, "
                    " commission, slippage, order_ref, ts) "
                    "VALUES (:run_id, :strategy_id, :symbol, :action, :qty, :price, "
                    "        :commission, :slippage, :order_ref, :ts)"
                ),
                {
                    "run_id":      run_id,
                    "strategy_id": strategy_id,
                    "symbol":      symbol,
                    "action":      action,
                    "qty":         float(qty),
                    "price":       float(price),
                    "commission":  float(commission),
                    "slippage":    float(slippage),
                    "order_ref":   order_ref,
                    "ts":          ts or time.time(),
                },
            )

    def bulk_log_trades(self, run_id: str, trade_df: pd.DataFrame) -> None:
        """
        Efficiently insert all trades from a BacktestResult.trade_log DataFrame.

        Expected columns: symbol, action, qty, price, commission, slippage, order_ref.
        """
        if trade_df.empty:
            return
        now = time.time()
        rows = []
        for r in trade_df.itertuples(index=False):
            rows.append({
                "run_id":      run_id,
                "strategy_id": "",
                "symbol":      r.symbol,
                "action":      r.action,
                "qty":         float(r.qty),
                "price":       float(r.price),
                "commission":  float(r.commission),
                "slippage":    float(r.slippage),
                "order_ref":   getattr(r, "order_ref", ""),
                "ts":          now,
            })
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO trades "
                    "(run_id, strategy_id, symbol, action, qty, price, "
                    " commission, slippage, order_ref, ts) "
                    "VALUES (:run_id, :strategy_id, :symbol, :action, :qty, :price, "
                    "        :commission, :slippage, :order_ref, :ts)"
                ),
                rows,
            )
        _log.debug("bulk_log_trades | %d rows | run_id=%s", len(rows), run_id)

    # ── Equity logging ─────────────────────────────────────────────────────

    def log_equity(self, run_id: str, equity: float, ts: float = None) -> None:
        """Insert a single NAV snapshot."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO equity_snapshots (run_id, ts, equity) "
                    "VALUES (:run_id, :ts, :equity)"
                ),
                {"run_id": run_id, "ts": ts or time.time(), "equity": float(equity)},
            )

    def bulk_log_equity(self, run_id: str, equity_df: pd.DataFrame) -> None:
        """
        Efficiently insert all equity snapshots from a BacktestResult.equity_curve.

        Expected columns: timestamp, equity.
        """
        if equity_df.empty:
            return
        rows = [
            {
                "run_id": run_id,
                "ts":     r.timestamp.timestamp() if hasattr(r.timestamp, "timestamp")
                          else float(r.timestamp),
                "equity": float(r.equity),
            }
            for r in equity_df.itertuples(index=False)
        ]
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO equity_snapshots (run_id, ts, equity) "
                    "VALUES (:run_id, :ts, :equity)"
                ),
                rows,
            )
        _log.debug("bulk_log_equity | %d rows | run_id=%s", len(rows), run_id)

    # ── Query helpers ──────────────────────────────────────────────────────

    def get_trades(self, run_id: str) -> pd.DataFrame:
        """Return all trades for a run as a DataFrame."""
        with self._engine.connect() as conn:
            return pd.read_sql(
                text("SELECT * FROM trades WHERE run_id = :run_id ORDER BY ts"),
                conn,
                params={"run_id": run_id},
            )

    def get_equity(self, run_id: str) -> pd.DataFrame:
        """Return equity snapshots for a run as a DataFrame."""
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT ts, equity FROM equity_snapshots "
                    "WHERE run_id = :run_id ORDER BY ts"
                ),
                conn,
                params={"run_id": run_id},
            )
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
        return df

    def get_runs(self, strategy: str = None, mode: str = None) -> pd.DataFrame:
        """List backtest runs, optionally filtered by strategy or mode."""
        where_clauses = []
        params: Dict[str, Any] = {}
        if strategy:
            where_clauses.append("strategy = :strategy")
            params["strategy"] = strategy
        if mode:
            where_clauses.append("mode = :mode")
            params["mode"] = mode

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        with self._engine.connect() as conn:
            return pd.read_sql(
                text(
                    f"SELECT * FROM backtest_runs {where} ORDER BY start_ts DESC"
                ),
                conn,
                params=params,
            )

    # ── Convenience: save a full BacktestResult ────────────────────────────

    def save_backtest_result(
        self,
        result,             # BacktestResult (avoid circular import with type hint)
        strategy_name: str = "combined",
        params: Dict = None,
    ) -> str:
        """
        Persist a complete BacktestResult (trades + equity) in one call.

        Returns the run_id.
        """
        from utils.dashboard_stats import (
            compute_sharpe, compute_sortino,
            compute_max_drawdown, compute_calmar, compute_var_dollars,
        )

        run_id = self.new_run(
            strategy        = strategy_name,
            params          = params,
            initial_capital = result.config.initial_capital,
            mode            = "backtest",
        )

        self.bulk_log_trades(run_id, result.trade_log)
        self.bulk_log_equity(run_id, result.equity_curve)

        stats: Dict[str, Any] = {"num_trades": len(result.trade_log)}
        if not result.daily_returns.empty:
            eq = result.equity_curve.set_index("timestamp")["equity"]
            stats["sharpe_ratio"]  = compute_sharpe(result.daily_returns)
            stats["sortino_ratio"] = compute_sortino(result.daily_returns)
            stats["max_drawdown"]  = compute_max_drawdown(eq)
            stats["calmar_ratio"]  = compute_calmar(eq)
            stats["var_99_30d"]    = compute_var_dollars(
                result.daily_returns, result.config.portfolio_value
            )

        final_eq = (
            float(result.equity_curve["equity"].iloc[-1])
            if not result.equity_curve.empty
            else result.config.initial_capital
        )
        self.close_run(run_id, final_equity=final_eq, stats=stats)
        _log.info("Saved backtest run %s | trades=%d", run_id, stats["num_trades"])
        return run_id

    def close(self) -> None:
        """Dispose of the connection pool."""
        self._engine.dispose()
