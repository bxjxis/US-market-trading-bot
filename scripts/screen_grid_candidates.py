"""
scripts/screen_grid_candidates.py
----------------------------------
CLI tool to screen a universe of US equities for grid-trading suitability.

What it computes
----------------
For each symbol in the universe (and SPY as the market benchmark) the script
fetches one year of daily OHLCV data and calculates five metrics defined in
utils/grid_screener.py:

    beta          — systematic-risk coefficient vs SPY (252-day OLS)
    div_yield_pct — trailing-12M dividends / current price × 100
    atr_cov       — ATR-14 Coefficient of Variation (std/mean, 63-day)
    atr_price_pct — mean ATR-14 / mean close × 100 (63-day)
    adx_14        — Average Directional Index (14-day)

Hard viability constraints (all must pass):
    beta          < 1.0
    atr_price_pct > (GRID_RATIO − 1) × 100     e.g. > 1.5 % for default step
    adx_14        < 25

Data sources
------------
Primary   : IBKR Gateway (reuses the project's existing DataFetcher).
            Connect to IBKR before running, just like download_cache.py.

Offline / Dividend data : yfinance (optional).
    pip install yfinance
    When available, yfinance supplies dividend yield data and can also
    serve as a full offline fallback via --offline.

Usage
-----
    # Screen default universe (IBKR must be running)
    python scripts/screen_grid_candidates.py

    # Offline — fetch everything from yfinance (no IBKR needed)
    python scripts/screen_grid_candidates.py --offline

    # Custom symbol list
    python scripts/screen_grid_candidates.py --symbols XOM CVX T VZ KO

    # Save results to CSV
    python scripts/screen_grid_candidates.py --output data/screen_results.csv

    # Use a non-default grid step for viability thresholds
    python scripts/screen_grid_candidates.py --grid-ratio 1.02
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.grid_screener import (
    compute_beta,
    compute_atr_cov,
    compute_atr_price_pct,
    compute_adx,
    score_candidates,
    print_screen_results,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("screen_grid")
_log.setLevel(logging.INFO)

# ── Candidate universe ────────────────────────────────────────────────────────
# A mix of dividend-paying, historically stable-beta US equities that are
# plausible grid-trading targets.  The current live strategies (CLF, AMZN)
# are included so you can see where they rank.
DEFAULT_UNIVERSE: List[str] = [
    # Current live strategies (for comparison)
    "CLF", "AMZN",
    # Utilities — classic low-beta, high-yield sector
    "SO", "DUK", "NEE", "XEL", "D",
    # Consumer Staples — defensive, slow-moving
    "KO", "PEP", "PG", "MO",
    # Energy — dividend payers, commodity-driven oscillation
    "XOM", "CVX", "OXY",
    # Telecoms — high yield, range-bound
    "T", "VZ",
    # Financials
    "JPM", "BAC",
    # Monthly-dividend / REIT (high yield but check beta & ADX carefully)
    "O", "STAG",
]

BENCHMARK: str = "SPY"

# ── Optional: yfinance for dividends and offline mode ─────────────────────────
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# ═════════════════════════════════════════════════════════════════════════════
# Data fetching helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_via_yfinance(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Download 1 year of daily OHLCV for all symbols in one batch call.
    Returns dict: symbol → DataFrame with columns [open, high, low, close, volume].
    """
    if not _HAS_YF:
        raise ImportError("yfinance is not installed.  Run: pip install yfinance")

    tickers = " ".join(symbols)
    _log.info("yfinance: downloading %d symbols …", len(symbols))
    raw = yf.download(
        tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    result: Dict[str, pd.DataFrame] = {}

    if len(symbols) == 1:
        # yfinance returns a flat DataFrame for a single ticker
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        result[symbols[0]] = df.dropna()
    else:
        for sym in symbols:
            try:
                df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index = pd.to_datetime(df.index)
                result[sym] = df.dropna()
            except (KeyError, TypeError):
                _log.warning("yfinance: no data for %s", sym)

    return result


def _fetch_div_yield_yf(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch trailing-12M dividend yield (%) for each symbol via yfinance.

    Returns dict: symbol → yield_pct.  Missing data → nan.
    """
    if not _HAS_YF:
        return {}

    yields: Dict[str, float] = {}
    for sym in symbols:
        try:
            info  = yf.Ticker(sym).info
            # dividendYield is already a fraction (e.g. 0.035 = 3.5 %)
            raw   = info.get("dividendYield", None) or info.get("trailingAnnualDividendYield", None)
            yields[sym] = float(raw) * 100.0 if raw else float("nan")
        except Exception as exc:
            _log.debug("yfinance dividend fetch failed for %s: %s", sym, exc)
            yields[sym] = float("nan")

    return yields


async def _fetch_via_ibkr(
    symbols:  List[str],
    duration: str = "1 Y",
    bar_size: str = "1 day",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch 1-year daily OHLCV from IBKR for all symbols.
    Returns dict: symbol → DataFrame with columns [open, high, low, close, volume].
    """
    from ib_insync import Stock, util as ib_util
    from core.connection import IBConnection
    from utils.data_fetcher import DataFetcher, IndicatorUtils

    conn = IBConnection()
    ib   = conn.connect()
    _log.info("IBKR: connected, downloading %d symbols …", len(symbols))

    result: Dict[str, pd.DataFrame] = {}
    fetcher = DataFetcher(ib)
    contracts = [Stock(s, "SMART", "USD") for s in symbols]

    try:
        qualified = await ib.qualifyContractsAsync(*contracts)
        missing   = {c.symbol for c in contracts} - {c.symbol for c in qualified}
        if missing:
            _log.warning("Could not qualify: %s", missing)

        for i, contract in enumerate(qualified, start=1):
            sym = contract.symbol
            _log.info("[%d/%d] %s …", i, len(qualified), sym)
            try:
                bars = await fetcher.fetch_historical(
                    contract,
                    duration     = duration,
                    bar_size     = bar_size,
                    what_to_show = "TRADES",
                    use_rth      = True,
                    keep_up_to_date = False,
                )
                df = IndicatorUtils.bars_to_df(bars)
                if not df.empty:
                    result[sym] = df
            except Exception as exc:
                _log.warning("IBKR fetch failed for %s: %s", sym, exc)

            if i < len(qualified):
                await asyncio.sleep(3.0)   # IBKR pacing
    finally:
        conn.disconnect()

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Core screening logic
# ═════════════════════════════════════════════════════════════════════════════

def _build_records(
    ohlcv_map:  Dict[str, pd.DataFrame],
    div_yields: Dict[str, float],
    symbols:    List[str],
    grid_ratio: float,
) -> list:
    """
    Compute all metrics for each symbol and return a list of record dicts.
    SPY must be present in ohlcv_map to compute Beta.
    """
    bench_prices: Optional[pd.Series] = None
    if BENCHMARK in ohlcv_map:
        bench_prices = ohlcv_map[BENCHMARK]["close"]

    records = []
    for sym in symbols:
        if sym not in ohlcv_map:
            _log.warning("No OHLCV data for %s — skipping.", sym)
            continue

        df  = ohlcv_map[sym]
        rec: Dict = {
            "symbol":        sym,
            "beta":          float("nan"),
            "div_yield_pct": div_yields.get(sym, float("nan")),
            "atr_cov":       float("nan"),
            "atr_price_pct": float("nan"),
            "adx_14":        float("nan"),
        }

        if bench_prices is not None:
            rec["beta"] = compute_beta(df["close"], bench_prices)

        if len(df) >= 20:
            rec["atr_cov"]       = compute_atr_cov(df)
            rec["atr_price_pct"] = compute_atr_price_pct(df)
            rec["adx_14"]        = compute_adx(df)

        records.append(rec)

    return records


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen stocks for grid-trading suitability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/screen_grid_candidates.py\n"
            "  python scripts/screen_grid_candidates.py --offline\n"
            "  python scripts/screen_grid_candidates.py --symbols XOM CVX T\n"
            "  python scripts/screen_grid_candidates.py --output data/screen.csv\n"
        ),
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None, metavar="SYM",
        help="Custom list of symbols to screen (default: built-in universe).",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help=(
            "Use yfinance for all data — no IBKR connection required.  "
            "Requires: pip install yfinance"
        ),
    )
    parser.add_argument(
        "--grid-ratio", type=float, default=1.015, dest="grid_ratio",
        metavar="RATIO",
        help=(
            "Geometric grid step to use for viability thresholds "
            "(default: 1.015 = 1.5%% step)."
        ),
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Save results to a CSV file at this path.",
    )
    parser.add_argument(
        "--top", type=int, default=20, metavar="N",
        help="Number of top candidates to display (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    symbols = args.symbols or DEFAULT_UNIVERSE

    # Always include SPY for Beta computation (deduplicate)
    fetch_list = list(dict.fromkeys([BENCHMARK] + symbols))

    # ── Data acquisition ──────────────────────────────────────────────────
    if args.offline:
        if not _HAS_YF:
            print(
                "ERROR: --offline requires yfinance.  "
                "Install it with:  pip install yfinance"
            )
            sys.exit(1)
        _log.info("Offline mode: fetching all data from yfinance.")
        ohlcv_map  = _fetch_via_yfinance(fetch_list)
        div_yields = _fetch_div_yield_yf(symbols)
    else:
        # IBKR for OHLCV price data
        from ib_insync import util as ib_util
        ohlcv_map = ib_util.run(_fetch_via_ibkr(fetch_list))

        # Dividend yields from yfinance (optional supplement)
        if _HAS_YF:
            _log.info("Fetching dividend yields from yfinance …")
            div_yields = _fetch_div_yield_yf(symbols)
        else:
            _log.warning(
                "yfinance not installed — dividend yield will be N/A.  "
                "Install with: pip install yfinance"
            )
            div_yields = {}

    # ── Compute metrics & score ───────────────────────────────────────────
    records = _build_records(ohlcv_map, div_yields, symbols, args.grid_ratio)

    if not records:
        print("No data retrieved for any symbol.  Check your connection / symbols.")
        sys.exit(1)

    ranked = score_candidates(records, grid_ratio=args.grid_ratio)

    # ── Display ───────────────────────────────────────────────────────────
    print_screen_results(ranked, grid_ratio=args.grid_ratio, top_n=args.top)

    # ── Optional CSV export ───────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_path, index=False, float_format="%.4f")
        print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
