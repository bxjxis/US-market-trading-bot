"""
scripts/backtest_grid.py
------------------------
Run a single-symbol grid backtest using the CLFGridStrategy engine.

Reads historical data from data/cache/<SYMBOL>/ (populated by the grid
screener or download_cache.py) and prints a full performance summary.

Usage
-----
    # Backtest CTMX with screener-suggested params
    python scripts/backtest_grid.py --symbol CTMX --grid-ratio 1.015 --levels 15

    # Backtest with a larger account allocation
    python scripts/backtest_grid.py --symbol CTMX --account-size 36000

    # Backtest the default CLF with tuned params
    python scripts/backtest_grid.py --symbol CLF --grid-ratio 1.020 --levels 10
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtester import BacktestConfig, BacktestEngine
from strategies.clf_grid import CLFGridStrategy
from utils.dashboard_stats import print_summary

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-symbol grid strategy backtest."
    )
    parser.add_argument(
        "--symbol", default="CTMX",
        help="Ticker symbol to backtest (must have data in data/cache/). Default: CTMX",
    )
    parser.add_argument(
        "--grid-ratio", type=float, default=1.015, dest="grid_ratio",
        help="Geometric step between grid levels (default: 1.015).",
    )
    parser.add_argument(
        "--levels", type=int, default=15,
        help="Number of buy levels below anchor (default: 15).",
    )
    parser.add_argument(
        "--account-size", type=float, default=36_000.0, dest="account_size",
        help="Capital allocated to this grid position in USD (default: 36000).",
    )
    parser.add_argument(
        "--safety-pct", type=float, default=0.30, dest="safety_pct",
        help="Safety switch threshold — halt new orders if price deviates this far "
             "from anchor (default: 0.30 = 30%%).",
    )
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of bars used to seed indicators before replay (default: 100).",
    )
    parser.add_argument(
        "--portfolio-value", type=float, default=180_000.0, dest="portfolio_value",
        help="Total portfolio value used for VaR dollar scaling (default: 180000).",
    )
    parser.add_argument(
        "--adx-halt", type=float, default=0.0, dest="adx_halt",
        help="Legacy single-threshold: pause buys when ADX > this (default: 0 = disabled).",
    )
    parser.add_argument(
        "--adx-resume", type=float, default=25.0, dest="adx_resume",
        help="Legacy single-threshold: resume when ADX < this (default: 25).",
    )
    parser.add_argument(
        "--adx-tiered", action="store_true", dest="adx_tiered",
        help="Enable three-tier adaptive mode (overrides --adx-halt).",
    )
    parser.add_argument(
        "--adx-stable-thr", type=float, default=25.0, dest="adx_stable_thr",
        help="ADX threshold for stable mode (default: 25).",
    )
    parser.add_argument(
        "--adx-trend-thr", type=float, default=35.0, dest="adx_trend_thr",
        help="ADX threshold for slope-sensitive zone (default: 35).",
    )
    parser.add_argument(
        "--adx-halt-thr", type=float, default=45.0, dest="adx_halt_thr",
        help="ADX circuit-breaker threshold, always halt (default: 45).",
    )
    parser.add_argument(
        "--adx-stable-ratio", type=float, default=1.012, dest="adx_stable_ratio",
        help="Grid ratio used in stable mode (default: 1.012).",
    )
    parser.add_argument(
        "--adx-stable-qty-scale", type=float, default=0.5, dest="adx_stable_qty_scale",
        help="Position size scale in stable mode (default: 0.5 = half position).",
    )
    parser.add_argument(
        "--adx-ultra-thr", type=float, default=20.0, dest="adx_ultra_thr",
        help="ADX below this triggers ultra-aggressive (pyramid) mode (default: 20).",
    )
    parser.add_argument(
        "--adx-ultra-qty-scale", type=float, default=2.0, dest="adx_ultra_qty_scale",
        help="Position size scale in ultra mode (default: 2.0 = double).",
    )
    parser.add_argument(
        "--adx-trend-ratio", type=float, default=1.015, dest="adx_trend_ratio",
        help="Grid ratio in trending zone (ADX_TREND_THR..ADX_HALT_THR, rising slope). Default: 1.015.",
    )
    parser.add_argument(
        "--adx-trend-qty-scale", type=float, default=0.25, dest="adx_trend_qty_scale",
        help="Position size scale in trending zone (default: 0.25).",
    )
    parser.add_argument(
        "--atr-adaptive", action="store_true", dest="atr_adaptive",
        help="Bidirectional ATR grid adaptation (Guasoni): widen sell targets in high-vol "
             "to capture larger moves; hold baseline in normal/low vol.",
    )
    parser.add_argument(
        "--atr-grid-coverage", type=float, default=0.5, dest="atr_grid_coverage",
        help="Fraction of 1 ATR to use as grid step in ATR-adaptive warmup (default: 0.5).",
    )
    parser.add_argument(
        "--atr-widen-max", type=float, default=1.8, dest="atr_widen_max",
        help="Max ATR-adaptive widening: baseline * this (default: 1.8). Guasoni bidirectional.",
    )
    parser.add_argument(
        "--atr-long-period", type=int, default=50, dest="atr_long_period",
        help="Bars for long-term ATR baseline used in Guasoni scaling (default: 50).",
    )
    parser.add_argument(
        "--max-open-loss", type=float, default=0.0, dest="max_open_loss",
        help="Cancel all pending buys when unrealized loss exceeds this $ amount (default: 0 = disabled).",
    )
    parser.add_argument(
        "--atr-sizing", action="store_true", dest="atr_sizing",
        help="Size positions by ATR risk rather than capital allocation.",
    )
    parser.add_argument(
        "--atr-risk-pct", type=float, default=0.5, dest="atr_risk_pct",
        help="When --atr-sizing: %% of (account/levels) to risk per 1 ATR move (default: 0.5).",
    )
    parser.add_argument(
        "--window-dd-cap", type=float, default=0.0, dest="window_dd_cap",
        help="Halt new fills when drawdown from peak exceeds this fraction (e.g. 0.015 = 1.5%%). 0 = disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Verify cache exists before attempting backtest
    cache_dir = Path("data/cache") / args.symbol
    if not cache_dir.exists() or not list(cache_dir.glob("*.parquet")):
        print(
            f"ERROR: No cached data for '{args.symbol}' in {cache_dir.resolve()}.\n"
            f"Run the grid screener first:\n"
            f"  python scripts/grid_screener.py\n"
            f"Or download directly:\n"
            f"  python scripts/download_cache.py  (add '{args.symbol}' to CONTRACTS)"
        )
        sys.exit(1)

    params = {
        "SYMBOL":               args.symbol,
        "GRID_RATIO":           args.grid_ratio,
        "NUM_BUY_LEVELS":       args.levels,
        "ACCOUNT_SIZE":         args.account_size,
        "SAFETY_PCT":           args.safety_pct,
        "ADX_HALT":             args.adx_halt,
        "ADX_RESUME":           args.adx_resume,
        "ADX_TIERED":           args.adx_tiered,
        "ADX_STABLE_THR":       args.adx_stable_thr,
        "ADX_TREND_THR":        args.adx_trend_thr,
        "ADX_HALT_THR":         args.adx_halt_thr,
        "ADX_STABLE_RATIO":     args.adx_stable_ratio,
        "ADX_STABLE_QTY_SCALE": args.adx_stable_qty_scale,
        "ADX_ULTRA_THR":        args.adx_ultra_thr,
        "ADX_ULTRA_QTY_SCALE":  args.adx_ultra_qty_scale,
        "ADX_TREND_RATIO":      args.adx_trend_ratio,
        "ADX_TREND_QTY_SCALE":  args.adx_trend_qty_scale,
        "MAX_OPEN_LOSS":        args.max_open_loss,
        "ATR_SIZING":           args.atr_sizing,
        "ATR_RISK_PCT":         args.atr_risk_pct,
        "ATR_ADAPTIVE":         args.atr_adaptive,
        "ATR_GRID_COVERAGE":    args.atr_grid_coverage,
        "ATR_WIDEN_MAX":        args.atr_widen_max,
        "ATR_LONG_PERIOD":      args.atr_long_period,
    }

    print(
        f"\nBacktesting {args.symbol} grid strategy"
        f"\n  GRID_RATIO    = {args.grid_ratio}"
        f"\n  NUM_BUY_LEVELS= {args.levels}"
        f"\n  ACCOUNT_SIZE  = ${args.account_size:,.0f}"
        f"\n  SAFETY_PCT    = {args.safety_pct * 100:.0f}%"
        f"\n  ADX_MODE      = {'tiered (ultra/aggressive/stable/trending/halt)' if args.adx_tiered else (f'halt>{args.adx_halt:.0f}' if args.adx_halt > 0 else 'disabled')}"
        f"\n  ATR_ADAPTIVE  = {'on (coverage='+str(args.atr_grid_coverage)+')' if args.atr_adaptive else 'off'}"
        f"\n  Warmup bars   = {args.warmup}"
    )
    print()

    cfg = BacktestConfig(
        initial_capital = args.account_size,
        portfolio_value = args.portfolio_value,
        warmup_bars     = args.warmup,
        drawdown_cap    = args.window_dd_cap,
    )
    engine = BacktestEngine(cfg)

    result = engine.run(
        strategies      = [CLFGridStrategy],
        symbols         = [args.symbol],
        strategy_params = {"CLFGridStrategy": params},
    )

    print(result.summary())

    # Detailed trade breakdown
    if not result.trade_log.empty:
        buys  = result.trade_log[result.trade_log["action"] == "BUY"]
        sells = result.trade_log[result.trade_log["action"] == "SELL"]
        total_comm = result.trade_log["commission"].sum()
        print(
            f"\nTrade breakdown:"
            f"\n  BUY fills : {len(buys)}"
            f"\n  SELL fills: {len(sells)}"
            f"\n  Total commission: ${total_comm:.2f}"
        )
        if not sells.empty:
            gross_pnl = (
                (sells["price"] * sells["qty"]).sum()
                - (buys["price"] * buys["qty"]).sum()
            )
            print(f"  Gross P&L (matched fills): ${gross_pnl:,.2f}")
    else:
        print("\nNo trades executed — check warmup bars vs available data, "
              "or widen the price range / lower grid ratio.")


if __name__ == "__main__":
    main()
