"""
scripts/optimize.py
--------------------
Hyperparameter optimisation using Optuna.

Objective
---------
Maximise the Calmar Ratio (CAGR / Max-Drawdown) subject to the constraint
that the 30-day 99% Historical VaR on the full $180,000 portfolio stays
below $15,000.  Trials that violate the VaR constraint receive a heavy
penalty so Optuna learns to avoid them.

Backtest window
---------------
2 years of daily OHLCV data for AMZN and CLF (downloaded via
scripts/download_cache.py).  Only the two active strategies are included;
SmallCapArb (IREN/WULF) is excluded from this backtest.

Parameter search space
----------------------
CLF Grid:
    GRID_RATIO    — geometric step between levels   [1.010, 1.030]
    NUM_BUY_LEVELS — number of grid levels          [6, 15]
    ACCOUNT_SIZE  — capital allocated to CLF        [36_000, 108_000]
                    (20 %–60 % of $180k)

AMZN Reversion:
    BB_PERIOD     — Bollinger Band SMA period       [15, 30]
    RSI_ENTRY     — RSI oversold threshold          [20.0, 30.0]
    TAKE_PROFIT   — exit on this % gain             [0.02, 0.05]

Usage
-----
    # Default: 100 trials, SQLite storage
    python scripts/optimize.py

    # Custom: 500 trials, parallel jobs, resume from existing study
    python scripts/optimize.py --trials 500 --jobs 4 --study-name my_study

    # Dry-run: single trial with default params (validates the pipeline)
    python scripts/optimize.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Make project root importable when run as a script ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from optuna.samplers import TPESampler

from core.backtester import BacktestConfig, BacktestEngine
from strategies.clf_grid import CLFGridStrategy
from strategies.amzn_reversion import AMZNReversionStrategy
from utils.dashboard_stats import compute_calmar, compute_var_dollars

# ── Constants ─────────────────────────────────────────────────────────────────
PORTFOLIO_VALUE  = 180_000.0
VAR_LIMIT_USD    = 15_000.0       # 30-day 99% VaR must stay below this
VAR_PENALTY_RATE = 10.0           # Calmar penalty per $1k of VaR overage
WARMUP_BARS      = 120            # bars used to seed indicators before replay
# 2-year backtest using only the two active strategies (AMZN + CLF).
SYMBOLS          = ["CLF", "AMZN"]

logging.basicConfig(
    level=logging.WARNING,   # suppress strategy chatter during optimisation
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger("optimize")
# Keep optimizer-level messages visible
_log.setLevel(logging.INFO)


# ═════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ═════════════════════════════════════════════════════════════════════════════

def objective(trial: optuna.Trial) -> float:
    """
    Run one backtest with the parameters sampled by Optuna and return the
    Calmar Ratio (penalised if VaR exceeds the $15k limit).
    """

    # ── CLF Grid ───────────────────────────────────────────────────────────
    clf_account = trial.suggest_float(
        "clf_account_size", 0.20 * PORTFOLIO_VALUE, 0.60 * PORTFOLIO_VALUE, step=1_000
    )
    clf_params = {
        "GRID_RATIO":      trial.suggest_float("clf_grid_ratio", 1.010, 1.030, step=0.001),
        "NUM_BUY_LEVELS":  trial.suggest_int("clf_num_levels", 6, 15),
        "ACCOUNT_SIZE":    clf_account,
        "MIN_EVAL_SECS":   0.0,    # not applicable to CLF but harmless
    }

    # ── AMZN Reversion ─────────────────────────────────────────────────────
    amzn_params = {
        "BB_PERIOD":   trial.suggest_int("amzn_bb_period", 15, 30),
        "RSI_ENTRY":   trial.suggest_float("amzn_rsi_entry", 20.0, 30.0),
        "TAKE_PROFIT": trial.suggest_float("amzn_take_profit", 0.02, 0.05),
    }

    # ── Run backtest ───────────────────────────────────────────────────────
    config = BacktestConfig(
        initial_capital = PORTFOLIO_VALUE,
        portfolio_value = PORTFOLIO_VALUE,
        warmup_bars     = WARMUP_BARS,
    )
    engine = BacktestEngine(config)

    try:
        result = engine.run(
            strategies      = [CLFGridStrategy, AMZNReversionStrategy],
            symbols         = SYMBOLS,
            strategy_params = {
                "CLFGridStrategy":       clf_params,
                "AMZNReversionStrategy": amzn_params,
            },
        )
    except FileNotFoundError as exc:
        # Parquet data missing — skip trial
        _log.warning("Trial %d pruned: %s", trial.number, exc)
        raise optuna.exceptions.TrialPruned() from exc
    except Exception as exc:
        _log.warning("Trial %d error: %s", trial.number, exc)
        return float("-inf")

    # ── Evaluate metrics ───────────────────────────────────────────────────
    if result.equity_curve.empty or result.trade_log.empty:
        # No trades were generated — uninformative trial
        return float("-inf")

    eq           = result.equity_curve.set_index("timestamp")["equity"]
    daily_returns = result.daily_returns

    if daily_returns.empty:
        return float("-inf")

    calmar  = compute_calmar(eq)
    var_usd = compute_var_dollars(daily_returns, portfolio_value=PORTFOLIO_VALUE)

    # ── VaR constraint: penalise proportionally to the overage ────────────
    var_overage = max(0.0, var_usd - VAR_LIMIT_USD)
    penalty     = var_overage / 1_000.0 * VAR_PENALTY_RATE   # -10 Calmar per $1k over

    score = calmar - penalty

    _log.info(
        "Trial %4d | calmar=%.4f  var=$%,.0f  penalty=%.4f  score=%.4f | "
        "clf_ratio=%.3f  amzn_rsi=%.1f",
        trial.number, calmar, var_usd, penalty, score,
        clf_params["GRID_RATIO"],
        amzn_params["RSI_ENTRY"],
    )

    # Store intermediate metrics for analysis
    trial.set_user_attr("calmar",       calmar)
    trial.set_user_attr("var_usd",      var_usd)
    trial.set_user_attr("num_trades",   len(result.trade_log))
    trial.set_user_attr("final_equity", float(eq.iloc[-1]))

    return score


# ═════════════════════════════════════════════════════════════════════════════
# Dry-run helper
# ═════════════════════════════════════════════════════════════════════════════

def dry_run() -> None:
    """
    Run a single backtest with default parameters to validate the pipeline
    end-to-end.  Prints the result summary and exits.
    """
    _log.info("Dry-run: testing pipeline with default parameters …")

    config = BacktestConfig(
        initial_capital = PORTFOLIO_VALUE,
        portfolio_value = PORTFOLIO_VALUE,
        warmup_bars     = WARMUP_BARS,
    )
    engine = BacktestEngine(config)
    result = engine.run(
        strategies = [CLFGridStrategy, AMZNReversionStrategy],
        symbols    = SYMBOLS,
    )
    print(result.summary())
    result.plot_equity_curve(save_path=Path("data/backtest_equity.html"))


# ═════════════════════════════════════════════════════════════════════════════
# Results reporting
# ═════════════════════════════════════════════════════════════════════════════

def report_best(study: optuna.Study) -> None:
    """Print the best trial's parameters and metrics."""
    best = study.best_trial

    print("\n" + "═" * 60)
    print("  OPTIMISATION COMPLETE")
    print("═" * 60)
    print(f"  Best trial    : #{best.number}")
    print(f"  Best score    : {best.value:.4f}  (Calmar − VaR penalty)")
    print(f"  Calmar Ratio  : {best.user_attrs.get('calmar', 'N/A')}")
    print(f"  30d 99% VaR   : ${best.user_attrs.get('var_usd', 0):,.0f}")
    print(f"  Trades        : {best.user_attrs.get('num_trades', 0)}")
    print(f"  Final Equity  : ${best.user_attrs.get('final_equity', 0):,.2f}")
    print("\n  Best Parameters:")
    for k, v in best.params.items():
        print(f"    {k:<30} = {v}")
    print("═" * 60)

    # Warn if VaR is still above limit after optimisation
    var = best.user_attrs.get("var_usd", 0)
    if var > VAR_LIMIT_USD:
        print(
            f"\n  WARNING: Best trial VaR (${var:,.0f}) still exceeds "
            f"the $15,000 limit.  Consider increasing n_trials or "
            f"tightening the parameter ranges."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimisation for the trading bot."
    )
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials (default: 100).",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel workers for Optuna (default: 1).",
    )
    parser.add_argument(
        "--study-name", default="trading_bot_optimisation",
        help="Optuna study name (default: trading_bot_optimisation).",
    )
    parser.add_argument(
        "--storage", default=None,
        help=(
            "Optuna storage URL for distributed / persistent studies, e.g. "
            "'sqlite:///data/optuna.db'.  Defaults to in-memory."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run a single backtest with default params to validate the pipeline.",
    )
    return parser.parse_args()


def _save_visualizations(study: optuna.Study) -> None:
    """
    Save interactive HTML charts to data/ using Optuna's plotly backend.

    Requires: pip install plotly
    Opens automatically in any browser — no server needed.

    Files produced
    --------------
    data/optuna_history.html     — objective value across all trials
    data/optuna_importances.html — which parameters matter most (fANOVA)
    data/optuna_contour.html     — 2-D contour of CLF ratio vs AMZN RSI
    """
    try:
        import optuna.visualization as vis

        plots = {
            "optuna_history.html":     vis.plot_optimization_history(study),
            "optuna_importances.html": vis.plot_param_importances(study),
            "optuna_contour.html":     vis.plot_contour(
                study,
                params=["clf_grid_ratio", "amzn_rsi_entry"],
            ),
        }
        out_dir = Path("data")
        out_dir.mkdir(exist_ok=True)
        for filename, fig in plots.items():
            path = out_dir / filename
            fig.write_html(str(path))
            _log.info("Saved visualisation → %s", path)

        print(f"\n  Charts saved to data/  — open any .html file in a browser.")

    except ImportError:
        _log.warning(
            "plotly not installed — skipping visualizations. "
            "Run: pip install plotly"
        )
    except Exception as exc:
        _log.warning("Could not generate visualizations: %s", exc)


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run()
        return

    # Suppress ib_insync / strategy noise during parallel optimisation
    logging.getLogger("ib_insync").setLevel(logging.CRITICAL)
    for name in ("CLFGridStrategy", "AMZNReversionStrategy", "DataFetcher"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    sampler = TPESampler(seed=42)   # reproducible results
    study   = optuna.create_study(
        study_name = args.study_name,
        direction  = "maximize",
        sampler    = sampler,
        storage    = args.storage,
        load_if_exists = True,      # resume if a study with this name exists
    )

    _log.info(
        "Starting optimisation | trials=%d  jobs=%d  study=%s",
        args.trials, args.jobs, args.study_name,
    )

    study.optimize(
        objective,
        n_trials         = args.trials,
        n_jobs           = args.jobs,
        show_progress_bar = True,
        catch            = (Exception,),   # don't abort on isolated errors
    )

    report_best(study)

    # ── Persist best params to disk for reference ──────────────────────────
    import json
    best_path = Path("data") / "best_params.json"
    best_path.parent.mkdir(exist_ok=True)
    with open(best_path, "w") as f:
        json.dump(
            {
                "params":     study.best_trial.params,
                "score":      study.best_value,
                "calmar":     study.best_trial.user_attrs.get("calmar"),
                "var_usd":    study.best_trial.user_attrs.get("var_usd"),
                "num_trades": study.best_trial.user_attrs.get("num_trades"),
            },
            f,
            indent=2,
        )
    _log.info("Best params saved → %s", best_path)

    # ── Interactive HTML visualizations (requires plotly) ──────────────────
    _save_visualizations(study)


if __name__ == "__main__":
    main()
