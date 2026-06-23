"""Run the full walk-forward backtest across all strategies and benchmarks.

Saves to results/:
    backtest_metrics.csv          one row per strategy/benchmark, all metrics
    equity_curves.csv             date x strategy
    rolling_sharpe.csv            date x strategy (6-month rolling Sharpe)
    drawdown_series.csv           date x strategy (underwater curve)
    rebalance_weight_history.csv  long: date, strategy, ticker, weight_pct

Run:  python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_opt import backtest as B  # noqa: E402
from portfolio_opt import data as D       # noqa: E402
from portfolio_opt import models as M     # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LOOKBACK_YEARS = 1          # chosen: ~1.86 yr out-of-sample, 23 rebalances
MAX_WEIGHT = 0.25           # diversification cap (<= 25% per name)
COST_BPS = 10


def main():
    cfg = D.load_config()
    rf = float(cfg.get("risk_free_rate", 0.06))
    rebal_freq = cfg.get("rebalance_frequency", "monthly")
    tickers = D.get_tickers(cfg)
    prices = D.load_combined_prices(tickers, min_observations=int(cfg["min_observations"]))
    tickers = list(prices.columns)

    # Strategy registry: name -> (model_fn, model_kwargs)
    strategies = {
        "min_variance":   (M.min_variance_target_return,
                           {"target_return": 0.12, "max_weight": MAX_WEIGHT}),
        "utility_ra2":    (M.mean_variance_utility,
                           {"risk_aversion": 2.0, "max_weight": MAX_WEIGHT}),
        "utility_ra10":   (M.mean_variance_utility,
                           {"risk_aversion": 10.0, "max_weight": MAX_WEIGHT}),
        "utility_ra50":   (M.mean_variance_utility,
                           {"risk_aversion": 50.0, "max_weight": MAX_WEIGHT}),
        "max_sharpe":     (M.max_sharpe,
                           {"max_weight": MAX_WEIGHT, "risk_free_rate": rf}),
        "risk_parity":    (M.risk_parity,
                           {"max_weight": MAX_WEIGHT}),
    }

    equity_curves, rolling, drawdowns = {}, {}, {}
    metrics_rows, weight_history = {}, []
    eval_start = None

    print(f"Running {len(strategies)} strategies | lookback={LOOKBACK_YEARS}y | "
          f"{rebal_freq} | cost={COST_BPS}bps\n")

    for name, (fn, kw) in strategies.items():
        bt = B.run_backtest(prices, fn, kw, rebalance_frequency=rebal_freq,
                            lookback_years=LOOKBACK_YEARS,
                            transaction_cost_bps=COST_BPS)
        eval_start = bt["first_rebalance"] if eval_start is None else eval_start
        perf = B.compute_performance_metrics(bt["equity_curve"], rf,
                                             turnover=bt["turnover"])
        equity_curves[name] = bt["equity_curve"]
        rolling[name] = perf["rolling_sharpe"]
        drawdowns[name] = perf["drawdown_series"]
        m = dict(perf["metrics"])
        m["n_skipped"] = bt["n_skipped"]
        metrics_rows[name] = m
        print(f"  {name:16s} rebal={bt['n_rebalances']:2d} "
              f"skipped={bt['n_skipped']:2d}  "
              f"Sharpe={m['sharpe_ratio']:5.2f}  CAGR={m['annualized_return']*100:6.2f}%  "
              f"maxDD={m['max_drawdown']*100:6.2f}%")

        # Rebalance weight history (long format) for the weight-drift chart.
        rw = bt["rebalance_weights"] * 100.0
        long = rw.reset_index().melt(id_vars="index", var_name="ticker",
                                     value_name="weight_pct")
        long = long.rename(columns={"index": "date"})
        long.insert(1, "strategy", name)
        weight_history.append(long)

    # --- Benchmarks ---
    print("\nBenchmarks:")
    benches = B.compute_benchmarks(prices, eval_start,
                                   end_date=prices.index.max(),
                                   transaction_cost_bps=COST_BPS)
    for name, curve in benches.items():
        equity_curves[name] = curve
        perf = B.compute_performance_metrics(curve, rf)
        rolling[name] = perf["rolling_sharpe"]
        drawdowns[name] = perf["drawdown_series"]
        metrics_rows[name] = dict(perf["metrics"])
        m = metrics_rows[name]
        print(f"  {name:18s} Sharpe={m['sharpe_ratio']:5.2f}  "
              f"CAGR={m['annualized_return']*100:6.2f}%  maxDD={m['max_drawdown']*100:6.2f}%")

    # --- Assemble & save ---
    RESULTS.mkdir(exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows).T
    # Pull the Sharpe-convention note out of the numeric table; write it as a
    # comment header on the CSV so anyone opening the file sees why the Sharpe
    # column does not equal (Return - rf) / vol. Read back with comment="#".
    sharpe_note = str(metrics_df["sharpe_note"].dropna().iloc[0])
    # The string note column makes the transposed frame object-dtype; drop it and
    # coerce the rest back to numeric so formatting/rounding works.
    metrics_df = metrics_df.drop(columns=["sharpe_note"]).apply(
        pd.to_numeric, errors="coerce")
    col_order = ["annualized_return", "annualized_return_arithmetic",
                 "total_return", "annualized_vol", "sharpe_ratio", "max_drawdown",
                 "final_value", "avg_turnover", "annualized_turnover",
                 "n_rebalances", "n_skipped", "years", "n_days"]
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]
    metrics_path = RESULTS / "backtest_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {sharpe_note}\n")
        metrics_df.to_csv(fh)

    pd.DataFrame(equity_curves).to_csv(RESULTS / "equity_curves.csv")
    pd.DataFrame(rolling).to_csv(RESULTS / "rolling_sharpe.csv")
    pd.DataFrame(drawdowns).to_csv(RESULTS / "drawdown_series.csv")
    pd.concat(weight_history, ignore_index=True).to_csv(
        RESULTS / "rebalance_weight_history.csv", index=False)

    # --- Final table ---
    show = metrics_df.copy()
    for c in ["annualized_return", "total_return", "annualized_vol", "max_drawdown"]:
        show[c] = (show[c] * 100).round(2)
    show["sharpe_ratio"] = show["sharpe_ratio"].round(2)
    show["final_value"] = show["final_value"].round(3)
    if "avg_turnover" in show:
        show["avg_turnover"] = show["avg_turnover"].round(3)
    print("\n" + "=" * 100)
    print("BACKTEST METRICS  (returns/vol/drawdown in %, out-of-sample "
          f"{pd.Timestamp(eval_start).date()} .. {prices.index.max().date()})")
    print("=" * 100)
    cols = ["annualized_return", "annualized_vol", "sharpe_ratio",
            "max_drawdown", "total_return", "final_value", "avg_turnover"]
    print(show[[c for c in cols if c in show.columns]].to_string())
    print("=" * 100)
    print(f"\nArtifacts written to {RESULTS}")


if __name__ == "__main__":
    main()
