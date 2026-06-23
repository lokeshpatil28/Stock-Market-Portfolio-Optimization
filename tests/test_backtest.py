"""Tests for the walk-forward backtest engine (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_opt import backtest as B


def _synthetic_prices(n_assets=4, n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    drift = 0.0003 + 0.0002 * rng.standard_normal(n_assets)
    daily = drift + 0.01 * rng.standard_normal((n_days, n_assets))
    prices = 100.0 * np.cumprod(1.0 + daily, axis=0)
    return pd.DataFrame(prices, index=dates,
                        columns=[f"S{i}" for i in range(n_assets)])


def test_cost_rate_units():
    # The headline check the user asked to confirm: 10 bps == 0.001.
    assert B.cost_rate_from_bps(10) == 0.001
    assert B.cost_rate_from_bps(0) == 0.0
    assert B.cost_rate_from_bps(50) == 0.005


def test_rebalance_dates_monthly_are_first_trading_days():
    idx = pd.bdate_range("2022-01-01", periods=200)
    rd = B.rebalance_dates(idx, "monthly")
    # One per calendar month present, all within the index, sorted, unique.
    assert rd.is_monotonic_increasing
    assert len(rd) == len(set(rd))
    assert set(rd).issubset(set(idx))
    months = {(d.year, d.month) for d in rd}
    assert len(months) == len(rd)            # exactly one rebalance per month


def test_run_backtest_equal_weight_drifts_and_compounds():
    prices = _synthetic_prices()
    n = prices.shape[1]

    def equal_weight(mu, Sigma, **kw):
        return {"weights": np.full(n, 1.0 / n), "converged": True}

    bt = B.run_backtest(prices, equal_weight, lookback_years=1,
                        transaction_cost_bps=0)
    eq = bt["equity_curve"]
    assert len(eq) > 0
    assert np.all(np.isfinite(eq.values))
    # Daily weights stay on the simplex (sum to 1) as they drift.
    wsum = bt["daily_weights"].sum(axis=1)
    np.testing.assert_allclose(wsum.values, 1.0, atol=1e-9)
    assert bt["n_rebalances"] >= 1
    assert bt["first_rebalance"] <= bt["last_rebalance"]


def test_transaction_costs_reduce_terminal_value():
    prices = _synthetic_prices(seed=2)
    n = prices.shape[1]

    # A churning strategy: alternate concentration between two assets so every
    # rebalance incurs large turnover.
    state = {"k": 0}

    def churn(mu, Sigma, **kw):
        w = np.zeros(n)
        w[state["k"] % n] = 1.0
        state["k"] += 1
        return {"weights": w, "converged": True}

    state["k"] = 0
    free = B.run_backtest(prices, churn, lookback_years=1, transaction_cost_bps=0)
    state["k"] = 0
    costly = B.run_backtest(prices, churn, lookback_years=1, transaction_cost_bps=50)
    assert costly["equity_curve"].iloc[-1] < free["equity_curve"].iloc[-1]


def test_infeasible_model_is_skipped_not_crashed():
    prices = _synthetic_prices(seed=3)

    def always_bad(mu, Sigma, **kw):
        return {"weights": np.full(prices.shape[1], np.nan), "converged": False}

    bt = B.run_backtest(prices, always_bad, lookback_years=1)
    # Every rebalance is skipped; weights remain all-cash, value stays at 1.0.
    assert bt["n_skipped"] == bt["n_rebalances"]
    np.testing.assert_allclose(bt["equity_curve"].values, 1.0, atol=1e-12)


def test_performance_metrics_on_known_curve():
    # Up-trending (0.1%/day mean) with tiny noise so volatility is nonzero and
    # Sharpe is well defined; drift dominates so it stays (near) monotone up.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=504)
    daily = 0.001 + 1e-5 * rng.standard_normal(504)
    equity = pd.Series(np.cumprod(1.0 + daily), index=dates)
    perf = B.compute_performance_metrics(equity, risk_free_rate=0.0)
    m = perf["metrics"]
    assert m["annualized_return"] > 0
    assert m["sharpe_ratio"] > 0
    assert m["max_drawdown"] > -0.01            # negligible dips only
    # Rolling Sharpe and drawdown series are returned for the dashboard.
    assert perf["rolling_sharpe"].notna().any()
    assert (perf["drawdown_series"] <= 1e-9).all()      # drawdown is <= 0


def test_sharpe_formula_is_pinned_to_arithmetic_convention():
    """Pin the exact Sharpe definition so it can't silently drift conventions.

    Sharpe = (mean(period_returns)*P - rf) / (std(period_returns, ddof=1)*sqrt(P)).
    This is the annualized ARITHMETIC mean excess return over annualized std --
    NOT (CAGR - rf)/vol.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-01", periods=600)
    daily = 0.0005 + 0.012 * rng.standard_normal(600)      # genuine variation
    equity = pd.Series(100.0 * np.cumprod(1.0 + daily), index=dates)

    rf, P = 0.06, 252
    perf = B.compute_performance_metrics(equity, risk_free_rate=rf,
                                         periods_per_year=P)
    m = perf["metrics"]

    # Recompute the expected Sharpe independently from the equity curve.
    d = equity.pct_change().dropna()
    expected_sharpe = (d.mean() * P - rf) / (d.std(ddof=1) * np.sqrt(P))
    assert m["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-12)

    # The arithmetic annualized return must be exactly the Sharpe numerator base.
    assert m["annualized_return_arithmetic"] == pytest.approx(d.mean() * P, rel=1e-12)

    # And it must NOT be the geometric CAGR-based Sharpe (guards against a silent
    # switch to (CAGR - rf)/vol). With nonzero vol the two differ by ~sigma/2.
    cagr_sharpe = (m["annualized_return"] - rf) / m["annualized_vol"]
    assert abs(m["sharpe_ratio"] - cagr_sharpe) > 1e-4
    assert m["sharpe_ratio"] > cagr_sharpe          # arithmetic >= geometric

    # The explanatory note travels with the metrics.
    assert "arithmetic" in m["sharpe_note"] and "CAGR" in m["sharpe_note"]


def test_metrics_max_drawdown_is_negative_on_dip():
    dates = pd.bdate_range("2022-01-01", periods=10)
    vals = [1.0, 1.1, 1.2, 0.9, 0.95, 1.0, 1.3, 1.25, 1.4, 1.5]
    equity = pd.Series(vals, index=dates)
    perf = B.compute_performance_metrics(equity, risk_free_rate=0.0)
    # Worst peak-to-trough: 1.2 -> 0.9 = -25%.
    assert perf["metrics"]["max_drawdown"] == pytest.approx(-0.25, abs=1e-9)
