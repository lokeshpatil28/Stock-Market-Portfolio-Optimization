"""Tests for the portfolio construction strategies in portfolio_opt.models."""

import numpy as np
import pytest

from portfolio_opt import models as M


def _toy_market(n=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = 0.08 + 0.20 * rng.random(n)            # annualized-ish, all positive
    B = rng.standard_normal((n, n))
    Sigma = B @ B.T / n + 0.02 * np.eye(n)      # SPD
    tickers = [f"T{i}" for i in range(n)]
    return mu, Sigma, tickers


def _assert_common_schema(res, tickers):
    for key in ("weights", "weights_pct", "expected_return", "volatility",
                "sharpe_ratio", "risk_contribution_pct"):
        assert key in res, f"missing key {key}"
    x = res["weights"]
    assert np.isclose(x.sum(), 1.0, atol=1e-6)
    # weights_pct is a {ticker: pct} dict aligned to tickers
    assert set(res["weights_pct"]) == set(tickers)
    # risk contributions sum to ~100 for every model
    assert abs(sum(res["risk_contribution_pct"].values()) - 100.0) < 0.1


def test_min_variance_target_return():
    mu, Sigma, tickers = _toy_market()
    target = float(mu.mean())
    res = M.min_variance_target_return(mu, Sigma, target_return=target,
                                       max_weight=0.4, tickers=tickers)
    _assert_common_schema(res, tickers)
    assert res["converged"]
    assert np.isclose(mu @ res["weights"], target, atol=1e-6)
    assert np.all(res["weights"] <= 0.4 + 1e-7)


def test_mean_variance_utility_basic():
    mu, Sigma, tickers = _toy_market(seed=1)
    res = M.mean_variance_utility(mu, Sigma, risk_aversion=5.0, tickers=tickers)
    _assert_common_schema(res, tickers)
    assert res["converged"]


def test_higher_risk_aversion_lowers_volatility():
    mu, Sigma, tickers = _toy_market(seed=2)
    lo = M.mean_variance_utility(mu, Sigma, risk_aversion=1.0, tickers=tickers)
    hi = M.mean_variance_utility(mu, Sigma, risk_aversion=100.0, tickers=tickers)
    assert hi["volatility"] <= lo["volatility"] + 1e-9


def test_turnover_limit_binds():
    mu, Sigma, tickers = _toy_market(seed=3)
    prev = np.full(len(mu), 1.0 / len(mu))
    unconstrained = M.mean_variance_utility(mu, Sigma, risk_aversion=5.0,
                                            tickers=tickers)
    base_turnover = np.abs(unconstrained["weights"] - prev).sum()
    limit = 0.5 * base_turnover                      # force the cap to bind
    res = M.mean_variance_utility(mu, Sigma, risk_aversion=5.0,
                                  turnover_limit=limit, prev_weights=prev,
                                  tickers=tickers)
    _assert_common_schema(res, tickers)
    assert res["converged"]
    assert res["turnover"] <= limit + 1e-5


def test_turnover_requires_prev_weights():
    mu, Sigma, tickers = _toy_market()
    with pytest.raises(ValueError):
        M.mean_variance_utility(mu, Sigma, risk_aversion=5.0, turnover_limit=0.5)


def test_tracking_error_constraint_is_respected():
    pytest.importorskip("scipy.optimize")
    mu, Sigma, tickers = _toy_market(seed=4)
    bench = np.full(len(mu), 1.0 / len(mu))
    te_limit = 0.05
    res = M.mean_variance_utility(mu, Sigma, risk_aversion=5.0,
                                  tracking_error_limit=te_limit,
                                  benchmark_weights=bench, tickers=tickers)
    _assert_common_schema(res, tickers)
    assert "SLSQP" in res["method"]                   # honestly not the QP solver
    assert res["tracking_error"] <= te_limit + 1e-4


def test_max_sharpe_beats_endpoints():
    mu, Sigma, tickers = _toy_market(seed=5)
    ms = M.max_sharpe(mu, Sigma, risk_free_rate=0.06, tickers=tickers)
    _assert_common_schema(ms, tickers)
    assert ms["converged"]
    lo = M.mean_variance_utility(mu, Sigma, risk_aversion=0.1, tickers=tickers,
                                 risk_free_rate=0.06)
    hi = M.mean_variance_utility(mu, Sigma, risk_aversion=300.0, tickers=tickers,
                                 risk_free_rate=0.06)
    assert ms["sharpe_ratio"] >= lo["sharpe_ratio"] - 1e-9
    assert ms["sharpe_ratio"] >= hi["sharpe_ratio"] - 1e-9


def test_transaction_cost_pulls_toward_prev():
    mu, Sigma, tickers = _toy_market(seed=6)
    prev = np.zeros(len(mu))
    prev[0] = 1.0                                     # fully in asset 0
    cheap = M.transaction_cost_aware(mu, Sigma, prev_weights=prev,
                                     risk_aversion=5.0, cost_penalty=0.0,
                                     tickers=tickers)
    pricey = M.transaction_cost_aware(mu, Sigma, prev_weights=prev,
                                      risk_aversion=5.0, cost_penalty=50.0,
                                      tickers=tickers)
    _assert_common_schema(pricey, tickers)
    # A large cost penalty must reduce turnover away from prev.
    assert pricey["turnover"] < cheap["turnover"]


def test_risk_parity_equalizes_contributions():
    pytest.importorskip("scipy.optimize")
    mu, Sigma, tickers = _toy_market(seed=7)
    res = M.risk_parity(mu, Sigma, tickers=tickers)
    _assert_common_schema(res, tickers)
    assert res["converged"]
    rc = np.array(list(res["risk_contribution_pct"].values()))
    # All contributions close to equal (100/n %).
    assert rc.std() < 0.5
    np.testing.assert_allclose(rc, 100.0 / len(mu), atol=1.0)


def test_sector_caps_applied():
    mu, Sigma, tickers = _toy_market(seed=8)
    sector_map = ["X", "X", "X", "Y", "Y", "Y", "Z", "Z"]
    caps = {"X": 0.25}
    res = M.mean_variance_utility(mu, Sigma, risk_aversion=2.0,
                                  sector_map=sector_map, sector_caps=caps,
                                  tickers=tickers)
    assert res["converged"]
    x = res["weights"]
    assert x[0] + x[1] + x[2] <= 0.25 + 1e-6


def test_compare_all_models_shapes():
    import pandas as pd
    mu, Sigma, tickers = _toy_market(n=6, seed=9)
    out = M.compare_all_models(mu, Sigma, tickers, target_return=float(mu.mean()),
                               risk_aversion=8.0)
    assert isinstance(out["summary"], pd.DataFrame)
    assert isinstance(out["weights"], pd.DataFrame)
    # five models, one row each
    assert out["summary"].shape[0] == 5
    assert list(out["weights"].index) == tickers
    assert out["weights"].shape == (6, 5)
    # each model's weight column sums to ~100%
    col_sums = out["weights"].sum(axis=0)
    assert np.allclose(col_sums.values, 100.0, atol=0.1)
