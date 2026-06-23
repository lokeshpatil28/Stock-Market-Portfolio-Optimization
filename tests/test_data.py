"""Tests for the data pipeline (no network access required).

Covers the alignment and estimation logic with synthetic CSVs:
* misaligned trading dates between tickers are intersected, not silently mixed;
* a ticker with insufficient history is excluded explicitly;
* the Ledoit-Wolf shrinkage covariance is positive semi-definite.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_opt import data as D


def _write_csv(path, dates, prices):
    pd.DataFrame({"Date": pd.to_datetime(dates), "Close": prices}).to_csv(
        path, index=False)


def test_missing_dates_between_tickers_are_intersected(tmp_path):
    # AAA trades Mon-Fri; BBB is missing Wednesday. The aligned panel should
    # keep only the 4 dates both tickers share.
    dates_a = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    dates_b = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]  # no 01-03
    _write_csv(tmp_path / "AAA.csv", dates_a, [10, 11, 12, 13, 14])
    _write_csv(tmp_path / "BBB.csv", dates_b, [20, 21, 23, 24])

    panel = D.load_combined_prices(["AAA", "BBB"], prices_dir=tmp_path)

    assert list(panel.columns) == ["AAA", "BBB"]
    # Only the 4 common dates survive; 2024-01-03 dropped (missing for BBB).
    assert len(panel) == 4
    assert pd.Timestamp("2024-01-03") not in panel.index
    assert not panel.isna().any().any()


def test_ticker_with_insufficient_history_is_excluded(tmp_path):
    # LONG has 300 obs, SHORT only 10. With min_observations=250, SHORT is out.
    long_dates = pd.bdate_range("2023-01-02", periods=300)
    short_dates = pd.bdate_range("2023-01-02", periods=10)
    _write_csv(tmp_path / "LONG.csv", long_dates, np.linspace(100, 130, 300))
    _write_csv(tmp_path / "SHORT.csv", short_dates, np.linspace(50, 51, 10))

    panel = D.load_combined_prices(["LONG", "SHORT"], prices_dir=tmp_path,
                                   min_observations=250)
    assert list(panel.columns) == ["LONG"]
    assert "SHORT" not in panel.columns


def test_missing_csv_is_reported_not_crashed(tmp_path):
    long_dates = pd.bdate_range("2023-01-02", periods=300)
    _write_csv(tmp_path / "LONG.csv", long_dates, np.linspace(100, 130, 300))
    # GHOST has no CSV -> excluded, but LONG still loads.
    panel = D.load_combined_prices(["LONG", "GHOST"], prices_dir=tmp_path)
    assert list(panel.columns) == ["LONG"]


def test_shrinkage_covariance_is_psd():
    rng = np.random.default_rng(7)
    T, n = 400, 12
    # Correlated returns via a random factor structure.
    factor = rng.standard_normal((T, 3))
    loadings = rng.standard_normal((3, n))
    returns = factor @ loadings * 0.01 + 0.002 * rng.standard_normal((T, n))
    rdf = pd.DataFrame(returns, columns=[f"S{i}" for i in range(n)])

    mu, Sigma, intensity = D.estimate_mu_sigma(rdf, method="shrinkage")

    assert Sigma.shape == (n, n)
    assert 0.0 <= intensity <= 1.0
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)  # symmetric
    eig = np.linalg.eigvalsh(Sigma)
    assert eig.min() >= -1e-10, f"not PSD: min eigenvalue {eig.min()}"


def test_sample_vs_shrinkage_differ_and_mu_matches():
    rng = np.random.default_rng(11)
    T, n = 300, 8
    returns = 0.01 * rng.standard_normal((T, n))
    rdf = pd.DataFrame(returns, columns=[f"S{i}" for i in range(n)])

    mu_s, cov_sample, none_int = D.estimate_mu_sigma(rdf, method="sample")
    mu_w, cov_shrink, intensity = D.estimate_mu_sigma(rdf, method="shrinkage")

    assert none_int is None
    assert intensity is not None
    # mu is the sample mean in both modes -> identical.
    np.testing.assert_allclose(mu_s, mu_w)
    # Shrinkage actually changes the covariance.
    assert not np.allclose(cov_sample, cov_shrink)


def test_compute_returns_matches_manual():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 55.0, 55.0]})
    rets = D.compute_returns(prices)
    assert len(rets) == 2
    np.testing.assert_allclose(rets["A"].values, [0.10, -0.10])
    np.testing.assert_allclose(rets["B"].values, [0.10, 0.0])


def test_config_helpers_round_trip():
    sectors = D.get_sector_map()
    caps = D.get_cap_tier_map()
    tickers = D.get_tickers()
    assert len(tickers) == 25
    assert sectors["RELIANCE.NS"] == "Energy"
    assert caps["GOLDBEES.NS"] == "Alt"
    assert set(sectors) == set(tickers) == set(caps)
