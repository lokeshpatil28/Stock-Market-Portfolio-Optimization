"""Walk-forward backtesting of portfolio strategies.

Engine design
-------------
* Rebalance on the first trading day of each period (monthly by default) once
  enough trailing history exists.
* On a rebalance day, ``mu``/``Sigma`` are estimated from a trailing
  ``lookback_years`` window ending at that day's close (no look-ahead), the
  model is called for target weights, and a turnover-proportional transaction
  cost is charged.
* Between rebalances the held weights **drift** with realized daily returns
  (we do not snap back to the target every day) and the portfolio compounds.

Transaction costs are quoted in basis points: ``transaction_cost_bps=10`` means
a cost rate of ``10 / 10_000 = 0.001`` (0.1%) charged on the L1 turnover at each
rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------
_FREQ_TO_PERIOD = {"monthly": "M", "weekly": "W", "quarterly": "Q"}


def rebalance_dates(index, frequency="monthly"):
    """First trading day of each period in ``index`` for the given frequency."""
    if frequency not in _FREQ_TO_PERIOD:
        raise ValueError(f"unsupported frequency {frequency!r}")
    index = pd.DatetimeIndex(index)
    periods = index.to_period(_FREQ_TO_PERIOD[frequency])
    s = pd.Series(index, index=periods)
    first = s.groupby(level=0).first()
    return pd.DatetimeIndex(sorted(first.values))


def cost_rate_from_bps(transaction_cost_bps):
    """Convert basis points to a fractional cost rate (10 bps -> 0.001)."""
    return float(transaction_cost_bps) / 10_000.0


# ---------------------------------------------------------------------------
# 1. Walk-forward backtest
# ---------------------------------------------------------------------------
def run_backtest(prices_df, model_fn, model_kwargs=None,
                 rebalance_frequency="monthly", lookback_years=2,
                 transaction_cost_bps=10, estimation_method="shrinkage",
                 initial_value=1.0):
    """Run a walk-forward, drifting-weight backtest of a single strategy.

    Parameters
    ----------
    prices_df : DataFrame
        Aligned close prices (index = dates, columns = tickers).
    model_fn : callable
        ``model_fn(mu, Sigma, **model_kwargs) -> {"weights": ndarray, ...}``.
    model_kwargs : dict, optional
        Extra keyword args passed to ``model_fn`` at every rebalance.
    rebalance_frequency, lookback_years, transaction_cost_bps, estimation_method
        See module docstring.

    Returns
    -------
    dict with keys:
        ``equity_curve`` (Series), ``daily_weights`` (wide DataFrame, drifted),
        ``rebalance_weights`` (wide DataFrame, targets at each rebalance),
        ``turnover`` (Series indexed by rebalance date),
        ``n_rebalances``, ``first_rebalance``, ``last_rebalance``,
        ``n_skipped`` (rebalances where the model was infeasible / non-convergent).
    """
    model_kwargs = dict(model_kwargs or {})
    prices = prices_df.sort_index()
    tickers = list(prices.columns)
    n = len(tickers)

    returns = prices.pct_change().dropna(how="any")
    idx = returns.index
    R = returns.values
    cost_rate = cost_rate_from_bps(transaction_cost_bps)
    lookback_days = int(round(lookback_years * TRADING_DAYS))

    # Rebalance dates with enough trailing history.
    all_rebals = rebalance_dates(idx, rebalance_frequency)
    pos = {d: i for i, d in enumerate(idx)}
    rebals = [d for d in all_rebals if d in pos and pos[d] >= lookback_days]
    if not rebals:
        raise ValueError("not enough history for any rebalance "
                         f"(need >= {lookback_days} trailing days)")
    rebal_set = set(rebals)
    start_i = pos[rebals[0]]

    value = float(initial_value)
    w = np.zeros(n)
    eq_dates, eq_vals, daily_w = [], [], []
    turnover_rec, rebal_w = {}, {}
    n_skipped = 0

    for i in range(start_i, len(idx)):
        d = idx[i]
        r = R[i]

        # --- 1. realize today's return with the (drifted) held weights ---
        port_ret = float(w @ r)
        value *= (1.0 + port_ret)
        if w.sum() > 0 and (1.0 + port_ret) != 0:
            w = w * (1.0 + r) / (1.0 + port_ret)   # weights drift with returns

        # --- 2. rebalance at the close if scheduled ---
        if d in rebal_set:
            window = returns.iloc[i - lookback_days + 1: i + 1]
            mu_w, Sigma_w, _ = D.estimate_mu_sigma(window, method=estimation_method)
            try:
                res = model_fn(mu_w, Sigma_w, **model_kwargs)
                w_target = np.asarray(res["weights"], dtype=float)
                ok = bool(res.get("converged", True)) and np.all(np.isfinite(w_target))
            except Exception:  # noqa: BLE001 - a bad window must not kill the run
                ok = False

            if ok:
                turnover = float(np.abs(w_target - w).sum())
                value *= (1.0 - cost_rate * turnover)   # charge turnover cost
                w = w_target
                turnover_rec[d] = turnover
                rebal_w[d] = w_target.copy()
            else:
                # Infeasible/non-convergent: hold current weights, no trade.
                n_skipped += 1
                turnover_rec[d] = 0.0
                rebal_w[d] = w.copy()

        eq_dates.append(d)
        eq_vals.append(value)
        daily_w.append(w.copy())

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates),
                       name="portfolio_value")
    daily_weights = pd.DataFrame(daily_w, index=equity.index, columns=tickers)
    rebalance_weights = pd.DataFrame(
        {d: wv for d, wv in rebal_w.items()}).T
    if not rebalance_weights.empty:
        rebalance_weights.columns = tickers
        rebalance_weights = rebalance_weights.sort_index()
    turnover = pd.Series(turnover_rec, name="turnover").sort_index()

    return {
        "equity_curve": equity,
        "daily_weights": daily_weights,
        "rebalance_weights": rebalance_weights,
        "turnover": turnover,
        "n_rebalances": len(rebals),
        "first_rebalance": rebals[0],
        "last_rebalance": rebals[-1],
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# 2. Benchmarks
# ---------------------------------------------------------------------------
def _equal_weight_curve(returns, eval_start, cost_rate, rebalance):
    """Equal-weight equity curve from ``eval_start``.

    ``rebalance=True`` -> reset to equal weights on the first trading day of each
    month; ``rebalance=False`` -> buy-and-hold (deploy once, then drift).
    """
    sub = returns.loc[eval_start:]
    n = sub.shape[1]
    eq_w = np.full(n, 1.0 / n)
    rebal_set = set(rebalance_dates(sub.index, "monthly")) if rebalance else set()

    value = 1.0 * (1.0 - cost_rate * 1.0)   # initial deployment from cash (turnover 1)
    w = eq_w.copy()
    vals = []
    for d, row in zip(sub.index, sub.values):
        port_ret = float(w @ row)
        value *= (1.0 + port_ret)
        if (1.0 + port_ret) != 0:
            w = w * (1.0 + row) / (1.0 + port_ret)
        if d in rebal_set:
            turnover = float(np.abs(eq_w - w).sum())
            value *= (1.0 - cost_rate * turnover)
            w = eq_w.copy()
        vals.append(value)
    return pd.Series(vals, index=sub.index)


def compute_benchmarks(prices_df, eval_start, end_date=None,
                       transaction_cost_bps=10):
    """Build benchmark equity curves aligned to the strategy evaluation period.

    Returns a dict of Series:
        ``equal_weight``           -- monthly-rebalanced equal weight,
        ``buy_and_hold_equal``     -- equal weight, never rebalanced,
        ``nifty50``                -- ^NSEI total path if downloadable, else
                                       omitted (a warning is printed -- not faked).
    """
    prices = prices_df.sort_index()
    returns = prices.pct_change().dropna(how="any")
    cost_rate = cost_rate_from_bps(transaction_cost_bps)
    eval_start = pd.Timestamp(eval_start)

    out = {
        "equal_weight": _equal_weight_curve(returns, eval_start, cost_rate, True),
        "buy_and_hold_equal": _equal_weight_curve(returns, eval_start, cost_rate, False),
    }

    # --- Nifty 50 (^NSEI) for the same window, if available ---
    start = eval_start
    end = pd.Timestamp(end_date) if end_date is not None else prices.index.max()
    nifty = _download_nifty(start, end)
    if nifty is not None and not nifty.empty:
        nifty = nifty.reindex(out["equal_weight"].index).ffill().dropna()
        if not nifty.empty:
            out["nifty50"] = nifty / nifty.iloc[0]   # normalize to 1.0 at start
        else:
            print("[benchmark] ^NSEI returned no overlapping dates - omitting Nifty 50.")
    else:
        print("[benchmark] ^NSEI download unavailable - omitting Nifty 50 "
              "(not faked).")
    return out


def _download_nifty(start, end):
    """Best-effort ^NSEI close series; returns None on any failure."""
    try:
        import yfinance as yf

        df = yf.download("^NSEI", start=start, end=end + pd.Timedelta(days=1),
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df["Close"].dropna()
    except Exception as exc:  # noqa: BLE001
        print(f"[benchmark] ^NSEI download error: {exc}")
        return None


# ---------------------------------------------------------------------------
# 3. Performance metrics
# ---------------------------------------------------------------------------
# The Sharpe ratio uses the *standard* definition: the annualized arithmetic
# mean of period (daily) excess returns over their annualized standard
# deviation. This is intentionally NOT (CAGR - rf)/vol: the reported
# ``annualized_return`` is the geometric CAGR, while the Sharpe numerator is the
# arithmetic mean. The two annualized returns differ by the volatility drag
# (~0.5 * sigma^2), so dividing the Return column by vol will not reproduce the
# Sharpe column exactly -- both figures are standard, they just use different
# (arithmetic vs geometric) return conventions.
SHARPE_NOTE = (
    "Sharpe uses annualized arithmetic mean excess return "
    "((mean(period_returns)*periods_per_year - risk_free_rate) / "
    "(std(period_returns)*sqrt(periods_per_year))); the Return column is "
    "geometric CAGR -- both are standard but won't divide exactly into each "
    "other (they differ by the ~0.5*sigma^2 volatility drag)."
)


def compute_performance_metrics(equity_curve, risk_free_rate=0.06,
                                turnover=None, periods_per_year=TRADING_DAYS,
                                rolling_window=126):
    """Summary stats plus rolling-Sharpe and drawdown series for the dashboard.

    Sharpe ratio convention (pinned; see :data:`SHARPE_NOTE`)::

        sharpe = (mean(period_returns) * periods_per_year - risk_free_rate)
                 / (std(period_returns, ddof=1) * sqrt(periods_per_year))

    i.e. the annualized **arithmetic** mean of daily excess returns divided by
    the annualized standard deviation -- the standard Sharpe definition. Note
    this does NOT equal ``(annualized_return - rf) / annualized_vol`` because
    ``annualized_return`` is the **geometric** CAGR; the two differ by the
    volatility drag. ``annualized_return_arithmetic`` is also returned so the
    Sharpe is exactly reproducible from the reported columns.

    Returns a dict with scalar metrics (``annualized_return`` [CAGR],
    ``annualized_return_arithmetic``, ``annualized_vol``, ``sharpe_ratio``,
    ``sharpe_note``, ``max_drawdown``, turnover stats, ``final_value``) and two
    Series: ``rolling_sharpe`` (6-month, 126-day) and ``drawdown_series``.
    """
    equity = equity_curve.dropna()
    daily = equity.pct_change().dropna()
    n_days = len(daily)
    years = n_days / periods_per_year if n_days else np.nan

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan
    ann_vol = daily.std(ddof=1) * np.sqrt(periods_per_year) if n_days > 1 else np.nan
    ann_ret_arith = daily.mean() * periods_per_year if n_days else np.nan
    sharpe = ((ann_ret_arith - risk_free_rate) / ann_vol
              if ann_vol and ann_vol > 1e-12 else np.nan)

    drawdown_series = equity / equity.cummax() - 1.0
    max_dd = float(drawdown_series.min()) if len(drawdown_series) else np.nan

    # Rolling 6-month annualized Sharpe.
    roll_mean = daily.rolling(rolling_window).mean()
    roll_std = daily.rolling(rolling_window).std(ddof=1)
    rolling_sharpe = ((roll_mean - risk_free_rate / periods_per_year) / roll_std
                      * np.sqrt(periods_per_year))
    rolling_sharpe.name = "rolling_sharpe_6m"

    metrics = {
        "annualized_return": float(cagr),                       # geometric CAGR
        "annualized_return_arithmetic": float(ann_ret_arith),   # Sharpe numerator base
        "total_return": float(total_return),
        "annualized_vol": float(ann_vol),
        "sharpe_ratio": float(sharpe),                          # arithmetic; see sharpe_note
        "sharpe_note": SHARPE_NOTE,
        "max_drawdown": max_dd,
        "final_value": float(equity.iloc[-1]) if len(equity) else np.nan,
        "n_days": int(n_days),
        "years": float(years),
    }
    if turnover is not None and len(turnover):
        t = turnover[turnover.index >= equity.index.min()]
        metrics.update({
            "avg_turnover": float(t.mean()),
            "total_turnover": float(t.sum()),
            "annualized_turnover": float(t.sum() / years) if years and years > 0 else np.nan,
            "n_rebalances": int(len(t)),
        })

    return {"metrics": metrics, "rolling_sharpe": rolling_sharpe,
            "drawdown_series": drawdown_series}
