"""Config-driven, idempotent market-data pipeline.

Supersedes the original procedural download/compute-stats scripts with
composable, testable functions:

* :func:`download_prices`     -- fetch per-ticker close prices (retry + backoff,
                                 idempotent, loud about failures).
* :func:`load_combined_prices`-- align per-ticker CSVs onto common trading dates
                                 with explicit NaN handling.
* :func:`compute_returns`     -- daily percentage returns.
* :func:`estimate_mu_sigma`   -- annualized mean and covariance (sample or
                                 Ledoit-Wolf shrinkage, returning the intensity).
* :func:`get_sector_map` / :func:`get_cap_tier_map` -- constraint/dashboard tags.

All paths and the universe are driven by ``config.yaml`` at the repo root.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# --- Repo-relative paths -----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
PROCESSED_DIR = DATA_DIR / "processed"

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path=CONFIG_PATH) -> dict:
    """Load and return the YAML config as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_tickers(config=None) -> list:
    """Return the list of ticker symbols from config."""
    config = config or load_config()
    return [t["symbol"] for t in config["tickers"]]


def get_sector_map(config=None) -> dict:
    """Return ``{symbol: sector}`` for constraint building and grouping."""
    config = config or load_config()
    return {t["symbol"]: t["sector"] for t in config["tickers"]}


def get_cap_tier_map(config=None) -> dict:
    """Return ``{symbol: cap_tier}`` for constraint building and grouping."""
    config = config or load_config()
    return {t["symbol"]: t["cap_tier"] for t in config["tickers"]}


def resolve_dates(config=None):
    """Resolve the (start, end) datetimes from config, computed dynamically.

    ``end_date`` defaults to today; ``start_date`` defaults to
    ``end_date - lookback_years``. Explicit ISO dates in config override.
    """
    config = config or load_config()
    dw = config.get("date_window", {})
    end = dw.get("end_date")
    start = dw.get("start_date")
    lookback = int(dw.get("lookback_years", 3))

    end_dt = pd.to_datetime(end).to_pydatetime() if end else datetime.today()
    if start:
        start_dt = pd.to_datetime(start).to_pydatetime()
    else:
        start_dt = end_dt - timedelta(days=365 * lookback)
    return start_dt, end_dt


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _price_csv_path(symbol, out_dir):
    return Path(out_dir) / f"{symbol}.csv"


def _existing_covers(path, start, end, tol_days=7):
    """True if an existing CSV already spans ~[start, end] (idempotency check)."""
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return False
    if df.empty or "Date" not in df.columns:
        return False
    first, last = df["Date"].min(), df["Date"].max()
    tol = timedelta(days=tol_days)
    return (first <= pd.Timestamp(start) + tol) and (last >= pd.Timestamp(end) - tol)


def _download_one(symbol, start, end):
    """Single yfinance fetch -> tidy DataFrame[Date, Close]. Raises on empty."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False,
                     auto_adjust=True, threads=False)
    if df is None or df.empty:
        raise ValueError("empty frame returned")
    # Newer yfinance returns a (Price, Ticker) MultiIndex even for one symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError(f"no Close column (got {list(df.columns)})")
    out = df.reset_index()[["Date", "Close"]].dropna()
    if out.empty:
        raise ValueError("all-NaN Close")
    return out


def download_prices(tickers, start, end, out_dir=PRICES_DIR,
                    max_retries=3, backoff=1.5, force=False):
    """Download per-ticker close prices to ``out_dir`` as ``<symbol>.csv``.

    Idempotent: a ticker whose CSV already spans the requested window is skipped
    unless ``force=True``. Each download is retried up to ``max_retries`` times
    with exponential backoff. Any ticker still failing afterward is logged at
    ERROR level and reported in the return value -- never silently dropped.

    Returns
    -------
    dict
        ``{"succeeded": [...], "skipped": [...], "failed": {symbol: reason}}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    succeeded, skipped, failed = [], [], {}

    for symbol in tickers:
        path = _price_csv_path(symbol, out_dir)
        if not force and path.exists() and _existing_covers(path, start, end):
            logger.info("skip %s (cached, range covered)", symbol)
            skipped.append(symbol)
            continue

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                df = _download_one(symbol, start, end)
                df.to_csv(path, index=False)
                logger.info("downloaded %s: %d rows [%s .. %s]", symbol, len(df),
                            df["Date"].min().date(), df["Date"].max().date())
                succeeded.append(symbol)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 - yfinance raises many types
                last_err = str(exc)
                wait = backoff ** attempt
                logger.warning("attempt %d/%d failed for %s: %s (retry in %.1fs)",
                               attempt, max_retries, symbol, last_err, wait)
                if attempt < max_retries:
                    time.sleep(wait)

        if last_err is not None:
            logger.error("FAILED to download %s after %d attempts: %s",
                         symbol, max_retries, last_err)
            failed[symbol] = last_err

    logger.info("download summary: %d ok, %d skipped, %d failed",
                len(succeeded), len(skipped), len(failed))
    return {"succeeded": succeeded, "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# Combine / align
# ---------------------------------------------------------------------------
def load_combined_prices(tickers, prices_dir=PRICES_DIR, min_observations=0):
    """Merge per-ticker CSVs into one Close-price panel on common trading dates.

    Missing tickers (no CSV) and tickers with fewer than ``min_observations``
    rows are reported and excluded explicitly (logged, not silently dropped).
    The remaining tickers are outer-joined on Date; the per-ticker count of
    misaligned dates (present for some tickers, missing for others) is logged,
    then rows with any NaN are dropped so every retained date has a price for
    every retained ticker.

    Returns
    -------
    pandas.DataFrame
        Index = Date (sorted), columns = retained ticker symbols.
    """
    prices_dir = Path(prices_dir)
    series = {}
    missing, too_short = [], []

    for symbol in tickers:
        path = _price_csv_path(symbol, prices_dir)
        if not path.exists():
            missing.append(symbol)
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).dropna(subset=["Close"])
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
        if len(df) < min_observations:
            too_short.append((symbol, len(df)))
            continue
        series[symbol] = df.set_index("Date")["Close"].rename(symbol)

    if missing:
        logger.warning("no CSV for %d tickers (excluded): %s",
                       len(missing), ", ".join(missing))
    if too_short:
        logger.warning("excluded %d tickers with < %d observations: %s",
                       len(too_short), min_observations,
                       ", ".join(f"{s}({n})" for s, n in too_short))
    if not series:
        raise ValueError("no usable price series found")

    # Outer join keeps every date any ticker traded -> exposes misalignment.
    combined = pd.concat(series.values(), axis=1, join="outer").sort_index()

    na_counts = combined.isna().sum()
    misaligned = na_counts[na_counts > 0]
    if not misaligned.empty:
        logger.info("misaligned dates per ticker (NaN before alignment):\n%s",
                    misaligned.to_string())

    n_before = len(combined)
    combined = combined.dropna(how="any")
    n_after = len(combined)
    logger.info("aligned panel: %d tickers x %d common dates (dropped %d "
                "partially-missing dates) [%s .. %s]",
                combined.shape[1], n_after, n_before - n_after,
                combined.index.min().date(), combined.index.max().date())
    return combined


# ---------------------------------------------------------------------------
# Returns & moments
# ---------------------------------------------------------------------------
def compute_returns(prices_df):
    """Daily simple (percentage-change) returns; first NaN row dropped."""
    if isinstance(prices_df, pd.DataFrame):
        return prices_df.pct_change().dropna(how="any")
    arr = np.asarray(prices_df, dtype=float)
    rets = arr[1:] / arr[:-1] - 1.0
    return rets


def estimate_mu_sigma(returns_df, method="shrinkage", periods_per_year=TRADING_DAYS):
    """Annualized expected returns and covariance.

    ``mu`` is always the annualized sample mean. ``Sigma`` is either the
    annualized sample covariance (``method="sample"``) or the annualized
    Ledoit-Wolf shrinkage covariance (``method="shrinkage"``).

    Returns
    -------
    mu : ndarray, shape (n,)
    Sigma : ndarray, shape (n, n)
    shrinkage_intensity : float or None
        Ledoit-Wolf shrinkage coefficient in [0, 1] (``None`` for sample).
    """
    R = returns_df.values if isinstance(returns_df, pd.DataFrame) else np.asarray(returns_df, float)
    if R.ndim != 2:
        raise ValueError("returns must be 2-D (T, n)")

    mu = R.mean(axis=0) * periods_per_year

    if method == "sample":
        Sigma = np.cov(R, rowvar=False) * periods_per_year
        return mu, Sigma, None

    if method == "shrinkage":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(R)
        Sigma = lw.covariance_ * periods_per_year
        return mu, Sigma, float(lw.shrinkage_)

    raise ValueError(f"unknown method {method!r} (expected 'sample'/'shrinkage')")
