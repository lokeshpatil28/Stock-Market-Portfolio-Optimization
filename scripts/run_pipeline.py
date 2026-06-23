"""End-to-end data pipeline driven by config.yaml.

Downloads prices, aligns them, computes returns, estimates moments (both sample
and Ledoit-Wolf shrinkage covariances), and writes processed artifacts to
``data/processed/``:

    mu.npy                  annualized expected returns (sample mean)
    cov_sample.npy          annualized sample covariance
    cov_shrinkage.npy       annualized Ledoit-Wolf covariance
    shrinkage_intensity.txt Ledoit-Wolf shrinkage coefficient
    universe.txt            retained tickers (row/column order of the matrices)

Run:  python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_opt import data as D  # noqa: E402


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_pipeline")

    config = D.load_config()
    tickers = D.get_tickers(config)
    start, end = D.resolve_dates(config)
    method = config.get("estimation_method", "shrinkage")
    min_obs = int(config.get("min_observations", 0))

    log.info("universe: %d tickers | window %s .. %s | method=%s",
             len(tickers), start.date(), end.date(), method)

    # 1. Download (idempotent, retrying) -----------------------------------
    report = D.download_prices(tickers, start, end)

    # 2. Align -------------------------------------------------------------
    prices = D.load_combined_prices(tickers, min_observations=min_obs)
    universe = list(prices.columns)

    # 3. Returns -----------------------------------------------------------
    returns = D.compute_returns(prices)

    # 4. Moments (both estimators) -----------------------------------------
    mu, cov_sample, _ = D.estimate_mu_sigma(returns, method="sample")
    _, cov_shrink, intensity = D.estimate_mu_sigma(returns, method="shrinkage")

    # 5. Persist -----------------------------------------------------------
    D.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(D.PROCESSED_DIR / "mu.npy", mu)
    np.save(D.PROCESSED_DIR / "cov_sample.npy", cov_sample)
    np.save(D.PROCESSED_DIR / "cov_shrinkage.npy", cov_shrink)
    (D.PROCESSED_DIR / "shrinkage_intensity.txt").write_text(f"{intensity:.6f}\n")
    (D.PROCESSED_DIR / "universe.txt").write_text("\n".join(universe) + "\n")

    # 6. Console summary ----------------------------------------------------
    n_ok = len(report["succeeded"]) + len(report["skipped"])
    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    print(f"Tickers requested      : {len(tickers)}")
    print(f"Downloaded fresh       : {len(report['succeeded'])}")
    print(f"Reused cached          : {len(report['skipped'])}")
    print(f"Failed                 : {len(report['failed'])}")
    if report["failed"]:
        for sym, reason in report["failed"].items():
            print(f"    - {sym}: {reason}")
    print(f"Usable in panel        : {len(universe)} of {len(tickers)}")
    print(f"Date range covered     : {prices.index.min().date()} .. "
          f"{prices.index.max().date()}  ({len(prices)} trading days)")
    print(f"Return observations    : {len(returns)}")
    print(f"Shrinkage intensity    : {intensity:.6f}")
    print(f"Artifacts written to   : {D.PROCESSED_DIR}")

    # Sample-vs-shrinkage covariance comparison (first few off-diagonals).
    print("\nSample vs shrinkage covariance (annualized) — sample entries:")
    k = min(4, len(universe))
    names = universe[:k]
    print("           " + "".join(f"{n[:9]:>11s}" for n in names))
    for i in range(k):
        row = "".join(f"{cov_sample[i, j]:11.5f}" for j in range(k))
        print(f"  {names[i][:9]:9s}{row}")
    print("\n... and the same block under shrinkage:")
    print("           " + "".join(f"{n[:9]:>11s}" for n in names))
    for i in range(k):
        row = "".join(f"{cov_shrink[i, j]:11.5f}" for j in range(k))
        print(f"  {names[i][:9]:9s}{row}")

    # How much did shrinkage pull off-diagonals toward zero / diag toward mean?
    off = ~np.eye(len(universe), dtype=bool)
    shrink_ratio = (np.abs(cov_shrink[off]).mean() /
                    np.abs(cov_sample[off]).mean())
    print(f"\nMean |off-diagonal| shrinkage: cov_shrink/cov_sample = "
          f"{shrink_ratio:.3f} (intensity={intensity:.3f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
