# Legacy coursework results (BEFORE artifact)

These files are the **original coursework outputs**, kept only as a "before"
artifact of the rebuild. They are **not** current results and are **not**
produced by any code still in the repo — the scripts that generated them
(`src/qp_solver.py`, `src/frontier.py`, `src/plot_frontier.py`) were removed
during the rebuild.

They are intentionally **not** in `results/`, which now holds only current
output from `scripts/run_backtest.py` (the installable `portfolio_opt` package).

Caveats (why you should not read these as current numbers):

- They were produced by the old clip-and-renormalize "solver", which did **not**
  correctly enforce the long-only / inequality constraints (the bug the rebuild
  fixed with the active-set QP solver).
- They cover the original **10-asset, daily-scale** universe, not the current
  25-asset annualized universe.
- `optimal_weights.csv` and `optimal_portfolio.csv` **disagree with each other**
  (different weights for the same stocks) — a symptom of the old pipeline's
  inconsistency, preserved here as documentation of the "before" state.

| File | What it was |
|------|-------------|
| `optimal_weights.csv` / `.txt` | old per-stock weights (one of two disagreeing versions) |
| `optimal_portfolio.csv` | old per-stock weights + percentages (the other version) |
| `summary.csv` | old portfolio return/variance/volatility (daily scale) |
| `frontier.csv` | old efficient-frontier sweep (daily scale) |
| `frontier.png` | old efficient-frontier plot |

See the corresponding legacy inputs under `data/` (`mu.npy`, `cov.npy`,
`combined_prices.csv`), likewise kept untouched as a before/after artifact.
