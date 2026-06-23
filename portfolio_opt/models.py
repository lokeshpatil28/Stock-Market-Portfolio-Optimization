"""Portfolio construction strategies.

Each strategy returns a dict with a common schema (see :func:`_metrics`):
``weights``, ``weights_pct``, ``expected_return``, ``volatility``,
``sharpe_ratio``, ``risk_contribution_pct`` (+ ``variance``, ``converged``,
``method``). The ``risk_contribution_pct`` field is computed for *every* model
(not just risk parity) so the dashboard's risk-contribution chart works
uniformly.

Which solver is used where
--------------------------
* Pure convex QPs solved by our hand-rolled active-set
  :func:`portfolio_opt.solver.solve_qp` / :func:`solve_portfolio_qp`:
  :func:`min_variance_target_return`, :func:`mean_variance_utility` (base,
  turnover via L1-as-QP reformulation), :func:`transaction_cost_aware`,
  and the sweep inside :func:`max_sharpe`.
* Solved by ``scipy.optimize.minimize`` (SLSQP) because the problem is **not**
  a linear-constraint QP: :func:`mean_variance_utility` with a
  ``tracking_error_limit`` (a *quadratic* constraint) and :func:`risk_parity`
  (a nonconvex equal-risk-contribution objective).

Note on signatures: several functions add ``sector_map=``, ``tickers=`` and
``risk_free_rate=`` keyword arguments beyond the originally requested set --
``sector_caps`` cannot be applied without knowing each asset's sector
(``sector_map``), Sharpe needs the risk-free rate, and ``weights_pct`` needs
ticker labels.
"""

from __future__ import annotations

import numpy as np

from .solver import solve_qp, solve_portfolio_qp

__all__ = [
    "min_variance_target_return",
    "mean_variance_utility",
    "max_sharpe",
    "transaction_cost_aware",
    "risk_parity",
    "build_inequality_constraints",
    "compare_all_models",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _default_tickers(n, tickers):
    if tickers is None:
        return [f"A{i}" for i in range(n)]
    return list(tickers)


def build_inequality_constraints(n, max_weight=1.0, sector_map=None,
                                 sector_caps=None):
    """Assemble ``A_ineq, b_ineq`` for the standard portfolio constraints.

    Always includes long-only (``-x <= 0``) and per-asset cap (``x <= max_weight``).
    Optionally appends one row per sector cap: ``sum_{i in sector} x_i <= cap``.
    Reused by every model that builds a raw QP for :func:`solve_qp`.

    Parameters
    ----------
    n : int
        Number of assets.
    max_weight : float or (n,) array_like
        Per-asset upper bound.
    sector_map : sequence of length n, optional
        Sector label for each asset (aligned to ``mu``/``Sigma`` order).
    sector_caps : dict, optional
        ``{sector_label: cap}``. Requires ``sector_map``.
    """
    mw = np.broadcast_to(np.asarray(max_weight, dtype=float), (n,)).copy()
    rows = [-np.eye(n), np.eye(n)]
    b = [np.zeros(n), mw]

    if sector_caps:
        if sector_map is None:
            raise ValueError("sector_caps requires sector_map")
        sector_map = list(sector_map)
        if len(sector_map) != n:
            raise ValueError("sector_map must have length n")
        for sector, cap in sector_caps.items():
            mask = np.array([1.0 if s == sector else 0.0 for s in sector_map])
            if mask.sum() == 0:
                continue
            rows.append(mask.reshape(1, n))
            b.append(np.array([float(cap)]))

    return np.vstack(rows), np.concatenate(b)


def _metrics(x, mu, Sigma, tickers, risk_free_rate, method, converged):
    """Build the common result dict from a weight vector.

    risk_contribution_pct[i] = x_i * (Sigma x)_i / (xᵀ Σ x) * 100  (sums to 100).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    tickers = _default_tickers(n, tickers)

    variance = float(x @ Sigma @ x)
    volatility = float(np.sqrt(max(variance, 0.0)))
    expected_return = float(mu @ x)
    sharpe = ((expected_return - risk_free_rate) / volatility
              if volatility > 1e-12 else float("nan"))

    # Marginal risk contributions: x_i * (Σx)_i, normalized to % of variance.
    if variance > 1e-18:
        rc = x * (Sigma @ x) / variance * 100.0
    else:
        rc = np.zeros(n)

    return {
        "weights": x,
        "weights_pct": {t: round(float(w) * 100.0, 2) for t, w in zip(tickers, x)},
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": float(sharpe),
        "risk_contribution_pct": {t: round(float(r), 2) for t, r in zip(tickers, rc)},
        "risk_contribution": rc,
        "converged": bool(converged),
        "method": method,
    }


# ---------------------------------------------------------------------------
# 1. Minimum-variance for a target return  (pure QP via active-set solver)
# ---------------------------------------------------------------------------
def min_variance_target_return(mu, Sigma, target_return, max_weight=1.0,
                               sector_caps=None, sector_map=None, tickers=None,
                               risk_free_rate=0.06):
    """Minimize portfolio variance subject to a required expected return.

    ``min xᵀΣx  s.t.  μᵀx = target_return, 1ᵀx = 1, 0<=x<=max_weight, sector caps``.
    A clean convex QP -> solved by our active-set solver via
    :func:`solve_portfolio_qp`.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    res = solve_portfolio_qp(mu, Sigma, target_return=target_return,
                             max_weight=max_weight, sector_map=sector_map,
                             sector_caps=sector_caps)
    out = _metrics(res["weights"], mu, Sigma, tickers, risk_free_rate,
                   method="active_set_qp (min-variance, target return)",
                   converged=res["converged"])
    out["solver_status"] = res.get("status")
    return out


# ---------------------------------------------------------------------------
# 2. Mean-variance utility  (QP; turnover via reformulation; TE via SLSQP)
# ---------------------------------------------------------------------------
def mean_variance_utility(mu, Sigma, risk_aversion, max_weight=1.0,
                          sector_caps=None, turnover_limit=None, prev_weights=None,
                          tracking_error_limit=None, benchmark_weights=None,
                          sector_map=None, tickers=None, risk_free_rate=0.06):
    """Maximize ``μᵀx - risk_aversion · xᵀΣx`` under various constraints.

    Equivalent minimization: ``min  risk_aversion·xᵀΣx - μᵀx``, i.e. a QP with
    ``P = 2·risk_aversion·Σ`` and ``q = -μ``.

    Optional extensions
    -------------------
    * ``turnover_limit`` (with ``prev_weights``): bound L1 turnover
      ``Σ_i |x_i - prev_i| <= turnover_limit``. The absolute value is *not*
      linear, so we use the standard **L1-as-QP reformulation**: introduce
      auxiliary variables ``u_i >= |x_i - prev_i|`` and solve over ``z = [x; u]``::

          x_i - prev_i <=  u_i        ( x_i - u_i <=  prev_i )
         -(x_i - prev_i) <= u_i       (-x_i - u_i <= -prev_i )
          Σ_i u_i <= turnover_limit
          u_i >= 0

      Since ``u`` does not enter the objective, any optimal ``u`` satisfies
      ``u_i >= |x_i-prev_i|`` and ``Σu <= L``, which together imply
      ``Σ|x_i-prev_i| <= L`` -- exactly the turnover budget. This stays a linear-
      constraint QP, so our active-set solver handles it. (A tiny ridge on the
      ``u`` block keeps the augmented Hessian well conditioned.)

    * ``tracking_error_limit`` (with ``benchmark_weights``): bound
      ``(x-b)ᵀΣ(x-b) <= limit²``. This is a **quadratic constraint**, which the
      active-set solver (linear constraints only) cannot represent. It is solved
      instead by ``scipy.optimize.minimize`` (SLSQP) -- see the dedicated branch.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    n = len(mu)

    # --- Quadratic-constraint branch: tracking error -> SLSQP --------------
    if tracking_error_limit is not None:
        return _mean_variance_tracking_error(
            mu, Sigma, risk_aversion, tracking_error_limit, benchmark_weights,
            max_weight, sector_caps, sector_map, tickers, risk_free_rate)

    # --- Base QP objective: P = 2λΣ, q = -μ --------------------------------
    P = 2.0 * risk_aversion * Sigma
    q = -mu

    if turnover_limit is not None:
        # ---- L1-as-QP turnover reformulation over z = [x; u] (size 2n) ----
        if prev_weights is None:
            raise ValueError("turnover_limit requires prev_weights")
        prev = np.asarray(prev_weights, float)

        ridge = 1e-8  # keeps the zero-curvature u-block from making KKT singular
        P_aug = np.zeros((2 * n, 2 * n))
        P_aug[:n, :n] = P
        P_aug[n:, n:] = ridge * np.eye(n)
        q_aug = np.concatenate([q, np.zeros(n)])

        A_eq = np.concatenate([np.ones(n), np.zeros(n)]).reshape(1, 2 * n)
        b_eq = np.array([1.0])

        I = np.eye(n)
        Z = np.zeros((n, n))
        A_base, b_base = build_inequality_constraints(n, max_weight, sector_map,
                                                      sector_caps)
        rows = [
            np.hstack([A_base, np.zeros((A_base.shape[0], n))]),  # x-only rows
            np.hstack([I, -I]),     #  x_i - u_i <=  prev_i
            np.hstack([-I, -I]),    # -x_i - u_i <= -prev_i
            np.hstack([np.zeros((1, n)), np.ones((1, n))]),       # Σ u_i <= L
            np.hstack([Z, -I]),     # u_i >= 0
        ]
        bvals = [b_base, prev, -prev, np.array([float(turnover_limit)]),
                 np.zeros(n)]
        A_ineq = np.vstack(rows)
        b_ineq = np.concatenate(bvals)

        res = solve_qp(P_aug, q_aug, A_eq, b_eq, A_ineq, b_ineq)
        x = res["x"][:n]
        method = "active_set_qp (utility, L1 turnover reformulation)"
        out = _metrics(x, mu, Sigma, tickers, risk_free_rate, method,
                       res["converged"])
        out["solver_status"] = res.get("status")
        out["turnover"] = float(np.abs(x - prev).sum())
        return out

    # --- Plain utility QP --------------------------------------------------
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    A_ineq, b_ineq = build_inequality_constraints(n, max_weight, sector_map,
                                                  sector_caps)
    res = solve_qp(P, q, A_eq, b_eq, A_ineq, b_ineq)
    out = _metrics(res["x"], mu, Sigma, tickers, risk_free_rate,
                   method="active_set_qp (mean-variance utility)",
                   converged=res["converged"])
    out["solver_status"] = res.get("status")
    return out


def _mean_variance_tracking_error(mu, Sigma, risk_aversion, te_limit, benchmark,
                                  max_weight, sector_caps, sector_map, tickers,
                                  risk_free_rate):
    """Quadratic constraint - solved via SLSQP since the active-set QP solver
    handles linear constraints only.

    ``max μᵀx - λ xᵀΣx  s.t.  (x-b)ᵀΣ(x-b) <= te_limit², 1ᵀx=1, 0<=x<=mw, caps``.
    """
    from scipy.optimize import minimize

    if benchmark is None:
        raise ValueError("tracking_error_limit requires benchmark_weights")
    benchmark = np.asarray(benchmark, float)
    n = len(mu)

    def neg_util(x):
        return float(risk_aversion * (x @ Sigma @ x) - mu @ x)

    def neg_util_jac(x):
        return 2.0 * risk_aversion * (Sigma @ x) - mu

    constraints = [
        {"type": "eq", "fun": lambda x: x.sum() - 1.0,
         "jac": lambda x: np.ones(n)},
        # Quadratic tracking-error constraint:  te_limit² - (x-b)ᵀΣ(x-b) >= 0.
        {"type": "ineq",
         "fun": lambda x: te_limit ** 2 - (x - benchmark) @ Sigma @ (x - benchmark),
         "jac": lambda x: -2.0 * Sigma @ (x - benchmark)},
    ]
    if sector_caps:
        if sector_map is None:
            raise ValueError("sector_caps requires sector_map")
        for sector, cap in sector_caps.items():
            mask = np.array([1.0 if s == sector else 0.0 for s in sector_map])
            constraints.append({"type": "ineq",
                                "fun": lambda x, m=mask, c=cap: c - m @ x,
                                "jac": lambda x, m=mask: -m})

    mw = np.broadcast_to(np.asarray(max_weight, float), (n,))
    bounds = [(0.0, float(mw[i])) for i in range(n)]
    x0 = benchmark if np.isclose(benchmark.sum(), 1.0) else np.full(n, 1.0 / n)

    res = minimize(neg_util, x0, jac=neg_util_jac, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-12})
    out = _metrics(res.x, mu, Sigma, tickers, risk_free_rate,
                   method="SLSQP (utility + quadratic tracking-error constraint)",
                   converged=bool(res.success))
    out["solver_status"] = res.message
    out["tracking_error"] = float(np.sqrt(max(
        (res.x - benchmark) @ Sigma @ (res.x - benchmark), 0.0)))
    return out


# ---------------------------------------------------------------------------
# 3. Maximum Sharpe  (frontier sweep of QPs)
# ---------------------------------------------------------------------------
def max_sharpe(mu, Sigma, risk_free_rate=0.06, max_weight=1.0, sector_caps=None,
               sector_map=None, tickers=None, n_grid=60):
    """Approximate the maximum-Sharpe (tangency) portfolio by a frontier sweep.

    The Sharpe ratio ``(μᵀx - r_f) / sqrt(xᵀΣx)`` is a *ratio* of a linear term
    to the square root of a quadratic -- a linear-fractional, non-quadratic
    objective. It is therefore not a QP in the form our solver accepts (and is
    only quasi-concave). However, the tangency portfolio lies on the efficient
    frontier, which we *can* trace as a family of QPs by sweeping the
    risk-aversion λ in :func:`mean_variance_utility`. We evaluate the Sharpe
    ratio at each frontier point and return the best.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)

    lambdas = np.logspace(-1, 2.5, n_grid)  # ~0.1 .. ~316, matches our scale
    best = None
    for lam in lambdas:
        r = mean_variance_utility(mu, Sigma, risk_aversion=lam,
                                  max_weight=max_weight, sector_caps=sector_caps,
                                  sector_map=sector_map, tickers=tickers,
                                  risk_free_rate=risk_free_rate)
        if not r["converged"] or not np.isfinite(r["sharpe_ratio"]):
            continue
        if best is None or r["sharpe_ratio"] > best["sharpe_ratio"]:
            best = r
            best["risk_aversion"] = float(lam)

    if best is None:  # nothing converged -> report gracefully
        n = len(mu)
        out = _metrics(np.full(n, 1.0 / n), mu, Sigma, tickers, risk_free_rate,
                       method="max_sharpe (sweep failed)", converged=False)
        return out
    best["method"] = "max_sharpe (frontier sweep of utility QPs)"
    return best


# ---------------------------------------------------------------------------
# 4. Transaction-cost-aware  (pure QP via active-set solver)
# ---------------------------------------------------------------------------
def transaction_cost_aware(mu, Sigma, prev_weights, risk_aversion, cost_penalty,
                           max_weight=1.0, sector_caps=None, sector_map=None,
                           tickers=None, risk_free_rate=0.06):
    """Utility with a quadratic transaction-cost penalty (stays a QP).

    Objective::

        max  μᵀx - risk_aversion·xᵀΣx - cost_penalty·(x - prev)ᵀ(x - prev)

    Expanding the penalty ``c·(x-p)ᵀ(x-p) = c·xᵀx - 2c·pᵀx + c·pᵀp`` and dropping
    the constant, the equivalent minimization is a QP with

        P = 2·risk_aversion·Σ + 2·cost_penalty·I
        q = -μ - 2·cost_penalty·prev

    so ``(1/2)xᵀP x = risk_aversion·xᵀΣx + cost_penalty·xᵀx`` and
    ``qᵀx = -μᵀx - 2·cost_penalty·prevᵀx``, matching the objective exactly.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    prev = np.asarray(prev_weights, float)
    n = len(mu)

    P = 2.0 * risk_aversion * Sigma + 2.0 * cost_penalty * np.eye(n)
    q = -mu - 2.0 * cost_penalty * prev

    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    A_ineq, b_ineq = build_inequality_constraints(n, max_weight, sector_map,
                                                  sector_caps)

    res = solve_qp(P, q, A_eq, b_eq, A_ineq, b_ineq)
    out = _metrics(res["x"], mu, Sigma, tickers, risk_free_rate,
                   method="active_set_qp (utility + quadratic transaction cost)",
                   converged=res["converged"])
    out["solver_status"] = res.get("status")
    out["turnover"] = float(np.abs(res["x"] - prev).sum())
    return out


# ---------------------------------------------------------------------------
# 5. Risk parity  (nonconvex -> SLSQP)
# ---------------------------------------------------------------------------
def risk_parity(mu, Sigma, max_weight=1.0, tickers=None, risk_free_rate=0.06):
    """Equal-risk-contribution portfolio (not a QP).

    Each asset's risk contribution ``RC_i = x_i·(Σx)_i`` is *quadratic* in x;
    equalizing all RC_i is a system of nonlinear equations and the natural
    objective ``Σ_i (RC_i - mean RC)²`` is quartic and nonconvex. There is no
    convex-QP form, so this is solved by ``scipy.optimize.minimize`` (SLSQP)
    subject to ``1ᵀx = 1`` and ``0 <= x <= max_weight``.
    """
    from scipy.optimize import minimize

    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    n = len(mu)

    def obj(x):
        rc = x * (Sigma @ x)            # absolute risk contributions
        return float(np.sum((rc - rc.mean()) ** 2))

    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                    "jac": lambda x: np.ones(n)}]
    mw = float(np.broadcast_to(np.asarray(max_weight, float), (n,)).max())
    bounds = [(0.0, mw)] * n
    x0 = np.full(n, 1.0 / n)

    res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000, "ftol": 1e-14})
    out = _metrics(res.x, mu, Sigma, tickers, risk_free_rate,
                   method="SLSQP (equal risk contribution)",
                   converged=bool(res.success))
    out["solver_status"] = res.message
    return out


# ---------------------------------------------------------------------------
# 7. Compare all models
# ---------------------------------------------------------------------------
def compare_all_models(mu, Sigma, tickers, target_return=None, risk_aversion=10.0,
                       cost_penalty=1.0, prev_weights=None, risk_free_rate=0.06,
                       max_weight=1.0, sector_map=None, sector_caps=None):
    """Run all five strategies on shared inputs for the dashboard comparison tab.

    Returns
    -------
    dict
        * ``summary`` -- DataFrame, one row per model, columns
          ``expected_return``, ``volatility``, ``sharpe_ratio``, ``converged``.
        * ``weights`` -- wide DataFrame, one row per ticker, one column per
          model, values in weight-% (sums to ~100 per column).
        * ``results`` -- dict of the full per-model result dicts.
    """
    import pandas as pd

    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    tickers = list(tickers)
    n = len(mu)
    if target_return is None:
        target_return = float(mu.mean())          # ~0.18 on our data
    if prev_weights is None:
        prev_weights = np.full(n, 1.0 / n)         # start from equal weights

    common = dict(max_weight=max_weight, sector_map=sector_map,
                  sector_caps=sector_caps, tickers=tickers,
                  risk_free_rate=risk_free_rate)

    results = {
        "min_variance": min_variance_target_return(
            mu, Sigma, target_return=target_return, **common),
        "mean_variance_utility": mean_variance_utility(
            mu, Sigma, risk_aversion=risk_aversion, **common),
        "max_sharpe": max_sharpe(
            mu, Sigma, **common),
        "transaction_cost_aware": transaction_cost_aware(
            mu, Sigma, prev_weights=prev_weights, risk_aversion=risk_aversion,
            cost_penalty=cost_penalty, **common),
        # risk_parity ignores sector_caps/sector_map (not part of its definition).
        "risk_parity": risk_parity(
            mu, Sigma, max_weight=max_weight, tickers=tickers,
            risk_free_rate=risk_free_rate),
    }

    summary = pd.DataFrame({
        name: {
            "expected_return": r["expected_return"],
            "volatility": r["volatility"],
            "sharpe_ratio": r["sharpe_ratio"],
            "converged": r["converged"],
        }
        for name, r in results.items()
    }).T
    summary = summary[["expected_return", "volatility", "sharpe_ratio", "converged"]]

    weights = pd.DataFrame(
        {name: r["weights"] * 100.0 for name, r in results.items()},
        index=tickers,
    ).round(2)

    return {"summary": summary, "weights": weights, "results": results}
