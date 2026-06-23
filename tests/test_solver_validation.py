"""Validation of the from-scratch active-set solver against cvxpy + OSQP.

This is the project's strongest correctness evidence: the same portfolio QP is
solved by

  (a) ``portfolio_opt.solve_portfolio_qp``  (our hand-rolled active-set method)
  (b) ``cvxpy`` with the OSQP backend       (an independent, widely-used solver)

and the resulting weight vectors / objective values are compared. The n=2 case
is additionally checked against the closed-form analytical KKT solution for the
global minimum-variance portfolio.

Run with ``pytest -s -v tests/test_solver_validation.py`` to see the numeric
side-by-side tables.
"""

import numpy as np
import pytest

from portfolio_opt.solver import solve_portfolio_qp, solve_qp

cp = pytest.importorskip("cvxpy")

WEIGHT_TOL = 1e-4       # max |w_mine - w_cvxpy| must be below this
OBJ_RTOL = 1e-4         # looser relative tolerance on objective value

# n -> per-asset max weight (chosen >= 1/n so equal weights stay feasible,
# small enough that the cap actually binds in several cases).
CASES = {2: 0.7, 5: 0.4, 10: 0.25, 25: 0.10}


def _make_market(n, seed):
    """Seeded PSD covariance and expected-return vector."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 1e-2 * np.eye(n)   # symmetric positive definite
    mu = 0.08 + 0.05 * rng.standard_normal(n)
    return mu, Sigma


def _cvxpy_min_variance(mu, Sigma, target, max_weight):
    """Reference: min xᵀΣx s.t. 1ᵀx=1, μᵀx=target, 0<=x<=max_weight (OSQP)."""
    n = len(mu)
    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma)))
    constraints = [cp.sum(x) == 1, mu @ x == target, x >= 0, x <= max_weight]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9,
               max_iter=200000, polishing=True, verbose=False)
    return np.asarray(x.value).ravel(), float(x.value @ Sigma @ x.value)


def _print_table(n, w_mine, w_ref, label):
    dev = np.abs(w_mine - w_ref)
    print(f"\n  [{label}] n={n}  weights (mine | reference | abs.diff):")
    for i in range(n):
        print(f"    x[{i:2d}]  {w_mine[i]: .8f}   {w_ref[i]: .8f}   {dev[i]:.2e}")
    print(f"    max|dw| = {dev.max():.3e}")
    return dev.max()


@pytest.mark.parametrize("n", list(CASES.keys()))
def test_portfolio_matches_cvxpy_osqp(n):
    """solve_portfolio_qp (target-return mode) vs cvxpy/OSQP for n=2,5,10,25."""
    seed = 100 + n
    mu, Sigma = _make_market(n, seed)
    max_weight = CASES[n]
    target = float(mu @ (np.ones(n) / n))   # equal-weight return: feasible

    res = solve_portfolio_qp(mu, Sigma, target_return=target,
                             max_weight=max_weight)
    assert res["converged"], f"our solver failed to converge for n={n}"
    w_mine = res["weights"]
    obj_mine = float(w_mine @ Sigma @ w_mine)

    w_ref, obj_ref = _cvxpy_min_variance(mu, Sigma, target, max_weight)

    max_dw = _print_table(n, w_mine, w_ref, "OSQP")
    print(f"    objective: mine={obj_mine:.10f}  ref={obj_ref:.10f}  "
          f"rel.diff={abs(obj_mine - obj_ref) / abs(obj_ref):.2e}")

    # Constraints are exactly honoured by our solver.
    assert abs(w_mine.sum() - 1.0) < 1e-7
    assert abs(mu @ w_mine - target) < 1e-6
    assert np.all(w_mine >= -1e-7)
    assert np.all(w_mine <= max_weight + 1e-7)

    # Weight vectors agree with the independent solver.
    assert max_dw < WEIGHT_TOL, f"weights differ by {max_dw:.3e} for n={n}"
    assert np.isclose(obj_mine, obj_ref, rtol=OBJ_RTOL, atol=1e-10)


def test_n2_closed_form_global_min_variance():
    """n=2 global min-variance vs closed-form KKT and vs cvxpy/OSQP.

    Global minimum-variance portfolio:  min xᵀΣx  s.t. 1ᵀx = 1.
    KKT closed form (long-only inactive):  x* = Σ⁻¹1 / (1ᵀΣ⁻¹1).
    """
    # A covariance whose GMV solution is strictly interior (both weights in
    # (0, 1)), so the long-only bounds are inactive and the analytic formula
    # is the true optimum.
    Sigma = np.array([[0.04, 0.006],
                      [0.006, 0.09]])
    mu = np.array([0.10, 0.14])  # unused by GMV objective; needed by builder

    # --- closed-form analytic KKT solution ---
    inv1 = np.linalg.solve(Sigma, np.ones(2))
    w_analytic = inv1 / inv1.sum()

    # --- our solver: pure min-variance s.t. sum=1, long-only (no return eq) ---
    res = solve_qp(P=2 * Sigma, q=np.zeros(2),
                   A_eq=np.ones((1, 2)), b_eq=np.array([1.0]),
                   A_ineq=np.vstack([-np.eye(2), np.eye(2)]),
                   b_ineq=np.array([0.0, 0.0, 1.0, 1.0]))
    assert res["converged"]
    w_mine = res["x"]

    # --- cvxpy / OSQP reference ---
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma))),
                      [cp.sum(x) == 1, x >= 0, x <= 1])
    prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10,
               max_iter=200000, polishing=True)
    w_osqp = np.asarray(x.value).ravel()

    print("\n  [n=2 GMV] weights (analytic | mine | OSQP):")
    for i in range(2):
        print(f"    x[{i}]  {w_analytic[i]: .10f}   {w_mine[i]: .10f}   "
              f"{w_osqp[i]: .10f}")
    print(f"    max|mine-analytic| = {np.abs(w_mine - w_analytic).max():.3e}")
    print(f"    max|mine-OSQP|     = {np.abs(w_mine - w_osqp).max():.3e}")

    assert np.all((w_analytic > 0) & (w_analytic < 1)), "GMV not interior"
    np.testing.assert_allclose(w_mine, w_analytic, atol=1e-8)
    np.testing.assert_allclose(w_mine, w_osqp, atol=WEIGHT_TOL)
