"""Tests for the active-set QP solver.

Correctness is cross-checked against ``scipy.optimize.minimize`` (SLSQP), which
is an *independent* implementation -- our solver does not use scipy.
"""

import numpy as np
import pytest

from portfolio_opt.solver import solve_qp, solve_portfolio_qp

scipy_opt = pytest.importorskip("scipy.optimize")


def _scipy_qp(P, q, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None, n=None):
    """Reference solution via SLSQP for cross-checking."""
    P = np.asarray(P, float)
    q = np.asarray(q, float)
    n = P.shape[0]

    def fun(x):
        return 0.5 * x @ P @ x + q @ x

    def jac(x):
        return P @ x + q

    cons = []
    if A_eq is not None:
        A_eq = np.atleast_2d(A_eq)
        b_eq = np.atleast_1d(b_eq)
        cons.append({"type": "eq", "fun": lambda x, A=A_eq, b=b_eq: A @ x - b})
    if A_ineq is not None:
        A_ineq = np.atleast_2d(A_ineq)
        b_ineq = np.atleast_1d(b_ineq)
        # scipy ineq constraints are g(x) >= 0  ->  b - A x >= 0
        cons.append({"type": "ineq", "fun": lambda x, A=A_ineq, b=b_ineq: b - A @ x})

    res = scipy_opt.minimize(fun, np.zeros(n), jac=jac, method="SLSQP",
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def test_unconstrained_minimum():
    P = np.array([[2.0, 0.0], [0.0, 4.0]])
    q = np.array([-2.0, -8.0])
    res = solve_qp(P, q)
    assert res["converged"]
    # minimizer of 1/2 xPx + qx is x = -P^{-1} q = [1, 2]
    np.testing.assert_allclose(res["x"], [1.0, 2.0], atol=1e-7)


def test_equality_constrained():
    P = np.eye(3)
    q = np.zeros(3)
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([3.0])
    res = solve_qp(P, q, A_eq=A_eq, b_eq=b_eq)
    assert res["converged"]
    np.testing.assert_allclose(res["x"], [1.0, 1.0, 1.0], atol=1e-7)


def test_inequality_active_at_bound():
    # min 1/2(x1^2 + x2^2) s.t. x1 + x2 >= 1 -> x = [0.5, 0.5]
    # write as -x1 - x2 <= -1
    P = np.eye(2)
    q = np.zeros(2)
    A_ineq = np.array([[-1.0, -1.0]])
    b_ineq = np.array([-1.0])
    res = solve_qp(P, q, A_ineq=A_ineq, b_ineq=b_ineq)
    assert res["converged"]
    np.testing.assert_allclose(res["x"], [0.5, 0.5], atol=1e-7)
    assert res["active_set"] == [0]


def test_box_constraint_clipping_is_wrong_way():
    # Unconstrained min is at [1, 2]; box x <= [0.5, 0.5] forces a *joint*
    # re-solve, not independent clipping. Both bounds bind here.
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])  # unconstrained min [1, 2]
    A_ineq = np.eye(2)
    b_ineq = np.array([0.5, 0.5])
    res = solve_qp(P, q, A_ineq=A_ineq, b_ineq=b_ineq)
    ref = _scipy_qp(P, q, A_ineq=A_ineq, b_ineq=b_ineq)
    assert res["converged"]
    np.testing.assert_allclose(res["x"], ref, atol=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_random_qps_match_scipy(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(2, 6)
    M = rng.standard_normal((n, n))
    P = M @ M.T + 0.5 * np.eye(n)      # SPD
    q = rng.standard_normal(n)

    # one equality, a few inequalities
    A_eq = rng.standard_normal((1, n))
    b_eq = rng.standard_normal(1)
    A_ineq = rng.standard_normal((n, n))
    b_ineq = np.abs(rng.standard_normal(n)) + 1.0

    res = solve_qp(P, q, A_eq, b_eq, A_ineq, b_ineq)
    ref = _scipy_qp(P, q, A_eq, b_eq, A_ineq, b_ineq)
    assert res["converged"]
    # Compare objective values (robust to non-unique x); they should match.
    f = lambda x: 0.5 * x @ P @ x + q @ x
    assert f(res["x"]) <= f(ref) + 1e-5
    np.testing.assert_allclose(res["x"], ref, atol=1e-5)


def test_infeasible_returns_gracefully():
    # x <= 0 and x >= 1 simultaneously -> infeasible.
    P = np.eye(1)
    q = np.zeros(1)
    A_ineq = np.array([[1.0], [-1.0]])
    b_ineq = np.array([0.0, -1.0])  # x <= 0 and -x <= -1 (x >= 1)
    res = solve_qp(P, q, A_ineq=A_ineq, b_ineq=b_ineq)
    assert not res["converged"]
    assert res["status"] == "infeasible"


def test_provided_x0_is_respected():
    P = np.eye(2)
    q = np.zeros(2)
    A_eq = np.array([[1.0, 1.0]])
    b_eq = np.array([2.0])
    res = solve_qp(P, q, A_eq=A_eq, b_eq=b_eq, x0=np.array([1.0, 1.0]))
    assert res["converged"]
    np.testing.assert_allclose(res["x"], [1.0, 1.0], atol=1e-7)


# --------------------------- portfolio model ---------------------------------

def _toy_market(seed=0):
    rng = np.random.default_rng(seed)
    n = 5
    mu = 0.05 + 0.1 * rng.random(n)
    B = rng.standard_normal((n, n))
    Sigma = B @ B.T / n + 0.01 * np.eye(n)
    return mu, Sigma


def test_portfolio_target_return_budget_and_bounds():
    mu, Sigma = _toy_market()
    target = float(np.mean(mu))
    res = solve_portfolio_qp(mu, Sigma, target_return=target, max_weight=0.5)
    assert res["converged"]
    x = res["weights"]
    assert np.isclose(x.sum(), 1.0, atol=1e-7)         # budget
    assert np.all(x >= -1e-7)                           # long-only
    assert np.all(x <= 0.5 + 1e-7)                      # max weight
    assert np.isclose(mu @ x, target, atol=1e-6)        # return target


def test_portfolio_risk_aversion_mode():
    mu, Sigma = _toy_market(seed=1)
    res = solve_portfolio_qp(mu, Sigma, risk_aversion=5.0)
    assert res["converged"]
    x = res["weights"]
    assert np.isclose(x.sum(), 1.0, atol=1e-7)
    assert np.all(x >= -1e-7)


def test_portfolio_sector_caps():
    mu, Sigma = _toy_market(seed=2)
    sectors = ["A", "A", "B", "B", "B"]
    caps = {"A": 0.3, "B": 0.8}
    res = solve_portfolio_qp(mu, Sigma, risk_aversion=2.0,
                             sector_map=sectors, sector_caps=caps)
    assert res["converged"]
    x = res["weights"]
    assert x[0] + x[1] <= 0.3 + 1e-6
    assert x[2] + x[3] + x[4] <= 0.8 + 1e-6


def test_portfolio_requires_exactly_one_mode():
    mu, Sigma = _toy_market()
    with pytest.raises(ValueError):
        solve_portfolio_qp(mu, Sigma)  # neither
    with pytest.raises(ValueError):
        solve_portfolio_qp(mu, Sigma, target_return=0.1, risk_aversion=1.0)  # both
