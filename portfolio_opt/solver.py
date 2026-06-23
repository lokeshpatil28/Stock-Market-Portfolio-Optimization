"""Active-set quadratic programming solver (pure NumPy).

This module implements a *primal active-set method* for convex quadratic
programs of the form::

    minimize    (1/2) xᵀ P x + qᵀ x
    subject to  A_eq   x  = b_eq
                A_ineq x <= b_ineq

with ``P`` symmetric positive semidefinite (so the problem is convex and any
local minimizer is global).

The implementation is deliberately written from scratch -- it does not wrap
``scipy.optimize`` or ``cvxpy`` -- so the KKT bookkeeping is fully visible.
See :func:`solve_qp` for the public entry point and the module-level notes
below for the mathematics.

Karush-Kuhn-Tucker (KKT) conditions
-----------------------------------
At an optimum ``x*`` with active inequality set ``A(x*)`` there exist
multipliers ``ν`` (equalities) and ``λ >= 0`` (inequalities) such that

1. Stationarity:  P x* + q + A_eqᵀ ν + A_ineqᵀ λ = 0
2. Primal feasibility:  A_eq x* = b_eq,  A_ineq x* <= b_ineq
3. Dual feasibility:  λ_i >= 0
4. Complementary slackness:  λ_i (a_iᵀ x* - b_i) = 0

The active-set method maintains primal feasibility at every iterate and works
to satisfy stationarity + dual feasibility by adding/removing inequality
constraints from a *working set* ``W`` (a guess at the active set).
"""

from __future__ import annotations

import numpy as np

__all__ = ["solve_qp", "solve_portfolio_qp"]


def _as_2d(A, n):
    """Coerce a constraint matrix to shape (m, n); ``None`` -> (0, n)."""
    if A is None:
        return np.zeros((0, n), dtype=float)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[1] != n:
        raise ValueError(f"constraint matrix has {A.shape[1]} cols, expected {n}")
    return A


def _as_1d(b, m):
    """Coerce a RHS vector to shape (m,); ``None`` -> length-0 vector."""
    if b is None:
        return np.zeros(0, dtype=float)
    b = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    if b.shape[0] != m:
        raise ValueError(f"rhs has length {b.shape[0]}, expected {m}")
    return b


def _kkt_step(P, C, g):
    """Solve the equality-constrained QP for the Newton step ``p``.

    Solves::

        minimize_p  (1/2) pᵀ P p + gᵀ p   s.t.  C p = 0

    via the KKT linear system

        [ P   Cᵀ ] [ p ]   [ -g ]
        [ C   0  ] [ y ] = [  0 ]

    Here ``g = P x + q`` is the gradient at the current iterate and ``C`` stacks
    the normals of the working-set constraints. Holding ``C p = 0`` keeps every
    working-set constraint satisfied as the iterate moves along ``p``.

    Returns
    -------
    p : ndarray, shape (n,)
        Step direction (the minimizer of the subproblem).
    y : ndarray, shape (m,)
        Lagrange multipliers of the working-set constraints. At ``p = 0`` these
        equal the original-problem multipliers (see module docstring).

    ``np.linalg.lstsq`` is used rather than ``solve`` so that a rank-deficient
    KKT matrix (degenerate / linearly dependent constraints, or ``P`` singular
    on the working-set null space) yields a graceful minimum-norm answer instead
    of raising ``LinAlgError``.
    """
    n = P.shape[0]
    m = C.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = P
    if m > 0:
        K[:n, n:] = C.T
        K[n:, :n] = C
    rhs = np.concatenate([-g, np.zeros(m)])
    sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    return sol[:n], sol[n:]


def _active_set_phase2(P, q, A_eq, b_eq, A_ineq, b_ineq, x, max_iter, tol):
    """Run the active-set iteration from a *feasible* starting point ``x``.

    Assumes ``x`` already satisfies ``A_eq x = b_eq`` and ``A_ineq x <= b_ineq``.
    Equality constraints are permanently in the working set; the working set
    ``W`` additionally tracks indices of inequality constraints currently held
    as equalities.
    """
    n = len(x)
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]

    # Initial working set: inequalities active at the starting point.
    resid = (A_ineq @ x - b_ineq) if m_ineq else np.zeros(0)
    W = [i for i in range(m_ineq) if abs(resid[i]) <= tol]

    for it in range(1, max_iter + 1):
        # C stacks the equality normals (always) and the active inequalities.
        if W:
            C = np.vstack([A_eq, A_ineq[W]]) if m_eq else A_ineq[W]
        else:
            C = A_eq
        g = P @ x + q

        # --- Step 1: minimize the objective over the working-set null space ---
        p, y = _kkt_step(P, C, g)

        if np.linalg.norm(p, np.inf) <= tol:
            # --- Step 2: x minimizes the EQP. Check inequality multipliers. ---
            # Multipliers are ordered [equalities..., working inequalities...].
            lam_ineq = y[m_eq:] if len(W) else np.zeros(0)
            if lam_ineq.size == 0 or np.min(lam_ineq) >= -tol:
                # All λ_i >= 0  ->  KKT satisfied  ->  global optimum.
                return {
                    "x": x,
                    "active_set": sorted(W),
                    "iterations": it,
                    "converged": True,
                    "multipliers_eq": y[:m_eq].copy(),
                    "multipliers_ineq_active": lam_ineq.copy(),
                    "status": "optimal",
                }
            # Some λ_i < 0: dropping the most negative one allows further
            # descent (complementary slackness lets us deactivate it).
            j = int(np.argmin(lam_ineq))
            W.pop(j)
            continue

        # --- Step 3: p != 0. Take the longest feasible step along p. ---
        # For each inactive inequality with a_iᵀ p > 0 the iterate moves toward
        # its boundary; the largest α keeping a_iᵀ(x+αp) <= b_i is
        # α_i = (b_i - a_iᵀ x) / (a_iᵀ p).
        alpha = 1.0
        blocking = -1
        not_in_W = [i for i in range(m_ineq) if i not in W]
        if not_in_W:
            Ap = A_ineq[not_in_W] @ p
            slack = b_ineq[not_in_W] - A_ineq[not_in_W] @ x
            for k, i in enumerate(not_in_W):
                if Ap[k] > tol:
                    ai = slack[k] / Ap[k]
                    if ai < alpha:
                        alpha = ai
                        blocking = i

        x = x + alpha * p
        if blocking >= 0:
            # A new constraint became active; add it to the working set.
            W.append(blocking)

    # Iteration budget exhausted without satisfying the KKT conditions.
    return {
        "x": x,
        "active_set": sorted(W),
        "iterations": max_iter,
        "converged": False,
        "status": "max_iter_reached",
    }


def _find_feasible_point(A_eq, b_eq, A_ineq, b_ineq, n, tol, max_iter):
    """Phase-1: find a point satisfying all constraints, or report infeasible.

    Strategy:

    1. Take the minimum-norm least-squares solution of ``A_eq x = b_eq`` as a
       base point. If the equalities cannot be satisfied, the problem is
       infeasible.
    2. If that base point already satisfies the inequalities, return it.
    3. Otherwise solve an auxiliary *strictly convex* QP in ``(x, t)`` that
       drives the worst inequality violation ``t`` to zero, reusing the Phase-2
       machinery::

           minimize   (1/2) ε‖x - x_base‖² + (1/2) t²
           s.t.       A_eq x = b_eq
                      A_ineq x - t·1 <= b_ineq
                      -t <= 0

       The start ``(x_base, t0)`` with ``t0 = max violation`` is feasible by
       construction, so Phase-2 can run on it. If the optimal ``t`` is ~0 the
       ``x`` part is feasible for the original problem.
    """
    m_eq = A_eq.shape[0]
    m_ineq = A_ineq.shape[0]

    if m_eq:
        x_base, *_ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
        if not np.allclose(A_eq @ x_base, b_eq, atol=1e-7, rtol=0):
            return None  # inconsistent equality constraints -> infeasible
    else:
        x_base = np.zeros(n)

    if m_ineq == 0 or np.all(A_ineq @ x_base - b_ineq <= tol):
        return x_base

    # --- Build the auxiliary QP in variable z = [x; t] (size n + 1). ---
    eps = 1e-6
    P_aux = np.zeros((n + 1, n + 1))
    P_aux[:n, :n] = eps * np.eye(n)
    P_aux[n, n] = 1.0
    q_aux = np.concatenate([-eps * x_base, [0.0]])

    if m_eq:
        A_eq_aux = np.hstack([A_eq, np.zeros((m_eq, 1))])
    else:
        A_eq_aux = np.zeros((0, n + 1))
    b_eq_aux = b_eq

    # A_ineq x - t <= b_ineq, stacked with -t <= 0.
    A_ineq_aux = np.vstack([
        np.hstack([A_ineq, -np.ones((m_ineq, 1))]),
        np.hstack([np.zeros((1, n)), [[-1.0]]]),
    ])
    b_ineq_aux = np.concatenate([b_ineq, [0.0]])

    t0 = max(0.0, float(np.max(A_ineq @ x_base - b_ineq)))
    # Nudge t0 up slightly so the start is strictly inside A_ineq x - t <= b.
    z0 = np.concatenate([x_base, [t0 + 1e-9]])

    res = _active_set_phase2(P_aux, q_aux, A_eq_aux, b_eq_aux,
                             A_ineq_aux, b_ineq_aux, z0,
                             max_iter=max_iter, tol=tol)
    z = res["x"]
    x = z[:n]
    t = z[n]
    if t <= 1e-6 and np.all(A_ineq @ x - b_ineq <= 1e-6):
        return x
    return None  # could not reach a feasible point -> infeasible


def solve_qp(P, q, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None,
             x0=None, max_iter=200, tol=1e-9):
    """Solve a convex quadratic program with an active-set method.

    Problem::

        minimize    (1/2) xᵀ P x + qᵀ x
        subject to  A_eq   x  = b_eq
                    A_ineq x <= b_ineq

    Parameters
    ----------
    P : (n, n) array_like
        Symmetric positive-semidefinite Hessian.
    q : (n,) array_like
        Linear term.
    A_eq, b_eq : array_like or None
        Equality constraints ``A_eq x = b_eq``. ``None`` means no equalities.
    A_ineq, b_ineq : array_like or None
        Inequality constraints ``A_ineq x <= b_ineq``. ``None`` means none.
    x0 : (n,) array_like or None
        Optional feasible starting point. If omitted (or infeasible) a Phase-1
        search constructs one.
    max_iter : int
        Maximum active-set iterations (per phase).
    tol : float
        Tolerance for "zero step" and multiplier-sign tests.

    Returns
    -------
    dict
        Keys:

        * ``x`` -- solution vector (best iterate found).
        * ``active_set`` -- sorted list of inequality-row indices active at the
          solution.
        * ``iterations`` -- Phase-2 iterations used.
        * ``converged`` -- ``True`` iff the KKT conditions were satisfied.
        * ``status`` -- ``"optimal"``, ``"infeasible"``, or
          ``"max_iter_reached"``.
        * ``objective`` -- objective value at ``x`` (NaN if no solution).
        * (when optimal) ``multipliers_eq`` / ``multipliers_ineq_active``.

    Notes
    -----
    The solver never raises on infeasible or non-converging inputs; it returns
    ``converged=False`` with an explanatory ``status`` instead.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P.shape[0]
    # Symmetrize defensively (the math assumes P = Pᵀ).
    P = 0.5 * (P + P.T)
    q = np.asarray(q, dtype=float).ravel() if q is not None else np.zeros(n)
    if q.shape[0] != n:
        raise ValueError(f"q has length {q.shape[0]}, expected {n}")

    A_eq = _as_2d(A_eq, n)
    b_eq = _as_1d(b_eq, A_eq.shape[0])
    A_ineq = _as_2d(A_ineq, n)
    b_ineq = _as_1d(b_ineq, A_ineq.shape[0])

    def _objective(x):
        return float(0.5 * x @ P @ x + q @ x)

    # --- Obtain a feasible starting point ---------------------------------
    x_start = None
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.shape[0] != n:
            raise ValueError(f"x0 has length {x0.shape[0]}, expected {n}")
        eq_ok = (A_eq.shape[0] == 0) or np.allclose(A_eq @ x0, b_eq, atol=1e-7)
        ineq_ok = (A_ineq.shape[0] == 0) or np.all(A_ineq @ x0 - b_ineq <= tol)
        if eq_ok and ineq_ok:
            x_start = x0

    if x_start is None:
        x_start = _find_feasible_point(A_eq, b_eq, A_ineq, b_ineq, n, tol, max_iter)

    if x_start is None:
        return {
            "x": np.full(n, np.nan),
            "active_set": [],
            "iterations": 0,
            "converged": False,
            "status": "infeasible",
            "objective": float("nan"),
        }

    # --- Phase 2: optimize from the feasible point ------------------------
    res = _active_set_phase2(P, q, A_eq, b_eq, A_ineq, b_ineq,
                             x_start.copy(), max_iter=max_iter, tol=tol)
    res["objective"] = _objective(res["x"])
    return res


def solve_portfolio_qp(mu, Sigma, target_return=None, risk_aversion=None,
                       max_weight=1.0, sector_map=None, sector_caps=None,
                       **solver_kwargs):
    """Build and solve a Markowitz mean-variance portfolio QP.

    Two mutually exclusive modes (supply exactly one):

    * ``target_return`` -- **minimum-variance** portfolio for a required return::

          minimize    xᵀ Σ x
          subject to  μᵀ x = target_return
                      1ᵀ x = 1
                      0 <= x_i <= max_weight
                      (optional sector caps)

      Implemented with ``P = 2Σ`` (so ``(1/2)xᵀP x = xᵀΣx``) and ``q = 0``.

    * ``risk_aversion`` (λ) -- **utility maximization**
      ``max μᵀx - λ xᵀΣx``, i.e.::

          minimize    λ xᵀ Σ x - μᵀ x
          subject to  1ᵀ x = 1
                      0 <= x_i <= max_weight
                      (optional sector caps)

      Implemented with ``P = 2λΣ`` and ``q = -μ``.

    Parameters
    ----------
    mu : (n,) array_like
        Expected returns.
    Sigma : (n, n) array_like
        Return covariance matrix (PSD).
    target_return : float, optional
        Required portfolio expected return (min-variance mode).
    risk_aversion : float, optional
        Risk-aversion coefficient λ > 0 (utility mode).
    max_weight : float or (n,) array_like, default 1.0
        Per-asset upper bound ``x_i <= max_weight``.
    sector_map : array_like of length n, optional
        Sector label for each asset (any hashable, e.g. strings). Required to
        apply ``sector_caps``.
    sector_caps : dict, optional
        Mapping ``{sector_label: cap}``; adds ``sum_{i in sector} x_i <= cap``.
    **solver_kwargs
        Forwarded to :func:`solve_qp` (e.g. ``x0``, ``max_iter``, ``tol``).

    Returns
    -------
    dict
        The :func:`solve_qp` result dict augmented with portfolio metrics:
        ``weights``, ``expected_return``, ``variance``, ``volatility``.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("Sigma must be (n, n) matching len(mu)")

    if (target_return is None) == (risk_aversion is None):
        raise ValueError("supply exactly one of target_return or risk_aversion")

    # --- Objective ---------------------------------------------------------
    if target_return is not None:
        P = 2.0 * Sigma
        q = np.zeros(n)
    else:
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be > 0")
        P = 2.0 * risk_aversion * Sigma
        q = -mu

    # --- Equality constraints: budget (+ return target) --------------------
    A_eq_rows = [np.ones(n)]
    b_eq_vals = [1.0]
    if target_return is not None:
        A_eq_rows.append(mu.copy())
        b_eq_vals.append(float(target_return))
    A_eq = np.vstack(A_eq_rows)
    b_eq = np.asarray(b_eq_vals, dtype=float)

    # --- Inequality constraints -------------------------------------------
    mw = np.broadcast_to(np.asarray(max_weight, dtype=float), (n,)).copy()
    A_ineq_rows = [-np.eye(n), np.eye(n)]      # long-only (-x<=0), cap (x<=mw)
    b_ineq_vals = [np.zeros(n), mw]

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
            A_ineq_rows.append(mask.reshape(1, n))
            b_ineq_vals.append(np.array([float(cap)]))

    A_ineq = np.vstack(A_ineq_rows)
    b_ineq = np.concatenate(b_ineq_vals)

    res = solve_qp(P, q, A_eq, b_eq, A_ineq, b_ineq, **solver_kwargs)

    x = res["x"]
    if res["converged"] and np.all(np.isfinite(x)):
        res["weights"] = x
        res["expected_return"] = float(mu @ x)
        res["variance"] = float(x @ Sigma @ x)
        res["volatility"] = float(np.sqrt(max(x @ Sigma @ x, 0.0)))
    else:
        res["weights"] = x
        res["expected_return"] = float("nan")
        res["variance"] = float("nan")
        res["volatility"] = float("nan")
    return res
