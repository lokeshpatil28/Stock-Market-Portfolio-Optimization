"""portfolio_opt: a from-scratch quadratic-programming portfolio optimizer.

Public API
----------
solve_qp
    General convex QP active-set solver.
solve_portfolio_qp
    Markowitz mean-variance model builder + solver.
"""

from __future__ import annotations

from .solver import solve_qp, solve_portfolio_qp

__version__ = "0.1.0"

__all__ = ["solve_qp", "solve_portfolio_qp", "__version__"]
