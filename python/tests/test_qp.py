"""Python tests for COSMO.rs. Run after `maturin develop --features python`."""

import numpy as np
import pytest
from scipy import sparse


def test_import():
    import cosmo_rs
    from cosmo_rs import CosmoSolver, solve


def test_qp_solve():
    from cosmo_rs import CosmoSolver

    P = sparse.triu(sparse.csc_matrix([[4.0, 1.0], [1.0, 2.0]])).tocsc()
    q = np.array([1.0, 1.0])
    A = sparse.csc_matrix(
        [
            [1.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    b = np.array([-1.0, 0.0, 0.0, 1.0, 0.7, 0.7])
    solver = CosmoSolver(P, q, A, b, [("nonnegative", 6)], verbose=False, eps_abs=1e-6)
    sol = solver.solve()
    assert sol.status == "Solved"
    np.testing.assert_allclose(sol.x, [0.3, 0.7], atol=1e-3)


def test_cvxpy_registration():
    pytest.importorskip("cvxpy")
    import cvxpy as cp
    from cosmo_rs.cvxpy_interface import COSMO_RUST, register

    register()
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0, cp.sum(x) == 1])
    prob.solve(solver="COSMO_RUST")
    assert prob.status in ("optimal", "optimal_inaccurate")
    np.testing.assert_allclose(x.value, [0.5, 0.5], atol=2e-2)

    x2 = cp.Variable(2)
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(x2)), [x2 >= 0, cp.sum(x2) == 1])
    prob2.solve(solver=COSMO_RUST())
    np.testing.assert_allclose(x2.value, [0.5, 0.5], atol=2e-2)
