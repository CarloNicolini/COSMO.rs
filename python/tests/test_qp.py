"""Python tests for COSMO.rs. Run after `maturin develop --features python`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


def textbook_qp():
    """Clarabel textbook QP: equality + box bounds."""
    P = sparse.triu(sparse.csc_matrix([[6.0, 0.0], [0.0, 4.0]])).tocsc()
    q = np.array([-1.0, -4.0])
    A = sparse.csc_matrix(
        [
            [1.0, -2.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    b = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    cones = [("zero", 1), ("nonnegative", 4)]
    return P, q, A, b, cones


def box_qp():
    """min ½‖x‖² − 1ᵀx  s.t. 0 ≤ x ≤ 1  → x* = 1."""
    P = sparse.eye(2, format="csc")
    q = np.array([-1.0, -1.0])
    A = sparse.csc_matrix(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    b = np.array([1.0, 1.0, 0.0, 0.0])
    cones = [("nonnegative", 4)]
    return P, q, A, b, cones


def test_import():
    import cosmo_rs
    from cosmo_rs import CosmoSolver, solve

    assert hasattr(CosmoSolver, "update_p")
    assert hasattr(CosmoSolver, "update_a")
    assert hasattr(CosmoSolver, "reset")


def test_qp_solve():
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = textbook_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-6)
    sol = solver.solve()
    assert sol.status == "Solved"
    np.testing.assert_allclose(sol.x, [0.42857142857, 0.21428571428], atol=1e-3)
    assert sol.obj_val == pytest.approx(-0.64285714285, abs=1e-3)


def test_update_p_same_sparsity():
    """Covariance / Hessian change with fixed sparsity → numerical refactor."""
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = box_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-6)
    s1 = solver.solve()
    assert s1.status == "Solved"

    # Stronger diagonal curvature; same CSC pattern.
    P2 = sparse.diags([4.0, 4.0], format="csc")
    solver.update_p(P2)
    solver.reset("factor")
    s2 = solver.solve()
    assert s2.status == "Solved"

    # Fresh solve must match the reused workspace.
    fresh = CosmoSolver(P2, q, A, b, cones, verbose=False, eps_abs=1e-6).solve()
    assert fresh.status == "Solved"
    np.testing.assert_allclose(s2.x, fresh.x, atol=1e-4)
    assert s2.obj_val == pytest.approx(fresh.obj_val, abs=1e-4)


def test_update_p_pattern_change():
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = box_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-6)
    assert solver.solve().status == "Solved"

    # Fill in an off-diagonal — sparsity pattern changes.
    P2 = sparse.triu(sparse.csc_matrix([[2.0, 0.5], [0.5, 2.0]])).tocsc()
    solver.update_p(P2)
    solver.reset("cold")
    s2 = solver.solve()
    assert s2.status == "Solved"
    fresh = CosmoSolver(P2, q, A, b, cones, verbose=False, eps_abs=1e-6).solve()
    np.testing.assert_allclose(s2.x, fresh.x, atol=1e-4)


def test_update_a_drops_factor_and_solves():
    """Scenario-return / constraint matrix change (CVaR-style)."""
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = box_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-6)
    assert solver.solve().status == "Solved"

    # Tighter upper bounds: 0 ≤ x ≤ 0.5
    A2 = A.copy()
    b2 = np.array([0.5, 0.5, 0.0, 0.0])
    solver.update_a(A2)
    solver.update_b(b2)
    solver.reset("cold")
    s2 = solver.solve()
    assert s2.status == "Solved"
    np.testing.assert_allclose(s2.x, [0.5, 0.5], atol=1e-3)

    fresh = CosmoSolver(P, q, A2, b2, cones, verbose=False, eps_abs=1e-6).solve()
    np.testing.assert_allclose(s2.x, fresh.x, atol=1e-4)


def test_reset_cold_vs_factor():
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = textbook_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-6)
    s1 = solver.solve()
    assert s1.status == "Solved"

    solver.reset("factor")
    s_factor = solver.solve()
    assert s_factor.status == "Solved"
    np.testing.assert_allclose(s_factor.x, s1.x, atol=1e-5)
    assert s_factor.obj_val == pytest.approx(s1.obj_val, abs=1e-6)

    solver.reset("cold")
    s_cold = solver.solve()
    assert s_cold.status == "Solved"
    np.testing.assert_allclose(s_cold.x, s1.x, atol=1e-5)

    with pytest.raises(Exception):
        solver.reset("not-a-mode")


def test_walkforward_covariance_then_scenarios():
    """Class-A (update_p) then class-B (update_a) walkforward sequence."""
    from cosmo_rs import CosmoSolver

    P, q, A, b, cones = box_qp()
    solver = CosmoSolver(P, q, A, b, cones, verbose=False, eps_abs=1e-5, max_iter=8000)
    assert solver.solve().status == "Solved"

    # Day t+1: new covariance, same constraints — keep factorisation path.
    P_t1 = sparse.diags([2.0, 3.0], format="csc")
    solver.update_p(P_t1)
    solver.reset("factor")
    s_t1 = solver.solve()
    assert s_t1.status == "Solved"

    # Day t+2: new scenario matrix (bounds) — drop factor + ADMM state.
    A_t2 = A.copy()
    b_t2 = np.array([0.8, 0.8, 0.0, 0.0])
    solver.update_a(A_t2)
    solver.update_b(b_t2)
    solver.reset("cold")
    s_t2 = solver.solve()
    assert s_t2.status == "Solved"
    assert np.all(np.asarray(s_t2.x) <= 0.8 + 1e-3)

    fresh = CosmoSolver(P_t1, q, A_t2, b_t2, cones, verbose=False, eps_abs=1e-5).solve()
    assert fresh.status == "Solved"
    np.testing.assert_allclose(s_t2.x, fresh.x, atol=2e-3)


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
