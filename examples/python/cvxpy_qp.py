"""CVXPY QP solved with COSMO_RUST (matches examples/rust/example_qp.rs)."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from cosmo_rs.cvxpy_interface import register

register()

# min  ½ xᵀ P x + qᵀ x
# s.t. x₁ − 2 x₂ = 0
#      −1 ≤ x ≤ 1
P = np.array([[6.0, 0.0], [0.0, 4.0]])
q = np.array([-1.0, -4.0])

x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
    [x[0] - 2 * x[1] == 0, x >= -1, x <= 1],
)
prob.solve(solver="COSMO_RUST", verbose=True)
print(f"status = {prob.status}")
print(f"x      = {x.value}")
print(f"obj    = {prob.value}")
