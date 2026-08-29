"""CVXPY SOCP solved with COSMO_RUST (matches examples/rust/example_socp.rs)."""

from __future__ import annotations

import cvxpy as cp
from cosmo_rs.cvxpy_interface import register

register()

# min  x₂²
# s.t. ||(2 − 2 x₁,  −2 − x₂)||₂ ≤ 1
x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(cp.square(x[1])),
    [cp.norm(cp.hstack([2 - 2 * x[0], -2 - x[1]]), 2) <= 1],
)
prob.solve(solver="COSMO_RUST", verbose=True)
print(f"status = {prob.status}")
print(f"x      = {x.value}")
print(f"obj    = {prob.value}")
