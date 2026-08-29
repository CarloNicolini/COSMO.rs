"""CVXPY LP and simplex projection with COSMO_RUST."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from cosmo_rs.cvxpy_interface import COSMO_RUST, register

register()

# Simple LP: min cᵀ x  s.t. A x ≤ b, x ≥ 0
c = np.array([1.0, 2.0])
A = np.array([[1.0, 1.0], [-1.0, 2.0]])
b = np.array([2.0, 2.0])

x = cp.Variable(2, nonneg=True)
lp = cp.Problem(cp.Minimize(c @ x), [A @ x <= b])
lp.solve(solver="COSMO_RUST")
print(f"LP  status={lp.status}  x={x.value}  obj={lp.value}")

# Projection onto the probability simplex
y = cp.Variable(3, nonneg=True)
target = np.array([0.8, 0.5, -0.1])
proj = cp.Problem(cp.Minimize(cp.sum_squares(y - target)), [cp.sum(y) == 1])
proj.solve(solver=COSMO_RUST())
print(f"simplex status={proj.status}  y={y.value}  obj={proj.value}")
