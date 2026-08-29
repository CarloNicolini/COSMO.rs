"""Generic COSMO vs Clarabel timing helper (Rust library comparison is in cargo tests)."""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse


def random_qp(n=20, m=30, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    P = sparse.triu(sparse.csc_matrix(B.T @ B)).tocsc()
    q = rng.normal(size=n)
    A = sparse.csc_matrix(rng.normal(size=(m, n)))
    x = rng.random(n)
    s = rng.random(m)
    b = A @ x + s
    return P, q, A, b


def main():
    from cosmo_rs import CosmoSolver

    P, q, A, b = random_qp()
    t0 = time.perf_counter()
    sol = CosmoSolver(P, q, A, b, [("nonnegative", b.size)], verbose=False).solve()
    dt = time.perf_counter() - t0
    print(f"status={sol.status} iter={sol.iter} obj={sol.obj_val:.6g} time={dt:.4f}s")
    print(f"setup={sol.setup_time:.4f} factor={sol.factor_time:.4f} iter={sol.iter_time:.4f}")


if __name__ == "__main__":
    main()
