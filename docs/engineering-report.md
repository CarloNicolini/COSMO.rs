# Engineering report

## A. Feasibility

A native Rust COSMO solver is feasible. COSMO's per-iteration work is a quasi-definite linear solve plus cone projection. Clarabel.rs already ships a production QDLDL and CSC type; those are reused. The ADMM state machine is independent.

## B. Clarabel.rs reuse

- `CscMatrix` (public fields, `to_triu`, `new_from_triplets`, `MatrixMath` / `MatrixMathMut`)
- `QDLDLFactorisation` (`new`, `solve`, `update_values`, `refactor`, `positive_inertia`)
- Clarabel as an **oracle** in the test suite (`DefaultSolver`)

Not reused: Clarabel cones, IPM residuals, homogeneous embedding, chordal PSD, Python module.

## C. Independently implemented

ADMM loop, x-update RHS, z-projection, w over-relaxation, ρ vector + equality/inequality scaling, Ruiz scaling, Anderson Type-II, Banjac infeasibility, COSMO termination, cone projections (zero, NN, SOC, exp, power, box), Python/CVXPY mapping.

## D. Multiproblem suite

`cargo test` includes:

- Hand LP / QP / SOCP / exp / power examples with known solutions
- Random feasible LPs and QPs (fixed seeds), box QPs, least-squares-with-bounds
- Equality QPs, SOCPs, mixed zero+NN+SOC, poorly scaled data
- Tiny through medium sizes (LP up to 80×40)
- Solver reuse, warm start, `update_q` vs a fresh object
- Cone projection unit tests (idempotence, Moreau, NaN, in-cone after project)
- `tests/clarabel_benchmark.rs`: 172 instances vs Clarabel.rs (see `docs/benchmarks.md`)

Each solved instance is compared with Clarabel.rs when Clarabel reports `Solved`.

## E–F. CVXPY

`COSMO_RUST` is a `ConicSolver` subclass. `register()` installs the name in CVXPY's conic solver map. Duals use `y = −μ`. SDP is rejected.

## G. Comparison vs other solvers

| Solver | Role |
|---|---|
| Clarabel.rs | primary reference (same crate dependency) |
| OSQP / HiGHS / SCS | not required to compile tests; intended for later Python benchmarks |

On generic small QPs, Clarabel typically uses fewer iterations (Newton vs ADMM). COSMO's advantage, when it appears, is cheap iterations and factorisation reuse under `update_q`/`update_b`.

## H. Problem classes that fit COSMO

- Repeated QPs/SOCPs with fixed sparsity and changing `q`/`b`
- Medium sparse LPs/QPs where a single LDL plus many cheap ADMM steps is acceptable
- Problems whose cones are cheap to project (NN, SOC)

Exp/power projections are relatively expensive per iteration.

## I. Failure modes

- Slow convergence / `Max_iter_reached` at tight tolerances
- Missed infeasibility
- Exp/power projection stalling on pathological points
- Nonconvex P (QDLDL inertia ≠ n) → `KktError::NotConvex`

## J. Persistence speedup

Qualitative (measured in solver timings fields):

- Warm solution: fewer ADMM iterations on nearby problems
- Full ADMM state: same plus ρ / w history
- Factorisation reuse: `update_q`/`update_b` skip AMD + numeric factor
- Scaling reuse: after the first `solve`, Ruiz is not recomputed

Quantitative tables belong in `benchmarks/results/` after `cargo test` and optional Python benches.

## K. CVXPY readiness

Suitable for broader testing on LP/QP/SOCP/exp/power problems. Not a drop-in for SDP or production IPM accuracy.

## L. Before skfolio / portfolio work

1. Keep the multiproblem suite green.
2. Add Python/CVXPY tests in a full environment with CVXPY, OSQP, SCS, Clarabel.
3. Document unmatched statuses instead of loosening tolerances silently.
4. Only then evaluate portfolio sequences.

## Tests run (this revision)

See the summary after `cargo test` in the PR description. The target is the full library test set plus the Clarabel comparison suite.
