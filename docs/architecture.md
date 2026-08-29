# Architecture report (Phase 0)

## Question

Is a native Rust COSMO implementation technically feasible using Clarabel.rs
infrastructure, without Julia, while preserving COSMO's ADMM mathematics?

**Yes.** COSMO and Clarabel share a problem statement and sparse numerical
types, but they are **not** algorithmically equivalent. COSMO is first-order
ADMM / operator splitting. Clarabel is a homogeneous interior-point method.

## COSMO.jl → Rust mapping

| COSMO.jl | Reuse from Clarabel.rs | New Rust component |
|---|---|---|
| `solver.jl` ADMM loop (`admm_x!`, `admm_z!`, `admm_w!`) | no (IPM state machine) | `solver/mod.rs` |
| KKT ` [P+σI, A'; A, −I/ρ] ` assembly | CSC storage | `linsys/mod.rs` |
| `QdldlKKTSolver` / `update_rho!` | `clarabel::qdldl::QDLDLFactorisation` (`update_values` + `refactor`) | wrapper |
| CSC `SparseMatrixCSC` | `clarabel::algebra::CscMatrix` | `algebra` gemv/symv (Clarabel's `gemv` is crate-private) |
| Cone **projections** (`project!`) | **no** — Clarabel cones are IPM barriers / NT scalings | `cones/*` |
| Exponential / power projection | not interchangeable with Clarabel barriers | `cones/exp.rs`, `cones/power.rs` (COSMO/SCS/Hien) |
| Ruiz scaling | conceptually similar, different API | `scaling/mod.rs` |
| Adaptive ρ, over-relaxation α | no | `solver/mod.rs` |
| Anderson acceleration | no | `accelerator/mod.rs` (from COSMOAccelerators.jl) |
| Infeasibility (Banjac conditions) | no (Clarabel uses embedding) | `solver/mod.rs` |
| Termination (prim/dual residual ∞-norm) | no | `solver/mod.rs` |
| Chordal PSD decomposition | not in this milestone | omitted (documented) |
| MOI / JuMP | n/a | Python / CVXPY adapter |
| Box cone | n/a | `cones::BoxCone` |

## CVXPY canonical data → Rust COSMO

CVXPY conic solvers receive:

| CVXPY | COSMO.rs |
|---|---|
| `data[s.P]` CSC (often triu) | `CscMatrix`, stored as symmetric triu |
| `data[s.C]` = `q` | `q` |
| `data[s.A]`, `data[s.B]` | `A`, `b` in `Ax + s = b` |
| `dims.zero, nonneg, soc, exp, p3d` | `Cone` list in that order |
| `dims.psd`, `dims.pnd` | rejected with a clear error |
| `OFFSET` | `Solution.obj_offset` |
| dual `y` | COSMO `y = −μ` (same convention as COSMO.jl / CVXPY COSMO) |
| statuses | `Solved→OPTIMAL`, `Primal_infeasible→INFEASIBLE`, `Dual_infeasible→UNBOUNDED`, `Max_iter_reached→USER_LIMIT` |

Cone order matches SCS/Clarabel: Zero, Nonneg, SOC, (PSD omitted), Exp, Power3D.

Exponential cone argument order is `(x, y, z)` (`EXP_CONE_ORDER = [0,1,2]`).

Power cone parameter is `α ∈ (0,1)` as in CVXPY `PowCone3D`.

## What cannot be reused

Clarabel's `Cone` trait (`unit_initialization`, `Nt_scaling`, `Hessian`, etc.) is an interior-point API. Calling it a projection would be incorrect. COSMO needs Euclidean projection / proximal maps of indicator functions.

Clarabel's KKT system is the IPM Newton matrix (including cone scalings that change every iteration). COSMO's KKT matrix is constant except for the `−1/ρ` diagonal when ρ is adapted.

## Module boundaries

```
algebra     Clarabel CscMatrix + COSMO gemv/symv
cones       ADMM projections (independent of Clarabel cones)
linsys      COSMO KKT + QDLDL
scaling     Ruiz equilibration
accelerator  Type-II Anderson
solver      ADMM state machine (persistent)
python      PyO3 (optional feature)
cvxpy       adapter only; no solver logic
```

## Numerical risks

- First-order residuals vs IPM accuracy: COSMO may need more iterations for tight tolerances.
- Exp/power Newton/bisection projections can fail on badly scaled points.
- Adaptive ρ + Anderson can interact; safeguarding is on by default.
- QDLDL regularisation (Clarabel default) is slightly different from Julia QDLDL; inertia is still checked (`positive_inertia == n`).
- Dual recovery uses Moreau: `μ = ρ (w_s − Π(w_s))`, `y = −μ`.

## Performance bottlenecks (expected)

1. Sparse triangular solves per ADMM iteration.
2. Cone projections (especially exp/power).
3. Refactor on ρ updates (symbolic AMD is reused).
4. Python/CVXPY canonicalisation (not solver time).

## Packaging

- Crate `cosmo`, Apache-2.0.
- Python package `cosmo-rs` via maturin / PyO3 (`cosmo_rs._cosmo`).
- CVXPY name `COSMO_RUST` registered by `cosmo_rs.cvxpy_interface.register()`.
- Julia is never a dependency.

## Omissions vs COSMO.jl (explicit)

- No PSD / chordal decomposition.
- No dual exponential/power in the CVXPY path by default (Rust supports them).
- No indirect (CG) KKT solver, no Pardiso.
- No MOI / JuMP.
- Accelerator is Type-II QR + restarted memory only (not Type-I / rolling / Tikhonov).
