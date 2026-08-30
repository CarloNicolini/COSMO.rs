# Hot-path performance notes

COSMO.rs spends most of its time in (1) the QDLDL solve, (2) cone
projection, and (3) sparse residual matvecs on termination / ρ / infeasibility
checks. This document records low-risk optimisations that preserve the ADMM
mathematics.

## Changes (this milestone)

| Optimisation | Where | Effect |
|---|---|---|
| Fuse residual + relative-norm components | `solver::calculate_residuals_and_cost` | One `Ax` / `Px` / `Aᵀμ` per check instead of two |
| Objective from `xᵀ(Px)` | same | Drop a separate `quad_form` sweep over `P` |
| Skip duplicate residual on same iteration | main ADMM loop | When termination and delayed infeasibility land together |
| KKT RHS written into `sol` | `admm_x` + `solve_inplace` | Drop an `(n+m)` dense copy every ADMM step |
| Buffered `-1/ρ` update | `QdldlKktSolver::update_rho` | No `Vec` alloc on every ρ adapt |
| `gemv_t` `b == 0` fast path | `algebra` | Residual / infeasibility path |
| Triplet `with_capacity` | `assemble_kkt_upper` | Fewer reallocs on factor rebuild |
| Reuse `Dwork` for identity scale | Ruiz rectification | Avoid a one-shot `vec![1; n]` |

## Measured (release, same host)

Harness: `cargo run --release --example bench_hotpath` — 24 random feasible
QPs (40×25, 80×40, 120×60), fixed seeds.

| | sum solve ms | wall ms | total ADMM iters | solved |
|---|---|---|---|---|
| Before | 25.902 | 33.704 | 1564 | 24/24 |
| After | 24.214 | 31.699 | 1564 | 24/24 |
| Ratio | **1.07×** | **1.06×** | identical | — |

Identical iteration counts and a green Clarabel comparison suite
(`172/172`) confirm that the numerical path is unchanged; the gain is
from fewer matvecs / copies per residual evaluation and cheaper ADMM steps.

On small problems the LDL solve still dominates, so absolute speedups are
modest. Residual fusion matters more when `check_termination` is frequent
or when `A`/`P` are denser.
