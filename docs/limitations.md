# Limitations

This file states what COSMO.rs does **not** claim.

## Algorithm

- COSMO is ADMM, not Clarabel's interior-point method. Iterates, residuals, and runtimes will differ even on identical data.
- Default tolerances (`1e-5`) are COSMO.jl defaults, looser than typical IPM `1e-8`.
- Extreme constraint scaling (e.g. multiplying `A` and `b` by `1e±8`) can make ADMM stall or report a wrong `Solved` while Clarabel.rs still converges — see `tests/stress_vs_clarabel.rs` and [`docs/benchmarks.md`](benchmarks.md).
- Freezing a tiny ρ (`rho=1e-6`, `adaptive_rho=false`) can prevent convergence within a few thousand iterations on otherwise easy QPs.
- First-order methods can require thousands of iterations on badly scaled or tightly constrained problems. Clarabel is often faster and more accurate on small/medium conic problems.

## Cones

- **SDP / PSD cones are not implemented.** CVXPY problems that require them fail with a clear error.
- ND power cones (`PowConeND`) are not implemented.
- Dual exponential and dual power cones exist in Rust; they are not registered as CVXPY capabilities.

## Warm starts and factorisation

- `update_q` / `update_b` never refactor.
- `update_p`: same CSC sparsity → numerical KKT refactor; pattern change → drop factorisation and rebuild on the next `solve`.
- `update_a`: always drops the KKT factorisation (scenario / constraint matrix change).
- `reset("cold")` zeros ADMM iterates and reseeds `ρ` on the next solve; `reset("factor")` zeros iterates but keeps `ρ` and the current factorisation.
- Warm starts help when the next problem is a small perturbation of a solved one. They can hurt if the previous iterate is far from the new solution; use `WarmStartMode::ColdStart` / `reset("cold")`.
- After ρ adaptation, the accelerator is restarted (COSMO.jl behaviour).

## Infeasibility / unboundedness

- Certificates follow Banjac-style ADMM conditions and are checked periodically. They can miss infeasibility (return `Max_iter_reached`) or, rarely, fire late. Clarabel's embedding is typically more decisive.
- Dual `y` is `−μ` after reverse scaling. Constraint dual recovery through CVXPY uses the standard zero/nonneg split; SOC/exp/power dual slices follow CVXPY's `extract_dual_value`.

## Acceleration

- Only Type-II Anderson with QR, restarted memory, and residual-norm safeguarding.
- Results can vary slightly if acceleration is toggled; disable with `accelerate = false` for stricter reproducibility.

## Packaging / platforms

- Python wheels require a Rust toolchain if built from source (`maturin`).
- The `clarabel` crate is a compile-time dependency (CSC + QDLDL + test reference). It is not invoked as a solver except in tests.
- No Julia runtime.

## Comparison protocol

Do not compare wall-clock alone. Tests compare status, objective, primal residual, and (when both report Solved) solution distance under documented tolerances. COSMO may return `Max_iter_reached` with small residuals where Clarabel reports `Solved`.
