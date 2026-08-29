# Numerical comparison vs Clarabel.rs

Rust integration tests solve each instance with COSMO.rs (ADMM) and, where
the cone type is supported, with Clarabel.rs (`DefaultSolver`, interior point).
SDP is not generated. Wall-clock is not the figure of merit: this is a
**correctness** suite.

Run:

```bash
cargo test --test clarabel_benchmark extensive -- --nocapture
cargo test
```

## Latest `cargo test --test clarabel_benchmark extensive -- --nocapture`

Recorded from a debug-profile run of `tests/clarabel_benchmark.rs`.

| Metric | Value |
|---|---|
| Problems | 172 |
| Clarabel.rs `Solved` | 169 |
| Both `Solved` | 169 |
| COSMO.rs agreement | **172 / 172** |
| Relative objective error (Clarabel-solved, p50) | 3.3e-10 |
| Relative objective error (p90) | 1.3e-8 |
| Failures | 0 |

The three cases Clarabel does not report `Solved` are constructed infeasible /
unbounded LPs. COSMO.rs either matches the infeasibility status or returns
`Max_iter_reached`; the suite treats that as agreement.

### Families

| Family | Count | Notes |
|---|---|---|
| Clarabel textbook QP / LP / SOCP / power | 4 | Exact examples from Clarabel.rs |
| Exponential cone (`max x`, `min z`) | 10 | `c ∈ {-1,0,1,2,5}` |
| Random feasible LPs | 37 | Including up to 80×40 |
| Random strictly convex QPs | 32 | KKT-constructed + diagonal shift |
| Equality-constrained QPs | 16 | Zero cone |
| Box QPs | 16 | Bounds as nonnegative slacks |
| Unconstrained QPs | 10 | Unique minimiser |
| SOCPs | 19 | Linear objective over a second-order cone |
| Least-squares with bounds | 12 | `P = FᵀF` |
| Mixed zero + NN + SOC | 12 | |
| Poorly scaled QP | 1 | `q` × 1e4, `b` × 1e-4 |
| Infeasible / unbounded | 3 | Farkas / unbounded ray |

Additional checks (same crate, other test files): solver reuse vs a fresh
object, `update_q` vs a new solve, warm start, cone projection idempotence,
and the COSMO.jl-style OSQP QP (`x = (0.3, 0.7)`, obj `1.88`).

## How agreement is decided

When Clarabel reports `Solved` / `AlmostSolved`:

- COSMO.rs must report `Solved`, **or** `Max_iter_reached` with both residuals
  `< 1e-2` and a matching objective.
- Relative objective `|f_cosmo − f_clar| / (1 + |f_clar|)` must be within a
  family-specific tolerance (typically `2e-3` for LP/QP, `8e-2` for exp/power),
  **or** `‖x_cosmo − x_clar‖_∞` within the family `x` tolerance (non-unique LPs).

When Clarabel reports primal/dual infeasible, COSMO.rs must report a compatible
status or `Max_iter_reached`.

## What this does not show

- COSMO.rs is not faster than Clarabel.rs on these small instances (Newton vs ADMM).
- SDP, ND power cones, and chordal decomposition are not tested.
- Portfolio / skfolio sequences are not in this suite.
