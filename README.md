<p align="center">
  <a href="https://github.com/CarloNicolini/COSMO.rs/actions"><img src="https://img.shields.io/github/actions/workflow/status/CarloNicolini/COSMO.rs/ci.yml?branch=main" alt="CI" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://crates.io/crates/cosmo"><img src="https://img.shields.io/badge/crates.io-cosmo-orange.svg" alt="crates.io" /></a>
  <a href="https://github.com/CarloNicolini/COSMO.rs/releases"><img src="https://img.shields.io/badge/Release-v0.1.0-blue.svg" alt="Release" /></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#examples">Examples</a> •
  <a href="#python--cvxpy">Python</a> •
  <a href="#citing">Citing</a> •
  <a href="#contributing">Contributing</a>
</p>

This is a **native Rust** implementation of the *Conic Operator Splitting Method* (COSMO). It solves large convex conic optimisation problems of the form

```text
minimize    (1/2) xᵀ P x + qᵀ x
subject to  A x + s = b
            s ∈ K
```

with decision variables `x ∈ ℝⁿ`, `s ∈ ℝᵐ` and data matrices `P = Pᵀ ⪰ 0`, `q ∈ ℝⁿ`, `A ∈ ℝᵐˣⁿ`, and `b ∈ ℝᵐ`. The convex set `K` is a Cartesian product of elementary convex sets and cones.

COSMO.rs is **not** a Julia wrapper and **not** Clarabel’s interior-point method. The ADMM iteration, cone projections, ρ-adaptation, over-relaxation, Anderson acceleration, and Banjac infeasibility certificates follow [COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl). Sparse CSC storage and QDLDL factorisation are reused from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) (Apache-2.0).

For design notes, limitations, and the Clarabel.rs numerical comparison, see [`docs/`](docs/).

## Features

* **Versatile** — linear programs, quadratic programs, second-order cone programs, and problems with exponential and 3D power cones
* **Quadratic objectives** — `P ⪰ 0` is handled natively in the ADMM KKT system (including `P = 0` for pure LPs)
* **Safeguarded acceleration** — Type-II Anderson acceleration with residual-norm safeguarding (from [COSMOAccelerators.jl](https://github.com/oxfordcontrol/COSMOAccelerators.jl))
* **Infeasibility detection** — Banjac-style certificates without a homogeneous self-dual embedding
* **CVXPY support** — optional Python bindings expose a `COSMO_RUST` conic solver for CVXPY
* **Warm starting** — warm-start `x` / `y`, and reuse the factorisation across `update_q` / `update_b`
* **Persistent workspace** — solve related problems without rebuilding the KKT pattern when only linear terms change
* **Open source** — Apache License 2.0

**Not in this release (see [`docs/limitations.md`](docs/limitations.md)):** SDP / PSD cones, chordal decomposition, clique merging, JuMP/MOI, and pluggable third-party linear solvers beyond Clarabel’s QDLDL.

## Installation

### Rust

Add the crate to your `Cargo.toml` (path or git until published):

```toml
[dependencies]
cosmo = { git = "https://github.com/CarloNicolini/COSMO.rs" }
```

Or clone and use locally:

```bash
git clone https://github.com/CarloNicolini/COSMO.rs
cd COSMO.rs
cargo build --release
cargo test
```

Requires a recent Rust toolchain (`rustc` ≥ 1.70).

### Python (optional)

```bash
pip install maturin numpy scipy
maturin develop --features python
# optional CVXPY interface
pip install cvxpy
```

## Examples

### Quadratic program

```rust
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    // min  ½ xᵀ P x + qᵀ x
    // s.t. x₁ − 2 x₂ = 0
    //      −1 ≤ x ≤ 1
    let P = CscMatrix::from(&[[6.0, 0.0], [0.0, 4.0]]);
    let q = vec![-1.0, -4.0];
    let A = CscMatrix::from(&[
        [1.0, -2.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ]);
    let b = vec![0.0, 1.0, 1.0, 1.0, 1.0];
    let cones = vec![Cone::zero(1), Cone::nonnegative(4)];

    let mut settings = Settings::default();
    settings.verbose = true;

    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, settings).unwrap();
    let sol = solver.solve().unwrap();

    println!("status = {}", sol.status);
    println!("x      = {:?}", sol.x);
    println!("obj    = {}", sol.obj_val);
}
```

Run the bundled examples:

```bash
cargo run --example qp
cargo run --example lp
cargo run --example socp
cargo run --example expcone
cargo run --example powcone
cargo run --example cosmo_jl_qp   # COSMO.jl examples/qp.jl
cargo run --example cosmo_jl_lp   # COSMO.jl examples/lp.jl

# Python / CVXPY (see also [#python--cvxpy](#python--cvxpy))
uv run python examples/python/cvxpy_qp.py
uv run python examples/python/cvxpy_socp.py
uv run python examples/python/cvxpy_lp.py
```

### Warm starts and updates

```rust
// Reuse the factorisation when only the linear cost changes.
solver.update_q(&q_new).unwrap();
solver.update_b(&b_new).unwrap();
solver.warm_start(Some(&x0), Some(&y0)).unwrap();
let sol = solver.solve().unwrap();
```

### Second-order cone

```rust
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

let P = CscMatrix::from(&[[0.0, 0.0], [0.0, 2.0]]);
let q = vec![0.0, 0.0];
let A = CscMatrix::from(&[[0.0, 0.0], [-2.0, 0.0], [0.0, -1.0]]);
let b = vec![1.0, -2.0, -2.0];

let mut solver = CosmoSolver::new(
    &P, &q, &A, &b,
    vec![Cone::second_order(3)],
    Settings::default(),
).unwrap();

println!("{:?}", solver.solve().unwrap().x);
```

### Supported cones

| Cone | Constructor | Dimension |
|---|---|---|
| Zero `{0}ⁿ` | `Cone::zero(n)` | `n` |
| Nonnegative `ℝ₊ⁿ` | `Cone::nonnegative(n)` | `n` |
| Second-order | `Cone::second_order(n)` | `n ≥ 2` |
| Exponential | `Cone::exponential()` | 3 |
| Dual exponential | `Cone::dual_exponential()` | 3 |
| Power (`0 < α < 1`) | `Cone::power(alpha)` | 3 |
| Dual power | `Cone::dual_power(alpha)` | 3 |
| Box `[l, u]` | `Cone::boxed(l, u)` | `l.len()` |

## Python & CVXPY

Low-level bindings (`cosmo_rs.CosmoSolver`) and a CVXPY conic interface (`COSMO_RUST`) are available after installing the Python package (see [Installation](#installation)).

### Low-level API

```python
import numpy as np
from scipy import sparse
from cosmo_rs import CosmoSolver

P = sparse.triu(sparse.csc_matrix([[6.0, 0.0], [0.0, 4.0]])).tocsc()
q = np.array([-1.0, -4.0])
A = sparse.csc_matrix(
    [[1.0, -2.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
)
b = np.array([0.0, 1.0, 1.0, 1.0, 1.0])

solver = CosmoSolver(
    P, q, A, b,
    [("zero", 1), ("nonnegative", 4)],
    eps_abs=1e-6,
)
print(solver.solve().x)
```

### CVXPY

Register once, then pass `solver="COSMO_RUST"` (or `solver=COSMO_RUST()`):

```python
import cvxpy as cp
import numpy as np
from cosmo_rs.cvxpy_interface import register

register()

# Quadratic program (same data as the Rust QP example)
P = np.array([[6.0, 0.0], [0.0, 4.0]])
q = np.array([-1.0, -4.0])
x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
    [x[0] - 2 * x[1] == 0, x >= -1, x <= 1],
)
prob.solve(solver="COSMO_RUST", verbose=True)
print(prob.status, x.value, prob.value)
```

Second-order cone:

```python
x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(cp.square(x[1])),
    [cp.norm(cp.hstack([2 - 2 * x[0], -2 - x[1]]), 2) <= 1],
)
prob.solve(solver="COSMO_RUST")
print(prob.status, x.value)
```

Simplex projection (instance API):

```python
from cosmo_rs.cvxpy_interface import COSMO_RUST

y = cp.Variable(3, nonneg=True)
target = np.array([0.8, 0.5, -0.1])
prob = cp.Problem(cp.Minimize(cp.sum_squares(y - target)), [cp.sum(y) == 1])
prob.solve(solver=COSMO_RUST())
print(prob.status, y.value)
```

Runnable scripts:

```bash
# after: uv sync && uv pip install cvxpy
#    or: maturin develop --features python && pip install cvxpy
uv run python examples/python/cvxpy_qp.py
uv run python examples/python/cvxpy_socp.py
uv run python examples/python/cvxpy_lp.py
uv run python examples/python/bench_qp.py
```

There is **no Julia dependency**.

## Numerical checks

```bash
cargo test
cargo test --test clarabel_benchmark extensive -- --nocapture
cargo test --test stress_vs_clarabel -- --nocapture

# optional: Julia reference (COSMO.jl)
julia --project=/tmp/cosmo-compare examples/julia/compare_textbook.jl
```

The Clarabel.rs suite covers 172 instances. A separate stress suite catalogs cases where Clarabel converges but COSMO.rs stalls (extreme `A` scaling, frozen tiny ρ). See [`docs/benchmarks.md`](docs/benchmarks.md).

## Citing

If you find COSMO useful in your project, please cite the COSMO paper (the algorithm this crate implements):

```bibtex
@Article{Garstka_2021,
  author    = {Michael Garstka and Mark Cannon and Paul Goulart},
  journal   = {Journal of Optimization Theory and Applications},
  title     = {{COSMO}: A Conic Operator Splitting Method for Convex Conic Problems},
  volume    = {190},
  number    = {3},
  pages     = {779--810},
  year      = {2021},
  publisher = {Springer},
  doi       = {10.1007/s10957-021-01896-x},
  url       = {https://doi.org/10.1007/s10957-021-01896-x}
}
```

Open access: [link.springer.com](https://link.springer.com/article/10.1007/s10957-021-01896-x).

## Contributing

Helpful contributions are welcome.

* How the ADMM loop maps from COSMO.jl is documented in [`docs/architecture.md`](docs/architecture.md).
* Please report issues or bugs on GitHub.
* Before large new features (SDP, chordal decomposition, alternate linear solvers), open an issue so the design can be discussed against [`docs/limitations.md`](docs/limitations.md).

```bash
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings   # optional
```

## Related projects

| Project | Role |
|---|---|
| [COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl) | Original Julia solver — algorithmic reference |
| [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) | CSC + QDLDL substrate; IPM reference in tests |
| [COSMOAccelerators.jl](https://github.com/oxfordcontrol/COSMOAccelerators.jl) | Anderson acceleration reference |
| [cosmo-python](https://github.com/oxfordcontrol/cosmo-python) | Julia COSMO from Python (separate from this crate’s PyO3 bindings) |

## Licence

This project is licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details and third-party attribution.
