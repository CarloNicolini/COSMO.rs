# COSMO.rs

Native Rust implementation of the **Conic Operator Splitting Method (COSMO)** — an ADMM solver for convex quadratic-conic programs:

```text
minimize    (1/2) xᵀ P x + qᵀ x
subject to  A x + s = b
            s ∈ K
```

`K` may be a product of zero, nonnegative, second-order, exponential, and 3D power cones. SDP is not implemented in this milestone.

This is **not** Clarabel and **not** a Julia wrapper. The ADMM iteration, cone projections, rho adaptation, over-relaxation, Anderson acceleration, and infeasibility certificates follow COSMO.jl. Sparse CSC storage and QDLDL factorisation are reused from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) (Apache-2.0).

## Direct Rust usage

```rust
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

let P = CscMatrix::from(&[[4., 1.], [0., 2.]]);
let q = vec![1., 1.];
let A = CscMatrix::from(&[
    [1., 1.], [-1., 0.], [0., -1.],
    [1., 1.], [1., 0.], [0., 1.],
]);
let b = vec![-1., 0., 0., 1., 0.7, 0.7];
let cones = vec![Cone::nonnegative(6)];
let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, Settings::default()).unwrap();
let sol = solver.solve().unwrap();
println!("{}  x = {:?}  obj = {}", sol.status, sol.x, sol.obj_val);
```

Persistent updates (no KKT rebuild for `q`/`b`):

```rust
solver.update_q(&q2).unwrap();
solver.update_b(&b2).unwrap();
solver.solve().unwrap();
```

## Python

```bash
pip install maturin
maturin develop --features python
```

```python
import numpy as np
from scipy import sparse
from cosmo_rs import CosmoSolver

P = sparse.triu(sparse.csc_matrix([[4., 1.], [1., 2.]])).tocsc()
q = np.array([1., 1.])
A = sparse.csc_matrix([[1., 1.], [-1., 0.], [0., -1.], [1., 1.], [1., 0.], [0., 1.]])
b = np.array([-1., 0., 0., 1., 0.7, 0.7])
solver = CosmoSolver(P, q, A, b, [("nonnegative", 6)], eps_abs=1e-6)
print(solver.solve().x)
```

## CVXPY

```python
import cvxpy as cp
from cosmo_rs.cvxpy_interface import COSMO_RUST, register
register()

x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0, cp.sum(x) == 1])
prob.solve(solver="COSMO_RUST")
# or: prob.solve(solver=COSMO_RUST())
```

There is **no Julia dependency**.

## Tests

```bash
cargo test
```

The `multiproblem` and `clarabel_benchmark` suites solve independent LPs, QPs,
SOCPs, mixed-cone, scaled, tiny, and medium problems, and compare objective /
status / residuals against Clarabel.rs (`docs/benchmarks.md`).

## License

Apache-2.0. See `LICENSE` and `NOTICE` for attribution of COSMO.jl, Clarabel.rs, and COSMOAccelerators.jl.

## Documentation

- [Architecture (Phase 0)](docs/architecture.md)
- [Limitations](docs/limitations.md)
- [Engineering report](docs/engineering-report.md)
