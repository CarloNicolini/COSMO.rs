//! COSMO.jl textbook QP (examples/qp.jl / OSQP classic).

#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    // min ½ xᵀ P x + qᵀ x
    // s.t. 0 ≤ x ≤ 0.7,  x₁ + x₂ = 1
    // expected: x* = [0.3, 0.7], obj ≈ 1.88
    let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
    let q = vec![1.0, 1.0];
    let A = CscMatrix::from(&[
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]);
    let b = vec![-1.0, 0.0, 0.0, 1.0, 0.7, 0.7];
    let mut settings = Settings::default();
    settings.verbose = true;
    let mut solver =
        CosmoSolver::new(&P, &q, &A, &b, vec![Cone::nonnegative(6)], settings).unwrap();
    let sol = solver.solve().unwrap();
    println!("status = {}", sol.status);
    println!("x      = {:?}", sol.x);
    println!("obj    = {}", sol.obj_val);
}
