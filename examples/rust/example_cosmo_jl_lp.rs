//! COSMO.jl textbook LP (examples/lp.jl) as a standalone Rust example.

#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    // min cᵀx  s.t. x ≤ 10, x ≥ 1, x₂ ≥ 5, x₁+x₃ ≥ 4
    // expected: x* = [3,5,1,1], obj = 20
    let P = CscMatrix::<f64>::zeros((4, 4));
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let A = CscMatrix::from(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, -1.0, 0.0],
    ]);
    let b = vec![10.0, 10.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0, -5.0, -4.0];
    let mut settings = Settings::default();
    settings.verbose = true;
    settings.eps_abs = 1e-5;
    let mut solver =
        CosmoSolver::new(&P, &q, &A, &b, vec![Cone::nonnegative(10)], settings).unwrap();
    let sol = solver.solve().unwrap();
    println!("status = {}", sol.status);
    println!("x      = {:?}", sol.x);
    println!("obj    = {}", sol.obj_val);
}
