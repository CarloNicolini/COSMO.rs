#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
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
    let mut solver = CosmoSolver::new(
        &P,
        &q,
        &A,
        &b,
        vec![Cone::nonnegative(10)],
        Settings::default(),
    )
    .unwrap();
    let sol = solver.solve().unwrap();
    println!("status={} x={:?} obj={}", sol.status, sol.x, sol.obj_val);
}
