#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    let P = CscMatrix::<f64>::zeros((3, 3));
    let q = vec![-1.0, 0.0, 0.0];
    let A = CscMatrix::from(&[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let b = vec![0.0, 0.0, 0.0, 1.0, 5f64.exp()];
    let mut solver = CosmoSolver::new(
        &P,
        &q,
        &A,
        &b,
        vec![Cone::exponential(), Cone::zero(2)],
        Settings::default(),
    )
    .unwrap();
    let sol = solver.solve().unwrap();
    println!("status={} x={:?} obj={}", sol.status, sol.x, sol.obj_val);
}
