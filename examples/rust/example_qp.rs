#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    // Clarabel.rs textbook QP
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
    let mut settings = Settings::default();
    settings.verbose = true;
    let mut solver = CosmoSolver::new(
        &P,
        &q,
        &A,
        &b,
        vec![Cone::zero(1), Cone::nonnegative(4)],
        settings,
    )
    .unwrap();
    let sol = solver.solve().unwrap();
    println!("status = {}", sol.status);
    println!("x = {:?}", sol.x);
    println!("obj = {}", sol.obj_val);
}
