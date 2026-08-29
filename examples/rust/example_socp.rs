#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    let P = CscMatrix::from(&[[0.0, 0.0], [0.0, 2.0]]);
    let q = vec![0.0, 0.0];
    let A = CscMatrix::from(&[[0.0, 0.0], [-2.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, -2.0, -2.0];
    let mut solver = CosmoSolver::new(
        &P,
        &q,
        &A,
        &b,
        vec![Cone::second_order(3)],
        Settings::default(),
    )
    .unwrap();
    println!("{:?}", solver.solve().unwrap().x);
}
