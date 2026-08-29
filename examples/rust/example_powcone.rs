#![allow(non_snake_case)]
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};

fn main() {
    let P = CscMatrix::<f64>::zeros((6, 6));
    let q = vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0];
    let A = CscMatrix::from(&[
        [-1., 0., 0., 0., 0., 0.],
        [0., -1., 0., 0., 0., 0.],
        [0., 0., -1., 0., 0., 0.],
        [0., 0., 0., -1., 0., 0.],
        [0., 0., 0., 0., -1., 0.],
        [0., 0., 0., 0., 0., -1.],
        [1., 2., 0., 3., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
    ]);
    let b = vec![0., 0., 0., 0., 0., 0., 3., 1.];
    let cones = vec![
        Cone::power(0.6),
        Cone::power(0.1),
        Cone::zero(1),
        Cone::zero(1),
    ];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, Settings::default()).unwrap();
    let sol = solver.solve().unwrap();
    println!("status={} obj={}", sol.status, sol.obj_val);
}
