#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};

#[test]
fn socp_clarabel_example() {
    // Clarabel example_socp
    let P = CscMatrix::from(&[[0.0, 0.0], [0.0, 2.0]]);
    let q = vec![0.0, 0.0];
    let A = CscMatrix::from(&[[0.0, 0.0], [-2.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, -2.0, -2.0];
    let cones = vec![Cone::second_order(3)];
    let mut solver =
        CosmoSolver::new(&P, &q, &A, &b, cones.clone(), default_test_settings()).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    compare_to_clarabel(&P, &q, &A, &b, cones, default_test_settings(), 1e-3, 1e-2);
}

#[test]
fn soc_feasibility() {
    // find x s.t. ||x|| <= 1,  minimise 0
    let n = 4;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![0.0; n];
    // s = (t, x), t=1,  A x + s = b with A mapping x into the x-part of the SOC
    // [0; -I] x + s = [1; 0]  => s = (1, x) ∈ SOC
    let mut rows = vec![vec![0.0; n]; n + 1];
    for i in 0..n {
        rows[i + 1][i] = -1.0;
    }
    let A = dense_to_csc(&rows);
    let mut b = vec![0.0; n + 1];
    b[0] = 1.0;
    let cones = vec![Cone::second_order(n + 1)];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    let nrm: f64 = sol.x.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(nrm <= 1.0 + 1e-3);
}
