#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};

#[test]
fn lp_cosmo_example() {
    // min c'x  s.t. x <= 10, x >= 1, x2 >= 5, x1+x3 >= 4
    // known solution [3,5,1,1], obj 20
    let P = CscMatrix::<f64>::zeros((4, 4));
    let q = vec![1.0, 2.0, 3.0, 4.0];
    // Ax + s = b, s in R_+
    // x <= 10  =>  x + s = 10, s >= 0
    // x >= 1   => -x + s = -1
    // x2 >= 5  => -x2 + s = -5
    // x1+x3>=4 => -x1-x3 + s = -4
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
    let cones = vec![Cone::nonnegative(10)];
    let mut settings = default_test_settings();
    settings.eps_abs = 1e-5;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, settings).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!((sol.x[0] - 3.0).abs() < 2e-2);
    assert!((sol.x[1] - 5.0).abs() < 2e-2);
    assert!((sol.x[2] - 1.0).abs() < 2e-2);
    assert!((sol.x[3] - 1.0).abs() < 2e-2);
    assert!((sol.obj_val - 20.0).abs() < 2e-2);
}

#[test]
fn equality_constrained_qp() {
    // min 1/2 ||x||^2  s.t. x1 + x2 = 1
    let P = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix::from(&[[1.0, 1.0]]);
    let b = vec![1.0];
    let cones = vec![Cone::zero(1)];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!((sol.x[0] - 0.5).abs() < 1e-3);
    assert!((sol.x[1] - 0.5).abs() < 1e-3);
}

#[test]
fn primal_infeasible_lp() {
    // x >= 1, x <= 0
    let P = CscMatrix::<f64>::zeros((1, 1));
    let q = vec![1.0];
    let A = CscMatrix::from(&[[-1.0], [1.0]]);
    let b = vec![-1.0, 0.0];
    let cones = vec![Cone::nonnegative(2)];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let sol = solver.solve().unwrap();
    assert!(
        sol.status == SolverStatus::PrimalInfeasible || sol.status == SolverStatus::MaxIterReached
    );
}
