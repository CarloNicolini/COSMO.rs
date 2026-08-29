#![allow(non_snake_case)]

use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};

mod common;
use common::*;

#[test]
fn qp_osqp_style() {
    // min  2 x1^2 + x2^2 + x1 x2 + x1 + x2
    // s.t. 0 <= x <= 0.7,  x1 + x2 == 1
    let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
    let q = vec![1.0, 1.0];
    let A = CscMatrix::from(&[
        [1.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]);
    let b = vec![1.0, -1.0, 0.0, 0.0, 0.7, 0.7];
    let cones = vec![Cone::zero(2), Cone::nonnegative(4)];
    let mut settings = default_test_settings();
    settings.eps_abs = 1e-6;
    settings.eps_rel = 1e-6;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, settings).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!((sol.x[0] - 0.3).abs() < 1e-3);
    assert!((sol.x[1] - 0.7).abs() < 1e-3);
    assert!((sol.obj_val - 1.88).abs() < 1e-3);
}

#[test]
fn qp_matches_clarabel() {
    // Clarabel.rs example_qp
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
    let cones = vec![Cone::zero(1), Cone::nonnegative(4)];
    compare_to_clarabel(&P, &q, &A, &b, cones, default_test_settings(), 1e-3, 1e-2);
}

#[test]
fn unconstrained_qp() {
    // min 1/2 (x-1)^2 = 1/2 x^2 - x + 1/2, ignore const
    let P = CscMatrix::identity(1);
    let q = vec![-1.0];
    let A = CscMatrix::<f64>::zeros((0, 1));
    let b: Vec<f64> = vec![];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, vec![], default_test_settings()).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!((sol.x[0] - 1.0).abs() < 1e-4);
}

#[test]
fn max_iter_status() {
    let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
    let q = vec![1.0, 1.0];
    let A = CscMatrix::from(&[
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]);
    let b = vec![-1.0, 0.0, 0.0, 1.0, 0.7, 0.7];
    let mut s = default_test_settings();
    s.max_iter = 5;
    s.check_termination = 1;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, vec![Cone::nonnegative(6)], s).unwrap();
    let sol = solver.solve().unwrap();
    assert_eq!(sol.status, SolverStatus::MaxIterReached);
}

#[test]
fn update_q_no_refactor_changes_solution() {
    let P = CscMatrix::identity(2);
    let q = vec![-1.0, -1.0];
    let A = CscMatrix::from(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, 1.0, 0.0, 0.0];
    let cones = vec![Cone::nonnegative(4)];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let sol1 = solver.solve().unwrap().clone();
    assert_eq!(sol1.status, SolverStatus::Solved);
    solver.update_q(&[-2.0, -1.0]).unwrap();
    let sol2 = solver.solve().unwrap();
    assert_eq!(sol2.status, SolverStatus::Solved);
    assert!((sol1.x[0] - sol2.x[0]).abs() > 1e-4 || (sol1.obj_val - sol2.obj_val).abs() > 1e-4);
}

#[test]
fn repeated_solve_same_problem() {
    let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
    let q = vec![1.0, 1.0];
    let A = CscMatrix::from(&[
        [1.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]);
    let b = vec![1.0, -1.0, 0.0, 0.0, 0.7, 0.7];
    let cones = vec![Cone::zero(2), Cone::nonnegative(4)];
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let a = solver.solve().unwrap().clone();
    let bsol = solver.solve().unwrap().clone();
    assert_eq!(a.status, SolverStatus::Solved);
    assert_eq!(bsol.status, SolverStatus::Solved);
    assert!((a.obj_val - bsol.obj_val).abs() < 1e-6);
    assert!((a.x[0] - bsol.x[0]).abs() < 1e-4);
    assert!((a.x[1] - bsol.x[1]).abs() < 1e-4);
}
