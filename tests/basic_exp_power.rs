#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};

#[test]
fn expcone_max_x() {
    // max x  s.t. y exp(x/y) <= z, y=1, z=exp(5)
    // canonical Clarabel form
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
    let cones = vec![Cone::exponential(), Cone::zero(2)];
    let mut settings = default_test_settings();
    settings.max_iter = 10_000;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones.clone(), settings.clone()).unwrap();
    let sol = solver.solve().unwrap();
    assert!(
        sol.status == SolverStatus::Solved || sol.r_prim < 1e-3,
        "status {:?} obj {} rp {} rd {}",
        sol.status,
        sol.obj_val,
        sol.r_prim,
        sol.r_dual
    );
    if sol.status == SolverStatus::Solved {
        assert!((sol.obj_val + 5.0).abs() < 5e-2);
        assert!((sol.x[0] - 5.0).abs() < 5e-2);
    }
    compare_to_clarabel(&P, &q, &A, &b, cones, settings, 5e-2, 1e-1);
}

#[test]
fn power_cone_opt() {
    let P = CscMatrix::<f64>::zeros((6, 6));
    let q = vec![0.0, 0.0, -1.0, 0.0, 0.0, -1.0];
    let A = CscMatrix::from(&[
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [1.0, 2.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]);
    let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0];
    let cones = vec![
        Cone::power(0.6),
        Cone::power(0.1),
        Cone::zero(1),
        Cone::zero(1),
    ];
    let mut settings = default_test_settings();
    settings.max_iter = 10_000;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones.clone(), settings.clone()).unwrap();
    let sol = solver.solve().unwrap();
    if sol.status == SolverStatus::Solved {
        assert!((sol.obj_val + 1.8458).abs() < 5e-2);
    }
    compare_to_clarabel(&P, &q, &A, &b, cones, settings, 5e-2, 1e-1);
}
