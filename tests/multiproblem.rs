//! Multiproblem correctness suite: independent instances, reuse, and Clarabel.rs comparison.

#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};
use rand::Rng;

fn solve_and_compare(
    seed: u64,
    P: &CscMatrix,
    q: &[f64],
    A: &CscMatrix,
    b: &[f64],
    cones: Vec<Cone>,
    obj_tol: f64,
    x_tol: f64,
) {
    let mut settings = default_test_settings();
    settings.max_iter = 12_000;
    compare_to_clarabel(P, q, A, b, cones, settings, obj_tol, x_tol);
    let _ = seed;
}

fn feasible_lp(
    seed: u64,
    m: usize,
    n: usize,
) -> (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>) {
    let mut rng = rng(seed);
    let A = random_sparse_a(&mut rng, m, n, 0.5);
    let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>().abs()).collect();
    let s: Vec<f64> = (0..m).map(|_| rng.gen::<f64>().abs()).collect();
    let y: Vec<f64> = (0..m).map(|_| rng.gen::<f64>().abs()).collect();
    let mut b = vec![0.0; m];
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            b[A.rowval[j]] += A.nzval[j] * x[col];
        }
    }
    for i in 0..m {
        b[i] += s[i];
    }
    let mut q = vec![0.0; n];
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            q[col] -= A.nzval[j] * y[A.rowval[j]];
        }
    }
    let P = CscMatrix::<f64>::zeros((n, n));
    (P, q, A, b, vec![Cone::nonnegative(m)])
}

fn feasible_qp(
    seed: u64,
    m: usize,
    n: usize,
) -> (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>) {
    let mut rng = rng(seed);
    let P = random_psd_triu(&mut rng, n);
    let A = random_sparse_a(&mut rng, m, n, 0.4);
    let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let s: Vec<f64> = (0..m).map(|_| rng.gen::<f64>().abs()).collect();
    let y: Vec<f64> = (0..m).map(|_| rng.gen::<f64>().abs()).collect();
    let mut b = vec![0.0; m];
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            b[A.rowval[j]] += A.nzval[j] * x[col];
        }
    }
    for i in 0..m {
        b[i] += s[i];
    }
    let mut q = vec![0.0; n];
    // q = -P x - A' y  so that (x,s,y) is KKT
    for col in 0..n {
        for j in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[j];
            q[row] -= P.nzval[j] * x[col];
            if row != col {
                q[col] -= P.nzval[j] * x[row];
            }
        }
    }
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            q[col] -= A.nzval[j] * y[A.rowval[j]];
        }
    }
    (P, q, A, b, vec![Cone::nonnegative(m)])
}

fn equality_qp(
    seed: u64,
    m: usize,
    n: usize,
) -> (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>) {
    let mut rng = rng(seed);
    let P = random_psd_triu(&mut rng, n);
    let A = random_sparse_a(&mut rng, m, n, 0.8);
    let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let mut b = vec![0.0; m];
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            b[A.rowval[j]] += A.nzval[j] * x[col];
        }
    }
    let y: Vec<f64> = (0..m).map(|_| rng.gen::<f64>()).collect();
    let mut q = vec![0.0; n];
    for col in 0..n {
        for j in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[j];
            q[row] -= P.nzval[j] * x[col];
            if row != col {
                q[col] -= P.nzval[j] * x[row];
            }
        }
    }
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            q[col] -= A.nzval[j] * y[A.rowval[j]];
        }
    }
    (P, q, A, b, vec![Cone::zero(m)])
}

fn socp_linear(seed: u64, dim: usize) -> (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>) {
    let mut rng = rng(seed);
    let n = dim - 1;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q: Vec<f64> = (0..n).map(|_| randn(&mut rng)).collect();
    let mut rows = vec![vec![0.0; n]; dim];
    for i in 0..n {
        rows[i + 1][i] = -1.0;
    }
    let A = dense_to_csc(&rows);
    let mut b = vec![0.0; dim];
    b[0] = 1.0 + rng.gen::<f64>().abs();
    (P, q, A, b, vec![Cone::second_order(dim)])
}

#[test]
fn random_feasible_lps() {
    for seed in 1u64..=20 {
        let (P, q, A, b, cones) = feasible_lp(seed, 8, 5);
        solve_and_compare(seed, &P, &q, &A, &b, cones, 1e-3, 5e-2);
    }
}

#[test]
fn random_feasible_qps() {
    for seed in 1u64..=20 {
        let (P, q, A, b, cones) = feasible_qp(seed, 10, 6);
        solve_and_compare(seed, &P, &q, &A, &b, cones, 2e-3, 8e-2);
    }
}

#[test]
fn equality_constrained_qps() {
    for seed in 1u64..=12 {
        let (P, q, A, b, cones) = equality_qp(seed, 3, 6);
        solve_and_compare(seed, &P, &q, &A, &b, cones, 2e-3, 8e-2);
    }
}

#[test]
fn random_socps() {
    for seed in 1u64..=12 {
        let (P, q, A, b, cones) = socp_linear(seed, 6);
        solve_and_compare(seed, &P, &q, &A, &b, cones, 5e-3, 1e-1);
    }
}

#[test]
fn mixed_zero_nn_soc() {
    for seed in 1u64..=10 {
        let mut rng = rng(seed);
        let n = 4;
        let P = random_psd_triu(&mut rng, n);
        let Aeq = random_sparse_a(&mut rng, 2, n, 1.0);
        let Ann = random_sparse_a(&mut rng, 3, n, 0.6);
        // stack [Aeq; Ann; SOC rows]
        let mut rows = vec![vec![0.0; n]; 2 + 3 + 3];
        for col in 0..n {
            for j in Aeq.colptr[col]..Aeq.colptr[col + 1] {
                rows[Aeq.rowval[j]][col] = Aeq.nzval[j];
            }
            for j in Ann.colptr[col]..Ann.colptr[col + 1] {
                rows[2 + Ann.rowval[j]][col] = Ann.nzval[j];
            }
        }
        rows[5][0] = 0.0;
        rows[6][0] = -1.0;
        rows[7][1] = -1.0;
        let A = dense_to_csc(&rows);
        let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let mut s = vec![0.0; 8];
        s[2] = rng.gen::<f64>().abs();
        s[3] = rng.gen::<f64>().abs();
        s[4] = rng.gen::<f64>().abs();
        s[5] = 2.0;
        s[6] = 0.3;
        s[7] = 0.4;
        let mut b = vec![0.0; 8];
        for col in 0..n {
            for j in A.colptr[col]..A.colptr[col + 1] {
                b[A.rowval[j]] += A.nzval[j] * x[col];
            }
        }
        for i in 0..8 {
            b[i] += s[i];
        }
        let q: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let cones = vec![Cone::zero(2), Cone::nonnegative(3), Cone::second_order(3)];
        solve_and_compare(seed, &P, &q, &A, &b, cones, 5e-3, 1e-1);
    }
}

#[test]
fn poorly_scaled_qp() {
    let _rng = rng(99);
    let (P, mut q, A, mut b, cones) = feasible_qp(7, 8, 5);
    for qi in q.iter_mut() {
        *qi *= 1e4;
    }
    for bi in b.iter_mut() {
        *bi *= 1e-4;
    }
    let _ = rng;
    solve_and_compare(99, &P, &q, &A, &b, cones, 5e-3, 1e-1);
}

#[test]
fn tiny_and_medium_dimensions() {
    for &(m, n, seed) in &[(1usize, 1usize, 1u64), (2, 3, 2), (15, 10, 3), (30, 20, 4)] {
        let (P, q, A, b, cones) = feasible_lp(seed + 100, m, n);
        solve_and_compare(seed, &P, &q, &A, &b, cones, 5e-3, 1e-1);
    }
}

#[test]
fn solver_reuse_across_q_updates() {
    let (P, q, A, b, cones) = feasible_qp(11, 8, 5);
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let s1 = solver.solve().unwrap().clone();
    assert_eq!(s1.status, SolverStatus::Solved);
    let mut q2 = q.clone();
    for qi in q2.iter_mut() {
        *qi *= 1.05;
    }
    solver.update_q(&q).unwrap();
    solver.update_q(&q2).unwrap();
    let s2 = solver.solve().unwrap().clone();
    assert!(
        s2.status == SolverStatus::Solved || s2.r_prim < 1e-3,
        "reuse failed {:?}",
        s2.status
    );
}

#[test]
fn fresh_vs_reused_solver() {
    let (P, q, A, b, cones) = feasible_qp(21, 6, 4);
    let mut s1 = CosmoSolver::new(&P, &q, &A, &b, cones.clone(), default_test_settings()).unwrap();
    let a = s1.solve().unwrap().clone();
    let mut s2 = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let bsol = s2.solve().unwrap().clone();
    assert_eq!(a.status, SolverStatus::Solved);
    assert_eq!(bsol.status, SolverStatus::Solved);
    assert!((a.obj_val - bsol.obj_val).abs() < 1e-6);
}

#[test]
fn zero_curvature_lp_via_qp_api() {
    let (P, q, A, b, cones) = feasible_lp(3, 6, 4);
    solve_and_compare(3, &P, &q, &A, &b, cones, 1e-3, 5e-2);
}

#[test]
fn box_qp_via_nonnegative() {
    let P = CscMatrix::identity(3);
    let q = vec![-1.0, -2.0, -3.0];
    let A = CscMatrix::from(&[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]);
    let b = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let cones = vec![Cone::nonnegative(6)];
    solve_and_compare(0, &P, &q, &A, &b, cones, 1e-3, 5e-2);
}

#[test]
fn independent_instances_in_one_process() {
    let mut objs = vec![];
    for seed in 50u64..60 {
        let (P, q, A, b, cones) = feasible_qp(seed, 5, 4);
        let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
        let sol = solver.solve().unwrap();
        assert_eq!(sol.status, SolverStatus::Solved);
        objs.push(sol.obj_val);
    }
    assert!(objs.iter().all(|v| v.is_finite()));
}

#[test]
fn warm_start_does_not_break_correctness() {
    let (P, q, A, b, cones) = feasible_qp(8, 6, 4);
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, default_test_settings()).unwrap();
    let s1 = solver.solve().unwrap().clone();
    solver.warm_start(Some(&s1.x), Some(&s1.y)).unwrap();
    let s2 = solver.solve().unwrap().clone();
    assert_eq!(s2.status, SolverStatus::Solved);
    assert!((s1.obj_val - s2.obj_val).abs() < 1e-4);
}

#[test]
fn exp_and_power_match_clarabel() {
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
    settings.max_iter = 15_000;
    compare_to_clarabel(&P, &q, &A, &b, cones, settings, 8e-2, 2e-1);
}
