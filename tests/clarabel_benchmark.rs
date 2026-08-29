//! Extensive numerical comparison of COSMO.rs against Clarabel.rs.
//!
//! Every instance is solved by both solvers. When Clarabel reports Solved,
//! COSMO.rs must match the objective (and, where the solution is unique, x).
//! SDP is out of scope; those problems are not generated.

#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, SolverStatus};
use rand::Rng;

type Problem = (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>);

fn settings_for(max_iter: usize) -> cosmo::Settings {
    let mut s = default_test_settings();
    s.max_iter = max_iter;
    s
}

fn feasible_lp(seed: u64, m: usize, n: usize) -> Problem {
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
    (
        CscMatrix::<f64>::zeros((n, n)),
        q,
        A,
        b,
        vec![Cone::nonnegative(m)],
    )
}

fn feasible_qp(seed: u64, m: usize, n: usize) -> Problem {
    let mut rng = rng(seed);
    let P = add_diag(&random_psd_triu(&mut rng, n), 0.25);
    let A = random_sparse_a(&mut rng, m, n, 0.45);
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

fn equality_qp(seed: u64, m: usize, n: usize) -> Problem {
    let mut rng = rng(seed);
    let P = add_diag(&random_psd_triu(&mut rng, n), 0.5);
    let A = random_sparse_a(&mut rng, m, n, 0.85);
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

fn box_qp(seed: u64, n: usize) -> Problem {
    let mut rng = rng(seed);
    let P = add_diag(&random_psd_triu(&mut rng, n), 0.5);
    let q: Vec<f64> = (0..n).map(|_| randn(&mut rng)).collect();
    let mut rows = vec![vec![0.0; n]; 2 * n];
    let mut b = vec![0.0; 2 * n];
    for i in 0..n {
        rows[i][i] = 1.0;
        rows[n + i][i] = -1.0;
        b[i] = 1.0 + rng.gen::<f64>().abs();
        b[n + i] = rng.gen::<f64>().abs();
    }
    (P, q, dense_to_csc(&rows), b, vec![Cone::nonnegative(2 * n)])
}

fn unconstrained_qp(seed: u64, n: usize) -> Problem {
    let mut rng = rng(seed);
    let P = add_diag(&random_psd_triu(&mut rng, n), 1.0);
    let q: Vec<f64> = (0..n).map(|_| randn(&mut rng)).collect();
    (P, q, CscMatrix::<f64>::zeros((0, n)), vec![], vec![])
}

fn socp_linear(seed: u64, dim: usize) -> Problem {
    let mut rng = rng(seed);
    let n = dim - 1;
    let q: Vec<f64> = (0..n).map(|_| randn(&mut rng)).collect();
    let mut rows = vec![vec![0.0; n]; dim];
    for i in 0..n {
        rows[i + 1][i] = -1.0;
    }
    let mut b = vec![0.0; dim];
    b[0] = 1.0 + rng.gen::<f64>().abs();
    (
        CscMatrix::<f64>::zeros((n, n)),
        q,
        dense_to_csc(&rows),
        b,
        vec![Cone::second_order(dim)],
    )
}

fn least_squares_box(seed: u64, m: usize, n: usize) -> Problem {
    let mut rng = rng(seed);
    let F = random_sparse_a(&mut rng, m, n, 0.8);
    let g: Vec<f64> = (0..m).map(|_| randn(&mut rng)).collect();
    // P = F'F (triu)
    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for row in 0..m {
                let mut fi = 0.0;
                let mut fj = 0.0;
                for pidx in F.colptr[i]..F.colptr[i + 1] {
                    if F.rowval[pidx] == row {
                        fi = F.nzval[pidx];
                    }
                }
                for pidx in F.colptr[j]..F.colptr[j + 1] {
                    if F.rowval[pidx] == row {
                        fj = F.nzval[pidx];
                    }
                }
                s += fi * fj;
            }
            p[i][j] = s;
        }
    }
    let P = add_diag(&dense_to_csc(&p), 1e-4);
    let mut q = vec![0.0; n];
    for col in 0..n {
        for j in F.colptr[col]..F.colptr[col + 1] {
            q[col] -= F.nzval[j] * g[F.rowval[j]];
        }
    }
    let mut rows = vec![vec![0.0; n]; 2 * n];
    let mut b = vec![0.0; 2 * n];
    for i in 0..n {
        rows[i][i] = 1.0;
        rows[n + i][i] = -1.0;
        b[i] = 1.0;
        b[n + i] = 0.0;
    }
    (P, q, dense_to_csc(&rows), b, vec![Cone::nonnegative(2 * n)])
}

fn mixed_zero_nn_soc(seed: u64) -> Problem {
    let mut rng = rng(seed);
    let n = 5;
    let P = add_diag(&random_psd_triu(&mut rng, n), 0.3);
    let Aeq = random_sparse_a(&mut rng, 2, n, 1.0);
    let Ann = random_sparse_a(&mut rng, 3, n, 0.7);
    let mut rows = vec![vec![0.0; n]; 2 + 3 + 4];
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
    rows[8][2] = -1.0;
    let A = dense_to_csc(&rows);
    let x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let mut s = vec![0.0; 9];
    s[2] = rng.gen::<f64>().abs();
    s[3] = rng.gen::<f64>().abs();
    s[4] = rng.gen::<f64>().abs();
    s[5] = 3.0;
    s[6] = 0.4;
    s[7] = 0.5;
    s[8] = 0.2;
    let mut b = vec![0.0; 9];
    for col in 0..n {
        for j in A.colptr[col]..A.colptr[col + 1] {
            b[A.rowval[j]] += A.nzval[j] * x[col];
        }
    }
    for i in 0..9 {
        b[i] += s[i];
    }
    let q: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    (
        P,
        q,
        A,
        b,
        vec![Cone::zero(2), Cone::nonnegative(3), Cone::second_order(4)],
    )
}

fn clarabel_qp() -> Problem {
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
    (P, q, A, b, vec![Cone::zero(1), Cone::nonnegative(4)])
}

fn clarabel_lp() -> Problem {
    let P = CscMatrix::<f64>::zeros((2, 2));
    let q = vec![1.0, -1.0];
    let A = CscMatrix::from(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, 1.0, 1.0, 1.0];
    (P, q, A, b, vec![Cone::nonnegative(4)])
}

fn clarabel_socp() -> Problem {
    let P = CscMatrix::from(&[[0.0, 0.0], [0.0, 2.0]]);
    let q = vec![0.0, 0.0];
    let A = CscMatrix::from(&[[0.0, 0.0], [-2.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, -2.0, -2.0];
    (P, q, A, b, vec![Cone::second_order(3)])
}

fn exp_max_x(c: f64) -> Problem {
    let P = CscMatrix::<f64>::zeros((3, 3));
    let q = vec![-1.0, 0.0, 0.0];
    let A = CscMatrix::from(&[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let b = vec![0.0, 0.0, 0.0, 1.0, c.exp()];
    (P, q, A, b, vec![Cone::exponential(), Cone::zero(2)])
}

fn exp_min_z(xval: f64) -> Problem {
    // min z  s.t. (x, 1, z) ∈ Kexp, x = xval  ⇒  z ≥ exp(xval)
    let P = CscMatrix::<f64>::zeros((3, 3));
    let q = vec![0.0, 0.0, 1.0];
    let A = CscMatrix::from(&[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let b = vec![0.0, 0.0, 0.0, xval, 1.0];
    (P, q, A, b, vec![Cone::exponential(), Cone::zero(2)])
}

fn clarabel_power() -> Problem {
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
    (
        P,
        q,
        A,
        b,
        vec![
            Cone::power(0.6),
            Cone::power(0.1),
            Cone::zero(1),
            Cone::zero(1),
        ],
    )
}

fn primal_infeasible_lp() -> Problem {
    let P = CscMatrix::<f64>::zeros((1, 1));
    let q = vec![1.0];
    let A = CscMatrix::from(&[[-1.0], [1.0]]);
    let b = vec![-1.0, 0.0];
    (P, q, A, b, vec![Cone::nonnegative(2)])
}

fn dual_infeasible_lp() -> Problem {
    // min -x  s.t. x >= 0  (unbounded)
    let P = CscMatrix::<f64>::zeros((1, 1));
    let q = vec![-1.0];
    let A = CscMatrix::from(&[[-1.0]]);
    let b = vec![0.0];
    (P, q, A, b, vec![Cone::nonnegative(1)])
}

fn inconsistent_eq() -> Problem {
    let P = CscMatrix::identity(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix::from(&[[1.0, 0.0], [1.0, 0.0]]);
    let b = vec![1.0, 2.0];
    (P, q, A, b, vec![Cone::zero(2)])
}

struct Case {
    name: String,
    p: Problem,
    obj_tol: f64,
    x_tol: f64,
    max_iter: usize,
}

fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    let (P, q, A, b, cones) = clarabel_qp();
    cases.push(Case {
        name: "clarabel_qp".into(),
        p: (P, q, A, b, cones),
        obj_tol: 1e-4,
        x_tol: 1e-3,
        max_iter: 8_000,
    });
    let (P, q, A, b, cones) = clarabel_lp();
    cases.push(Case {
        name: "clarabel_lp".into(),
        p: (P, q, A, b, cones),
        obj_tol: 1e-4,
        x_tol: 1e-3,
        max_iter: 8_000,
    });
    let (P, q, A, b, cones) = clarabel_socp();
    cases.push(Case {
        name: "clarabel_socp".into(),
        p: (P, q, A, b, cones),
        obj_tol: 1e-3,
        x_tol: 1e-2,
        max_iter: 8_000,
    });
    let (P, q, A, b, cones) = clarabel_power();
    cases.push(Case {
        name: "clarabel_power".into(),
        p: (P, q, A, b, cones),
        obj_tol: 8e-2,
        x_tol: 2e-1,
        max_iter: 15_000,
    });

    for (i, c) in [-1.0, 0.0, 1.0, 2.0, 5.0].iter().enumerate() {
        let (P, q, A, b, cones) = exp_max_x(*c);
        cases.push(Case {
            name: format!("exp_max_x_{i}"),
            p: (P, q, A, b, cones),
            obj_tol: 8e-2,
            x_tol: 2e-1,
            max_iter: 15_000,
        });
        let (P, q, A, b, cones) = exp_min_z(*c);
        cases.push(Case {
            name: format!("exp_min_z_{i}"),
            p: (P, q, A, b, cones),
            obj_tol: 8e-2,
            x_tol: 2e-1,
            max_iter: 15_000,
        });
    }

    for seed in 1u64..=30 {
        let (P, q, A, b, cones) = feasible_lp(seed, 8, 5);
        cases.push(Case {
            name: format!("lp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 2e-3,
            x_tol: 8e-2,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=30 {
        let (P, q, A, b, cones) = feasible_qp(seed, 10, 6);
        cases.push(Case {
            name: format!("qp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 2e-3,
            x_tol: 8e-2,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=16 {
        let (P, q, A, b, cones) = equality_qp(seed, 3, 6);
        cases.push(Case {
            name: format!("eqqp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 2e-3,
            x_tol: 8e-2,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=16 {
        let (P, q, A, b, cones) = box_qp(seed, 5);
        cases.push(Case {
            name: format!("boxqp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 2e-3,
            x_tol: 8e-2,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=10 {
        let (P, q, A, b, cones) = unconstrained_qp(seed, 4);
        cases.push(Case {
            name: format!("uncqp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 1e-4,
            x_tol: 1e-3,
            max_iter: 4_000,
        });
    }
    for seed in 1u64..=16 {
        let (P, q, A, b, cones) = socp_linear(seed, 6);
        cases.push(Case {
            name: format!("socp_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 5e-3,
            x_tol: 1e-1,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=12 {
        let (P, q, A, b, cones) = least_squares_box(seed, 8, 4);
        cases.push(Case {
            name: format!("lsbox_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 3e-3,
            x_tol: 8e-2,
            max_iter: 12_000,
        });
    }
    for seed in 1u64..=12 {
        let (P, q, A, b, cones) = mixed_zero_nn_soc(seed);
        cases.push(Case {
            name: format!("mixed_s{seed}"),
            p: (P, q, A, b, cones),
            obj_tol: 5e-3,
            x_tol: 1e-1,
            max_iter: 12_000,
        });
    }

    for &(m, n, seed) in &[
        (1usize, 1usize, 1u64),
        (2, 3, 2),
        (15, 10, 3),
        (25, 15, 4),
        (40, 20, 5),
        (60, 30, 6),
        (80, 40, 7),
    ] {
        let (P, q, A, b, cones) = feasible_lp(seed + 200, m, n);
        cases.push(Case {
            name: format!("lp_{m}x{n}"),
            p: (P, q, A, b, cones),
            obj_tol: 5e-3,
            x_tol: 1e-1,
            max_iter: 20_000,
        });
    }
    for &(m, n, seed) in &[(20usize, 15usize, 8u64), (35, 22, 9)] {
        let (P, q, A, b, cones) = feasible_qp(seed + 300, m, n);
        cases.push(Case {
            name: format!("qp_{m}x{n}"),
            p: (P, q, A, b, cones),
            obj_tol: 5e-3,
            x_tol: 1e-1,
            max_iter: 20_000,
        });
    }
    for dim in [8usize, 12, 16] {
        let (P, q, A, b, cones) = socp_linear(40 + dim as u64, dim);
        cases.push(Case {
            name: format!("socp_dim{dim}"),
            p: (P, q, A, b, cones),
            obj_tol: 5e-3,
            x_tol: 1e-1,
            max_iter: 15_000,
        });
    }

    let (P, mut q, A, mut b, cones) = feasible_qp(7, 8, 5);
    for qi in q.iter_mut() {
        *qi *= 1e4;
    }
    for bi in b.iter_mut() {
        *bi *= 1e-4;
    }
    cases.push(Case {
        name: "poorly_scaled_qp".into(),
        p: (P, q, A, b, cones),
        obj_tol: 5e-3,
        x_tol: 1e-1,
        max_iter: 12_000,
    });

    for (name, p) in [
        ("primal_infeasible_lp", primal_infeasible_lp()),
        ("dual_infeasible_lp", dual_infeasible_lp()),
        ("inconsistent_eq", inconsistent_eq()),
    ] {
        cases.push(Case {
            name: name.into(),
            p,
            obj_tol: 1.0,
            x_tol: 1.0,
            max_iter: 8_000,
        });
    }

    cases
}

fn run_suite(cases: &[Case]) -> (usize, usize, Vec<CompareReport>) {
    let mut reports = Vec::with_capacity(cases.len());
    let mut n_clar_solved = 0;
    let mut n_ok = 0;
    for c in cases {
        let (P, q, A, b, cones) = &c.p;
        let r = compare_report(
            &c.name,
            P,
            q,
            A,
            b,
            cones.clone(),
            settings_for(c.max_iter),
            c.obj_tol,
            c.x_tol,
        );
        if r.clar_solved {
            n_clar_solved += 1;
        }
        if r.ok {
            n_ok += 1;
        } else {
            eprintln!(
                "FAIL {} status={:?} obj={} vs {} rel={:.3e} dx={:.3e} rp={:.3e} rd={:.3e} iter={} {}",
                r.name,
                r.cosmo_status,
                r.obj_cosmo,
                r.obj_clar,
                r.rel_obj,
                r.dx,
                r.rp,
                r.rd,
                r.iter,
                r.note
            );
        }
        reports.push(r);
    }
    (n_ok, n_clar_solved, reports)
}

#[test]
fn clarabel_textbook_examples() {
    for (name, p, ot, xt) in [
        ("qp", clarabel_qp(), 1e-4, 1e-3),
        ("lp", clarabel_lp(), 1e-4, 1e-3),
        ("socp", clarabel_socp(), 1e-3, 1e-2),
        ("power", clarabel_power(), 8e-2, 2e-1),
        ("exp5", exp_max_x(5.0), 8e-2, 2e-1),
    ] {
        let (P, q, A, b, cones) = p;
        let mut s = default_test_settings();
        s.max_iter = 15_000;
        let r = compare_report(name, &P, &q, &A, &b, cones, s, ot, xt);
        assert!(r.ok, "{} failed: {}", r.name, r.note);
    }
}

#[test]
fn extensive_clarabel_correctness() {
    let cases = all_cases();
    let n = cases.len();
    let (n_ok, n_clar, reports) = run_suite(&cases);
    let n_fail = n - n_ok;
    let solved_both = reports
        .iter()
        .filter(|r| r.clar_solved && r.cosmo_status == SolverStatus::Solved)
        .count();
    let mut rels: Vec<f64> = reports
        .iter()
        .filter(|r| r.clar_solved && r.ok)
        .map(|r| r.rel_obj)
        .collect();
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = rels.get(rels.len() / 2).copied().unwrap_or(0.0);
    let p90 = rels.get(rels.len() * 9 / 10).copied().unwrap_or(0.0);
    let mut families: Vec<(&str, usize, usize)> = Vec::new();
    for prefix in [
        "clarabel_",
        "exp_",
        "lp_",
        "qp_",
        "eqqp_",
        "boxqp_",
        "uncqp_",
        "socp_",
        "lsbox_",
        "mixed_",
        "poorly_",
        "primal_",
        "dual_",
        "inconsistent_",
    ] {
        let subset: Vec<_> = reports
            .iter()
            .filter(|r| r.name.starts_with(prefix))
            .collect();
        if subset.is_empty() {
            continue;
        }
        let ok = subset.iter().filter(|r| r.ok).count();
        families.push((prefix, ok, subset.len()));
    }
    eprintln!(
        "Clarabel benchmark: {n} problems, {n_clar} Clarabel-solved, {solved_both} both Solved, {n_ok} ok, {n_fail} fail; rel_obj p50={p50:.3e} p90={p90:.3e}"
    );
    for (prefix, ok, tot) in families {
        eprintln!("  family {prefix:<16} {ok}/{tot} ok");
    }
    assert_eq!(
        n_fail, 0,
        "{n_fail}/{n} problems disagreed with Clarabel.rs"
    );
}

#[test]
fn reuse_matches_fresh_and_clarabel() {
    let (P, q, A, b, cones) = feasible_qp(42, 8, 5);
    let mut s = default_test_settings();
    s.max_iter = 12_000;
    let mut reused = CosmoSolver::new(&P, &q, &A, &b, cones.clone(), s.clone()).unwrap();
    let a = reused.solve().unwrap().clone();
    let b1 = reused.solve().unwrap().clone();
    assert_eq!(a.status, SolverStatus::Solved);
    assert_eq!(b1.status, SolverStatus::Solved);
    assert!((a.obj_val - b1.obj_val).abs() < 1e-6);

    let mut q2 = q.clone();
    for qi in q2.iter_mut() {
        *qi *= 1.1;
    }
    reused.update_q(&q2).unwrap();
    let c = reused.solve().unwrap().clone();
    let mut fresh = CosmoSolver::new(&P, &q2, &A, &b, cones.clone(), s.clone()).unwrap();
    let d = fresh.solve().unwrap().clone();
    assert_eq!(c.status, SolverStatus::Solved);
    assert_eq!(d.status, SolverStatus::Solved);
    assert!((c.obj_val - d.obj_val).abs() < 5e-5);
    compare_to_clarabel(&P, &q2, &A, &b, cones, s, 2e-3, 8e-2);
}

#[test]
fn warm_start_matches_clarabel() {
    let (P, q, A, b, cones) = feasible_qp(9, 6, 4);
    let mut s = default_test_settings();
    s.max_iter = 12_000;
    let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones.clone(), s.clone()).unwrap();
    let s1 = solver.solve().unwrap().clone();
    solver.warm_start(Some(&s1.x), Some(&s1.y)).unwrap();
    let s2 = solver.solve().unwrap().clone();
    assert_eq!(s2.status, SolverStatus::Solved);
    assert!((s1.obj_val - s2.obj_val).abs() < 1e-4);
    compare_to_clarabel(&P, &q, &A, &b, cones, s, 2e-3, 8e-2);
}
