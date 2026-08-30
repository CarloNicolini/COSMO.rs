//! Stress cases: Clarabel.rs often converges where ADMM (COSMO) struggles.
//!
//! Also re-checks Clarabel textbook problems and COSMO.jl example ports.
//!
//! Run with:
//! ```bash
//! cargo test --test stress_vs_clarabel -- --nocapture
//! ```

#![allow(non_snake_case)]

mod common;
use common::*;
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings, SolverStatus};

type Problem = (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>);

fn osqp_qp() -> Problem {
    // COSMO.jl examples/qp.jl  /  Clarabel basic_qp_data
    let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
    let q = vec![1.0, 1.0];
    let A = CscMatrix::from(&[
        [-1.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]);
    let b = vec![-1.0, 0.0, 0.0, 1.0, 0.7, 0.7];
    (P, q, A, b, vec![Cone::nonnegative(6)])
}

fn cosmo_jl_lp() -> Problem {
    // COSMO.jl examples/lp.jl → Clarabel form Ax + s = b
    let P = CscMatrix::<f64>::zeros((4, 4));
    let q = vec![1.0, 2.0, 3.0, 4.0];
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
    (P, q, A, b, vec![Cone::nonnegative(10)])
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

fn clarabel_exp() -> Problem {
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
    (P, q, A, b, vec![Cone::exponential(), Cone::zero(2)])
}

fn box_qp_cosmo_jl() -> Problem {
    // COSMO.jl qp-box.jl: min ½‖x‖² + [1,-1]ᵀx, x ∈ [0,1]² → obj ≈ -0.5
    let P = CscMatrix::identity(2);
    let q = vec![1.0, -1.0];
    let A = CscMatrix::from(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
    let b = vec![1.0, 1.0, 0.0, 0.0];
    (P, q, A, b, vec![Cone::nonnegative(4)])
}

/// Scale all nonzero entries of A by `factor` (extreme dynamic range).
fn scale_a(p: Problem, factor: f64) -> Problem {
    let (P, q, mut A, mut b, cones) = p;
    for v in A.nzval.iter_mut() {
        *v *= factor;
    }
    for v in b.iter_mut() {
        *v *= factor;
    }
    (P, q, A, b, cones)
}

fn scale_q(p: Problem, factor: f64) -> Problem {
    let (P, mut q, A, b, cones) = p;
    for v in q.iter_mut() {
        *v *= factor;
    }
    (P, q, A, b, cones)
}

fn scale_p_diag(p: Problem, factor: f64) -> Problem {
    let (mut P, q, A, b, cones) = p;
    for col in 0..P.n {
        for j in P.colptr[col]..P.colptr[col + 1] {
            if P.rowval[j] == col {
                P.nzval[j] *= factor;
            }
        }
    }
    (P, q, A, b, cones)
}

fn near_singular_eq() -> Problem {
    // Ill-conditioned equalities: rows nearly parallel.
    let P = CscMatrix::identity(2);
    let q = vec![-1.0, -1.0];
    let eps = 1e-10;
    let A = CscMatrix::from(&[[1.0, 1.0], [1.0 + eps, 1.0]]);
    let b = vec![1.0, 1.0 + eps];
    (P, q, A, b, vec![Cone::zero(2)])
}

fn huge_exp() -> Problem {
    // max x s.t. y·exp(x/y) ≤ z, y=1, z=exp(12)
    let P = CscMatrix::<f64>::zeros((3, 3));
    let q = vec![-1.0, 0.0, 0.0];
    let A = CscMatrix::from(&[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let b = vec![0.0, 0.0, 0.0, 1.0, 12f64.exp()];
    (P, q, A, b, vec![Cone::exponential(), Cone::zero(2)])
}

fn large_soc(dim: usize) -> Problem {
    let n = dim - 1;
    let q: Vec<f64> = (0..n).map(|i| if i == 0 { -1.0 } else { 0.1 }).collect();
    let mut rows = vec![vec![0.0; n]; dim];
    for i in 0..n {
        rows[i + 1][i] = -1.0;
    }
    let mut b = vec![0.0; dim];
    b[0] = 10.0;
    (
        CscMatrix::<f64>::zeros((n, n)),
        q,
        dense_to_csc(&rows),
        b,
        vec![Cone::second_order(dim)],
    )
}

fn tiny_rho_settings() -> Settings {
    let mut s = default_test_settings();
    s.rho = 1e-6;
    s.adaptive_rho = false;
    s.accelerate = false;
    s.max_iter = 2_000;
    s.eps_abs = 1e-6;
    s.eps_rel = 1e-6;
    s
}

fn tight_tol_settings() -> Settings {
    let mut s = default_test_settings();
    s.eps_abs = 1e-8;
    s.eps_rel = 1e-8;
    s.max_iter = 3_000;
    s
}

fn no_scaling_settings() -> Settings {
    let mut s = default_test_settings();
    s.scaling = 0;
    s.max_iter = 5_000;
    s
}

struct StressCase {
    name: &'static str,
    problem: Problem,
    settings: Settings,
    /// If true, COSMO is *expected* to struggle (documented gap).
    expect_cosmo_hard: bool,
}

fn textbook_cases() -> Vec<StressCase> {
    let s = default_test_settings();
    let mut s_long = default_test_settings();
    s_long.max_iter = 15_000;
    vec![
        StressCase {
            name: "cosmo_jl_qp",
            problem: osqp_qp(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "cosmo_jl_lp",
            problem: cosmo_jl_lp(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "cosmo_jl_box_qp",
            problem: box_qp_cosmo_jl(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "clarabel_qp",
            problem: clarabel_qp(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "clarabel_lp",
            problem: clarabel_lp(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "clarabel_socp",
            problem: clarabel_socp(),
            settings: s.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "clarabel_exp",
            problem: clarabel_exp(),
            settings: s_long.clone(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "clarabel_power",
            problem: clarabel_power(),
            settings: s_long,
            expect_cosmo_hard: false,
        },
    ]
}

fn stress_cases() -> Vec<StressCase> {
    vec![
        // Documented gaps: Clarabel Solved, COSMO MaxIter / wrong obj.
        StressCase {
            name: "scale_A_1e8",
            problem: scale_a(osqp_qp(), 1e8),
            settings: default_test_settings(),
            expect_cosmo_hard: true,
        },
        StressCase {
            name: "scale_A_1e-8",
            problem: scale_a(osqp_qp(), 1e-8),
            settings: default_test_settings(),
            expect_cosmo_hard: true,
        },
        StressCase {
            name: "tiny_rho_no_adapt_qp",
            problem: osqp_qp(),
            settings: tiny_rho_settings(),
            expect_cosmo_hard: true,
        },
        // Clarabel also struggles (InsufficientProgress) — not a COSMO-only gap.
        StressCase {
            name: "scale_P_1e-12",
            problem: scale_p_diag(osqp_qp(), 1e-12),
            settings: default_test_settings(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "scale_A_1e12_no_ruiz",
            problem: scale_a(osqp_qp(), 1e12),
            settings: no_scaling_settings(),
            expect_cosmo_hard: false,
        },
        // These look stressful but currently agree with Clarabel.
        StressCase {
            name: "scale_q_1e6",
            problem: scale_q(osqp_qp(), 1e6),
            settings: default_test_settings(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "near_singular_eq",
            problem: near_singular_eq(),
            settings: default_test_settings(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "huge_exp_12",
            problem: huge_exp(),
            settings: {
                let mut s = default_test_settings();
                s.max_iter = 15_000;
                s
            },
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "large_soc_64",
            problem: large_soc(64),
            settings: tight_tol_settings(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "tight_tol_1e8_qp",
            problem: osqp_qp(),
            settings: tight_tol_settings(),
            expect_cosmo_hard: false,
        },
        StressCase {
            name: "illcond_A_eps_on_qp",
            problem: {
                let (P, q, mut A, b, cones) = osqp_qp();
                if !A.nzval.is_empty() {
                    A.nzval[0] *= f64::EPSILON;
                }
                (P, q, A, b, cones)
            },
            settings: default_test_settings(),
            expect_cosmo_hard: false,
        },
    ]
}

fn run_one(c: &StressCase) -> (CompareReport, cosmo::Solution) {
    let (P, q, A, b, cones) = &c.problem;
    let mut solver = CosmoSolver::new(P, q, A, b, cones.clone(), c.settings.clone()).unwrap();
    let sol = solver.solve().unwrap().clone();
    let r = compare_report(
        c.name,
        P,
        q,
        A,
        b,
        cones.clone(),
        c.settings.clone(),
        5e-2,
        2e-1,
    );
    (r, sol)
}

#[test]
fn clarabel_and_cosmo_jl_examples_agree() {
    let mut failures = Vec::new();
    for c in textbook_cases() {
        let (r, sol) = run_one(&c);

        // Golden checks for known solutions.
        match c.name {
            "cosmo_jl_qp" => {
                if sol.status == SolverStatus::Solved {
                    assert!((sol.x[0] - 0.3).abs() < 2e-2, "x0={}", sol.x[0]);
                    assert!((sol.x[1] - 0.7).abs() < 2e-2, "x1={}", sol.x[1]);
                    assert!((sol.obj_val - 1.88).abs() < 2e-2, "obj={}", sol.obj_val);
                }
            }
            "cosmo_jl_lp" => {
                if sol.status == SolverStatus::Solved {
                    assert!((sol.obj_val - 20.0).abs() < 5e-2, "obj={}", sol.obj_val);
                }
            }
            "cosmo_jl_box_qp" => {
                if sol.status == SolverStatus::Solved {
                    assert!((sol.obj_val + 0.5).abs() < 5e-2, "obj={}", sol.obj_val);
                }
            }
            "clarabel_exp" => {
                if sol.status == SolverStatus::Solved {
                    assert!((sol.obj_val + 5.0).abs() < 8e-2, "obj={}", sol.obj_val);
                }
            }
            "clarabel_power" => {
                if sol.status == SolverStatus::Solved {
                    assert!((sol.obj_val + 1.8458).abs() < 8e-2, "obj={}", sol.obj_val);
                }
            }
            _ => {}
        }

        eprintln!(
            "OK?={} {} cosmo={:?} clar_solved={} obj={:.6} vs {:.6} rel={:.2e} dx={:.2e} iter={} {}",
            r.ok,
            r.name,
            r.cosmo_status,
            r.clar_solved,
            r.obj_cosmo,
            r.obj_clar,
            r.rel_obj,
            r.dx,
            r.iter,
            r.note
        );
        if !r.ok {
            failures.push(r.name.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "textbook examples failed vs Clarabel: {failures:?}"
    );
}

#[test]
fn stress_cosmo_fail_while_clarabel_ok() {
    let mut gaps = Vec::new();
    let mut unexpected = Vec::new();
    let mut recovered = Vec::new();

    for c in stress_cases() {
        let (r, _) = run_one(&c);
        let clar_ok = r.clar_solved;
        let cosmo_ok = r.ok && r.cosmo_status == SolverStatus::Solved;
        let gap = clar_ok && !cosmo_ok;

        eprintln!(
            "stress {} | clar_ok={} cosmo_ok={} gap={} | status={:?} rel={:.2e} dx={:.2e} rp={:.2e} rd={:.2e} iter={} | {}",
            r.name,
            clar_ok,
            cosmo_ok,
            gap,
            r.cosmo_status,
            r.rel_obj,
            r.dx,
            r.rp,
            r.rd,
            r.iter,
            r.note
        );

        if gap {
            gaps.push(r.name.clone());
            if !c.expect_cosmo_hard {
                unexpected.push(r.name.clone());
            }
        } else if c.expect_cosmo_hard && clar_ok && cosmo_ok {
            recovered.push(r.name.clone());
        }
    }

    eprintln!("\n=== COSMO fail / Clarabel ok ({}) ===", gaps.len());
    for g in &gaps {
        eprintln!("  - {g}");
    }
    if !recovered.is_empty() {
        eprintln!("Note: previously hard cases now agree (good): {recovered:?}");
    }

    assert!(
        unexpected.is_empty(),
        "unexpected COSMO failure on easy/tight-tol cases: {unexpected:?}; known gaps: {gaps:?}"
    );
    // Ensure the stress suite actually exercises Clarabel-succeeds-COSMO-struggles territory.
    assert!(
        !gaps.is_empty(),
        "expected at least one documented stress gap (Clarabel ok, COSMO fail); none found — update expect_cosmo_hard flags"
    );
}
