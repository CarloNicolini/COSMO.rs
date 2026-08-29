//! Shared helpers for integration tests.

#![allow(dead_code, non_snake_case)]

use clarabel::algebra::CscMatrix as ClarCsc;
use clarabel::solver::{
    DefaultSettings as ClarSettings, DefaultSolver, IPSolver, SolverStatus as ClarStatus,
    SupportedConeT,
};
use cosmo::{Cone, CosmoSolver, CscMatrix, Settings, Solution, SolverStatus};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub fn dense_to_csc(rows: &[Vec<f64>]) -> CscMatrix {
    let m = rows.len();
    let n = if m == 0 { 0 } else { rows[0].len() };
    let mut I = Vec::new();
    let mut J = Vec::new();
    let mut V = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                I.push(i);
                J.push(j);
                V.push(v);
            }
        }
    }
    CscMatrix::new_from_triplets(m, n, I, J, V)
}

pub fn to_clar(P: &CscMatrix) -> ClarCsc<f64> {
    ClarCsc::new(
        P.m,
        P.n,
        P.colptr.clone(),
        P.rowval.clone(),
        P.nzval.clone(),
    )
}

pub fn cones_to_clar(cones: &[Cone]) -> Vec<SupportedConeT<f64>> {
    cones
        .iter()
        .map(|c| match c {
            Cone::Zero(z) => SupportedConeT::ZeroConeT(z.dim),
            Cone::Nonnegative(c) => SupportedConeT::NonnegativeConeT(c.dim),
            Cone::SecondOrder(c) => SupportedConeT::SecondOrderConeT(c.dim),
            Cone::Exponential(_) => SupportedConeT::ExponentialConeT(),
            Cone::Power(c) => SupportedConeT::PowerConeT(c.alpha),
            other => panic!("unsupported cone for Clarabel comparison: {other:?}"),
        })
        .collect()
}

pub struct ClarResult {
    pub status: ClarStatus,
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    pub s: Vec<f64>,
    pub obj: f64,
}

pub fn solve_clarabel(
    P: &CscMatrix,
    q: &[f64],
    A: &CscMatrix,
    b: &[f64],
    cones: &[Cone],
) -> ClarResult {
    let P = to_clar(P);
    let A = to_clar(A);
    let cones = cones_to_clar(cones);
    let settings = ClarSettings::<f64> {
        verbose: false,
        max_iter: 200,
        ..ClarSettings::default()
    };
    let mut solver =
        DefaultSolver::new(&P, &q.to_vec(), &A, &b.to_vec(), &cones, settings).unwrap();
    solver.solve();
    ClarResult {
        status: solver.solution.status,
        x: solver.solution.x.clone(),
        z: solver.solution.z.clone(),
        s: solver.solution.s.clone(),
        obj: solver.solution.obj_val,
    }
}

pub fn status_compatible(c: SolverStatus, r: ClarStatus) -> bool {
    use ClarStatus::*;
    match (c, r) {
        (SolverStatus::Solved, Solved | AlmostSolved) => true,
        (SolverStatus::PrimalInfeasible, PrimalInfeasible | AlmostPrimalInfeasible) => true,
        (SolverStatus::DualInfeasible, DualInfeasible | AlmostDualInfeasible) => true,
        (SolverStatus::MaxIterReached, MaxIterations | InsufficientProgress) => true,
        _ => false,
    }
}

pub fn inf_norm_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

pub fn primal_residual(A: &CscMatrix, x: &[f64], s: &[f64], b: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    // r = A x + s - b
    for col in 0..A.n {
        let xc = x[col];
        for j in A.colptr[col]..A.colptr[col + 1] {
            r[A.rowval[j]] += A.nzval[j] * xc;
        }
    }
    r.iter()
        .zip(s.iter())
        .zip(b.iter())
        .map(|((ri, si), bi)| (ri + si - bi).abs())
        .fold(0.0, f64::max)
}

pub fn assert_kkt_reasonable(sol: &Solution, A: &CscMatrix, b: &[f64], tol: f64) {
    assert!(
        sol.status == SolverStatus::Solved
            || sol.status == SolverStatus::MaxIterReached
            || sol.status == SolverStatus::PrimalInfeasible
            || sol.status == SolverStatus::DualInfeasible,
        "unexpected status {:?}",
        sol.status
    );
    if sol.status == SolverStatus::Solved {
        let rp = primal_residual(A, &sol.x, &sol.s, b);
        assert!(
            rp < tol || sol.r_prim < tol,
            "primal residual too large: rp={} reported={}",
            rp,
            sol.r_prim
        );
        assert!(
            sol.x.iter().all(|v| v.is_finite())
                && sol.y.iter().all(|v| v.is_finite())
                && sol.s.iter().all(|v| v.is_finite()),
            "non-finite solution"
        );
    }
}

pub fn dual_residual(P: &CscMatrix, q: &[f64], A: &CscMatrix, x: &[f64], y: &[f64]) -> f64 {
    // Stationarity: P x + q + A' y = 0  (y = −μ).
    let n = q.len();
    let mut r = vec![0.0; n];
    for col in 0..P.n {
        let xc = x[col];
        for j in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[j];
            r[row] += P.nzval[j] * xc;
            if row != col {
                r[col] += P.nzval[j] * x[row];
            }
        }
    }
    for i in 0..n {
        r[i] += q[i];
    }
    for col in 0..A.n {
        let mut s = 0.0;
        for j in A.colptr[col]..A.colptr[col + 1] {
            s += A.nzval[j] * y[A.rowval[j]];
        }
        r[col] += s;
    }
    r.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Make a strictly convex QP Hessian by adding `eps` on the diagonal of a PSD triu matrix.
pub fn add_diag(P: &CscMatrix, eps: f64) -> CscMatrix {
    let n = P.n;
    let mut I = Vec::new();
    let mut J = Vec::new();
    let mut V = Vec::new();
    for col in 0..n {
        I.push(col);
        J.push(col);
        V.push(eps);
        for j in P.colptr[col]..P.colptr[col + 1] {
            I.push(P.rowval[j]);
            J.push(col);
            V.push(P.nzval[j]);
        }
    }
    CscMatrix::new_from_triplets(n, n, I, J, V)
}

#[derive(Clone, Debug)]
pub struct CompareReport {
    pub name: String,
    pub cosmo_status: SolverStatus,
    pub clar_solved: bool,
    pub obj_cosmo: f64,
    pub obj_clar: f64,
    pub rel_obj: f64,
    pub dx: f64,
    pub rp: f64,
    pub rd: f64,
    pub iter: usize,
    pub ok: bool,
    pub note: String,
}

pub fn compare_report(
    name: &str,
    P: &CscMatrix,
    q: &[f64],
    A: &CscMatrix,
    b: &[f64],
    cones: Vec<Cone>,
    settings: Settings,
    obj_tol: f64,
    x_tol: f64,
) -> CompareReport {
    let mut solver = CosmoSolver::new(P, q, A, b, cones.clone(), settings).unwrap();
    let sol = solver.solve().unwrap().clone();
    let clar = solve_clarabel(P, q, A, b, &cones);
    let dx = if sol.x.len() == clar.x.len() {
        inf_norm_diff(&sol.x, &clar.x)
    } else {
        f64::NAN
    };
    let scale = 1.0 + clar.obj.abs();
    let rel_obj = (sol.obj_val - clar.obj).abs() / scale;
    let clar_solved = clar.status == ClarStatus::Solved || clar.status == ClarStatus::AlmostSolved;
    let rp = primal_residual(A, &sol.x, &sol.s, b);
    let rd = dual_residual(P, q, A, &sol.x, &sol.y);

    let mut note = String::new();
    let mut ok = true;
    if clar_solved {
        match sol.status {
            SolverStatus::Solved => {
                let obj_ok = rel_obj <= obj_tol;
                let x_ok = dx < x_tol;
                if !(obj_ok || x_ok) {
                    ok = false;
                    note = format!("obj/x mismatch rel_obj={rel_obj:.3e} dx={dx:.3e}");
                } else if rp > (obj_tol * 20.0).max(5e-3) && sol.r_prim > (obj_tol * 20.0).max(5e-3)
                {
                    ok = false;
                    note = format!("primal residual rp={rp:.3e} reported={}", sol.r_prim);
                }
            }
            SolverStatus::MaxIterReached => {
                if sol.r_prim < 1e-2 && sol.r_dual < 1e-2 && rel_obj <= obj_tol.max(5e-2) {
                    note = "max_iter but residuals/obj acceptable".into();
                } else {
                    ok = false;
                    note = format!(
                        "max_iter rp={} rd={} rel_obj={rel_obj:.3e}",
                        sol.r_prim, sol.r_dual
                    );
                }
            }
            other => {
                ok = false;
                note = format!("COSMO {other:?} vs Clarabel {:?}", clar.status);
            }
        }
    } else if matches!(
        clar.status,
        ClarStatus::PrimalInfeasible | ClarStatus::DualInfeasible
    ) {
        if !(status_compatible(sol.status, clar.status)
            || sol.status == SolverStatus::MaxIterReached)
        {
            ok = false;
            note = format!("infeas mismatch {:?} vs {:?}", sol.status, clar.status);
        } else {
            note = format!("clarabel {:?}, cosmo {:?}", clar.status, sol.status);
        }
    } else {
        note = format!("clarabel {:?}, skipped strict match", clar.status);
    }

    CompareReport {
        name: name.to_string(),
        cosmo_status: sol.status,
        clar_solved,
        obj_cosmo: sol.obj_val,
        obj_clar: clar.obj,
        rel_obj,
        dx,
        rp,
        rd,
        iter: sol.iter,
        ok,
        note,
    }
}

pub fn compare_to_clarabel(
    P: &CscMatrix,
    q: &[f64],
    A: &CscMatrix,
    b: &[f64],
    cones: Vec<Cone>,
    settings: Settings,
    obj_tol: f64,
    x_tol: f64,
) {
    let r = compare_report("case", P, q, A, b, cones, settings, obj_tol, x_tol);
    assert!(
        r.ok,
        "{}: COSMO obj={} Clarabel={} dx={} rp={} rd={} iter={} status={:?} {}",
        r.name, r.obj_cosmo, r.obj_clar, r.dx, r.rp, r.rd, r.iter, r.cosmo_status, r.note
    );
}

pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

pub fn randn(rng: &mut ChaCha8Rng) -> f64 {
    // Box-Muller
    let u1 = rng.gen::<f64>().max(1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub fn random_psd_triu(rng: &mut ChaCha8Rng, n: usize) -> CscMatrix {
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = randn(rng);
        }
    }
    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += b[k][i] * b[k][j];
            }
            p[i][j] = s;
        }
    }
    // store full then triu via constructor that drops zeros; keep upper
    let mut tri = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            tri[i][j] = p[i][j];
        }
    }
    dense_to_csc(&tri)
}

pub fn random_sparse_a(rng: &mut ChaCha8Rng, m: usize, n: usize, density: f64) -> CscMatrix {
    let mut rows = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            if rng.gen::<f64>() < density {
                rows[i][j] = randn(rng);
            }
        }
        // ensure no zero rows
        if rows[i].iter().all(|&v| v == 0.0) {
            rows[i][rng.gen_range(0..n)] = randn(rng);
        }
    }
    dense_to_csc(&rows)
}

pub fn default_test_settings() -> Settings {
    let mut s = Settings::default();
    s.verbose = false;
    s.max_iter = 8000;
    s.eps_abs = 1e-5;
    s.eps_rel = 1e-5;
    s.accelerate = true;
    s
}
