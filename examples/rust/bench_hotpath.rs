//! Quick release timing harness for ADMM hot-path comparison (not a criterion bench).
#![allow(non_snake_case)]

use cosmo::{Cone, CosmoSolver, CscMatrix, Settings};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn dense_to_csc(rows: &[Vec<f64>]) -> CscMatrix {
    let m = rows.len();
    let n = rows[0].len();
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

fn random_psd_triu(rng: &mut ChaCha8Rng, n: usize) -> CscMatrix {
    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let u1 = rng.gen::<f64>().max(1e-12);
            let u2 = rng.gen::<f64>();
            b[i][j] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
    let mut tri = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.25;
            for k in 0..n {
                s += b[k][i] * b[k][j];
            }
            tri[i][j] = s;
        }
    }
    dense_to_csc(&tri)
}

fn random_sparse_a(rng: &mut ChaCha8Rng, m: usize, n: usize, density: f64) -> CscMatrix {
    let mut rows = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            if rng.gen::<f64>() < density {
                let u1 = rng.gen::<f64>().max(1e-12);
                let u2 = rng.gen::<f64>();
                rows[i][j] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
        if rows[i].iter().all(|&v| v == 0.0) {
            rows[i][rng.gen_range(0..n)] = 1.0;
        }
    }
    dense_to_csc(&rows)
}

fn feasible_qp(
    seed: u64,
    m: usize,
    n: usize,
) -> (CscMatrix, Vec<f64>, CscMatrix, Vec<f64>, Vec<Cone>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let P = random_psd_triu(&mut rng, n);
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

fn main() {
    let sizes = [(40usize, 25usize), (80, 40), (120, 60)];
    let seeds = 1u64..=8;
    let mut settings = Settings::default();
    settings.verbose = false;
    settings.max_iter = 12_000;
    settings.eps_abs = 1e-5;
    settings.eps_rel = 1e-5;

    // warmup
    {
        let (P, q, A, b, cones) = feasible_qp(1, 20, 12);
        let mut s = CosmoSolver::new(&P, &q, &A, &b, cones, settings.clone()).unwrap();
        let _ = s.solve().unwrap();
    }

    let mut total_ms = 0.0;
    let mut total_iter = 0usize;
    let mut n_ok = 0usize;
    let t_all = Instant::now();
    for &(m, n) in &sizes {
        for seed in seeds.clone() {
            let (P, q, A, b, cones) = feasible_qp(seed + (m as u64) * 1000, m, n);
            let mut solver = CosmoSolver::new(&P, &q, &A, &b, cones, settings.clone()).unwrap();
            let t0 = Instant::now();
            let sol = solver.solve().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            total_ms += ms;
            total_iter += sol.iter;
            if sol.status.is_solved() {
                n_ok += 1;
            }
            println!(
                "qp_{m}x{n}_s{seed} status={} iter={} time_ms={:.3} obj={:.6e}",
                sol.status, sol.iter, ms, sol.obj_val
            );
        }
    }
    let wall = t_all.elapsed().as_secs_f64() * 1e3;
    println!(
        "SUMMARY solved={n_ok}/{} sum_ms={total_ms:.3} wall_ms={wall:.3} total_iter={total_iter}",
        sizes.len() * 8
    );
}
