//! Modified Ruiz equilibration (COSMO.jl `scale_ruiz!`).

#![allow(non_snake_case)]

use crate::algebra::{clip, CscMatrix, MatrixMath, MatrixMathMut};
use crate::cones::CompositeCone;
use crate::settings::Settings;

#[derive(Clone, Debug)]
pub struct ScaleMatrices {
    pub D: Vec<f64>,
    pub Dinv: Vec<f64>,
    pub E: Vec<f64>,
    pub Einv: Vec<f64>,
    pub c: f64,
    pub cinv: f64,
    pub enabled: bool,
}

impl ScaleMatrices {
    pub fn identity(m: usize, n: usize) -> Self {
        Self {
            D: vec![1.0; n],
            Dinv: vec![1.0; n],
            E: vec![1.0; m],
            Einv: vec![1.0; m],
            c: 1.0,
            cinv: 1.0,
            enabled: false,
        }
    }
}

pub fn scale_ruiz(
    P: &mut CscMatrix<f64>,
    A: &mut CscMatrix<f64>,
    q: &mut [f64],
    b: &mut [f64],
    cones: &CompositeCone,
    settings: &Settings,
) -> ScaleMatrices {
    let n = P.n;
    let m = A.m;
    let mut sm = ScaleMatrices::identity(m, n);
    if settings.scaling == 0 {
        return sm;
    }
    sm.enabled = true;

    let mut Dwork = vec![0.0; n];
    let mut Ework = vec![0.0; m];

    for _ in 0..settings.scaling {
        kkt_col_norms(P, A, &mut Dwork, &mut Ework);
        limit_scaling(&mut Dwork, settings);
        limit_scaling(&mut Ework, settings);
        inv_sqrt(&mut Dwork);
        inv_sqrt(&mut Ework);
        scale_data(P, A, q, b, &Dwork, &Ework, 1.0);
        for i in 0..n {
            sm.D[i] *= Dwork[i];
        }
        for i in 0..m {
            sm.E[i] *= Ework[i];
        }

        P.col_norms_sym(&mut Dwork);
        let mean_col = if n == 0 {
            0.0
        } else {
            Dwork.iter().sum::<f64>() / n as f64
        };
        let inf_q = q.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        if mean_col != 0.0 && inf_q != 0.0 {
            let inf_q = clip(inf_q, settings.min_scaling, settings.max_scaling, 1.0, 1.0);
            let mut scale_cost = inf_q.max(mean_col);
            scale_cost = clip(
                scale_cost,
                settings.min_scaling,
                settings.max_scaling,
                1.0,
                1.0,
            );
            let ctmp = 1.0 / scale_cost;
            P.scale(ctmp);
            for qi in q.iter_mut() {
                *qi *= ctmp;
            }
            sm.c *= ctmp;
        }
    }

    Ework.fill(1.0);
    if cones.rectify_scaling(&sm.E, &mut Ework) {
        scale_data(P, A, q, b, &vec![1.0; n], &Ework, 1.0);
        for i in 0..m {
            sm.E[i] *= Ework[i];
        }
    }

    for i in 0..n {
        sm.Dinv[i] = 1.0 / sm.D[i];
    }
    for i in 0..m {
        sm.Einv[i] = 1.0 / sm.E[i];
    }
    sm.cinv = 1.0 / sm.c;
    sm
}

fn kkt_col_norms(P: &CscMatrix<f64>, A: &CscMatrix<f64>, lhs: &mut [f64], rhs: &mut [f64]) {
    P.col_norms_sym(lhs);
    A.col_norms_no_reset(lhs);
    A.row_norms(rhs);
}

fn limit_scaling(s: &mut [f64], set: &Settings) {
    for si in s.iter_mut() {
        *si = clip(*si, set.min_scaling, set.max_scaling, 1.0, 1.0);
    }
}

fn inv_sqrt(a: &mut [f64]) {
    for ai in a.iter_mut() {
        *ai = 1.0 / ai.sqrt();
    }
}

fn scale_data(
    P: &mut CscMatrix<f64>,
    A: &mut CscMatrix<f64>,
    q: &mut [f64],
    b: &mut [f64],
    Ds: &[f64],
    Es: &[f64],
    cs: f64,
) {
    P.lrscale(Ds, Ds);
    A.lrscale(Es, Ds);
    for (qi, &d) in q.iter_mut().zip(Ds.iter()) {
        *qi *= d;
    }
    for (bi, &e) in b.iter_mut().zip(Es.iter()) {
        *bi *= e;
    }
    if cs != 1.0 {
        P.scale(cs);
        for qi in q.iter_mut() {
            *qi *= cs;
        }
    }
}

pub fn scale_variables(
    x: &mut [f64],
    mu: &mut [f64],
    s: &mut [f64],
    Dinv: &[f64],
    Einv: &[f64],
    E: &[f64],
    c: f64,
) {
    for (xi, &d) in x.iter_mut().zip(Dinv.iter()) {
        *xi *= d;
    }
    for i in 0..mu.len() {
        mu[i] *= Einv[i] * c;
        s[i] *= E[i];
    }
}

pub fn reverse_scaling(x: &mut [f64], mu: &mut [f64], s: &mut [f64], sm: &ScaleMatrices) {
    if !sm.enabled {
        return;
    }
    for (xi, &d) in x.iter_mut().zip(sm.D.iter()) {
        *xi *= d;
    }
    for i in 0..s.len() {
        s[i] *= sm.Einv[i];
        mu[i] *= sm.E[i] * sm.cinv;
    }
}
