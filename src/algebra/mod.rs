//! Sparse CSC operations used by the COSMO ADMM iteration.
//!
//! [`CscMatrix`] is reused from Clarabel.rs. Clarabel's `gemv`/`symv` traits
//! are crate-private, so the kernels COSMO needs are implemented here on top of
//! the public CSC storage.

#![allow(non_snake_case)]

pub use clarabel::algebra::{
    CscMatrix, MatrixMath, MatrixMathMut, ScalarMath, TriangularMatrixChecks, VectorMath,
};

/// y := a * A * x + b * y  (A is m-by-n CSC)
pub fn gemv(A: &CscMatrix<f64>, y: &mut [f64], x: &[f64], a: f64, b: f64) {
    debug_assert_eq!(y.len(), A.m);
    debug_assert_eq!(x.len(), A.n);
    if b == 0.0 {
        y.fill(0.0);
    } else if b != 1.0 {
        for yi in y.iter_mut() {
            *yi *= b;
        }
    }
    if a == 0.0 {
        return;
    }
    for col in 0..A.n {
        let xc = x[col];
        if xc == 0.0 {
            continue;
        }
        let ax = a * xc;
        for j in A.colptr[col]..A.colptr[col + 1] {
            y[A.rowval[j]] += A.nzval[j] * ax;
        }
    }
}

/// y := a * A' * x + b * y  (A is m-by-n CSC, y is n, x is m)
pub fn gemv_t(A: &CscMatrix<f64>, y: &mut [f64], x: &[f64], a: f64, b: f64) {
    debug_assert_eq!(y.len(), A.n);
    debug_assert_eq!(x.len(), A.m);
    for col in 0..A.n {
        let mut s = 0.0;
        for j in A.colptr[col]..A.colptr[col + 1] {
            s += A.nzval[j] * x[A.rowval[j]];
        }
        y[col] = a * s + b * y[col];
    }
}

/// y := a * P * x + b * y for symmetric P stored as the upper triangle.
pub fn symv(P: &CscMatrix<f64>, y: &mut [f64], x: &[f64], a: f64, b: f64) {
    debug_assert_eq!(P.m, P.n);
    debug_assert_eq!(y.len(), P.n);
    debug_assert_eq!(x.len(), P.n);
    if b == 0.0 {
        y.fill(0.0);
    } else if b != 1.0 {
        for yi in y.iter_mut() {
            *yi *= b;
        }
    }
    if a == 0.0 {
        return;
    }
    for col in 0..P.n {
        let xc = x[col];
        for j in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[j];
            let pij = P.nzval[j];
            y[row] += a * pij * xc;
            if row != col {
                y[col] += a * pij * x[row];
            }
        }
    }
}

/// x' P x for symmetric P stored as the upper triangle.
pub fn quad_form(P: &CscMatrix<f64>, x: &[f64]) -> f64 {
    debug_assert_eq!(P.m, P.n);
    debug_assert_eq!(x.len(), P.n);
    let mut q = 0.0;
    for col in 0..P.n {
        let xc = x[col];
        for j in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[j];
            let pij = P.nzval[j];
            if row == col {
                q += pij * xc * xc;
            } else {
                q += 2.0 * pij * x[row] * xc;
            }
        }
    }
    q
}

pub fn inf_norm(x: &[f64]) -> f64 {
    let mut m = 0.0;
    for &v in x {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

pub fn inf_norm_scaled(e: &[f64], x: &[f64]) -> f64 {
    debug_assert_eq!(e.len(), x.len());
    let mut m = 0.0;
    for (&ei, &xi) in e.iter().zip(x.iter()) {
        let a = (ei * xi).abs();
        if a > m {
            m = a;
        }
    }
    m
}

pub fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

pub fn copy(dst: &mut [f64], src: &[f64]) {
    dst.copy_from_slice(src);
}

pub fn scale(x: &mut [f64], c: f64) {
    for xi in x.iter_mut() {
        *xi *= c;
    }
}

pub fn clip(s: f64, min_thresh: f64, max_thresh: f64, min_new: f64, max_new: f64) -> f64 {
    if s < min_thresh {
        min_new
    } else if s > max_thresh {
        max_new
    } else {
        s
    }
}

/// Ensure P is stored as the upper triangle of a symmetric matrix.
pub fn to_symmetric_triu(P: &CscMatrix<f64>) -> CscMatrix<f64> {
    assert_eq!(P.m, P.n, "P must be square");
    if P.is_triu() {
        P.clone()
    } else {
        P.to_triu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_identity() {
        let A = CscMatrix::identity(3);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        gemv(&A, &mut y, &x, 1.0, 0.0);
        assert_eq!(y, x);
        let mut z = [0.0; 3];
        gemv_t(&A, &mut z, &x, 1.0, 0.0);
        assert_eq!(z, x);
    }

    #[test]
    fn symv_triu() {
        let P = CscMatrix::from(&[[4.0, 1.0], [0.0, 2.0]]);
        let x = [1.0, 2.0];
        let mut y = [0.0; 2];
        symv(&P, &mut y, &x, 1.0, 0.0);
        // P x = [4,1; 1,2] [1,2] = [6, 5]
        assert!((y[0] - 6.0).abs() < 1e-14);
        assert!((y[1] - 5.0).abs() < 1e-14);
        assert!((quad_form(&P, &x) - 16.0).abs() < 1e-14);
    }
}
