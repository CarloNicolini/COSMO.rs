//! COSMO KKT linear system:
//!
//! ```text
//! [ P + σ I    A' ] [ x̃ ]   =  [ σ w_x - q ]
//! [ A       -I/ρ  ] [ ν ]      [ b - 2s + w_s ]
//! ```
//!
//! The coefficient matrix is quasi-definite. Factorisation is performed with
//! Clarabel's QDLDL (AMD + LDL^T), which supports numerical refactorisation
//! after diagonal rho updates without repeating symbolic analysis.

#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLError, QDLDLFactorisation, QDLDLSettingsBuilder};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KktError {
    #[error("KKT matrix is not convex (expected {expected} positive inertia, got {got})")]
    NotConvex { expected: usize, got: usize },
    #[error("QDLDL factorisation failed: {0}")]
    Factor(#[from] QDLDLError),
    #[error("dimension mismatch in KKT system")]
    Dimension,
}

pub struct QdldlKktSolver {
    pub m: usize,
    pub n: usize,
    ldl: QDLDLFactorisation<f64>,
    /// indices of the -1/ρ diagonal entries in the assembled upper-triangular K.nzval
    rho_diag_idx: Vec<usize>,
    /// last assembled K, retained so we can rebuild if P/A change
    k_nnz: usize,
}

impl QdldlKktSolver {
    pub fn new(
        P: &CscMatrix<f64>,
        A: &CscMatrix<f64>,
        sigma: f64,
        rho: &[f64],
    ) -> Result<Self, KktError> {
        let n = P.n;
        let m = A.m;
        if P.m != n || A.n != n || rho.len() != m {
            return Err(KktError::Dimension);
        }
        let K = assemble_kkt_upper(P, A, sigma, rho);
        let rho_diag_idx = rho_diag_indices(&K, n, m);

        let mut dsigns = vec![1i8; n + m];
        for s in dsigns.iter_mut().skip(n) {
            *s = -1;
        }
        let opts = QDLDLSettingsBuilder::<f64>::default()
            .Dsigns(dsigns)
            .regularize_enable(true)
            .build()
            .unwrap();

        let ldl = QDLDLFactorisation::new(&K, Some(opts))?;
        let pos = ldl.positive_inertia();
        if pos != n {
            return Err(KktError::NotConvex {
                expected: n,
                got: pos,
            });
        }
        Ok(Self {
            m,
            n,
            ldl,
            rho_diag_idx,
            k_nnz: K.nnz(),
        })
    }

    /// Solve K sol = rhs in place into `sol` (copies rhs first).
    pub fn solve(&mut self, sol: &mut [f64], rhs: &[f64]) {
        sol.copy_from_slice(rhs);
        self.ldl.solve(sol);
    }

    /// Update the -1/ρ block and numerically refactor (symbolic analysis reused).
    pub fn update_rho(&mut self, rho: &[f64]) -> Result<(), KktError> {
        if rho.len() != self.m {
            return Err(KktError::Dimension);
        }
        let vals: Vec<f64> = rho.iter().map(|&r| -1.0 / r).collect();
        self.ldl.update_values(&self.rho_diag_idx, &vals);
        self.ldl.refactor()?;
        Ok(())
    }

    /// Rebuild the factorisation from new P, A (or new numerical values).
    /// Symbolic AMD ordering is recomputed.
    pub fn rebuild(
        &mut self,
        P: &CscMatrix<f64>,
        A: &CscMatrix<f64>,
        sigma: f64,
        rho: &[f64],
    ) -> Result<(), KktError> {
        *self = Self::new(P, A, sigma, rho)?;
        Ok(())
    }

    pub fn nnz_k(&self) -> usize {
        self.k_nnz
    }

    pub fn nnz_l(&self) -> usize {
        self.ldl.nnzL()
    }
}

/// Assemble the upper triangle of
/// K = [ P+σI , A' ; A , -I/ρ ].
pub fn assemble_kkt_upper(
    P: &CscMatrix<f64>,
    A: &CscMatrix<f64>,
    sigma: f64,
    rho: &[f64],
) -> CscMatrix<f64> {
    let n = P.n;
    let m = A.m;
    let mut I = Vec::new();
    let mut J = Vec::new();
    let mut V = Vec::new();

    for col in 0..n {
        I.push(col);
        J.push(col);
        V.push(sigma);
        for p in P.colptr[col]..P.colptr[col + 1] {
            let row = P.rowval[p];
            if row <= col {
                I.push(row);
                J.push(col);
                V.push(P.nzval[p]);
            }
        }
    }
    for col in 0..n {
        for p in A.colptr[col]..A.colptr[col + 1] {
            let row = A.rowval[p];
            I.push(col);
            J.push(n + row);
            V.push(A.nzval[p]);
        }
    }
    for i in 0..m {
        I.push(n + i);
        J.push(n + i);
        V.push(-1.0 / rho[i]);
    }

    CscMatrix::new_from_triplets(n + m, n + m, I, J, V)
}

fn rho_diag_indices(K: &CscMatrix<f64>, n: usize, m: usize) -> Vec<usize> {
    let mut idx = Vec::with_capacity(m);
    for i in 0..m {
        let col = n + i;
        idx.push(K.colptr[col + 1] - 1);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kkt_solve_identity() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let A = CscMatrix::identity(2);
        let rho = vec![1.0, 1.0];
        let mut kkt = QdldlKktSolver::new(&P, &A, 1.0, &rho).unwrap();
        // K = [I, I; I, -I], rhs = [1, 2, 3, 4]
        let rhs = [1.0, 2.0, 3.0, 4.0];
        let mut sol = [0.0; 4];
        kkt.solve(&mut sol, &rhs);
        // (I) x + ν = [1,2], x - ν = [3,4] => 2x = [4,6] => x=[2,3], ν=[-1,-1]
        assert!((sol[0] - 2.0).abs() < 1e-10);
        assert!((sol[1] - 3.0).abs() < 1e-10);
        assert!((sol[2] + 1.0).abs() < 1e-10);
        assert!((sol[3] + 1.0).abs() < 1e-10);
    }
}
