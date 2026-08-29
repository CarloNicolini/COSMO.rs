//! Nonnegative orthant.

#[derive(Clone, Debug)]
pub struct NonnegativeCone {
    pub dim: usize,
    /// true if the corresponding b-entry is a loose (numerically infinite) bound
    pub constr_type: Vec<bool>,
}

impl NonnegativeCone {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            constr_type: vec![false; dim],
        }
    }

    pub fn project(&self, x: &mut [f64]) {
        for xi in x.iter_mut() {
            if *xi < 0.0 {
                *xi = 0.0;
            }
        }
    }

    pub fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        x.iter().all(|&v| v >= -tol)
    }

    pub fn in_dual(&self, x: &[f64], tol: f64) -> bool {
        x.iter().all(|&v| v >= -tol)
    }

    pub fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        x.iter().all(|&v| v <= tol)
    }

    pub fn classify_constraints(&mut self, b: &[f64], cosmo_infty: f64, min_scaling: f64) {
        for (i, &bi) in b.iter().enumerate() {
            self.constr_type[i] = bi > cosmo_infty * min_scaling;
        }
    }
}
