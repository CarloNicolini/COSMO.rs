//! Zero cone {0}^dim.

#[derive(Clone, Debug)]
pub struct ZeroCone {
    pub dim: usize,
}

impl ZeroCone {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn project(&self, x: &mut [f64]) {
        x.fill(0.0);
    }

    pub fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        x.iter().all(|&v| v.abs() <= tol)
    }

    pub fn in_dual(&self, _x: &[f64], _tol: f64) -> bool {
        true
    }

    pub fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        x.iter().all(|&v| v.abs() <= tol)
    }
}
