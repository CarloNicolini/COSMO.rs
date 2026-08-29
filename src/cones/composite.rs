//! Cartesian product of convex cones.

use super::Cone;

#[derive(Clone, Debug)]
pub struct CompositeCone {
    pub cones: Vec<Cone>,
    pub offsets: Vec<usize>,
    pub dim: usize,
}

impl CompositeCone {
    pub fn new(cones: Vec<Cone>) -> Self {
        let mut offsets = Vec::with_capacity(cones.len());
        let mut dim = 0;
        for c in &cones {
            offsets.push(dim);
            dim += c.dim();
        }
        Self {
            cones,
            offsets,
            dim,
        }
    }

    pub fn project(&mut self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.dim);
        for (cone, &off) in self.cones.iter_mut().zip(self.offsets.iter()) {
            let d = cone.dim();
            cone.project(&mut x[off..off + d]);
        }
    }

    pub fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        self.cones.iter().zip(self.offsets.iter()).all(|(c, &off)| {
            let d = c.dim();
            c.in_cone(&x[off..off + d], tol)
        })
    }

    pub fn in_dual(&self, x: &[f64], tol: f64) -> bool {
        self.cones.iter().zip(self.offsets.iter()).all(|(c, &off)| {
            let d = c.dim();
            c.in_dual(&x[off..off + d], tol)
        })
    }

    pub fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        self.cones.iter().zip(self.offsets.iter()).all(|(c, &off)| {
            let d = c.dim();
            c.in_polar_recession(&x[off..off + d], tol)
        })
    }

    pub fn support_function(&self, y: &mut [f64], tol: f64) -> f64 {
        let mut s = 0.0;
        for (cone, &off) in self.cones.iter().zip(self.offsets.iter()) {
            let d = cone.dim();
            let si = cone.support_function(&mut y[off..off + d], tol);
            if !si.is_finite() {
                return f64::INFINITY;
            }
            s += si;
        }
        s
    }

    pub fn rectify_scaling(&self, e: &[f64], work: &mut [f64]) -> bool {
        let mut any = false;
        for (cone, &off) in self.cones.iter().zip(self.offsets.iter()) {
            let d = cone.dim();
            any |= cone.rectify_scaling(&e[off..off + d], &mut work[off..off + d]);
        }
        any
    }

    pub fn classify_constraints(&mut self, b: &[f64], cosmo_infty: f64, min_scaling: f64) {
        for (cone, &off) in self.cones.iter_mut().zip(self.offsets.iter()) {
            let d = cone.dim();
            cone.classify_constraints(&b[off..off + d], cosmo_infty, min_scaling);
        }
    }

    pub fn apply_rho_scaling(&self, rho: &mut [f64], rho_min: f64, rho_eq_over_ineq: f64) {
        for (cone, &off) in self.cones.iter().zip(self.offsets.iter()) {
            let d = cone.dim();
            match cone {
                Cone::Zero(_) => {
                    for r in &mut rho[off..off + d] {
                        *r *= rho_eq_over_ineq;
                    }
                }
                Cone::Nonnegative(c) => {
                    for (j, &loose) in c.constr_type.iter().enumerate() {
                        if loose {
                            rho[off + j] = rho_min;
                        }
                    }
                }
                Cone::Box(c) => {
                    for (j, &ct) in c.constr_type.iter().enumerate() {
                        if ct == -1 {
                            rho[off + j] = rho_min;
                        } else if ct == 1 {
                            rho[off + j] *= rho_eq_over_ineq;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
