//! Convex cones and Euclidean projections used by COSMO.
//!
//! These are **projection** operators for ADMM, not Clarabel's interior-point
//! barrier / Nesterov-Todd scaling maps. The algorithms follow COSMO.jl.

mod composite;
mod exp;
mod nonnegative;
mod power;
mod soc;
mod zero;

pub use composite::CompositeCone;
pub use exp::{DualExponentialCone, ExponentialCone};
pub use nonnegative::NonnegativeCone;
pub use power::{DualPowerCone, PowerCone};
pub use soc::SecondOrderCone;
pub use zero::ZeroCone;

use crate::algebra::clip;

/// A primitive convex set that COSMO can project onto.
#[derive(Clone, Debug)]
pub enum Cone {
    Zero(ZeroCone),
    Nonnegative(NonnegativeCone),
    SecondOrder(SecondOrderCone),
    Exponential(ExponentialCone),
    DualExponential(DualExponentialCone),
    Power(PowerCone),
    DualPower(DualPowerCone),
    Box(BoxCone),
}

impl Cone {
    pub fn zero(dim: usize) -> Self {
        Cone::Zero(ZeroCone::new(dim))
    }
    pub fn nonnegative(dim: usize) -> Self {
        Cone::Nonnegative(NonnegativeCone::new(dim))
    }
    pub fn second_order(dim: usize) -> Self {
        Cone::SecondOrder(SecondOrderCone::new(dim))
    }
    pub fn exponential() -> Self {
        Cone::Exponential(ExponentialCone::new())
    }
    pub fn dual_exponential() -> Self {
        Cone::DualExponential(DualExponentialCone::new())
    }
    pub fn power(alpha: f64) -> Self {
        Cone::Power(PowerCone::new(alpha))
    }
    pub fn dual_power(alpha: f64) -> Self {
        Cone::DualPower(DualPowerCone::new(alpha))
    }
    pub fn boxed(l: Vec<f64>, u: Vec<f64>) -> Self {
        Cone::Box(BoxCone::new(l, u))
    }

    pub fn dim(&self) -> usize {
        match self {
            Cone::Zero(c) => c.dim,
            Cone::Nonnegative(c) => c.dim,
            Cone::SecondOrder(c) => c.dim,
            Cone::Exponential(_) | Cone::DualExponential(_) => 3,
            Cone::Power(_) | Cone::DualPower(_) => 3,
            Cone::Box(c) => c.l.len(),
        }
    }

    pub fn is_equality(&self) -> bool {
        matches!(self, Cone::Zero(_))
    }

    pub fn project(&mut self, x: &mut [f64]) {
        debug_assert_eq!(x.len(), self.dim());
        match self {
            Cone::Zero(c) => c.project(x),
            Cone::Nonnegative(c) => c.project(x),
            Cone::SecondOrder(c) => c.project(x),
            Cone::Exponential(c) => c.project(x),
            Cone::DualExponential(c) => c.project(x),
            Cone::Power(c) => c.project(x),
            Cone::DualPower(c) => c.project(x),
            Cone::Box(c) => c.project(x),
        }
    }

    pub fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        match self {
            Cone::Zero(c) => c.in_cone(x, tol),
            Cone::Nonnegative(c) => c.in_cone(x, tol),
            Cone::SecondOrder(c) => c.in_cone(x, tol),
            Cone::Exponential(c) => c.in_cone(x, tol),
            Cone::DualExponential(c) => c.in_cone(x, tol),
            Cone::Power(c) => c.in_cone(x, tol),
            Cone::DualPower(c) => c.in_cone(x, tol),
            Cone::Box(c) => c.in_cone(x, tol),
        }
    }

    pub fn in_dual(&self, x: &[f64], tol: f64) -> bool {
        match self {
            Cone::Zero(c) => c.in_dual(x, tol),
            Cone::Nonnegative(c) => c.in_dual(x, tol),
            Cone::SecondOrder(c) => c.in_dual(x, tol),
            Cone::Exponential(c) => c.in_dual(x, tol),
            Cone::DualExponential(c) => c.in_dual(x, tol),
            Cone::Power(c) => c.in_dual(x, tol),
            Cone::DualPower(c) => c.in_dual(x, tol),
            Cone::Box(c) => c.in_dual(x, tol),
        }
    }

    pub fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        match self {
            Cone::Zero(c) => c.in_polar_recession(x, tol),
            Cone::Nonnegative(c) => c.in_polar_recession(x, tol),
            Cone::SecondOrder(c) => c.in_polar_recession(x, tol),
            Cone::Exponential(c) => c.in_polar_recession(x, tol),
            Cone::DualExponential(c) => c.in_polar_recession(x, tol),
            Cone::Power(c) => c.in_polar_recession(x, tol),
            Cone::DualPower(c) => c.in_polar_recession(x, tol),
            Cone::Box(c) => c.in_polar_recession(x, tol),
        }
    }

    /// In-place support of C̃ = -K: 0 if -y ∈ K*, else +∞. Mutates y to -y.
    pub fn support_function(&self, y: &mut [f64], tol: f64) -> f64 {
        match self {
            Cone::Box(c) => c.support_function(y, tol),
            _ => {
                for yi in y.iter_mut() {
                    *yi = -*yi;
                }
                if self.in_dual(y, tol) {
                    0.0
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// Scalar rectification for cones that only admit a single scaling.
    pub fn rectify_scaling(&self, e: &[f64], work: &mut [f64]) -> bool {
        match self {
            Cone::Zero(_) | Cone::Nonnegative(_) | Cone::Box(_) => false,
            _ => {
                let mean = e.iter().sum::<f64>() / (e.len() as f64);
                for (w, &ei) in work.iter_mut().zip(e.iter()) {
                    *w = mean / ei;
                }
                true
            }
        }
    }

    pub fn classify_constraints(&mut self, b: &[f64], cosmo_infty: f64, min_scaling: f64) {
        if let Cone::Nonnegative(c) = self {
            c.classify_constraints(b, cosmo_infty, min_scaling);
        }
        if let Cone::Box(c) = self {
            c.classify_constraints(cosmo_infty, min_scaling);
        }
    }
}

/// Axis-aligned box {x | l ≤ x ≤ u}.
#[derive(Clone, Debug)]
pub struct BoxCone {
    pub l: Vec<f64>,
    pub u: Vec<f64>,
    pub constr_type: Vec<i8>,
}

impl BoxCone {
    pub fn new(l: Vec<f64>, u: Vec<f64>) -> Self {
        assert_eq!(l.len(), u.len());
        let n = l.len();
        Self {
            l,
            u,
            constr_type: vec![0; n],
        }
    }

    fn project(&self, x: &mut [f64]) {
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = clip(*xi, self.l[i], self.u[i], self.l[i], self.u[i]);
        }
    }

    fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        x.iter()
            .zip(self.l.iter().zip(self.u.iter()))
            .all(|(&xi, (&lo, &up))| xi >= lo - tol && xi <= up + tol)
    }

    fn in_dual(&self, _x: &[f64], _tol: f64) -> bool {
        true
    }

    fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        for (i, &xi) in x.iter().enumerate() {
            if self.u[i].is_infinite() && xi > tol {
                return false;
            }
            if self.l[i].is_infinite() && self.l[i].is_sign_negative() && xi < -tol {
                return false;
            }
        }
        true
    }

    fn support_function(&self, x: &[f64], tol: f64) -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() {
            if x[i].abs() > tol && x[i] > 0.0 {
                s += x[i] * self.u[i];
            } else {
                s += x[i] * self.l[i];
            }
        }
        s
    }

    fn classify_constraints(&mut self, cosmo_infty: f64, min_scaling: f64) {
        for i in 0..self.l.len() {
            if self.u[i] > cosmo_infty * min_scaling && self.l[i] < -cosmo_infty * min_scaling {
                self.constr_type[i] = -1;
            } else if (self.u[i] - self.l[i]).abs() < 1e-4 {
                self.constr_type[i] = 1;
            } else {
                self.constr_type[i] = 0;
            }
        }
    }
}
