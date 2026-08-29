//! 3D power cone projection (Hien 2015 / COSMO.jl).
//!
//! K_pow(α) = { (x,y,z) | x^α y^(1-α) ≥ |z|, x ≥ 0, y ≥ 0 },  0 < α < 1.

#[derive(Clone, Debug)]
pub struct PowerCone {
    pub alpha: f64,
    max_iter: usize,
    tol: f64,
}

impl PowerCone {
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "power cone exponent α must be in (0, 1)"
        );
        Self {
            alpha,
            max_iter: 20,
            tol: 1e-8,
        }
    }

    pub fn project(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), 3);
        if !v.iter().all(|x| x.is_finite()) {
            v.fill(0.0);
            return;
        }
        if self.in_cone(v, 0.0) {
            return;
        }
        let neg = [-v[0], -v[1], -v[2]];
        if self.in_dual(&neg, 0.0) {
            v.fill(0.0);
            return;
        }
        if v[2].abs() <= self.tol {
            v[0] = v[0].max(0.0);
            v[1] = v[1].max(0.0);
            return;
        }
        self.project_pow(v);
    }

    fn project_pow(&self, v: &mut [f64]) {
        let x0 = v[0];
        let y0 = v[1];
        let z0 = v[2];
        let az = z0.abs();
        let mut r = az / 2.0;
        let mut phix = 0.0;
        let mut phiy = 0.0;
        for _ in 0..self.max_iter {
            phix = phi_c(x0, z0, r, self.alpha);
            phiy = phi_c(y0, z0, r, 1.0 - self.alpha);
            let phi = phix.powf(self.alpha) * phiy.powf(1.0 - self.alpha) - r;
            if phi.abs() < self.tol {
                break;
            }
            let dphix = dphi_c_dr(phix, x0, z0, r, self.alpha);
            let dphiy = dphi_c_dr(phiy, y0, z0, r, 1.0 - self.alpha);
            let dphi = phix.powf(self.alpha)
                * phiy.powf(1.0 - self.alpha)
                * (self.alpha * dphix / phix + (1.0 - self.alpha) * dphiy / phiy)
                - 1.0;
            if dphi == 0.0 || !dphi.is_finite() {
                break;
            }
            r -= phi / dphi;
            r = r.clamp(0.0, az);
        }
        v[0] = phix;
        v[1] = phiy;
        v[2] = z0 * r / az;
    }

    pub fn in_cone(&self, v: &[f64], tol: f64) -> bool {
        let (x, y, z) = (v[0], v[1], v[2]);
        x >= 0.0 && y >= 0.0 && x.powf(self.alpha) * y.powf(1.0 - self.alpha) >= z.abs() - tol
    }

    pub fn in_dual(&self, v: &[f64], tol: f64) -> bool {
        let (s, t, w) = (v[0], v[1], v[2]);
        let a = self.alpha;
        s >= -tol
            && t >= -tol
            && s.powf(a) * t.powf(1.0 - a) >= w.abs() * a.powf(a) * (1.0 - a).powf(1.0 - a) - tol
    }

    pub fn in_polar_recession(&self, v: &[f64], tol: f64) -> bool {
        self.in_dual(&[-v[0], -v[1], -v[2]], tol)
    }
}

fn phi_c(x0: f64, z0: f64, r: f64, alpha: f64) -> f64 {
    let inner = x0 * x0 + 4.0 * alpha * r * (z0.abs() - r);
    (0.5 * (x0 + inner.max(0.0).sqrt())).max(1e-10)
}

fn dphi_c_dr(phix: f64, x0: f64, z0: f64, r: f64, alpha: f64) -> f64 {
    let den = 2.0 * phix - x0;
    if den.abs() < 1e-16 {
        0.0
    } else {
        alpha / den * (z0.abs() - 2.0 * r)
    }
}

#[derive(Clone, Debug)]
pub struct DualPowerCone {
    v0: [f64; 3],
    primal: PowerCone,
}

impl DualPowerCone {
    pub fn new(alpha: f64) -> Self {
        Self {
            v0: [0.0; 3],
            primal: PowerCone::new(alpha),
        }
    }

    pub fn project(&mut self, v: &mut [f64]) {
        self.v0.copy_from_slice(v);
        for vi in v.iter_mut() {
            *vi = -*vi;
        }
        self.primal.project(v);
        for i in 0..3 {
            v[i] += self.v0[i];
        }
    }

    pub fn in_cone(&self, v: &[f64], tol: f64) -> bool {
        self.primal.in_dual(v, tol)
    }

    pub fn in_dual(&self, v: &[f64], tol: f64) -> bool {
        self.primal.in_cone(v, tol)
    }

    pub fn in_polar_recession(&self, v: &[f64], tol: f64) -> bool {
        self.in_dual(&[-v[0], -v[1], -v[2]], tol)
    }
}
