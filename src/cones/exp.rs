//! Exponential cone projection (COSMO.jl / SCS bisection method).
//!
//! K_exp = { (x,y,z) | y > 0, y exp(x/y) ≤ z } ∪ { (x,0,z) | x ≤ 0, z ≥ 0 }

#[derive(Clone, Debug)]
pub struct ExponentialCone {
    v0: [f64; 3],
    max_iter: usize,
    tol: f64,
}

impl ExponentialCone {
    pub fn new() -> Self {
        Self {
            v0: [0.0; 3],
            max_iter: 100,
            tol: 1e-8,
        }
    }

    pub fn project(&mut self, v: &mut [f64]) {
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
        if v[0] < 0.0 && v[1] < 0.0 {
            v[1] = 0.0;
            v[2] = v[2].max(0.0);
            return;
        }
        self.project_exp(v);
    }

    fn project_exp(&mut self, v: &mut [f64]) {
        self.v0.copy_from_slice(v);
        let (mut l, mut u) = self.bisection_bounds();
        for _ in 0..self.max_iter {
            let lambda = 0.5 * (u + l);
            let g = self.grad_dual(lambda, v);
            if g > 0.0 {
                l = lambda;
            } else {
                u = lambda;
            }
            if u - l < self.tol {
                break;
            }
        }
    }

    fn bisection_bounds(&self) -> (f64, f64) {
        let mut l = 0.0;
        let mut lambda = 0.125;
        let mut tmp = [0.0; 3];
        let mut g = self.grad_dual_into(lambda, &mut tmp);
        let mut guard = 0;
        while g > 0.0 && guard < 60 {
            l = lambda;
            lambda *= 2.0;
            g = self.grad_dual_into(lambda, &mut tmp);
            guard += 1;
        }
        (l, lambda)
    }

    fn grad_dual(&self, lambda: f64, v: &mut [f64]) -> f64 {
        self.grad_dual_into(lambda, v)
    }

    fn grad_dual_into(&self, lambda: f64, v: &mut [f64]) -> f64 {
        // COSMO.jl / SCS: g(λ) = r*  if s* == 0, else r* + s* log(s*/t*).
        // v = (r, s, t) = (x, y, z). The Julia code checks v[2] (1-based y).
        self.find_minimizers(lambda, v);
        if v[1] == 0.0 {
            v[0]
        } else if v[1] < 0.0 || v[2] <= 0.0 || !v[1].is_finite() || !v[2].is_finite() {
            v[0]
        } else {
            v[0] + v[1] * (v[1] / v[2]).ln()
        }
    }

    fn find_minimizers(&self, lambda: f64, v: &mut [f64]) {
        v[2] = find_min_t(lambda, self.v0[1], self.v0[2], self.tol);
        v[1] = (1.0 / lambda) * (v[2] - self.v0[2]) * v[2];
        v[0] = self.v0[0] - lambda;
    }

    pub fn in_cone(&self, v: &[f64], tol: f64) -> bool {
        let (x, y, z) = (v[0], v[1], v[2]);
        (y > 0.0 && y * (x / y).exp() <= z + tol) || (x <= tol && y == 0.0 && z >= -tol)
    }

    pub fn in_dual(&self, v: &[f64], tol: f64) -> bool {
        let (x, y, z) = (v[0], v[1], v[2]);
        (x < 0.0 && -x * (y / x).exp() - std::f64::consts::E * z <= tol)
            || (x.abs() <= tol && y >= -tol && z >= -tol)
    }

    pub fn in_polar_recession(&self, v: &[f64], tol: f64) -> bool {
        self.in_dual(&[-v[0], -v[1], -v[2]], tol)
    }
}

impl Default for ExponentialCone {
    fn default() -> Self {
        Self::new()
    }
}

fn find_min_t(lambda: f64, s0: f64, t0: f64, tol: f64) -> f64 {
    let mut dt = (-t0).max(tol);
    for _ in 0..150 {
        let f = dt * (dt + t0) / (lambda * lambda) - s0 / lambda + (dt / lambda).ln() + 1.0;
        let grad_f = (2.0 * dt + t0) / (lambda * lambda) + 1.0 / dt;
        if !grad_f.is_finite() || grad_f == 0.0 {
            break;
        }
        dt -= f / grad_f;
        if dt <= -t0 {
            dt = -t0;
            break;
        } else if dt <= 0.0 {
            dt = 0.0;
            break;
        } else if f.abs() < tol {
            break;
        }
    }
    dt + t0
}

#[derive(Clone, Debug)]
pub struct DualExponentialCone {
    v0: [f64; 3],
    primal: ExponentialCone,
}

impl DualExponentialCone {
    pub fn new() -> Self {
        Self {
            v0: [0.0; 3],
            primal: ExponentialCone::new(),
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

impl Default for DualExponentialCone {
    fn default() -> Self {
        Self::new()
    }
}
