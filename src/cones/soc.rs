//! Second-order (Lorentz) cone { (t, x) | ||x||_2 ≤ t }.

#[derive(Clone, Debug)]
pub struct SecondOrderCone {
    pub dim: usize,
}

impl SecondOrderCone {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "SOC dimension must be at least 1");
        Self { dim }
    }

    pub fn project(&self, x: &mut [f64]) {
        if x.len() == 1 {
            if x[0] < 0.0 {
                x[0] = 0.0;
            }
            return;
        }
        let t = x[0];
        let mut norm_x = 0.0;
        for &v in &x[1..] {
            norm_x += v * v;
        }
        norm_x = norm_x.sqrt();
        if norm_x <= t {
            return;
        }
        if norm_x <= -t {
            x.fill(0.0);
            return;
        }
        let scale = (norm_x + t) / (2.0 * norm_x);
        x[0] = (norm_x + t) / 2.0;
        for v in &mut x[1..] {
            *v *= scale;
        }
    }

    pub fn in_cone(&self, x: &[f64], tol: f64) -> bool {
        if x.is_empty() {
            return true;
        }
        let mut n = 0.0;
        for &v in &x[1..] {
            n += v * v;
        }
        n.sqrt() <= x[0] + tol
    }

    pub fn in_dual(&self, x: &[f64], tol: f64) -> bool {
        self.in_cone(x, tol)
    }

    pub fn in_polar_recession(&self, x: &[f64], tol: f64) -> bool {
        if x.is_empty() {
            return true;
        }
        let mut n = 0.0;
        for &v in &x[1..] {
            n += v * v;
        }
        n.sqrt() <= tol - x[0]
    }
}
