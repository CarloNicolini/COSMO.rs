//! Type-II Anderson accelerator with QR and restarted memory.
//!
//! Reimplementation of COSMOAccelerators.jl (MIT License, Michael Garstka).
//! The ADMM operator g is accelerated; safeguarding lives in the solver loop.

#[derive(Clone, Debug)]
pub struct AndersonAccelerator {
    init_phase: bool,
    mem: usize,
    min_mem: usize,
    dim: usize,
    iter: usize,
    pub g_last: Vec<f64>,
    pub f: Vec<f64>,
    f_last: Vec<f64>,
    eta: Vec<f64>,
    G: Vec<f64>, // dim x mem, column-major
    Q: Vec<f64>, // dim x mem
    R: Vec<f64>, // mem x mem
    pub success: bool,
}

impl AndersonAccelerator {
    pub fn new(dim: usize, mem: usize, min_mem: usize) -> Self {
        let mem = mem.max(2).min(dim.max(2));
        let min_mem = min_mem.max(2).min(mem);
        Self {
            init_phase: true,
            mem,
            min_mem,
            dim,
            iter: 0,
            g_last: vec![0.0; dim],
            f: vec![0.0; dim],
            f_last: vec![0.0; dim],
            eta: vec![0.0; mem],
            G: vec![0.0; dim * mem],
            Q: vec![0.0; dim * mem],
            R: vec![0.0; mem * mem],
            success: false,
        }
    }

    pub fn restart(&mut self) {
        self.f_last.fill(0.0);
        self.g_last.fill(0.0);
        self.eta.fill(0.0);
        self.iter = 0;
        self.init_phase = true;
        self.success = false;
    }

    pub fn was_successful(&self) -> bool {
        self.success
    }

    /// g = current operator output (w), x = previous operator input (w_prev).
    pub fn update(&mut self, g: &[f64], x: &[f64]) {
        debug_assert_eq!(g.len(), self.dim);
        debug_assert_eq!(x.len(), self.dim);
        for i in 0..self.dim {
            self.f[i] = x[i] - g[i];
        }
        if self.init_phase {
            self.g_last.copy_from_slice(g);
            self.f_last.copy_from_slice(&self.f);
            self.init_phase = false;
            return;
        }
        let j = self.iter % self.mem;
        if j == 0 && self.iter != 0 {
            self.iter = 0;
        }
        let j = self.iter % self.mem;
        for i in 0..self.dim {
            self.G[i + j * self.dim] = g[i] - self.g_last[i];
            self.f_last[i] = self.f[i] - self.f_last[i];
        }
        qr_add_column(
            &mut self.Q,
            &mut self.R,
            &mut self.f_last,
            self.dim,
            self.mem,
            j,
        );
        self.g_last.copy_from_slice(g);
        self.f_last.copy_from_slice(&self.f);
        self.iter += 1;
    }

    /// Overwrite `g` with the accelerated candidate if successful.
    pub fn accelerate(&mut self, g: &mut [f64]) {
        self.success = false;
        let l = self.iter.min(self.mem);
        if l < self.min_mem {
            return;
        }
        let dim = self.dim;
        let eta = &mut self.eta[..l];
        // eta = Q' f
        for j in 0..l {
            let mut s = 0.0;
            for i in 0..dim {
                s += self.Q[i + j * dim] * self.f[i];
            }
            eta[j] = s;
        }
        if backsolve_upper(&self.R, self.mem, eta).is_err() {
            return;
        }
        let nrm: f64 = eta.iter().map(|e| e * e).sum::<f64>().sqrt();
        if !nrm.is_finite() || nrm > 1e4 {
            return;
        }
        // g := g_last - G eta
        for i in 0..dim {
            let mut s = 0.0;
            for j in 0..l {
                s += self.G[i + j * dim] * eta[j];
            }
            g[i] = self.g_last[i] - s;
        }
        self.success = true;
    }
}

fn qr_add_column(Q: &mut [f64], R: &mut [f64], df: &mut [f64], dim: usize, mem: usize, j: usize) {
    for k in 0..j {
        let mut ip = 0.0;
        for i in 0..dim {
            ip += Q[i + k * dim] * df[i];
        }
        R[k + j * mem] = ip;
        for i in 0..dim {
            df[i] -= ip * Q[i + k * dim];
        }
    }
    let mut nrm = 0.0;
    for &v in df.iter() {
        nrm += v * v;
    }
    nrm = nrm.sqrt();
    R[j + j * mem] = nrm;
    if nrm > 0.0 {
        let inv = 1.0 / nrm;
        for i in 0..dim {
            Q[i + j * dim] = df[i] * inv;
        }
    } else {
        for i in 0..dim {
            Q[i + j * dim] = 0.0;
        }
    }
}

fn backsolve_upper(R: &[f64], mem: usize, eta: &mut [f64]) -> Result<(), ()> {
    let l = eta.len();
    for i in (0..l).rev() {
        let mut s = eta[i];
        for j in (i + 1)..l {
            s -= R[i + j * mem] * eta[j];
        }
        let rii = R[i + i * mem];
        if rii.abs() < 1e-16 || !rii.is_finite() {
            return Err(());
        }
        eta[i] = s / rii;
        if !eta[i].is_finite() {
            return Err(());
        }
    }
    Ok(())
}

/// Residual-norm of an accelerated candidate: ||w_prev - w||.
pub fn residual_norm(f: &mut [f64], w: &[f64], w_prev: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..w.len() {
        f[i] = w_prev[i] - w[i];
        s += f[i] * f[i];
    }
    s.sqrt()
}
