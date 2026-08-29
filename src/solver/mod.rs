//! COSMO ADMM solver: persistent workspace, iteration, updates, and warm starts.

#![allow(non_snake_case)]

use std::time::Instant;

use crate::accelerator::{residual_norm, AndersonAccelerator};
use crate::algebra::{
    copy, gemv, gemv_t, inf_norm, inf_norm_scaled, quad_form, scale, symv, to_symmetric_triu,
    CscMatrix, MatrixMathMut,
};
use crate::cones::{CompositeCone, Cone};
use crate::linsys::{KktError, QdldlKktSolver};
use crate::scaling::{reverse_scaling, scale_ruiz, scale_variables, ScaleMatrices};
use crate::settings::{Settings, WarmStartMode};
use crate::solution::{Solution, SolverStatus, Timings};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CosmoError {
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("KKT error: {0}")]
    Kkt(#[from] KktError),
    #[error("unsupported cone: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Other(String),
}

/// Persistent COSMO solver.
///
/// The object may be reused across related problems via [`CosmoSolver::update_q`],
/// [`CosmoSolver::update_b`], [`CosmoSolver::update_p`], [`CosmoSolver::update_a`],
/// and [`CosmoSolver::warm_start`].
pub struct CosmoSolver {
    n: usize,
    m: usize,
    P: CscMatrix<f64>,
    A: CscMatrix<f64>,
    q: Vec<f64>,
    b: Vec<f64>,
    cones: CompositeCone,
    settings: Settings,
    sm: ScaleMatrices,
    kkt: Option<QdldlKktSolver>,
    accelerator: Option<AndersonAccelerator>,
    rho: f64,
    rho_vec: Vec<f64>,
    rho_updates: Vec<f64>,
    w: Vec<f64>,
    w_prev: Vec<f64>,
    s: Vec<f64>,
    mu: Vec<f64>,
    s_tl: Vec<f64>,
    ls: Vec<f64>,
    sol: Vec<f64>,
    dx: Vec<f64>,
    dy: Vec<f64>,
    work_m: Vec<f64>,
    work_n: Vec<f64>,
    work_n2: Vec<f64>,
    is_scaled: bool,
    kkt_factored: bool,
    has_solved: bool,
    obj_offset: f64,
    solution: Solution,
    safeguarding_iter: usize,
}

impl CosmoSolver {
    pub fn new(
        P: &CscMatrix<f64>,
        q: &[f64],
        A: &CscMatrix<f64>,
        b: &[f64],
        cones: Vec<Cone>,
        settings: Settings,
    ) -> Result<Self, CosmoError> {
        Self::new_with_offset(P, q, A, b, cones, settings, 0.0)
    }

    pub fn new_with_offset(
        P: &CscMatrix<f64>,
        q: &[f64],
        A: &CscMatrix<f64>,
        b: &[f64],
        cones: Vec<Cone>,
        settings: Settings,
        obj_offset: f64,
    ) -> Result<Self, CosmoError> {
        check_dims(P, q, A, b)?;
        let n = q.len();
        let m = b.len();
        let cones = CompositeCone::new(cones);
        if cones.dim != m {
            return Err(CosmoError::Dimension(format!(
                "cone dimension {} does not match m = {}",
                cones.dim, m
            )));
        }
        let P = to_symmetric_triu(P);
        let A = A.clone();
        let mut solver = Self {
            n,
            m,
            P,
            A,
            q: q.to_vec(),
            b: b.to_vec(),
            cones,
            settings,
            sm: ScaleMatrices::identity(m, n),
            kkt: None,
            accelerator: None,
            rho: 0.0,
            rho_vec: vec![0.0; m],
            rho_updates: vec![],
            w: vec![0.0; n + m],
            w_prev: vec![0.0; n + m],
            s: vec![0.0; m],
            mu: vec![0.0; m],
            s_tl: vec![0.0; m],
            ls: vec![0.0; n + m],
            sol: vec![0.0; n + m],
            dx: vec![0.0; n],
            dy: vec![0.0; m],
            work_m: vec![0.0; m],
            work_n: vec![0.0; n],
            work_n2: vec![0.0; n],
            is_scaled: false,
            kkt_factored: false,
            has_solved: false,
            obj_offset,
            solution: Solution::empty(),
            safeguarding_iter: 0,
        };
        solver.solution.x = vec![0.0; n];
        solver.solution.y = vec![0.0; m];
        solver.solution.s = vec![0.0; m];
        Ok(solver)
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn m(&self) -> usize {
        self.m
    }

    pub fn solution(&self) -> &Solution {
        &self.solution
    }

    /// Solve the current problem. Reuses factorisation when still valid.
    pub fn solve(&mut self) -> Result<&Solution, CosmoError> {
        let t0 = Instant::now();
        let mut times = Timings::default();
        self.safeguarding_iter = 0;
        self.dx.fill(0.0);
        self.dy.fill(0.0);

        let t_setup = Instant::now();
        self.setup(&mut times)?;
        times.setup_time = t_setup.elapsed().as_secs_f64();

        let m = self.m;
        let n = self.n;
        let alpha = self.settings.alpha;
        let sigma = self.settings.sigma;

        // Warm-start the operator variable from (x, s, μ).
        // x lives in w_prev[0..n] after the first copy below; initialise w from current x/s/μ.
        self.w[..n].copy_from_slice(&self.w_prev[..n]);
        for i in 0..m {
            self.w[n + i] = self.s[i] + self.mu[i] / self.rho_vec[i];
        }

        // One ADMM step so the remaining loop agrees with standard ADMM.
        self.admm_x();
        self.admm_w(alpha);

        let mut status = SolverStatus::Undetermined;
        let mut cost = f64::INFINITY;
        let mut r_prim = f64::INFINITY;
        let mut r_dual = f64::INFINITY;
        let mut max_norm_prim = 0.0;
        let mut max_norm_dual = 0.0;
        let mut iter = 0usize;
        let mut infeas_due = false;
        let mut rho_due = false;

        let iter_start = Instant::now();
        let time_limit_start = Instant::now();

        while iter + self.safeguarding_iter < self.settings.max_iter {
            iter += 1;

            self.acceleration_pre(iter, &mut times);

            if infeas_due && self.accel_allows_update() {
                self.recover_mu();
                self.dy.copy_from_slice(&self.mu);
            }

            copy(&mut self.w_prev, &self.w);

            let tproj = Instant::now();
            self.admm_z();
            times.proj_time += tproj.elapsed().as_secs_f64();

            self.apply_rho_rules(iter, iter_start, &mut rho_due, &mut times)?;

            self.admm_x();
            self.admm_w(alpha);

            self.acceleration_post(&mut times, sigma);

            if iter == 1 || iter % self.settings.check_termination == 0 {
                self.recover_mu();
                let (rp, rd, mnp, mnd) = self.calculate_residuals(false);
                r_prim = rp;
                r_dual = rd;
                max_norm_prim = mnp;
                max_norm_dual = mnd;
                cost = self.calculate_cost();
                if cost.abs() > 1e20 {
                    status = SolverStatus::Unsolved;
                    break;
                }
                if self.settings.verbose && (iter == 1 || iter % 100 == 0) {
                    eprintln!("iter {iter:5}  cost {cost:.6e}  rp {r_prim:.3e}  rd {r_dual:.3e}");
                }
                if self.has_converged(r_prim, r_dual, max_norm_prim, max_norm_dual, cost) {
                    status = SolverStatus::Solved;
                    break;
                }
            }

            if iter % self.settings.check_infeasibility == 0 {
                infeas_due = true;
            } else if infeas_due && self.accel_allows_update() {
                infeas_due = false;
                self.recover_mu();
                let (rp, rd, mnp, mnd) = self.calculate_residuals(false);
                r_prim = rp;
                r_dual = rd;
                max_norm_prim = mnp;
                max_norm_dual = mnd;
                cost = self.calculate_cost();
                if self.has_converged(r_prim, r_dual, max_norm_prim, max_norm_dual, cost) {
                    status = SolverStatus::Solved;
                    break;
                }
                // Banjac certificates are necessary conditions; they can fire on
                // feasible problems when the iterate is already nearly feasible
                // and δy is numerical noise. Only test infeasibility when the
                // current point is clearly not primal-feasible.
                let prim_tol =
                    self.settings.eps_abs + self.settings.eps_rel * max_norm_prim.max(1.0);
                if r_prim > 50.0 * prim_tol {
                    for i in 0..m {
                        self.dy[i] -= self.mu[i];
                    }
                    for i in 0..n {
                        self.dx[i] = self.w[i] - self.w_prev[i];
                    }
                    if self.is_primal_infeasible() {
                        status = SolverStatus::PrimalInfeasible;
                        cost = f64::INFINITY;
                        break;
                    }
                    if self.is_dual_infeasible() {
                        status = SolverStatus::DualInfeasible;
                        cost = f64::NEG_INFINITY;
                        break;
                    }
                }
            }

            if self.settings.time_limit > 0.0
                && time_limit_start.elapsed().as_secs_f64() > self.settings.time_limit
            {
                let (rp, rd, mnp, mnd) = self.calculate_residuals(false);
                r_prim = rp;
                r_dual = rd;
                max_norm_prim = mnp;
                max_norm_dual = mnd;
                status = SolverStatus::TimeLimitReached;
                break;
            }
            let _ = sigma;
        }

        if status == SolverStatus::Undetermined {
            self.recover_mu();
            let (rp, rd, mnp, mnd) = self.calculate_residuals(false);
            r_prim = rp;
            r_dual = rd;
            max_norm_prim = mnp;
            max_norm_dual = mnd;
            cost = self.calculate_cost();
            status = SolverStatus::MaxIterReached;
        }

        times.iter_time = iter_start.elapsed().as_secs_f64();
        let tpost = Instant::now();

        // x is w_prev[0..n] (COSMO: x is a view onto w_prev). Unscale copies only;
        // keep the workspace in scaled coordinates so a subsequent solve() is
        // a true warm start and does not mix scaled data with unscaled iterates.
        let (x, y, s_unscaled) = self.unscaled_copies();

        times.post_time = tpost.elapsed().as_secs_f64();
        times.solver_time = t0.elapsed().as_secs_f64();

        if self.settings.verbose {
            eprintln!(
                "COSMO.rs  status={}  iter={} (safeguard={})  obj={:.6e}  time={:.3}ms",
                status,
                iter + self.safeguarding_iter,
                self.safeguarding_iter,
                cost,
                times.solver_time * 1e3
            );
        }

        self.solution = Solution {
            x,
            y,
            s: s_unscaled,
            obj_val: cost,
            iter: iter + self.safeguarding_iter,
            safeguarding_iter: self.safeguarding_iter,
            status,
            obj_offset: self.obj_offset,
            r_prim,
            r_dual,
            max_norm_prim,
            max_norm_dual,
            rho_updates: self.rho_updates.clone(),
            times,
        };
        self.has_solved = true;
        Ok(&self.solution)
    }

    fn setup(&mut self, times: &mut Timings) -> Result<(), CosmoError> {
        if self.settings.scaling != 0 && !self.is_scaled {
            let t = Instant::now();
            self.sm = scale_ruiz(
                &mut self.P,
                &mut self.A,
                &mut self.q,
                &mut self.b,
                &self.cones,
                &self.settings,
            );
            scale_variables(
                &mut self.w_prev[..self.n],
                &mut self.mu,
                &mut self.s,
                &self.sm.Dinv,
                &self.sm.Einv,
                &self.sm.E,
                self.sm.c,
            );
            self.is_scaled = true;
            times.scaling_time = t.elapsed().as_secs_f64();
        }

        self.cones.classify_constraints(
            &self.b,
            self.settings.cosmo_infty,
            self.settings.min_scaling,
        );

        if !self.has_solved {
            self.set_rho_vec();
        }

        if self.settings.accelerate {
            match self.accelerator.as_mut() {
                Some(aa) => aa.restart(),
                None => {
                    self.accelerator = Some(AndersonAccelerator::new(
                        self.m + self.n,
                        self.settings.accelerator_memory,
                        self.settings.accelerator_min_mem,
                    ));
                }
            }
        } else {
            self.accelerator = None;
        }

        if !self.kkt_factored {
            let t = Instant::now();
            self.kkt = Some(QdldlKktSolver::new(
                &self.P,
                &self.A,
                self.settings.sigma,
                &self.rho_vec,
            )?);
            self.kkt_factored = true;
            times.init_factor_time = t.elapsed().as_secs_f64();
        }
        Ok(())
    }

    fn set_rho_vec(&mut self) {
        self.rho = self.settings.rho;
        self.rho_vec.fill(self.rho);
        self.cones.apply_rho_scaling(
            &mut self.rho_vec,
            self.settings.rho_min,
            self.settings.rho_eq_over_ineq,
        );
        self.rho_updates.clear();
        self.rho_updates.push(self.rho);
    }

    fn admm_z(&mut self) {
        let n = self.n;
        self.s.copy_from_slice(&self.w[n..]);
        self.cones.project(&mut self.s);
    }

    fn recover_mu(&mut self) {
        let n = self.n;
        for i in 0..self.m {
            self.mu[i] = self.rho_vec[i] * (self.w_prev[n + i] - self.s[i]);
        }
    }

    fn admm_x(&mut self) {
        let n = self.n;
        let m = self.m;
        let sigma = self.settings.sigma;
        for i in 0..n {
            self.ls[i] = sigma * self.w[i] - self.q[i];
        }
        for i in 0..m {
            self.ls[n + i] = self.b[i] - 2.0 * self.s[i] + self.w[n + i];
        }
        let kkt = self.kkt.as_mut().expect("KKT solver");
        kkt.solve(&mut self.sol, &self.ls);
        for i in 0..m {
            self.s_tl[i] = 2.0 * self.s[i] - self.w[n + i] - self.sol[n + i] / self.rho_vec[i];
        }
    }

    fn admm_w(&mut self, alpha: f64) {
        let n = self.n;
        let m = self.m;
        for i in 0..n {
            self.w[i] += alpha * (self.sol[i] - self.w[i]);
        }
        for i in 0..m {
            self.w[n + i] += alpha * (self.s_tl[i] - self.s[i]);
        }
    }

    fn accel_allows_update(&self) -> bool {
        match &self.accelerator {
            Some(aa) => !aa.was_successful(),
            None => true,
        }
    }

    fn acceleration_pre(&mut self, iter: usize, times: &mut Timings) {
        if iter < 2 {
            return;
        }
        if let Some(aa) = self.accelerator.as_mut() {
            let t = Instant::now();
            aa.update(&self.w, &self.w_prev);
            aa.accelerate(&mut self.w);
            times.accelerate_time += t.elapsed().as_secs_f64();
        }
    }

    fn acceleration_post(&mut self, times: &mut Timings, _sigma: f64) {
        let Some(aa) = self.accelerator.as_mut() else {
            return;
        };
        if !aa.was_successful() {
            return;
        }
        if !self.settings.safeguard {
            return;
        }
        let nrm_f = aa.f.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nrm_tol = nrm_f * self.settings.safeguard_tol;
        let nrm_acc = residual_norm(&mut aa.f, &self.w, &self.w_prev);
        if nrm_acc > nrm_tol {
            self.w.copy_from_slice(&aa.g_last);
            self.w_prev.copy_from_slice(&aa.g_last);
            let tproj = Instant::now();
            self.admm_z();
            times.proj_time += tproj.elapsed().as_secs_f64();
            self.admm_x();
            self.admm_w(self.settings.alpha);
            self.safeguarding_iter += 1;
        }
    }

    fn apply_rho_rules(
        &mut self,
        iter: usize,
        iter_start: Instant,
        rho_due: &mut bool,
        times: &mut Timings,
    ) -> Result<(), CosmoError> {
        let s = &self.settings;
        if !s.adaptive_rho {
            return Ok(());
        }
        let mut interval = s.adaptive_rho_interval;
        if interval == 0 {
            if iter_start.elapsed().as_secs_f64()
                > s.adaptive_rho_fraction * times.setup_time.max(1e-9)
            {
                interval = iter.max(s.check_termination.max(25));
            } else {
                return Ok(());
            }
        }
        if interval > 0
            && iter % interval == 0
            && self.rho_updates.len() - 1 < s.adaptive_rho_max_adaptions
        {
            *rho_due = true;
        }
        if *rho_due && self.accel_allows_update() {
            *rho_due = false;
            self.recover_mu();
            if self.adapt_rho_vec(times)? {
                if let Some(aa) = self.accelerator.as_mut() {
                    aa.restart();
                }
                let n = self.n;
                for i in 0..self.m {
                    self.w[n + i] = self.s[i] + self.mu[i] / self.rho_vec[i];
                }
            }
        }
        Ok(())
    }

    fn adapt_rho_vec(&mut self, times: &mut Timings) -> Result<bool, CosmoError> {
        let (r_prim, r_dual, max_p, max_d) = self.calculate_residuals(true);
        let rp = r_prim / (max_p + 1e-10);
        let rd = r_dual / (max_d + 1e-10);
        let mut new_rho = self.rho * (rp / (rd + 1e-10)).sqrt();
        new_rho = new_rho.clamp(self.settings.rho_min, self.settings.rho_max);
        let tol = self.settings.adaptive_rho_tolerance;
        if new_rho > tol * self.rho || new_rho < self.rho / tol {
            self.update_rho_vec(new_rho, times)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn update_rho_vec(&mut self, new_rho: f64, times: &mut Timings) -> Result<(), CosmoError> {
        self.rho = new_rho;
        self.rho_vec.fill(new_rho);
        self.cones.apply_rho_scaling(
            &mut self.rho_vec,
            self.settings.rho_min,
            self.settings.rho_eq_over_ineq,
        );
        self.rho_updates.push(new_rho);
        let t = Instant::now();
        self.kkt.as_mut().unwrap().update_rho(&self.rho_vec)?;
        times.factor_update_time += t.elapsed().as_secs_f64();
        Ok(())
    }

    fn unscaled_copies(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = self.w_prev[..self.n].to_vec();
        let mut mu = self.mu.clone();
        let mut s = self.s.clone();
        reverse_scaling(&mut x, &mut mu, &mut s, &self.sm);
        let y = mu.iter().map(|mi| -mi).collect();
        (x, y, s)
    }

    fn calculate_cost(&mut self) -> f64 {
        let n = self.n;
        let x = &self.w_prev[..n];
        0.5 * quad_form(&self.P, x) * self.sm.cinv
            + self.sm.cinv * dot(x, &self.q)
            + self.obj_offset
    }

    fn calculate_residuals(&mut self, ignore_scaling: bool) -> (f64, f64, f64, f64) {
        let n = self.n;
        let x = &self.w_prev[..n];
        // r_prim = A x + s - b
        gemv(&self.A, &mut self.work_m, x, 1.0, 0.0);
        for i in 0..self.m {
            self.work_m[i] += self.s[i] - self.b[i];
        }
        if self.sm.enabled && !ignore_scaling {
            for i in 0..self.m {
                self.work_m[i] *= self.sm.Einv[i];
            }
        }
        let r_prim = inf_norm(&self.work_m);

        // r_dual = P x + q - A' μ
        symv(&self.P, &mut self.work_n, x, 1.0, 0.0);
        for i in 0..n {
            self.work_n[i] += self.q[i];
        }
        gemv_t(&self.A, &mut self.work_n2, &self.mu, 1.0, 0.0);
        for i in 0..n {
            self.work_n[i] -= self.work_n2[i];
        }
        if self.sm.enabled && !ignore_scaling {
            for i in 0..n {
                self.work_n[i] *= self.sm.Dinv[i] * self.sm.cinv;
            }
        }
        let r_dual = inf_norm(&self.work_n);

        let (max_p, max_d) = self.max_res_component_norm(ignore_scaling);
        (r_prim, r_dual, max_p, max_d)
    }

    fn max_res_component_norm(&mut self, ignore_scaling: bool) -> (f64, f64) {
        let n = self.n;
        let unscale = self.sm.enabled && !ignore_scaling;
        let x = &self.w_prev[..n];

        gemv(&self.A, &mut self.work_m, x, 1.0, 0.0);
        if unscale {
            for i in 0..self.m {
                self.work_m[i] *= self.sm.Einv[i];
            }
        }
        let mut max_p = inf_norm(&self.work_m);

        self.work_m.copy_from_slice(&self.s);
        if unscale {
            for i in 0..self.m {
                self.work_m[i] *= self.sm.Einv[i];
            }
        }
        max_p = max_p.max(inf_norm(&self.work_m));

        self.work_m.copy_from_slice(&self.b);
        if unscale {
            for i in 0..self.m {
                self.work_m[i] *= self.sm.Einv[i];
            }
        }
        max_p = max_p.max(inf_norm(&self.work_m));

        let x = &self.w_prev[..n];
        symv(&self.P, &mut self.work_n, x, 1.0, 0.0);
        if unscale {
            for i in 0..n {
                self.work_n[i] *= self.sm.Dinv[i] * self.sm.cinv;
            }
        }
        let mut max_d = inf_norm(&self.work_n);

        self.work_n.copy_from_slice(&self.q);
        if unscale {
            for i in 0..n {
                self.work_n[i] *= self.sm.Dinv[i] * self.sm.cinv;
            }
        }
        max_d = max_d.max(inf_norm(&self.work_n));

        gemv_t(&self.A, &mut self.work_n, &self.mu, 1.0, 0.0);
        if unscale {
            for i in 0..n {
                self.work_n[i] *= self.sm.Dinv[i] * self.sm.cinv;
            }
        }
        max_d = max_d.max(inf_norm(&self.work_n));
        (max_p, max_d)
    }

    fn has_converged(&self, r_prim: f64, r_dual: f64, max_p: f64, max_d: f64, cost: f64) -> bool {
        let s = &self.settings;
        let prim_ok = r_prim < s.eps_abs + s.eps_rel * max_p;
        let dual_ok = r_dual < s.eps_abs + s.eps_rel * max_d;
        let obj_ok = if s.obj_true.is_nan() {
            true
        } else {
            (s.obj_true - cost).abs() <= s.obj_true_tol
        };
        prim_ok && dual_ok && obj_ok
    }

    fn is_primal_infeasible(&mut self) -> bool {
        let eps = self.settings.eps_prim_inf;
        let norm_dy = if self.sm.enabled {
            inf_norm_scaled(&self.sm.E, &self.dy)
        } else {
            inf_norm(&self.dy)
        };
        if norm_dy <= eps {
            return false;
        }
        gemv_t(&self.A, &mut self.work_n, &self.dy, 1.0, 0.0);
        if self.sm.enabled {
            for i in 0..self.n {
                self.work_n[i] *= self.sm.Dinv[i];
            }
        }
        if inf_norm(&self.work_n) > eps * norm_dy {
            return false;
        }
        for yi in self.dy.iter_mut() {
            *yi *= -1.0 / norm_dy;
        }
        let dy_b = dot(&self.dy, &self.b);
        let sf = self.cones.support_function(&mut self.dy, eps);
        sf - dy_b <= eps
    }

    fn is_dual_infeasible(&mut self) -> bool {
        let eps = self.settings.eps_dual_inf;
        let norm_dx = if self.sm.enabled {
            inf_norm_scaled(&self.sm.D, &self.dx)
        } else {
            inf_norm(&self.dx)
        };
        if norm_dx <= eps {
            return false;
        }
        let c = if self.sm.enabled { self.sm.c } else { 1.0 };
        if dot(&self.q, &self.dx) / (norm_dx * c) >= -eps {
            return false;
        }
        symv(&self.P, &mut self.work_n, &self.dx, 1.0, 0.0);
        if self.sm.enabled {
            for i in 0..self.n {
                self.work_n[i] *= self.sm.Dinv[i];
            }
        }
        if inf_norm(&self.work_n) / (norm_dx * c) > eps {
            return false;
        }
        gemv(&self.A, &mut self.work_m, &self.dx, 1.0, 0.0);
        if self.sm.enabled {
            for i in 0..self.m {
                self.work_m[i] *= self.sm.Einv[i];
            }
        }
        scale(&mut self.work_m, 1.0 / norm_dx);
        self.cones.in_polar_recession(&self.work_m, eps)
    }

    pub fn warm_start(&mut self, x: Option<&[f64]>, y: Option<&[f64]>) -> Result<(), CosmoError> {
        // Callers always pass *unscaled* (x, y). If the stored (P, A, q, b) are
        // already Ruiz-scaled, map the warm start into the scaled workspace.
        if let Some(x0) = x {
            if x0.len() != self.n {
                return Err(CosmoError::Dimension("warm start x".into()));
            }
            if self.is_scaled {
                for i in 0..self.n {
                    self.w_prev[i] = x0[i] * self.sm.Dinv[i];
                }
            } else {
                self.w_prev[..self.n].copy_from_slice(x0);
            }
            gemv(&self.A, &mut self.work_m, &self.w_prev[..self.n], 1.0, 0.0);
            for i in 0..self.m {
                self.s[i] = self.b[i] - self.work_m[i];
            }
        }
        if let Some(y0) = y {
            if y0.len() != self.m {
                return Err(CosmoError::Dimension("warm start y".into()));
            }
            if self.is_scaled {
                for i in 0..self.m {
                    self.mu[i] = -y0[i] * self.sm.Einv[i] * self.sm.c;
                }
            } else {
                for i in 0..self.m {
                    self.mu[i] = -y0[i];
                }
            }
        }
        Ok(())
    }

    /// Reset ADMM iterates to zero; keep data and (optionally) the factorisation.
    pub fn reset(&mut self, mode: WarmStartMode) {
        match mode {
            WarmStartMode::ColdStart => {
                self.w.fill(0.0);
                self.w_prev.fill(0.0);
                self.s.fill(0.0);
                self.mu.fill(0.0);
                self.has_solved = false;
                if let Some(aa) = self.accelerator.as_mut() {
                    aa.restart();
                }
            }
            WarmStartMode::WarmStartSolution | WarmStartMode::WarmStartFullState => {}
            WarmStartMode::PersistentFactorization => {
                self.w.fill(0.0);
                self.w_prev.fill(0.0);
                self.s.fill(0.0);
                self.mu.fill(0.0);
            }
        }
    }

    /// Update q (unscaled). Does not refactor.
    pub fn update_q(&mut self, q: &[f64]) -> Result<(), CosmoError> {
        if q.len() != self.n {
            return Err(CosmoError::Dimension("q".into()));
        }
        if self.is_scaled {
            for i in 0..self.n {
                self.q[i] = self.sm.D[i] * q[i] * self.sm.c;
            }
        } else {
            self.q.copy_from_slice(q);
        }
        Ok(())
    }

    /// Update b (unscaled). Does not refactor.
    pub fn update_b(&mut self, b: &[f64]) -> Result<(), CosmoError> {
        if b.len() != self.m {
            return Err(CosmoError::Dimension("b".into()));
        }
        if self.is_scaled {
            for i in 0..self.m {
                self.b[i] = self.sm.E[i] * b[i];
            }
        } else {
            self.b.copy_from_slice(b);
        }
        Ok(())
    }

    /// Update P (unscaled). Same sparsity: numerical refactor. Pattern change: full rebuild.
    pub fn update_p(&mut self, P: &CscMatrix<f64>) -> Result<(), CosmoError> {
        if P.n != self.n || P.m != self.n {
            return Err(CosmoError::Dimension("P".into()));
        }
        let mut P = to_symmetric_triu(P);
        if self.is_scaled {
            P.lrscale(&self.sm.D, &self.sm.D);
            P.scale(self.sm.c);
        }
        let same = P.colptr == self.P.colptr && P.rowval == self.P.rowval;
        self.P = P;
        self.kkt_factored = false;
        if same {
            if let Some(kkt) = self.kkt.as_mut() {
                kkt.rebuild(&self.P, &self.A, self.settings.sigma, &self.rho_vec)?;
                self.kkt_factored = true;
            }
        } else {
            self.kkt = None;
        }
        Ok(())
    }

    /// Update A (unscaled). Refactors the KKT system.
    pub fn update_a(&mut self, A: &CscMatrix<f64>) -> Result<(), CosmoError> {
        if A.m != self.m || A.n != self.n {
            return Err(CosmoError::Dimension("A".into()));
        }
        let mut A = A.clone();
        if self.is_scaled {
            A.lrscale(&self.sm.E, &self.sm.D);
        }
        self.A = A;
        self.kkt_factored = false;
        self.kkt = None;
        Ok(())
    }

    pub fn update_qb(&mut self, q: Option<&[f64]>, b: Option<&[f64]>) -> Result<(), CosmoError> {
        if let Some(q) = q {
            self.update_q(q)?;
        }
        if let Some(b) = b {
            self.update_b(b)?;
        }
        Ok(())
    }
}

fn check_dims(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
) -> Result<(), CosmoError> {
    if P.m != P.n {
        return Err(CosmoError::Dimension("P must be square".into()));
    }
    if P.n != q.len() {
        return Err(CosmoError::Dimension("P and q".into()));
    }
    if A.n != q.len() {
        return Err(CosmoError::Dimension("A and q".into()));
    }
    if A.m != b.len() {
        return Err(CosmoError::Dimension("A and b".into()));
    }
    Ok(())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
