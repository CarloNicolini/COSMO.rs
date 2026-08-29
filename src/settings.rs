//! Solver settings. Defaults match COSMO.jl where practical, with
//! correctness-oriented choices (safeguarding on, adaptive rho on).

#[derive(Clone, Debug)]
pub struct Settings {
    /// ADMM penalty ρ (scalar seed; per-constraint ρvec is derived).
    pub rho: f64,
    /// ADMM primal proximal parameter σ.
    pub sigma: f64,
    /// Over-relaxation parameter α. COSMO default is 1.6.
    pub alpha: f64,
    pub eps_abs: f64,
    pub eps_rel: f64,
    pub eps_prim_inf: f64,
    pub eps_dual_inf: f64,
    pub max_iter: usize,
    pub verbose: bool,
    pub verbose_timing: bool,
    pub check_termination: usize,
    pub check_infeasibility: usize,
    /// Number of Ruiz scaling iterations. 0 disables scaling.
    pub scaling: usize,
    pub min_scaling: f64,
    pub max_scaling: f64,
    pub adaptive_rho: bool,
    pub adaptive_rho_interval: usize,
    pub adaptive_rho_tolerance: f64,
    pub adaptive_rho_fraction: f64,
    pub adaptive_rho_max_adaptions: usize,
    pub rho_min: f64,
    pub rho_max: f64,
    pub rho_eq_over_ineq: f64,
    pub cosmo_infty: f64,
    pub time_limit: f64,
    /// Enable Type-II Anderson acceleration of the ADMM operator.
    pub accelerate: bool,
    pub accelerator_memory: usize,
    pub accelerator_min_mem: usize,
    pub safeguard: bool,
    pub safeguard_tol: f64,
    pub obj_true: f64,
    pub obj_true_tol: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            rho: 0.1,
            sigma: 1e-6,
            alpha: 1.6,
            eps_abs: 1e-5,
            eps_rel: 1e-5,
            eps_prim_inf: 1e-4,
            eps_dual_inf: 1e-4,
            max_iter: 5000,
            verbose: false,
            verbose_timing: true,
            check_termination: 25,
            check_infeasibility: 40,
            scaling: 10,
            min_scaling: 1e-4,
            max_scaling: 1e4,
            adaptive_rho: true,
            adaptive_rho_interval: 40,
            adaptive_rho_tolerance: 5.0,
            adaptive_rho_fraction: 0.4,
            adaptive_rho_max_adaptions: usize::MAX,
            rho_min: 1e-6,
            rho_max: 1e6,
            rho_eq_over_ineq: 1e3,
            cosmo_infty: 1e20,
            time_limit: 0.0,
            accelerate: true,
            accelerator_memory: 15,
            accelerator_min_mem: 3,
            safeguard: true,
            safeguard_tol: 2.0,
            obj_true: f64::NAN,
            obj_true_tol: 1e-3,
        }
    }
}

impl Settings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings biased toward reproducible, slightly slower solves.
    pub fn robust() -> Self {
        let mut s = Self::default();
        s.accelerate = false;
        s.eps_abs = 1e-6;
        s.eps_rel = 1e-6;
        s.max_iter = 10_000;
        s
    }
}

/// How a subsequent solve should reuse solver state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmStartMode {
    /// Zero x, s, duals; keep factorisation if the KKT pattern is valid.
    ColdStart,
    /// Keep x / s / y from the previous solve.
    WarmStartSolution,
    /// Keep the full ADMM operator state (w, rho, accelerator history).
    WarmStartFullState,
    /// Keep the numerical KKT factorisation (valid when P, A, rho, σ unchanged).
    PersistentFactorization,
}

impl Default for WarmStartMode {
    fn default() -> Self {
        WarmStartMode::WarmStartSolution
    }
}
