//! Solution, status, and timing records.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverStatus {
    Undetermined,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterReached,
    TimeLimitReached,
    Unsolved,
    NumericalError,
}

impl SolverStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SolverStatus::Undetermined => "Undetermined",
            SolverStatus::Solved => "Solved",
            SolverStatus::PrimalInfeasible => "Primal_infeasible",
            SolverStatus::DualInfeasible => "Dual_infeasible",
            SolverStatus::MaxIterReached => "Max_iter_reached",
            SolverStatus::TimeLimitReached => "Time_limit_reached",
            SolverStatus::Unsolved => "Unsolved",
            SolverStatus::NumericalError => "Numerical_error",
        }
    }

    pub fn is_solved(self) -> bool {
        self == SolverStatus::Solved
    }
}

impl fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub solver_time: f64,
    pub setup_time: f64,
    pub scaling_time: f64,
    pub init_factor_time: f64,
    pub factor_update_time: f64,
    pub iter_time: f64,
    pub proj_time: f64,
    pub post_time: f64,
    pub accelerate_time: f64,
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub s: Vec<f64>,
    pub obj_val: f64,
    pub iter: usize,
    pub safeguarding_iter: usize,
    pub status: SolverStatus,
    pub obj_offset: f64,
    pub r_prim: f64,
    pub r_dual: f64,
    pub max_norm_prim: f64,
    pub max_norm_dual: f64,
    pub rho_updates: Vec<f64>,
    pub times: Timings,
}

impl Solution {
    pub fn empty() -> Self {
        Self {
            x: vec![],
            y: vec![],
            s: vec![],
            obj_val: f64::NAN,
            iter: 0,
            safeguarding_iter: 0,
            status: SolverStatus::Undetermined,
            obj_offset: 0.0,
            r_prim: f64::NAN,
            r_dual: f64::NAN,
            max_norm_prim: f64::NAN,
            max_norm_dual: f64::NAN,
            rho_updates: vec![],
            times: Timings::default(),
        }
    }

    /// Objective including a constant offset (CVXPY `OFFSET`).
    pub fn obj_val_with_offset(&self) -> f64 {
        self.obj_val + self.obj_offset
    }
}
