//! Native Rust COSMO solver: ADMM / operator splitting for convex quadratic-conic programs.
//!
//! COSMO.rs is **not** an interior-point method. It reuses Clarabel.rs sparse CSC
//! storage and QDLDL factorisation as a numerical substrate, and independently
//! implements the COSMO ADMM iteration, cone projections, rho adaptation,
//! over-relaxation, Anderson acceleration, and infeasibility certificates.
//!
//! Canonical problem:
//! ```text
//! minimize    (1/2) x' P x + q' x
//! subject to  A x + s = b
//!             s ∈ K
//! ```
//! with P ⪰ 0 and K a product of zero, nonnegative, second-order, exponential,
//! and power cones (and optional boxes). SDP is not implemented in this milestone.

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

pub mod accelerator;
pub mod algebra;
pub mod cones;
pub mod linsys;
pub mod scaling;
pub mod settings;
pub mod solution;
pub mod solver;

pub use algebra::CscMatrix;
pub use cones::{
    CompositeCone, Cone, DualExponentialCone, DualPowerCone, ExponentialCone, NonnegativeCone,
    PowerCone, SecondOrderCone, ZeroCone,
};
pub use settings::{Settings, WarmStartMode};
pub use solution::{Solution, SolverStatus, Timings};
pub use solver::{CosmoError, CosmoSolver};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _cosmo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
