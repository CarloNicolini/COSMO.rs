//! PyO3 bindings. The numerical core does not depend on this module.

#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use crate::cones::Cone;
use crate::settings::Settings;
use crate::solution::{Solution, SolverStatus};
use crate::solver::CosmoSolver;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct PyCsc(CscMatrix<f64>);

impl<'a> FromPyObject<'a> for PyCsc {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> PyResult<Self> {
        let nzval: Vec<f64> = obj.getattr("data")?.extract()?;
        let rowval: Vec<usize> = obj.getattr("indices")?.extract()?;
        let colptr: Vec<usize> = obj.getattr("indptr")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
        Ok(PyCsc(CscMatrix::new(
            shape[0], shape[1], colptr, rowval, nzval,
        )))
    }
}

fn cones_from_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Cone>> {
    let mut cones = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let kind: String = if let Ok(s) = item.get_item(0) {
            s.extract()?
        } else {
            item.getattr("kind")?.extract()?
        };
        let kind = kind.to_lowercase();
        match kind.as_str() {
            "zero" | "z" | "eq" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::zero(dim));
            }
            "nonnegative" | "nonneg" | "l" | "nn" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::nonnegative(dim));
            }
            "soc" | "q" | "secondorder" => {
                let dim: usize = item.get_item(1)?.extract()?;
                cones.push(Cone::second_order(dim));
            }
            "exp" | "exponential" | "ep" => cones.push(Cone::exponential()),
            "dualexp" | "ed" => cones.push(Cone::dual_exponential()),
            "power" | "pow" | "p" => {
                let alpha: f64 = item.get_item(1)?.extract()?;
                cones.push(Cone::power(alpha));
            }
            "dualpower" => {
                let alpha: f64 = item.get_item(1)?.extract()?;
                cones.push(Cone::dual_power(alpha));
            }
            "psd" | "sdp" => {
                return Err(PyValueError::new_err(
                    "SDP / PSD cones are not implemented in this COSMO.rs milestone",
                ));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown cone kind '{other}'"
                )));
            }
        }
    }
    Ok(cones)
}

fn settings_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Settings> {
    let mut s = Settings::default();
    s.verbose = false;
    let Some(kwargs) = kwargs else {
        return Ok(s);
    };
    for (k, v) in kwargs.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "rho" => s.rho = v.extract()?,
            "sigma" => s.sigma = v.extract()?,
            "alpha" => s.alpha = v.extract()?,
            "eps_abs" => s.eps_abs = v.extract()?,
            "eps_rel" => s.eps_rel = v.extract()?,
            "max_iter" => s.max_iter = v.extract()?,
            "verbose" => s.verbose = v.extract()?,
            "scaling" => s.scaling = v.extract()?,
            "adaptive_rho" => s.adaptive_rho = v.extract()?,
            "accelerate" => s.accelerate = v.extract()?,
            "time_limit" => s.time_limit = v.extract()?,
            "check_termination" => s.check_termination = v.extract()?,
            "eps_prim_inf" => s.eps_prim_inf = v.extract()?,
            "eps_dual_inf" => s.eps_dual_inf = v.extract()?,
            "use_quad_obj" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "unrecognized COSMO.rs setting '{other}'"
                )));
            }
        }
    }
    Ok(s)
}

#[pyclass(name = "Solution")]
struct PySolution {
    #[pyo3(get)]
    x: Vec<f64>,
    #[pyo3(get)]
    y: Vec<f64>,
    #[pyo3(get)]
    s: Vec<f64>,
    #[pyo3(get)]
    obj_val: f64,
    #[pyo3(get)]
    iter: usize,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    r_prim: f64,
    #[pyo3(get)]
    r_dual: f64,
    #[pyo3(get)]
    setup_time: f64,
    #[pyo3(get)]
    solve_time: f64,
    #[pyo3(get)]
    factor_time: f64,
    #[pyo3(get)]
    proj_time: f64,
    #[pyo3(get)]
    iter_time: f64,
}

impl From<&Solution> for PySolution {
    fn from(sol: &Solution) -> Self {
        Self {
            x: sol.x.clone(),
            y: sol.y.clone(),
            s: sol.s.clone(),
            obj_val: sol.obj_val + sol.obj_offset,
            iter: sol.iter,
            status: sol.status.as_str().to_string(),
            r_prim: sol.r_prim,
            r_dual: sol.r_dual,
            setup_time: sol.times.setup_time,
            solve_time: sol.times.solver_time,
            factor_time: sol.times.init_factor_time + sol.times.factor_update_time,
            proj_time: sol.times.proj_time,
            iter_time: sol.times.iter_time,
        }
    }
}

#[pyclass(name = "CosmoSolver")]
struct PyCosmoSolver {
    inner: CosmoSolver,
}

#[pymethods]
impl PyCosmoSolver {
    #[new]
    #[pyo3(signature = (P, q, A, b, cones, **kwargs))]
    fn new(
        P: PyCsc,
        q: Vec<f64>,
        A: PyCsc,
        b: Vec<f64>,
        cones: Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let settings = settings_from_kwargs(kwargs)?;
        let cones = cones_from_list(&cones)?;
        let inner = CosmoSolver::new(&P.0, &q, &A.0, &b, cones, settings)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn solve(&mut self, py: Python<'_>) -> PyResult<PySolution> {
        let sol = py
            .allow_threads(|| self.inner.solve())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PySolution::from(sol))
    }

    fn update_q(&mut self, q: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_q(&q)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn update_b(&mut self, b: Vec<f64>) -> PyResult<()> {
        self.inner
            .update_b(&b)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn warm_start(&mut self, x: Option<Vec<f64>>, y: Option<Vec<f64>>) -> PyResult<()> {
        self.inner
            .warm_start(x.as_deref(), y.as_deref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
#[pyo3(signature = (P, q, A, b, cones, **kwargs))]
fn solve(
    py: Python<'_>,
    P: PyCsc,
    q: Vec<f64>,
    A: PyCsc,
    b: Vec<f64>,
    cones: Bound<'_, PyAny>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolution> {
    let mut solver = PyCosmoSolver::new(P, q, A, b, cones, kwargs)?;
    solver.solve(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCosmoSolver>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add("STATUS_SOLVED", SolverStatus::Solved.as_str())?;
    Ok(())
}
