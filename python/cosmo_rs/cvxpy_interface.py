"""CVXPY adapter for the native Rust COSMO solver.

This module does not invoke Julia. It extracts CVXPY canonical conic data and
calls the Rust COSMO implementation through the PyO3 extension.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    import cvxpy as cp
    import cvxpy.settings as s
    from cvxpy.constraints import ExpCone, PowCone3D, SOC
    from cvxpy.reductions.solution import Solution, failure_solution
    from cvxpy.reductions.solvers import utilities
    from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
    from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
except ImportError as exc:  # pragma: no cover
    raise ImportError("cvxpy is required for cosmo_rs.cvxpy_interface") from exc

from cosmo_rs._cosmo import CosmoSolver as RustSolver


class COSMO_RUST(ConicSolver):
    """CVXPY interface for COSMO.rs (native Rust)."""

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]
    REQUIRED_MODULES = ("cosmo_rs",)
    EXP_CONE_ORDER = [0, 1, 2]

    STATUS_MAP = {
        "Solved": s.OPTIMAL,
        "Primal_infeasible": s.INFEASIBLE,
        "Dual_infeasible": s.UNBOUNDED,
        "Max_iter_reached": s.USER_LIMIT,
        "Time_limit_reached": s.USER_LIMIT,
        "Unsolved": s.SOLVER_ERROR,
        "Numerical_error": s.SOLVER_ERROR,
        "Undetermined": s.SOLVER_ERROR,
    }

    def name(self):
        return "COSMO_RUST"

    def import_solver(self) -> None:
        import cosmo_rs  # noqa: F401

    def supports_quad_obj(self) -> bool:
        return True

    def invert(self, solution, inverse_data):
        attr = {
            s.SOLVE_TIME: getattr(solution, "solve_time", 0.0),
            s.SETUP_TIME: getattr(solution, "setup_time", 0.0),
            s.NUM_ITERS: getattr(solution, "iter", 0),
            s.EXTRA_STATS: {
                "r_prim": getattr(solution, "r_prim", None),
                "r_dual": getattr(solution, "r_dual", None),
                "factor_time": getattr(solution, "factor_time", None),
                "proj_time": getattr(solution, "proj_time", None),
            },
        }
        status = self.STATUS_MAP.get(str(solution.status), s.SOLVER_ERROR)
        y = np.array(solution.y, dtype=float) if solution.y is not None else None
        dual_vars = {}
        if y is not None:
            zero_idx = inverse_data[ConicSolver.DIMS].zero
            eq_dual_vars = utilities.get_dual_values(
                y[:zero_idx],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual_vars = utilities.get_dual_values(
                y[zero_idx:],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = eq_dual_vars | ineq_dual_vars

        if status in s.SOLUTION_PRESENT:
            primal_val = float(solution.obj_val)
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: np.array(solution.x, dtype=float)}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        return failure_solution(status, attr, dual_vars)

    @staticmethod
    def dims_to_cones(dims):
        cones = []
        if dims.zero > 0:
            cones.append(("zero", int(dims.zero)))
        if dims.nonneg > 0:
            cones.append(("nonnegative", int(dims.nonneg)))
        for dim in dims.soc:
            cones.append(("soc", int(dim)))
        if getattr(dims, "psd", None):
            if len(dims.psd) > 0:
                raise ValueError("COSMO.rs does not implement SDP in this milestone")
        for _ in range(int(dims.exp)):
            cones.append(("exp",))
        for alpha in dims.p3d:
            cones.append(("power", float(alpha)))
        if getattr(dims, "pnd", None) and len(dims.pnd) > 0:
            raise ValueError("COSMO.rs does not implement ND power cones yet")
        return cones

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        A = sp.csc_matrix(data[s.A])
        b = np.array(data[s.B], dtype=float)
        q = np.array(data[s.C], dtype=float)
        if s.P in data:
            P = sp.csc_matrix(sp.triu(data[s.P]))
        else:
            P = sp.csc_matrix((q.size, q.size))

        cones = self.dims_to_cones(data[self.DIMS])
        opts = dict(solver_opts)
        opts.pop("use_quad_obj", None)
        opts["verbose"] = bool(verbose)
        solver = RustSolver(P, q, A, b, cones, **opts)

        if warm_start and solver_cache is not None and self.name() in solver_cache:
            old = solver_cache[self.name()]
            try:
                solver.warm_start(x=list(old.x), y=list(old.y))
            except Exception:
                pass

        result = solver.solve()
        if solver_cache is not None:
            solver_cache[self.name()] = result
        return result

    def cite(self, data):
        return (
            "@Article{Garstka_2021,\n"
            "  author  = {Michael Garstka and Mark Cannon and Paul Goulart},\n"
            "  journal = {Journal of Optimization Theory and Applications},\n"
            "  title   = {{COSMO}: A Conic Operator Splitting Method for Convex Conic Problems},\n"
            "  volume  = {190},\n"
            "  number  = {3},\n"
            "  pages   = {779--810},\n"
            "  year    = {2021},\n"
            "}"
        )


def register() -> None:
    """Register COSMO_RUST so ``problem.solve(solver='COSMO_RUST')`` works."""
    import cvxpy.reductions.solvers.defines as ds

    name = "COSMO_RUST"
    s.COSMO_RUST = name
    inst = COSMO_RUST()
    ds.SOLVER_MAP_CONIC[name] = inst
    if name not in ds.INSTALLED_SOLVERS:
        ds.INSTALLED_SOLVERS.append(name)
    if name not in ds.CONIC_SOLVERS:
        ds.CONIC_SOLVERS.append(name)
    if hasattr(cp, "COSMO_RUST"):
        return
    setattr(cp, "COSMO_RUST", name)
