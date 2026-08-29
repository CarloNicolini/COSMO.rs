"""Native Rust COSMO solver (no Julia)."""

from . import _cosmo
from ._cosmo import CosmoSolver, Solution, solve

try:
    from .cvxpy_interface import COSMO_RUST, register
except Exception:  # cvxpy is optional
    COSMO_RUST = None

    def register():
        raise ImportError("cvxpy is required to register COSMO_RUST")

__all__ = [
    "CosmoSolver",
    "Solution",
    "solve",
    "COSMO_RUST",
    "register",
]
