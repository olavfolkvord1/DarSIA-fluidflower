"""Module containing linear solvers for the mixed form of Beckmann's problem."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections import Counter
from enum import StrEnum
from functools import lru_cache
from threading import Lock
from typing import Optional
from warnings import warn

import numpy as np
import pyamg
import scipy.sparse as sps

import darsia


_RESIDUAL_FACTOR = 10
"""Accept symmetric scaled backward error within ten times the general error,
with a machine-epsilon floor for the general error."""


def _direct_factor(matrix, symmetric: bool):
    if symmetric:
        return sps.linalg.splu(
            matrix,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
    return sps.linalg.splu(matrix)


@lru_cache(maxsize=4)
def _darcy_lu(shape, data, indices, indptr, symmetric: bool = False):
    """Factor the exact CSC entries of a Darcy start in their existing order."""
    arrays = [
        np.frombuffer(raw, dtype).copy() for dtype, raw in (data, indices, indptr)
    ]
    matrix = sps.csc_matrix(tuple(arrays), shape=shape)
    return _direct_factor(matrix, symmetric), Lock()


# Define BeckmannLinearSolverType
class BeckmannLinearSolverType(StrEnum):
    """Enumeration of the linear-solver backends for the Beckmann problem."""

    DIRECT = "direct"
    AMG = "amg"
    CG = "cg"
    KSP = "ksp"


class BeckmannLinearSolver:
    """Class providing linear solver options for Beckmann's problem."""

    @abstractmethod
    def __call__(
        self,
        A: sps.csr_matrix,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """Solve linear system Ax = b.

        Parameters
        ----------
        A : sps.csr_matrix
            System matrix.
        b : np.ndarray
            Right hand side.
        x0 : Optional[np.ndarray]
            Initial guess.

        Returns
        -------
        tuple[np.ndarray, dict]
            Solution and info dictionary.
        """
        pass


class BeckmannDirectSolver(BeckmannLinearSolver):
    """Direct solver for Beckmann's problem.

    Parameters
    ----------
    options : dict
        ``direct_solver_ordering`` is ``"general"`` (default) or ``"symmetric"``.
        Symmetric ordering is eligible only for explicitly identified positive
        pressure reductions. A failed factorization, failed solve or a scaled
        backward error above ten times the general error (with an epsilon
        floor) falls back to general LU.
    """

    def __init__(self, options: dict) -> None:
        self.linear_solver: Optional[sps.linalg.splu] = None
        """Initialize direct solver."""
        self.ordering = options.get("direct_solver_ordering", "general")
        if self.ordering not in ("general", "symmetric"):
            raise ValueError("Direct solver ordering must be general or symmetric.")
        self.symmetric_fallbacks = Counter()
        self.symmetric_factorizations = 0
        self._factor_lock = None
        self._symmetric_matrix = None

    def setup(
        self,
        matrix: sps.csc_matrix,
        *,
        cache_darcy: bool = False,
        symmetric_pressure: bool = False,
    ) -> None:
        """Setup direct solver.

        Parameters
        ----------
        matrix : sps.csr_matrix
            System matrix.
        cache_darcy : bool
            Reuse an exactly matching Darcy-start factorisation.
        symmetric_pressure : bool
            The matrix is a pressure reduction built with a positive flux block.
        """
        self._symmetric_matrix = None
        symmetric = symmetric_pressure and self.ordering == "symmetric"
        try:
            self._setup_factor(matrix, cache_darcy, symmetric)
        except (RuntimeError, ValueError):
            if not symmetric:
                raise
            self.symmetric_fallbacks["factorization"] += 1
            self._setup_factor(matrix, cache_darcy, False)
        else:
            if symmetric:
                self.symmetric_factorizations += 1
                # Keep the actual operator for the reused solves and for fallback.
                self._symmetric_matrix = matrix
                self._matrix_norm = float(np.asarray(abs(matrix).sum(axis=1)).max())
                self._general_fallback = None

    def _setup_factor(self, matrix, cache_darcy, symmetric):
        if cache_darcy:
            csc = matrix.tocsc(copy=False)
            # splu normalizes the input CSC in place. Preserve that state on
            # both cache hits and misses for the next pressure reduction.
            csc.sum_duplicates()
            entries = [
                (array.dtype.str, array.tobytes())
                for array in (csc.data, csc.indices, csc.indptr)
            ]
            self.linear_solver, self._factor_lock = _darcy_lu(
                csc.shape, *entries, symmetric
            )
        else:
            self.linear_solver = _direct_factor(matrix, symmetric)
            self._factor_lock = None

    def __call__(
        self,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve linear system Ax = b using a direct solver.

        Parameters
        ----------
        rhs : np.ndarray
            Right hand side.

        Returns
        -------
        np.ndarray
            Solution.
        """
        if self._symmetric_matrix is None:
            return self._solve(rhs)

        matrix = self._symmetric_matrix
        try:
            solution = self._solve(rhs)
            error = self._backward_error(matrix, solution, rhs)
        except (RuntimeError, ValueError, FloatingPointError):
            solution, error = None, np.inf

        floor = np.finfo(matrix.dtype).eps
        if error <= _RESIDUAL_FACTOR * floor:
            return solution
        if self._general_fallback is None:
            # Use the same storage order as the operator used by this factor.
            general_matrix = matrix.copy()
            general_matrix.has_canonical_format = matrix.has_canonical_format
            general_matrix.has_sorted_indices = matrix.has_sorted_indices
            self._general_fallback = _direct_factor(general_matrix, False)
        general = self._general_fallback.solve(rhs)
        reference = self._backward_error(matrix, general, rhs)
        if not np.isfinite(error) or error > _RESIDUAL_FACTOR * max(reference, floor):
            reason = (
                "solve" if solution is None or not np.isfinite(error) else "residual"
            )
            self.symmetric_fallbacks[reason] += 1
            return general
        return solution

    def _solve(self, rhs):
        if self._factor_lock is None:
            return self.linear_solver.solve(rhs)
        with self._factor_lock:
            return self.linear_solver.solve(rhs)

    def _backward_error(self, matrix, solution, rhs) -> float:
        if not np.isfinite(solution).all():
            return np.inf
        residual = float(np.linalg.norm(matrix.dot(solution) - rhs, ord=np.inf))
        if residual == 0.0:
            return 0.0
        scale = self._matrix_norm * np.linalg.norm(solution, ord=np.inf)
        scale += np.linalg.norm(rhs, ord=np.inf)
        if not np.isfinite(residual) or not np.isfinite(scale) or scale == 0.0:
            return np.inf
        return residual / scale


class BeckmannAMGSolver(BeckmannLinearSolver):
    def __init__(self, amg_options: dict, linear_solver_options: dict) -> None:
        # Setup AMG options
        self.amg_options = {
            "strength": "symmetric",  # change the strength of connection
            "aggregate": "standard",  # use a standard aggregation method
            "smooth": ("jacobi"),  # prolongation smoother
            "presmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "postsmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "coarse_solver": "pinv2",  # pseudo inverse via SVD
            "max_coarse": 100,  # maximum number on a coarse level
        }
        """dict: options for the AMG solver"""

        # Allow to overwrite default options.
        self.amg_options.update(amg_options)

        # Setup linear solver options
        atol = linear_solver_options.get("atol", 1e-6)
        rtol = linear_solver_options.get("rtol", None)
        if not rtol:
            warn("rtol not used for AMG solver.")
        maxiter = linear_solver_options.get("maxiter", 100)
        self.residual_history: list[float] = []
        """list: history of residuals for the AMG solver"""
        self.solver_options = {
            "tol": atol,
            "maxiter": maxiter,
            "residuals": self.residual_history,
        }
        """dict: options for the iterative linear solver"""

        self.linear_solver: Optional[pyamg.amg_core.solve] = None
        """Initialize AMG solver."""

    def setup(self, matrix: sps.csc_matrix) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            self.linear_solver = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            )

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using an AMG solver.

        Parameters
        ----------
        rhs : np.ndarray
            Right hand side.
        x0 : Optional[np.ndarray]
            Initial guess.

        Returns
        -------
        np.ndarray
            Solution.
        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannCGSolver(BeckmannLinearSolver):
    def __init__(self, amg_options: dict, linear_solver_options: dict) -> None:
        # Setup AMG options for preconditioner
        self.amg_options = {
            "strength": "symmetric",  # change the strength of connection
            "aggregate": "standard",  # use a standard aggregation method
            "smooth": ("jacobi"),  # prolongation smoother
            "presmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "postsmoother": (
                "block_gauss_seidel",
                {"sweep": "symmetric", "iterations": 1},
            ),
            "coarse_solver": "pinv2",  # pseudo inverse via SVD
            "max_coarse": 100,  # maximum number on a coarse level
        }
        """dict: options for the AMG solver"""

        # Allow to overwrite default options.
        self.amg_options.update(amg_options)

        # Define solver options
        linear_solver_options = linear_solver_options.get("linear_solver_options", {})
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        self.solver_options = {
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            # "M": amg, # Set in setup
        }
        """dict: options for the iterative linear solver"""

        self.linear_solver: Optional[darsia.linalg.CG] = None
        """Initialize CG solver with AMG preconditioner."""

    def setup(self, matrix: sps.csc_matrix) -> None:
        """Setup an CG solver with AMG preconditioner for the given matrix.

        Parameters
        ----------
        matrix : sps.csc_matrix
            Matrix.

        Notes
        -----
        Sets up the AMG solver (``pyamg``) and a dict of AMG solver options.
        """
        # Define CG solver
        self.linear_solver = darsia.linalg.CG(matrix)

        # Define AMG preconditioner
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Implicit conversion of A to CSR")
            amg = pyamg.smoothed_aggregation_solver(
                matrix, **self.amg_options
            ).aspreconditioner(cycle="V")
        self.solver_options["M"] = amg

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using a CG solver with AMG preconditioner.

        Parameters
        ----------
        rhs : np.ndarray
            Right hand side.
        x0 : Optional[np.ndarray]
            Initial guess.

        Returns
        -------
        np.ndarray
            Solution.
        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannKSPSolver(BeckmannLinearSolver):
    def __init__(
        self,
        linear_solver_options: dict,
    ) -> None:
        # Define solver options
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        approach = linear_solver_options.get("approach", "direct")

        if approach == "direct":
            self.solver_options = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            prec = linear_solver_options.get("pc_type", "hypre")
            self.solver_options = {
                "ksp_type": approach,
                # "ksp_monitor_true_residual": None,
                "ksp_rtol": rtol,
                "ksp_atol": atol,
                "ksp_max_it": maxiter,
                "pc_type": prec,
            }

        # self.field_ises = None
        # """list: list of (field name, indices) tuples to define block structure"""

        # self.nullspace = None
        # """ list of nullspace vectors of the matrix"""

    def _setup(
        self,
        matrix: sps.csc_matrix,
    ) -> None:
        """Setup an KSP solver from PETSc for the given matrix.

        Parameters
        ----------
        matrix : sps.csc_matrix
            Matrix.

        Notes
        -----
        Sets up the PETSc KSP solver and a dict of KSP solver options.
        """
        # Define CG solver
        self.linear_solver = darsia.linalg.KSP(
            matrix
            # , field_ises=self.field_ises, nullspace=self.nullspace
        )
        self.linear_solver.setup(self.solver_options)
        """dict: options for the iterative linear solver"""

    def setup(self, matrix: sps.csc_matrix) -> None:
        """Setup KSP solver for the given matrix.

        Parameters
        ----------
        matrix : sps.csc_matrix
            System matrix.
        """
        if not hasattr(self, "linear_solver"):
            self._setup(matrix)

        # if reuse_solver:
        #    self.linear_solver.setup(self.solver_options)

        ## Free memory if solver needs to be re-setup
        # if hasattr(self, "linear_solver"):
        #    if not reuse_solver:
        #        self.linear_solver.kill()
        #        self._setup(self.fully_reduced_matrix)
        # else:
        #    self._setup(self.fully_reduced_matrix)

    def __call__(
        self,
        rhs: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve linear system Ax = b using an AMG solver.

        Parameters
        ----------
        rhs : np.ndarray
            Right hand side.
        x0 : Optional[np.ndarray]
            Initial guess.

        Returns
        -------
        np.ndarray
            Solution.
        """
        if x0 is None:
            x0 = np.zeros_like(rhs)

        # Solve system
        solution = self.linear_solver.solve(rhs, x0=x0, **self.solver_options)

        return solution


class BeckmannKSPFieldSplitSolver(BeckmannKSPSolver):
    def __init__(
        self,
        linear_solver_options: dict,
        field_ises: list[tuple[str, np.ndarray]],
        nullspace: list[np.ndarray],
    ) -> None:
        # Define solver options
        rtol = linear_solver_options.get("rtol", 1e-6)
        atol = linear_solver_options.get("atol", 0)
        maxiter = linear_solver_options.get("maxiter", 100)
        approach = linear_solver_options.get("approach", "direct")

        if approach == "direct":
            self.solver_options = {
                "ksp_type": "preonly",  # do not apply Krylov iterations
                "pc_type": "lu",
                "pc_factor_shift_type": "inblocks",  # for the zero entries
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            prec = linear_solver_options.get("pc_type", "hypre")
            # Block preconditioning approach
            # the the 0 and 1 in fieldsplit_0 and fieldsplit_1 are the strings
            # passed to the field_ises
            self.solver_options = {
                "ksp_type": approach,
                "ksp_rtol": rtol,
                "ksp_atol": atol,
                "ksp_max_it": maxiter,
                "ksp_monitor": None,
                # "ksp_monitor_true_residual": None, #this is for debugging
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                # use a full factorization of the Schur complement
                # other options are "diag","lower","upper"
                "pc_fieldsplit_schur_precondition": "selfp",
                # selfp -> form an approximate Schur complement
                # using S=-B diag(A)^{-1} B^T
                # which is the exact Schur complement in our case
                # https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
                "fieldsplit_flux_ksp_type": "preonly",
                "fieldsplit_flux_pc_type": "jacobi",
                # use the diagonal of the flux (it is the inverse)
                "fieldsplit_pressure": {
                    "ksp_type": "preonly",
                    "pc_type": prec,
                },
                # an example of the nested dictionary
            }

        self.linear_solver.setup(self.solver_options)
        """dict: options for the iterative linear solver"""

        self.field_ises = field_ises
        """list: list of (field name, indices) tuples to define block structure"""

        self.nullspace: list[np.ndarray] = nullspace
        """ list of nullspace vectors of the matrix"""


class BeckmannLinearSolverFactory:
    """Factory constructing a :class:`BeckmannLinearSolver` from a type and options."""

    @staticmethod
    def create(
        solver_type: BeckmannLinearSolverType,
        options: dict,
    ) -> BeckmannLinearSolver:
        """Factory method to create a linear solver for Beckmann's problem.

        Parameters
        ----------
        solver_type : BeckmannLinearSolverType
            Type of linear solver.
        options : dict
            Options for the linear solver.

        Returns
        -------
        BeckmannLinearSolver
            Instance of the linear solver.
        """

        if solver_type == darsia.BeckmannLinearSolverType.DIRECT:
            return darsia.BeckmannDirectSolver(options)
        elif solver_type == darsia.BeckmannLinearSolverType.AMG:
            return darsia.BeckmannAMGSolver(
                options.get("amg_options", {}),
                options.get("linear_solver_options", {}),
            )
        elif solver_type == darsia.BeckmannLinearSolverType.CG:
            return darsia.BeckmannCGSolver(
                options.get("amg_options", {}),
                options.get("linear_solver_options", {}),
            )
        elif solver_type == darsia.BeckmannLinearSolverType.KSP:
            return darsia.BeckmannKSPSolver(options.get("linear_solver_options", {}))
        # TODO: Use FieldSplit when full formulation... Not implemented right now
        # elif self.linear_solver_type == darsia.BeckmannLinearSolverType.KSP:
        #    self.linear_solver = darsia.BeckmannKSPSolver(
        #        self.options.get("linear_solver_options", {}),
        #        field_ises=[
        #            ("flux", self.flux_indices),
        #            ("pressure", self.pressure_indices),
        #        ],
        #    )
        else:
            raise NotImplementedError("Linear solver not implemented.")
