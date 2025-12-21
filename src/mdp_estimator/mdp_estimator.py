"""MDP Structural Parameter Estimator using NFXP.

This module implements:
- compute_log_likelihood: Compute log-likelihood for candidate parameters
- estimate_mle: Main estimation routine using Nelder-Mead
- compute_standard_errors: Numerical Hessian-based standard errors
"""

import numpy as np
import torch
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from mdp_solver import solve_value_function, compute_choice_probability
from mdp_simulator import PanelData


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EstimationResult:
    """Container for estimation results.
    
    Attributes
    ----------
    theta_hat : np.ndarray
        Estimated parameters (beta, gamma, delta)
    std_errors : np.ndarray
        Standard errors for each parameter
    cov_matrix : np.ndarray
        Variance-covariance matrix (3x3)
    log_likelihood : float
        Log-likelihood at the optimum
    n_iterations : int
        Number of outer loop iterations
    converged : bool
        Whether the optimizer converged
    optimization_result : Any
        Full scipy.optimize.OptimizeResult object
    """
    theta_hat: np.ndarray
    std_errors: np.ndarray
    cov_matrix: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool
    optimization_result: Any


# =============================================================================
# Core Functions
# =============================================================================

def compute_log_likelihood(
    theta: np.ndarray,
    data: PanelData,
    solver_params: Dict[str, Any],
) -> float:
    """Compute log-likelihood for candidate parameters.
    
    This function is called once per outer loop iteration. It solves
    the dynamic programming problem for the given parameters and
    computes the log-likelihood of the observed data.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameters (beta, gamma, delta)
    data : PanelData
        Panel data with states and actions arrays
    solver_params : dict
        Solver hyperparameters containing:
        - s_min: float, minimum state value
        - s_max: float, maximum state value
        - hidden_sizes: list[int], network architecture
        - learning_rate: float
        - batch_size: int
        - tolerance: float, convergence tolerance for inner loop
        - max_iterations: int, max iterations for inner loop
        - target_update_freq: int (optional, default 100)
    
    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid parameters.
    """
    # TODO: Implement following pseudocode in estimate_mdp.qmd
    raise NotImplementedError("compute_log_likelihood not yet implemented")


def estimate_mle(
    data: PanelData,
    theta_init: Tuple[float, float, float],
    solver_params: Dict[str, Any],
    maxiter: int = 200,
    verbose: bool = True,
) -> EstimationResult:
    """Estimate structural parameters via MLE using NFXP.
    
    Uses Nelder-Mead optimization in the outer loop, calling
    solve_value_function for each candidate parameter vector.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    theta_init : tuple
        Initial guess (beta, gamma, delta)
    solver_params : dict
        Solver hyperparameters (see compute_log_likelihood)
    maxiter : int
        Maximum outer loop iterations (default 200)
    verbose : bool
        Print progress during optimization (default True)
    
    Returns
    -------
    EstimationResult
        Estimation results including estimates, standard errors,
        and optimization diagnostics
    """
    # TODO: Implement following pseudocode in estimate_mdp.qmd
    raise NotImplementedError("estimate_mle not yet implemented")


def compute_standard_errors(
    theta_hat: np.ndarray,
    data: PanelData,
    solver_params: Dict[str, Any],
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standard errors via numerical Hessian.
    
    Uses central differences to approximate the Hessian of the
    log-likelihood at the MLE, then inverts to get the covariance.
    
    Parameters
    ----------
    theta_hat : np.ndarray
        MLE estimates (beta, gamma, delta)
    data : PanelData
        Panel data
    solver_params : dict
        Solver hyperparameters
    eps : float
        Step size for finite differences (default 1e-4)
    
    Returns
    -------
    std_errors : np.ndarray
        Standard errors for each parameter (shape: (3,))
    cov_matrix : np.ndarray
        Variance-covariance matrix (shape: (3, 3))
    """
    # TODO: Implement following pseudocode in estimate_mdp.qmd
    raise NotImplementedError("compute_standard_errors not yet implemented")

