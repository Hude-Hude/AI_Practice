"""MDP Structural Parameter Estimator using NFXP.

This module implements:
- compute_log_likelihood: Compute log-likelihood for candidate parameters
- estimate_mle: Estimation using Nelder-Mead optimization
- grid_search_mle: Estimation using explicit grid search
- compute_standard_errors: Numerical Hessian-based standard errors
"""

import os
import warnings
import numpy as np
import torch
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

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
    v0_init_state: Optional[dict] = None,
    v1_init_state: Optional[dict] = None,
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
    v0_init_state : dict, optional
        Initial state dict for v0 network (warm-start)
    v1_init_state : dict, optional
        Initial state dict for v1 network (warm-start)
    
    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid parameters.
    """
    # Unpack parameters
    beta, gamma, delta = theta
    
    # Validate parameter bounds (return -inf for invalid)
    if beta <= 0 or gamma <= 0 or gamma >= 1 or delta <= 0 or delta >= 1:
        return -np.inf
    
    # === INNER LOOP: Solve value functions for this theta ===
    v0_net, v1_net, losses, n_iter = solve_value_function(
        beta=beta,
        gamma=gamma,
        delta=delta,
        s_min=solver_params['s_min'],
        s_max=solver_params['s_max'],
        hidden_sizes=solver_params['hidden_sizes'],
        learning_rate=solver_params['learning_rate'],
        batch_size=solver_params['batch_size'],
        tolerance=solver_params['tolerance'],
        max_iterations=solver_params['max_iterations'],
        target_update_freq=solver_params.get('target_update_freq', 100),
        v0_init_state=v0_init_state,
        v1_init_state=v1_init_state,
    )
    
    # Check convergence (optional: warn if not converged)
    if n_iter == solver_params['max_iterations']:
        warnings.warn(f"Inner loop did not converge for theta = {theta}")
    
    # === Compute choice probabilities at all observed states ===
    # Flatten panel data for batch computation
    states_flat = data.states.flatten()  # shape: (N * T,)
    actions_flat = data.actions.flatten()  # shape: (N * T,)
    
    # Convert to tensor
    s_tensor = torch.tensor(states_flat, dtype=torch.float32)
    
    # Compute P(a=1 | s) for all observations
    with torch.no_grad():
        _, p1 = compute_choice_probability(v0_net, v1_net, s_tensor)
        p1 = p1.numpy()
    
    # Clip probabilities for numerical stability
    eps = 1e-10
    p1 = np.clip(p1, eps, 1 - eps)
    
    # === Compute log-likelihood ===
    # L = Σ [a * log(p1) + (1-a) * log(1-p1)]
    log_lik = np.sum(
        actions_flat * np.log(p1) + 
        (1 - actions_flat) * np.log(1 - p1)
    )
    
    return log_lik


def estimate_mle(
    data: PanelData,
    theta_init: Tuple[float, float, float],
    solver_params: Dict[str, Any],
    maxiter: int = 200,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
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
    pretrained_path : str, optional
        Path to directory containing pre-trained networks (v0_net.pt, v1_net.pt).
        If provided, uses these as warm-start initialization for all solver calls.
    
    Returns
    -------
    EstimationResult
        Estimation results including estimates, standard errors,
        and optimization diagnostics
    """
    theta_init = np.array(theta_init)
    
    # Load pre-trained networks for warm-starting (if provided)
    v0_init_state = None
    v1_init_state = None
    if pretrained_path is not None:
        v0_path = os.path.join(pretrained_path, 'v0_net.pt')
        v1_path = os.path.join(pretrained_path, 'v1_net.pt')
        if os.path.exists(v0_path) and os.path.exists(v1_path):
            v0_init_state = torch.load(v0_path, weights_only=True)
            v1_init_state = torch.load(v1_path, weights_only=True)
            if verbose:
                print(f"Loaded pre-trained networks from {pretrained_path} for warm-starting")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}, using random init")
    
    # Track evaluations
    eval_count = [0]
    
    # Define objective function (negative log-likelihood)
    def neg_log_likelihood(theta):
        eval_count[0] += 1
        log_lik = compute_log_likelihood(
            theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
        if verbose and eval_count[0] % 10 == 0:
            print(f"Eval {eval_count[0]}: theta={theta}, LL={log_lik:.2f}")
        return -log_lik
    
    # Run Nelder-Mead optimization
    result = optimize.minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        method='Nelder-Mead',
        options={
            'maxiter': maxiter,
            'xatol': 1e-4,      # Parameter tolerance
            'fatol': 1e-4,      # Function value tolerance
            'disp': verbose,
            'adaptive': True,   # Adapt simplex to parameter scales
        }
    )
    
    theta_hat = result.x
    log_lik_hat = -result.fun
    
    # Compute standard errors (also use warm-start)
    if verbose:
        print("Computing standard errors...")
    std_errors, cov_matrix = compute_standard_errors(
        theta_hat, data, solver_params,
        v0_init_state=v0_init_state,
        v1_init_state=v1_init_state,
    )
    
    return EstimationResult(
        theta_hat=theta_hat,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        log_likelihood=log_lik_hat,
        n_iterations=result.nit,
        converged=result.success,
        optimization_result=result,
    )


def grid_search_mle(
    data: PanelData,
    bounds: Dict[str, Tuple[float, float]],
    n_points: int,
    solver_params: Dict[str, Any],
    verbose: bool = True,
    compute_se: bool = True,
    pretrained_path: Optional[str] = None,
) -> EstimationResult:
    """Estimate structural parameters via grid search.
    
    Evaluates log-likelihood on a grid of parameter values and
    returns the grid point with highest likelihood.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    bounds : dict
        Parameter bounds: {'beta': (min, max), 'gamma': (min, max), 'delta': (min, max)}
    n_points : int
        Number of grid points per dimension (total evaluations = n_points^3)
    solver_params : dict
        Solver hyperparameters (see compute_log_likelihood)
    verbose : bool
        Print progress during search (default True)
    compute_se : bool
        Whether to compute standard errors at best point (default True)
    pretrained_path : str, optional
        Path to directory containing pre-trained networks (v0_net.pt, v1_net.pt).
        If provided, uses these as warm-start initialization for all solver calls.
    
    Returns
    -------
    EstimationResult
        Estimation results including estimates and optionally standard errors
    """
    # Load pre-trained networks for warm-starting (if provided)
    v0_init_state = None
    v1_init_state = None
    if pretrained_path is not None:
        v0_path = os.path.join(pretrained_path, 'v0_net.pt')
        v1_path = os.path.join(pretrained_path, 'v1_net.pt')
        if os.path.exists(v0_path) and os.path.exists(v1_path):
            v0_init_state = torch.load(v0_path, weights_only=True)
            v1_init_state = torch.load(v1_path, weights_only=True)
            if verbose:
                print(f"Loaded pre-trained networks from {pretrained_path} for warm-starting")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}, using random init")
    
    # Create grid for each parameter
    beta_grid = np.linspace(bounds['beta'][0], bounds['beta'][1], n_points)
    gamma_grid = np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_points)
    delta_grid = np.linspace(bounds['delta'][0], bounds['delta'][1], n_points)
    
    # Total evaluations
    total_evals = n_points ** 3
    
    if verbose:
        print(f"Grid search: {n_points} points per dimension = {total_evals} total evaluations")
        print(f"  beta  in [{bounds['beta'][0]:.3f}, {bounds['beta'][1]:.3f}]")
        print(f"  gamma in [{bounds['gamma'][0]:.3f}, {bounds['gamma'][1]:.3f}]")
        print(f"  delta in [{bounds['delta'][0]:.3f}, {bounds['delta'][1]:.3f}]")
    
    # Store all results
    results = []
    best_ll = -np.inf
    best_theta = None
    eval_count = 0
    
    # Iterate over grid
    for beta in beta_grid:
        for gamma in gamma_grid:
            for delta in delta_grid:
                eval_count += 1
                theta = np.array([beta, gamma, delta])
                
                # Compute log-likelihood (with warm-start)
                ll = compute_log_likelihood(
                    theta, data, solver_params,
                    v0_init_state=v0_init_state,
                    v1_init_state=v1_init_state,
                )
                results.append({'theta': theta, 'log_likelihood': ll})
                
                # Track best
                if ll > best_ll:
                    best_ll = ll
                    best_theta = theta.copy()
                
                if verbose and eval_count % 10 == 0:
                    print(f"  Eval {eval_count}/{total_evals}: best LL = {best_ll:.2f}")
    
    if verbose:
        print(f"\nGrid search complete.")
        print(f"  Best theta: beta={best_theta[0]:.4f}, gamma={best_theta[1]:.4f}, delta={best_theta[2]:.4f}")
        print(f"  Best LL: {best_ll:.2f}")
    
    # Compute standard errors at best point (optional, with warm-start)
    if compute_se:
        if verbose:
            print("Computing standard errors...")
        std_errors, cov_matrix = compute_standard_errors(
            best_theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
    else:
        std_errors = np.full(3, np.nan)
        cov_matrix = np.full((3, 3), np.nan)
    
    # Create result object
    # Store grid results in optimization_result for inspection
    grid_results = {
        'grid': {
            'beta': beta_grid,
            'gamma': gamma_grid,
            'delta': delta_grid,
        },
        'evaluations': results,
    }
    
    return EstimationResult(
        theta_hat=best_theta,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        log_likelihood=best_ll,
        n_iterations=total_evals,
        converged=True,  # Grid search always "converges"
        optimization_result=grid_results,
    )


def compute_standard_errors(
    theta_hat: np.ndarray,
    data: PanelData,
    solver_params: Dict[str, Any],
    eps: float = 1e-4,
    v0_init_state: Optional[dict] = None,
    v1_init_state: Optional[dict] = None,
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
    v0_init_state : dict, optional
        Initial state dict for v0 network (warm-start)
    v1_init_state : dict, optional
        Initial state dict for v1 network (warm-start)
    
    Returns
    -------
    std_errors : np.ndarray
        Standard errors for each parameter (shape: (3,))
    cov_matrix : np.ndarray
        Variance-covariance matrix (shape: (3, 3))
    """
    n_params = 3
    theta_hat = np.array(theta_hat)
    
    # Compute numerical Hessian via central differences
    hessian = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            # f(θ + ei*eps + ej*eps)
            theta_pp = theta_hat.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps
            f_pp = compute_log_likelihood(theta_pp, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ + ei*eps - ej*eps)
            theta_pm = theta_hat.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps
            f_pm = compute_log_likelihood(theta_pm, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ - ei*eps + ej*eps)
            theta_mp = theta_hat.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps
            f_mp = compute_log_likelihood(theta_mp, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ - ei*eps - ej*eps)
            theta_mm = theta_hat.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps
            f_mm = compute_log_likelihood(theta_mm, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # Second derivative approximation
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
    
    # Symmetrize Hessian (numerical differentiation can introduce asymmetry)
    hessian = (hessian + hessian.T) / 2
    
    # Covariance matrix = inverse of negative Hessian (information matrix)
    info_matrix = -hessian
    
    # Invert to get covariance matrix
    try:
        cov_matrix = np.linalg.inv(info_matrix)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # Singular matrix - return NaN
        warnings.warn("Information matrix is singular, cannot compute standard errors")
        cov_matrix = np.full((n_params, n_params), np.nan)
        std_errors = np.full(n_params, np.nan)
    
    return std_errors, cov_matrix

