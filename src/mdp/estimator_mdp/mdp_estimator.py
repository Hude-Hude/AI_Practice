"""MDP Structural Parameter Estimator using NFXP.

This module implements:
- estimate_gamma_ols: Estimate γ directly from transition data (OLS)
- estimate_beta_1d: Estimate β with γ from OLS and δ calibrated (1D grid search)
- compute_log_likelihood: Compute log-likelihood for candidate parameters
- estimate_mle: Estimation using Nelder-Mead optimization (joint, not recommended)
- grid_search_mle: Estimation using explicit grid search (joint, not recommended)
- compute_standard_errors: Numerical Hessian-based standard errors
"""

import os
import warnings
import numpy as np
import torch
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from mdp.solver_mdp import solve_value_function, compute_choice_probability
from mdp.simulator_mdp import PanelData


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


@dataclass
class TwoStepResult:
    """Container for two-step estimation results.
    
    Attributes
    ----------
    beta_hat : float
        Estimated reward coefficient
    gamma_hat : float
        Estimated state decay rate (from OLS)
    delta_calibrated : float
        Calibrated discount factor (fixed)
    beta_se : float
        Standard error for beta
    gamma_se : float
        Standard error for gamma (from OLS)
    log_likelihood : float
        Log-likelihood at the optimum
    n_evaluations : int
        Number of likelihood evaluations for beta
    gamma_ols_details : dict
        Details from gamma OLS estimation
    """
    beta_hat: float
    gamma_hat: float
    delta_calibrated: float
    beta_se: float
    gamma_se: float
    log_likelihood: float
    n_evaluations: int
    gamma_ols_details: dict


# =============================================================================
# Two-Step Estimation (Recommended)
# =============================================================================

def estimate_gamma_ols(data: PanelData) -> Tuple[float, float, dict]:
    """Estimate γ directly from observed state transitions using OLS.
    
    The transition equation is deterministic:
        s_{t+1} = (1 - γ) * s_t + a_t
    
    Rearranging:
        (s_{t+1} - a_t) = (1 - γ) * s_t
    
    This is a regression through the origin: Y = (1 - γ) * X
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    
    Returns
    -------
    gamma_hat : float
        Estimated decay rate
    std_error : float
        Standard error of the estimate
    details : dict
        Additional details (n_obs, r_squared, residual_std)
    """
    # Extract current states, next states, and actions
    # states has shape (n_agents, n_periods), we need transitions
    s_current = data.states[:, :-1].flatten()    # s_t (exclude last period)
    s_next = data.states[:, 1:].flatten()        # s_{t+1} (exclude first period)
    a_current = data.actions[:, :-1].flatten()   # a_t (exclude last period)
    
    # Construct regression variables
    Y = s_next - a_current    # Dependent variable: s_{t+1} - a_t
    X = s_current             # Independent variable: s_t
    
    # Filter out observations where s_t ≈ 0 (avoid division issues)
    valid = X > 1e-6
    Y = Y[valid]
    X = X[valid]
    n_obs = len(X)
    
    # OLS through origin: coef = Σ(X*Y) / Σ(X²)
    # This estimates (1 - γ)
    coef = np.sum(X * Y) / np.sum(X ** 2)
    gamma_hat = 1.0 - coef
    
    # Compute residuals and standard error
    residuals = Y - coef * X
    residual_ss = np.sum(residuals ** 2)
    mse = residual_ss / (n_obs - 1)  # Mean squared error (df = n - 1 for intercept-free)
    
    # Variance of coefficient: Var(coef) = σ² / Σ(X²)
    var_coef = mse / np.sum(X ** 2)
    std_error = np.sqrt(var_coef)  # SE for (1-γ), same magnitude as SE for γ
    
    # R-squared (for regression through origin)
    total_ss = np.sum(Y ** 2)  # Note: no mean subtraction for origin regression
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    
    details = {
        'n_obs': n_obs,
        'coef_1_minus_gamma': coef,
        'residual_std': np.sqrt(mse),
        'r_squared': r_squared,
    }
    
    return gamma_hat, std_error, details


def estimate_beta_1d(
    data: PanelData,
    gamma_fixed: float,
    delta_fixed: float,
    solver_params: Dict[str, Any],
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    n_points: int = 20,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
) -> Tuple[float, float, float, dict]:
    """Estimate β via 1D grid search with γ and δ fixed.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    gamma_fixed : float
        Fixed γ value (from OLS estimation)
    delta_fixed : float
        Fixed δ value (calibrated)
    solver_params : dict
        Solver hyperparameters
    beta_bounds : tuple
        (min, max) bounds for β grid search
    n_points : int
        Number of grid points for β (default 20)
    verbose : bool
        Print progress (default True)
    pretrained_path : str, optional
        Path to pre-trained networks for warm-start
    
    Returns
    -------
    beta_hat : float
        Estimated β
    std_error : float
        Standard error of β estimate
    log_likelihood : float
        Log-likelihood at optimum
    details : dict
        Grid search details (all evaluations)
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
                print(f"Loaded pre-trained networks from {pretrained_path}")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}")
    
    # Create 1D grid for β
    beta_grid = np.linspace(beta_bounds[0], beta_bounds[1], n_points)
    
    if verbose:
        print(f"1D Grid search for β:")
        print(f"  β in [{beta_bounds[0]:.3f}, {beta_bounds[1]:.3f}], {n_points} points")
        print(f"  γ fixed at {gamma_fixed:.4f} (from OLS)")
        print(f"  δ fixed at {delta_fixed:.4f} (calibrated)")
    
    # Store results
    results = []
    best_ll = -np.inf
    best_beta = None
    
    # Search over grid
    for i, beta in enumerate(beta_grid):
        theta = np.array([beta, gamma_fixed, delta_fixed])
        
        ll = compute_log_likelihood(
            theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
        results.append({'beta': beta, 'log_likelihood': ll})
        
        if ll > best_ll:
            best_ll = ll
            best_beta = beta
        
        if verbose:
            print(f"  [{i+1}/{n_points}] β={beta:.4f}, LL={ll:.2f}" + 
                  (" *" if beta == best_beta else ""))
    
    if verbose:
        print(f"\nBest: β={best_beta:.4f}, LL={best_ll:.2f}")
    
    # Compute standard error for β via numerical second derivative
    # d²L/dβ² approximated by central differences
    eps = 0.01
    theta_plus = np.array([best_beta + eps, gamma_fixed, delta_fixed])
    theta_minus = np.array([best_beta - eps, gamma_fixed, delta_fixed])
    theta_center = np.array([best_beta, gamma_fixed, delta_fixed])
    
    ll_plus = compute_log_likelihood(theta_plus, data, solver_params,
                                     v0_init_state, v1_init_state)
    ll_minus = compute_log_likelihood(theta_minus, data, solver_params,
                                      v0_init_state, v1_init_state)
    ll_center = compute_log_likelihood(theta_center, data, solver_params,
                                       v0_init_state, v1_init_state)
    
    # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
    d2_ll = (ll_plus - 2 * ll_center + ll_minus) / (eps ** 2)
    
    # Standard error: sqrt(-1 / d²L/dβ²)
    if d2_ll < 0:
        std_error = np.sqrt(-1.0 / d2_ll)
    else:
        warnings.warn("Second derivative is non-negative, SE may be invalid")
        std_error = np.nan
    
    details = {
        'beta_grid': beta_grid,
        'evaluations': results,
        'n_evaluations': n_points + 3,  # grid + 3 for SE computation
    }
    
    return best_beta, std_error, best_ll, details


def estimate_two_step(
    data: PanelData,
    delta_calibrated: float,
    solver_params: Dict[str, Any],
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    n_points: int = 20,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
) -> TwoStepResult:
    """Two-step structural estimation.
    
    Step 1: Estimate γ from transition data (OLS)
    Step 2: Estimate β via 1D grid search with γ and δ fixed
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    delta_calibrated : float
        Calibrated discount factor (fixed at this value)
    solver_params : dict
        Solver hyperparameters
    beta_bounds : tuple
        (min, max) bounds for β grid search
    n_points : int
        Number of grid points for β
    verbose : bool
        Print progress
    pretrained_path : str, optional
        Path to pre-trained networks for warm-start
    
    Returns
    -------
    TwoStepResult
        Complete estimation results
    """
    if verbose:
        print("=" * 60)
        print("TWO-STEP ESTIMATION")
        print("=" * 60)
    
    # Step 1: Estimate γ from transitions
    if verbose:
        print("\n--- Step 1: Estimate γ from transitions (OLS) ---")
    
    gamma_hat, gamma_se, gamma_details = estimate_gamma_ols(data)
    
    if verbose:
        print(f"  γ_hat = {gamma_hat:.6f} (SE = {gamma_se:.6f})")
        print(f"  R² = {gamma_details['r_squared']:.6f}")
        print(f"  N obs = {gamma_details['n_obs']}")
    
    # Step 2: Estimate β with γ and δ fixed
    if verbose:
        print(f"\n--- Step 2: Estimate β (γ={gamma_hat:.4f}, δ={delta_calibrated:.4f} fixed) ---")
    
    beta_hat, beta_se, log_lik, beta_details = estimate_beta_1d(
        data=data,
        gamma_fixed=gamma_hat,
        delta_fixed=delta_calibrated,
        solver_params=solver_params,
        beta_bounds=beta_bounds,
        n_points=n_points,
        verbose=verbose,
        pretrained_path=pretrained_path,
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("ESTIMATION COMPLETE")
        print("=" * 60)
        print(f"  β_hat = {beta_hat:.4f} (SE = {beta_se:.4f})")
        print(f"  γ_hat = {gamma_hat:.4f} (SE = {gamma_se:.6f})")
        print(f"  δ     = {delta_calibrated:.4f} (calibrated)")
        print(f"  Log-likelihood = {log_lik:.2f}")
    
    return TwoStepResult(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        delta_calibrated=delta_calibrated,
        beta_se=beta_se,
        gamma_se=gamma_se,
        log_likelihood=log_lik,
        n_evaluations=beta_details['n_evaluations'],
        gamma_ols_details=gamma_details,
    )


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

