"""MDP Structural Parameter Estimator using NFXP.

This module provides maximum likelihood estimation of structural
parameters (beta, gamma, delta) from panel data using the Nested
Fixed Point (NFXP) algorithm.

Recommended approach (two-step):
- estimate_gamma_ols: Estimate γ from transition data (instant)
- estimate_beta_1d: Estimate β with γ from OLS and δ calibrated
- estimate_two_step: Combined two-step estimation

Alternative methods (joint estimation, not recommended due to identification):
- estimate_mle: Nelder-Mead optimization
- grid_search_mle: Explicit 3D grid search
"""

from mdp_estimator.mdp_estimator import (
    EstimationResult,
    TwoStepResult,
    estimate_gamma_ols,
    estimate_beta_1d,
    estimate_two_step,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)

__all__ = [
    # Data structures
    "EstimationResult",
    "TwoStepResult",
    # Two-step estimation (recommended)
    "estimate_gamma_ols",
    "estimate_beta_1d",
    "estimate_two_step",
    # Core functions
    "compute_log_likelihood",
    "compute_standard_errors",
    # Joint estimation (not recommended)
    "estimate_mle",
    "grid_search_mle",
]

