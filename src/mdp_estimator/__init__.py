"""MDP Structural Parameter Estimator using NFXP.

This module provides maximum likelihood estimation of structural
parameters (beta, gamma, delta) from panel data using the Nested
Fixed Point (NFXP) algorithm.

Two estimation methods are available:
- estimate_mle: Nelder-Mead optimization (adaptive, fewer evaluations)
- grid_search_mle: Explicit grid search (predictable, parallelizable)
"""

from mdp_estimator.mdp_estimator import (
    EstimationResult,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)

__all__ = [
    "EstimationResult",
    "compute_log_likelihood",
    "estimate_mle",
    "grid_search_mle",
    "compute_standard_errors",
]

