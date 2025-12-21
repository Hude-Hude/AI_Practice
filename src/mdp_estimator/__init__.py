"""MDP Structural Parameter Estimator using NFXP.

This module provides maximum likelihood estimation of structural
parameters (beta, gamma, delta) from panel data using the Nested
Fixed Point (NFXP) algorithm with Nelder-Mead optimization.
"""

from mdp_estimator.mdp_estimator import (
    EstimationResult,
    compute_log_likelihood,
    estimate_mle,
    compute_standard_errors,
)

__all__ = [
    "EstimationResult",
    "compute_log_likelihood",
    "estimate_mle",
    "compute_standard_errors",
]

