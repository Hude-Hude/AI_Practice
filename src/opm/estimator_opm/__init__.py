"""OPM Estimator: Two-step estimation (GMM + cost recovery)."""

from .opm_estimator import (
    EstimationResult,
    berry_inversion,
    compute_demand_shocks,
    gmm_objective,
    estimate_alpha_gmm,
    recover_costs,
    construct_cost_instruments,
    construct_blp_instruments,
    estimate_opm,
)

__all__ = [
    "EstimationResult",
    "berry_inversion",
    "compute_demand_shocks",
    "gmm_objective",
    "estimate_alpha_gmm",
    "recover_costs",
    "construct_cost_instruments",
    "construct_blp_instruments",
    "estimate_opm",
]
