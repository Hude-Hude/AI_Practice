"""MDP Solver package.

A neural network-based solver for dynamic discrete choice models
with Type-I Extreme Value shocks.
"""

from mdp_solver.model import clamp_states, compute_next_state, compute_reward
from mdp_solver.network import (
    MonotonicLinear,
    SoftplusActivation,
    build_monotonic_network,
    evaluate_network,
)
from mdp_solver.solver import (
    check_convergence,
    initialize_networks,
    sample_states,
    solve_value_function,
)
from mdp_solver.value_function import (
    compute_bellman_loss,
    compute_bellman_targets,
    compute_choice_probability,
    compute_integrated_value,
)

__all__ = [
    # Model primitives
    "compute_reward",
    "compute_next_state",
    "clamp_states",
    # Network
    "MonotonicLinear",
    "SoftplusActivation",
    "build_monotonic_network",
    "evaluate_network",
    # Value function
    "compute_integrated_value",
    "compute_bellman_targets",
    "compute_bellman_loss",
    "compute_choice_probability",
    # Solver
    "initialize_networks",
    "sample_states",
    "check_convergence",
    "solve_value_function",
]

