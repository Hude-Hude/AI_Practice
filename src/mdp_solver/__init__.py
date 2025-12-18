"""MDP Solver package.

A neural network-based solver for dynamic discrete choice models
with Type-I Extreme Value shocks.
"""

# Import everything from the consolidated module
from mdp_solver.mdp_solver import (
    # Type definitions
    Scalar,
    Vector,
    Matrix,
    TensorVector,
    TensorMatrix,
    Action,
    HiddenSizes,
    NetworkPair,
    TargetPair,
    # Neural network components
    MonotonicLinear,
    SoftplusActivation,
    build_monotonic_network,
    evaluate_network,
    # Model primitives
    compute_reward,
    compute_next_state,
    clamp_states,
    # Value function computations
    compute_integrated_value,
    compute_bellman_targets,
    compute_bellman_loss,
    compute_choice_probability,
    # Solver
    initialize_networks,
    sample_states,
    check_convergence,
    solve_value_function,
)

__all__ = [
    # Type definitions
    "Scalar",
    "Vector",
    "Matrix",
    "TensorVector",
    "TensorMatrix",
    "Action",
    "HiddenSizes",
    "NetworkPair",
    "TargetPair",
    # Neural network components
    "MonotonicLinear",
    "SoftplusActivation",
    "build_monotonic_network",
    "evaluate_network",
    # Model primitives
    "compute_reward",
    "compute_next_state",
    "clamp_states",
    # Value function computations
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
