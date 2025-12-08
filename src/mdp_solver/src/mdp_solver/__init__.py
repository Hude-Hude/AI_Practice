"""
MDP Solver with Type-I Extreme Value Shocks.

A package for solving Markov Decision Processes with:
- Continuous state space
- Binary action choice
- Type-I extreme value (Gumbel) distributed shocks
- Monotonic neural network approximation
"""

from mdp_solver.types import Scalar, Vector, Matrix, StateGrid, Action
from mdp_solver.model import compute_reward, compute_next_state
from mdp_solver.value_iteration import (
    solve_value_function,
    compute_choice_value,
    compute_integrated_value,
    compute_choice_probability,
)
from mdp_solver.neural_network import (
    MonotonicLayer,
    MonotonicValueNetwork,
    ValueFunctionApproximator,
)

__all__ = [
    # Types
    "Scalar",
    "Vector",
    "Matrix",
    "StateGrid",
    "Action",
    # Model primitives
    "compute_reward",
    "compute_next_state",
    # Value iteration
    "solve_value_function",
    "compute_choice_value",
    "compute_integrated_value",
    "compute_choice_probability",
    # Neural networks
    "MonotonicLayer",
    "MonotonicValueNetwork",
    "ValueFunctionApproximator",
]


def main() -> None:
    """Entry point for the package."""
    print("MDP Solver v0.1.0")
