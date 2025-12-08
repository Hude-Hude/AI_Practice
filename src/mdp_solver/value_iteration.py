"""Value function iteration algorithm."""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import logsumexp

from mdp_solver.model import compute_next_state, compute_reward
from mdp_solver.types import Action, Scalar, StateGrid, Vector


def initialize_value_function(state_grid: StateGrid) -> Vector:
    """
    Initialize value function to zeros.

    Parameters
    ----------
    state_grid : StateGrid
        Discretized state space

    Returns
    -------
    Vector
        Initial value function (zeros)
    """
    n: int = len(state_grid)
    return np.zeros(shape=n)


def interpolate_value(
    V: Vector,
    state_grid: StateGrid,
    query_states: Vector,
) -> Vector:
    """
    Interpolate value function at query states.

    Uses linear interpolation with boundary extrapolation.

    Parameters
    ----------
    V : Vector
        Value function on grid
    state_grid : StateGrid
        Grid points
    query_states : Vector
        Points to interpolate

    Returns
    -------
    Vector
        Interpolated values
    """
    interpolator = interp1d(
        x=state_grid,
        y=V,
        kind="linear",
        bounds_error=False,
        fill_value=(V[0], V[-1]),
    )
    return interpolator(query_states)


def compute_choice_value(
    V: Vector,
    state_grid: StateGrid,
    action: Action,
    beta: Scalar,
    gamma: Scalar,
    delta: Scalar,
) -> Vector:
    """
    Compute choice-specific value function.

    v(s, a) = u(s, a) + δ * E[V(s')]

    Parameters
    ----------
    V : Vector
        Current integrated value function
    state_grid : StateGrid
        Discretized state space
    action : Action
        Action (0 or 1)
    beta : Scalar
        Reward coefficient on state
    gamma : Scalar
        State decay rate
    delta : Scalar
        Discount factor

    Returns
    -------
    Vector
        Choice-specific value for given action
    """
    # Step 1: Compute flow reward
    reward: Vector = compute_reward(s=state_grid, action=action, beta=beta)

    # Step 2: Compute next states
    next_states: Vector = compute_next_state(s=state_grid, action=action, gamma=gamma)

    # Step 3: Interpolate continuation value at next states
    continuation: Vector = interpolate_value(
        V=V, state_grid=state_grid, query_states=next_states
    )

    # Step 4: Combine flow reward and discounted continuation
    return reward + delta * continuation


def compute_integrated_value(v0: Vector, v1: Vector) -> Vector:
    """
    Compute integrated value function (Emax).

    V(s) = log(exp(v0) + exp(v1))

    This is the expected maximum value with Type-I EV shocks.

    Parameters
    ----------
    v0 : Vector
        Choice-specific value for action 0
    v1 : Vector
        Choice-specific value for action 1

    Returns
    -------
    Vector
        Integrated value function
    """
    return logsumexp(a=np.stack([v0, v1], axis=1), axis=1)


def check_convergence(
    V_old: Vector,
    V_new: Vector,
    tolerance: Scalar,
) -> bool:
    """
    Check if value function has converged.

    Parameters
    ----------
    V_old : Vector
        Previous value function
    V_new : Vector
        Updated value function
    tolerance : Scalar
        Convergence threshold

    Returns
    -------
    bool
        True if converged
    """
    diff: Scalar = np.max(np.abs(V_new - V_old))
    return diff < tolerance


def compute_choice_probability(v0: Vector, v1: Vector) -> Vector:
    """
    Compute probability of choosing action 1.

    P(a=1|s) = exp(v1) / (exp(v0) + exp(v1)) = sigmoid(v1 - v0)

    Parameters
    ----------
    v0 : Vector
        Choice-specific value for action 0
    v1 : Vector
        Choice-specific value for action 1

    Returns
    -------
    Vector
        Probability of choosing action 1
    """
    # Numerically stable softmax
    v_max: Vector = np.maximum(v0, v1)
    exp_v0: Vector = np.exp(v0 - v_max)
    exp_v1: Vector = np.exp(v1 - v_max)
    return exp_v1 / (exp_v0 + exp_v1)


def solve_value_function(
    state_grid: StateGrid,
    beta: Scalar,
    gamma: Scalar,
    delta: Scalar,
    tolerance: Scalar = 1e-10,
    max_iterations: int = 1000,
) -> Tuple[Vector, Vector, Vector, int]:
    """
    Solve for the integrated value function using contraction mapping.

    Parameters
    ----------
    state_grid : StateGrid
        Discretized state space
    beta : Scalar
        Reward coefficient on state
    gamma : Scalar
        State decay rate
    delta : Scalar
        Discount factor
    tolerance : Scalar
        Convergence tolerance
    max_iterations : int
        Maximum iterations

    Returns
    -------
    V : Vector
        Integrated value function
    v0 : Vector
        Choice-specific value for action 0
    v1 : Vector
        Choice-specific value for action 1
    n_iter : int
        Number of iterations to converge
    """
    # Initialize
    V: Vector = initialize_value_function(state_grid=state_grid)

    for iteration in range(max_iterations):
        # Compute choice-specific values
        v0: Vector = compute_choice_value(
            V=V,
            state_grid=state_grid,
            action=0,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )
        v1: Vector = compute_choice_value(
            V=V,
            state_grid=state_grid,
            action=1,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # Compute integrated value
        V_new: Vector = compute_integrated_value(v0=v0, v1=v1)

        # Check convergence
        if check_convergence(V_old=V, V_new=V_new, tolerance=tolerance):
            # Recompute final choice-specific values
            v0 = compute_choice_value(
                V=V_new,
                state_grid=state_grid,
                action=0,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            v1 = compute_choice_value(
                V=V_new,
                state_grid=state_grid,
                action=1,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            return V_new, v0, v1, iteration + 1

        V = V_new

    # Did not converge
    print(f"Warning: Did not converge after {max_iterations} iterations")
    return V, v0, v1, max_iterations

