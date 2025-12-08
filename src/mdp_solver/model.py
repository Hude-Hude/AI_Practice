"""Model primitives for the MDP."""

import numpy as np

from mdp_solver.types import Action, Scalar, Vector


def compute_reward(
    s: Vector,
    action: Action,
    beta: Scalar,
) -> Vector:
    """
    Compute flow reward with diminishing marginal returns.

    u(s, a) = β * log(1 + s) - a

    Parameters
    ----------
    s : Vector
        Current state(s)
    action : Action
        Action (0 or 1)
    beta : Scalar
        Reward coefficient on state

    Returns
    -------
    Vector
        Flow reward for each state
    """
    return beta * np.log(1 + s) - action


def compute_next_state(
    s: Vector,
    action: Action,
    gamma: Scalar,
) -> Vector:
    """
    Compute next state given current state and action.

    s' = (1 - γ) * s + a

    Parameters
    ----------
    s : Vector
        Current state(s)
    action : Action
        Action (0 or 1)
    gamma : Scalar
        State decay rate

    Returns
    -------
    Vector
        Next state(s)
    """
    return (1 - gamma) * s + action

