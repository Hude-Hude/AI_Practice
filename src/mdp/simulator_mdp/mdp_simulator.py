"""MDP Simulator with Monte Carlo Panel Data Generation.

Implements the SIMULATE_MDP_PANEL algorithm for generating panel data
from trained MDP value functions.

Based on pseudocode from simulate_mdp.qmd.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Import from mdp.solver for shared functionality
from mdp.solver_mdp import (
    compute_choice_probability,
    compute_reward as _compute_reward_torch,
    compute_next_state as _compute_next_state_torch,
)


# =============================================================================
# Type Definitions
# =============================================================================

Scalar = float
Vector = np.ndarray
Matrix = np.ndarray
TensorVector = torch.Tensor


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PanelData:
    """Panel data from MDP simulation.

    Attributes
    ----------
    states : np.ndarray
        State history, shape (n_agents, n_periods)
    actions : np.ndarray
        Action history, shape (n_agents, n_periods)
    rewards : np.ndarray
        Reward history, shape (n_agents, n_periods)
    n_agents : int
        Number of agents
    n_periods : int
        Number of time periods
    """

    states: Matrix
    actions: Matrix
    rewards: Matrix
    n_agents: int
    n_periods: int


# =============================================================================
# Subroutines (wrappers around mdp_solver functions)
# =============================================================================


def compute_reward(
    s: Vector,
    a: Vector,
    beta: Scalar,
) -> Vector:
    """Compute flow rewards: r = β * log(1 + s) - a.

    Wrapper around mdp_solver.compute_reward for numpy arrays.

    Parameters
    ----------
    s : np.ndarray
        States, shape (n,)
    a : np.ndarray
        Actions, shape (n,)
    beta : float
        Reward coefficient

    Returns
    -------
    np.ndarray
        Rewards, shape (n,)
    """
    s_tensor = torch.tensor(s, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32)
    reward_tensor = _compute_reward_torch(s=s_tensor, action=a_tensor, beta=beta)
    return reward_tensor.numpy()


def compute_next_state(
    s: Vector,
    a: Vector,
    gamma: Scalar,
) -> Vector:
    """Compute next states: s' = (1 - γ) * s + a.

    Wrapper around mdp_solver.compute_next_state for numpy arrays.

    Parameters
    ----------
    s : np.ndarray
        Current states, shape (n,)
    a : np.ndarray
        Actions, shape (n,)
    gamma : float
        State decay rate

    Returns
    -------
    np.ndarray
        Next states, shape (n,)
    """
    s_tensor = torch.tensor(s, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32)
    next_state_tensor = _compute_next_state_torch(s=s_tensor, action=a_tensor, gamma=gamma)
    return next_state_tensor.numpy()


# =============================================================================
# Main Algorithm
# =============================================================================


def simulate_mdp_panel(
    v0_net: nn.Module,
    v1_net: nn.Module,
    n_agents: int,
    n_periods: int,
    s_init: Union[Scalar, Vector],
    beta: Scalar,
    gamma: Scalar,
    seed: Optional[int] = None,
) -> PanelData:
    """Simulate MDP panel data using trained value functions.

    Implements SIMULATE_MDP_PANEL from pseudocode.

    Uses closed-form logit choice probabilities from compute_choice_probability
    to draw actions directly, rather than simulating Gumbel shocks.

    Parameters
    ----------
    v0_net : nn.Module
        Trained network for action 0
    v1_net : nn.Module
        Trained network for action 1
    n_agents : int
        Number of agents to simulate
    n_periods : int
        Number of time periods
    s_init : float or np.ndarray
        Initial states. If scalar, all agents start at the same state.
        If array, must have shape (n_agents,).
    beta : float
        Reward coefficient
    gamma : float
        State decay rate
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    PanelData
        Panel data containing states, actions, and rewards
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Initialize storage matrices
    states = np.zeros((n_agents, n_periods))
    actions = np.zeros((n_agents, n_periods), dtype=np.int32)
    rewards = np.zeros((n_agents, n_periods))

    # Set initial states
    if np.isscalar(s_init):
        states[:, 0] = s_init
    else:
        states[:, 0] = s_init

    # Simulate forward
    for t in range(n_periods):
        # Get current states as tensor
        s_t = torch.tensor(states[:, t], dtype=torch.float32)

        # Compute choice probabilities using closed-form logit formula
        with torch.no_grad():
            _, p1 = compute_choice_probability(v0_net, v1_net, s_t)
            p1_np = p1.numpy()

        # Draw actions from Bernoulli distribution
        u = np.random.uniform(size=n_agents)
        actions[:, t] = (u < p1_np).astype(np.int32)

        # Compute flow rewards
        rewards[:, t] = compute_reward(
            s=states[:, t],
            a=actions[:, t],
            beta=beta,
        )

        # State transition (if not last period)
        if t < n_periods - 1:
            states[:, t + 1] = compute_next_state(
                s=states[:, t],
                a=actions[:, t],
                gamma=gamma,
            )

    return PanelData(
        states=states,
        actions=actions,
        rewards=rewards,
        n_agents=n_agents,
        n_periods=n_periods,
    )

