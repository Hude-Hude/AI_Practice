"""Model primitives for the MDP.

Based on pseudo code subroutines:
- COMPUTE_REWARD
- COMPUTE_NEXT_STATE
- CLAMP_STATES
"""

import torch

from mdp_solver.types import Action, Scalar, TensorVector


def compute_reward(
    s: TensorVector,
    action: Action,
    beta: Scalar,
) -> TensorVector:
    """Compute flow reward.
    
    Implements COMPUTE_REWARD from pseudo code:
        r[i] = β * log(1 + s[i]) - action
    
    Parameters
    ----------
    s : TensorVector
        States of shape (N,)
    action : Action
        Action (0 or 1)
    beta : Scalar
        Reward coefficient
        
    Returns
    -------
    TensorVector
        Rewards of shape (N,)
    """
    return beta * torch.log(1 + s) - action


def compute_next_state(
    s: TensorVector,
    action: Action,
    gamma: Scalar,
) -> TensorVector:
    """Compute next state.
    
    Implements COMPUTE_NEXT_STATE from pseudo code:
        s_next[i] = (1 - γ) * s[i] + action
    
    Parameters
    ----------
    s : TensorVector
        Current states of shape (N,)
    action : Action
        Action (0 or 1)
    gamma : Scalar
        Decay rate
        
    Returns
    -------
    TensorVector
        Next states of shape (N,)
    """
    return (1 - gamma) * s + action


def clamp_states(
    s: TensorVector,
    s_min: Scalar,
    s_max: Scalar,
) -> TensorVector:
    """Clamp states to valid range.
    
    Implements CLAMP_STATES from pseudo code:
        s_clamped[i] = max(s_min, min(s_max, s[i]))
    
    Parameters
    ----------
    s : TensorVector
        States of shape (N,)
    s_min : Scalar
        Minimum bound
    s_max : Scalar
        Maximum bound
        
    Returns
    -------
    TensorVector
        Clamped states of shape (N,)
    """
    return torch.clamp(s, min=s_min, max=s_max)

