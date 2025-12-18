"""Value function computations.

Based on pseudo code subroutines:
- COMPUTE_INTEGRATED_VALUE
- COMPUTE_BELLMAN_TARGETS
- COMPUTE_BELLMAN_LOSS
- COMPUTE_CHOICE_PROBABILITY
"""

import torch
import torch.nn as nn

from mdp_solver.model import clamp_states, compute_next_state, compute_reward
from mdp_solver.network import evaluate_network
from mdp_solver.types import Scalar, TargetPair, TensorVector


def compute_integrated_value(
    v0_net: nn.Module,
    v1_net: nn.Module,
    s: TensorVector,
) -> TensorVector:
    """Compute integrated value function (log-sum-exp).
    
    Implements COMPUTE_INTEGRATED_VALUE from pseudo code:
        v_max = max(v0[i], v1[i])
        V_bar[i] = v_max + log(exp(v0[i] - v_max) + exp(v1[i] - v_max))
    
    Parameters
    ----------
    v0_net : nn.Module
        Network for action 0
    v1_net : nn.Module
        Network for action 1
    s : TensorVector
        States of shape (N,)
        
    Returns
    -------
    TensorVector
        Integrated value of shape (N,)
    """
    v0: TensorVector = evaluate_network(v0_net, s)
    v1: TensorVector = evaluate_network(v1_net, s)
    
    # Numerically stable log-sum-exp
    v_max: TensorVector = torch.maximum(v0, v1)
    V_bar: TensorVector = v_max + torch.log(
        torch.exp(v0 - v_max) + torch.exp(v1 - v_max)
    )
    
    return V_bar


def compute_bellman_targets(
    v0_net: nn.Module,
    v1_net: nn.Module,
    s: TensorVector,
    beta: Scalar,
    gamma: Scalar,
    delta: Scalar,
    s_min: Scalar,
    s_max: Scalar,
) -> TargetPair:
    """Compute Bellman target values.
    
    Implements COMPUTE_BELLMAN_TARGETS from pseudo code.
    
    Parameters
    ----------
    v0_net : nn.Module
        Network for action 0
    v1_net : nn.Module
        Network for action 1
    s : TensorVector
        Current states of shape (N,)
    beta : Scalar
        Reward coefficient
    gamma : Scalar
        Decay rate
    delta : Scalar
        Discount factor
    s_min : Scalar
        Minimum state bound
    s_max : Scalar
        Maximum state bound
        
    Returns
    -------
    TargetPair
        (target0, target1) - Target values for each action
    """
    # Compute targets for action 0
    r0: TensorVector = compute_reward(s, action=0, beta=beta)
    s_next0: TensorVector = compute_next_state(s, action=0, gamma=gamma)
    s_next0 = clamp_states(s_next0, s_min, s_max)
    V_next0: TensorVector = compute_integrated_value(v0_net, v1_net, s_next0)
    target0: TensorVector = r0 + delta * V_next0
    
    # Compute targets for action 1
    r1: TensorVector = compute_reward(s, action=1, beta=beta)
    s_next1: TensorVector = compute_next_state(s, action=1, gamma=gamma)
    s_next1 = clamp_states(s_next1, s_min, s_max)
    V_next1: TensorVector = compute_integrated_value(v0_net, v1_net, s_next1)
    target1: TensorVector = r1 + delta * V_next1
    
    return (target0, target1)


def compute_bellman_loss(
    v0_pred: TensorVector,
    v1_pred: TensorVector,
    target0: TensorVector,
    target1: TensorVector,
) -> torch.Tensor:
    """Compute mean squared Bellman residual loss.
    
    Implements COMPUTE_BELLMAN_LOSS from pseudo code:
        loss = sum((v0_pred - target0)^2 + (v1_pred - target1)^2) / (2 * N)
    
    Parameters
    ----------
    v0_pred : TensorVector
        Predicted values for action 0
    v1_pred : TensorVector
        Predicted values for action 1
    target0 : TensorVector
        Target values for action 0
    target1 : TensorVector
        Target values for action 1
        
    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    error0: TensorVector = v0_pred - target0
    error1: TensorVector = v1_pred - target1
    
    loss: torch.Tensor = torch.mean(error0 ** 2 + error1 ** 2) / 2
    
    return loss


def compute_choice_probability(
    v0_net: nn.Module,
    v1_net: nn.Module,
    s: TensorVector,
) -> TensorVector:
    """Compute probability of choosing action 1.
    
    Implements COMPUTE_CHOICE_PROBABILITY from pseudo code:
        prob[i] = 1 / (1 + exp(-(v1[i] - v0[i])))
    
    Parameters
    ----------
    v0_net : nn.Module
        Network for action 0
    v1_net : nn.Module
        Network for action 1
    s : TensorVector
        States of shape (N,)
        
    Returns
    -------
    TensorVector
        Probability of action 1 of shape (N,)
    """
    v0: TensorVector = evaluate_network(v0_net, s)
    v1: TensorVector = evaluate_network(v1_net, s)
    
    diff: TensorVector = v1 - v0
    prob: TensorVector = torch.sigmoid(diff)
    
    return prob

