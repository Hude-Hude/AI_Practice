"""MDP Solver with Neural Network Value Iteration.

A complete implementation for solving dynamic discrete choice models
with Type-I Extreme Value shocks using monotonic neural networks.

Based on pseudo code from solve_mdp.qmd.
"""

import copy
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# Type Definitions
# =============================================================================

Scalar = float
Vector = np.ndarray
Matrix = np.ndarray
TensorVector = torch.Tensor
TensorMatrix = torch.Tensor
Action = int  # 0 or 1
HiddenSizes = List[int]
NetworkPair = Tuple[nn.Module, nn.Module]
TargetPair = Tuple[TensorVector, TensorVector]


# =============================================================================
# Neural Network Components
# =============================================================================

class MonotonicLinear(nn.Module):
    """Linear layer with non-negative weights via softplus parameterization.
    
    Implements APPLY_MONOTONIC_LINEAR from pseudo code:
        W = log(1 + exp(W_raw))
        y = W @ x + b
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        
        # Unconstrained weight parameters (W_raw in pseudo code)
        self.weight_raw: nn.Parameter = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        # Bias vector (unconstrained)
        self.bias: nn.Parameter = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x: TensorVector) -> TensorVector:
        # Enforce non-negative weights via softplus
        weight: torch.Tensor = F.softplus(self.weight_raw)
        # Linear transformation: y = W @ x + b
        return F.linear(x, weight, self.bias)


class SoftplusActivation(nn.Module):
    """Softplus activation function: y = log(1 + exp(x))."""

    def forward(self, x: TensorVector) -> TensorVector:
        return F.softplus(x)


def build_monotonic_network(hidden_sizes: HiddenSizes) -> nn.Sequential:
    """Build a monotonic neural network.
    
    Implements BUILD_MONOTONIC_NETWORK from pseudo code.
    """
    layers: List[nn.Module] = []
    in_features: int = 1  # Single state input
    
    for i in range(len(hidden_sizes)):
        out_features: int = hidden_sizes[i]
        layers.append(MonotonicLinear(in_features, out_features))
        layers.append(SoftplusActivation())
        in_features = out_features
    
    # Output layer (no activation)
    layers.append(MonotonicLinear(in_features, 1))
    
    return nn.Sequential(*layers)


def evaluate_network(network: nn.Module, s: TensorVector) -> TensorVector:
    """Evaluate network at given states.
    
    Implements EVALUATE_NETWORK from pseudo code.
    """
    # Reshape to (N, 1) if needed
    if s.dim() == 1:
        x: torch.Tensor = s.unsqueeze(1)
    else:
        x = s
    
    # Forward pass through network
    output: torch.Tensor = network(x)
    
    # Reshape to (N,)
    return output.squeeze(-1)


# =============================================================================
# Model Primitives
# =============================================================================

def compute_reward(
    s: TensorVector,
    action: Action,
    beta: Scalar,
) -> TensorVector:
    """Compute flow reward: r = β * log(1 + s) - action.
    
    Implements COMPUTE_REWARD from pseudo code.
    """
    return beta * torch.log(1 + s) - action


def compute_next_state(
    s: TensorVector,
    action: Action,
    gamma: Scalar,
) -> TensorVector:
    """Compute next state: s' = (1 - γ) * s + action.
    
    Implements COMPUTE_NEXT_STATE from pseudo code.
    """
    return (1 - gamma) * s + action


def clamp_states(
    s: TensorVector,
    s_min: Scalar,
    s_max: Scalar,
) -> TensorVector:
    """Clamp states to valid range [s_min, s_max].
    
    Implements CLAMP_STATES from pseudo code.
    """
    return torch.clamp(s, min=s_min, max=s_max)


# =============================================================================
# Value Function Computations
# =============================================================================

def compute_integrated_value(
    v0_net: nn.Module,
    v1_net: nn.Module,
    s: TensorVector,
) -> TensorVector:
    """Compute integrated value function (log-sum-exp).
    
    Implements COMPUTE_INTEGRATED_VALUE from pseudo code:
        V(s) = log(exp(v0(s)) + exp(v1(s)))
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
) -> TargetPair:
    """Compute Bellman target values.
    
    Implements COMPUTE_BELLMAN_TARGETS from pseudo code:
        target_a = r(s, a) + δ * V(s')
    
    No clamping - unbounded state space. Networks must extrapolate.
    """
    # Compute targets for action 0 (no clamping - unbounded state space)
    r0: TensorVector = compute_reward(s, action=0, beta=beta)
    s_next0: TensorVector = compute_next_state(s, action=0, gamma=gamma)
    V_next0: TensorVector = compute_integrated_value(v0_net, v1_net, s_next0)
    target0: TensorVector = r0 + delta * V_next0
    
    # Compute targets for action 1 (no clamping - unbounded state space)
    r1: TensorVector = compute_reward(s, action=1, beta=beta)
    s_next1: TensorVector = compute_next_state(s, action=1, gamma=gamma)
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
    
    Implements COMPUTE_BELLMAN_LOSS from pseudo code.
    """
    error0: TensorVector = v0_pred - target0
    error1: TensorVector = v1_pred - target1
    
    loss: torch.Tensor = torch.mean(error0 ** 2 + error1 ** 2) / 2
    
    return loss


def compute_choice_probability(
    v0_net: nn.Module,
    v1_net: nn.Module,
    s: TensorVector,
) -> Tuple[TensorVector, TensorVector]:
    """Compute choice probabilities for both actions.
    
    Implements COMPUTE_CHOICE_PROBABILITY from pseudo code.
    Uses numerically stable softmax computation.
    
    Returns
    -------
    Tuple[TensorVector, TensorVector]
        (prob0, prob1) - P(a=0|s) and P(a=1|s)
    """
    v0: TensorVector = evaluate_network(v0_net, s)
    v1: TensorVector = evaluate_network(v1_net, s)
    
    # Numerically stable softmax
    v_max: TensorVector = torch.maximum(v0, v1)
    exp0: TensorVector = torch.exp(v0 - v_max)
    exp1: TensorVector = torch.exp(v1 - v_max)
    sum_exp: TensorVector = exp0 + exp1
    
    prob0: TensorVector = exp0 / sum_exp
    prob1: TensorVector = exp1 / sum_exp
    
    return (prob0, prob1)


# =============================================================================
# Solver
# =============================================================================

def copy_network(source: nn.Module) -> nn.Module:
    """Create a deep copy of a network with frozen gradients.
    
    Implements COPY_NETWORK from pseudo code.
    Used to create target networks that provide stable Bellman targets.
    
    Parameters
    ----------
    source : nn.Module
        Network to copy from
        
    Returns
    -------
    nn.Module
        Deep copy of source network with requires_grad=False
    """
    target: nn.Module = copy.deepcopy(source)
    
    # Freeze the target network (no gradient computation)
    for param in target.parameters():
        param.requires_grad = False
    
    return target


def initialize_networks(hidden_sizes: HiddenSizes) -> NetworkPair:
    """Initialize two monotonic neural networks.
    
    Implements INITIALIZE_NETWORKS from pseudo code.
    """
    v0_net: nn.Module = build_monotonic_network(hidden_sizes)
    v1_net: nn.Module = build_monotonic_network(hidden_sizes)
    
    return (v0_net, v1_net)


def sample_states(
    n: int,
    s_min: Scalar,
    s_max: Scalar,
) -> TensorVector:
    """Sample states uniformly from [s_min, s_max].
    
    Implements SAMPLE_STATES from pseudo code.
    Note: This is kept for backward compatibility.
    Prefer generate_state_grid for stable training.
    """
    u: torch.Tensor = torch.rand(n)
    s: TensorVector = s_min + u * (s_max - s_min)
    
    return s


def generate_state_grid(
    n: int,
    s_min: Scalar,
    s_max: Scalar,
) -> TensorVector:
    """Generate a fixed grid of evenly spaced states.
    
    Implements GENERATE_STATE_GRID from pseudo code.
    Using a fixed grid eliminates sampling variance and provides
    stable loss signals for optimization.
    
    Parameters
    ----------
    n : int
        Number of grid points
    s_min : float
        Minimum state value
    s_max : float
        Maximum state value
        
    Returns
    -------
    TensorVector
        Fixed state grid of shape (n,) with evenly spaced points
    """
    s: TensorVector = torch.linspace(s_min, s_max, n)
    
    return s


def check_convergence(loss: torch.Tensor, tolerance: Scalar) -> bool:
    """Check if algorithm has converged.
    
    Implements CHECK_CONVERGENCE from pseudo code:
        converged = sqrt(loss) < tolerance
    """
    rmse: float = math.sqrt(loss.item())
    converged: bool = rmse < tolerance
    
    return converged


def solve_value_function(
    beta: Scalar,
    gamma: Scalar,
    delta: Scalar,
    s_min: Scalar,
    s_max: Scalar,
    hidden_sizes: HiddenSizes,
    learning_rate: Scalar,
    batch_size: int,
    tolerance: Scalar,
    max_iterations: int,
    target_update_freq: int = 100,
) -> Tuple[nn.Module, nn.Module, List[float], int]:
    """Solve for value functions using neural network iteration.
    
    Implements SOLVE_VALUE_FUNCTION from pseudo code.
    Uses target networks to provide stable Bellman targets.
    
    Parameters
    ----------
    beta : float
        Reward coefficient
    gamma : float
        State decay rate
    delta : float
        Discount factor
    s_min : float
        Minimum state
    s_max : float
        Maximum state
    hidden_sizes : List[int]
        Hidden layer sizes for networks
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Number of states per batch
    tolerance : float
        Convergence tolerance (on RMSE)
    max_iterations : int
        Maximum number of iterations
    target_update_freq : int
        How often to update target networks (default: 100)
        
    Returns
    -------
    Tuple[nn.Module, nn.Module, List[float], int]
        (v0_net, v1_net, losses, n_iterations)
    """
    # Step 1: Initialize policy networks (updated every iteration)
    v0_net: nn.Module
    v1_net: nn.Module
    v0_net, v1_net = initialize_networks(hidden_sizes)
    
    # Initialize target networks (frozen copies, updated every T iterations)
    v0_target: nn.Module = copy_network(v0_net)
    v1_target: nn.Module = copy_network(v1_net)
    
    # Generate fixed state grid (same states used every iteration)
    s_grid: TensorVector = generate_state_grid(batch_size, s_min, s_max)
    
    # Create optimizer for policy networks only
    optimizer: optim.Optimizer = optim.Adam(
        list(v0_net.parameters()) + list(v1_net.parameters()),
        lr=learning_rate,
    )
    
    # Track losses
    losses: List[float] = []
    
    # Step 2: Iterate
    for k in range(max_iterations):
        # Predictions from policy networks (using fixed grid)
        v0_pred: TensorVector = evaluate_network(v0_net, s_grid)
        v1_pred: TensorVector = evaluate_network(v1_net, s_grid)
        
        # Targets computed from TARGET networks (stable, not moving)
        target0: TensorVector
        target1: TensorVector
        target0, target1 = compute_bellman_targets(
            v0_target, v1_target, s_grid,
            beta, gamma, delta,
        )
        
        # Compute loss
        loss: torch.Tensor = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        # Track loss
        losses.append(loss.item())
        
        # Update policy networks via gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Periodically sync target networks with policy networks
        if (k + 1) % target_update_freq == 0:
            v0_target = copy_network(v0_net)
            v1_target = copy_network(v1_net)
        
        # Check convergence
        if check_convergence(loss, tolerance):
            return (v0_net, v1_net, losses, k + 1)
    
    # Did not converge
    return (v0_net, v1_net, losses, max_iterations)

