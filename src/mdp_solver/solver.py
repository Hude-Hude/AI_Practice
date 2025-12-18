"""Main solver for the MDP.

Based on pseudo code:
- SOLVE_VALUE_FUNCTION (main algorithm)
- INITIALIZE_NETWORKS
- SAMPLE_STATES
- UPDATE_NETWORK_WEIGHTS (via optimizer)
- CHECK_CONVERGENCE
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from mdp_solver.network import build_monotonic_network, evaluate_network
from mdp_solver.types import HiddenSizes, NetworkPair, Scalar, TensorVector
from mdp_solver.value_function import compute_bellman_loss, compute_bellman_targets


def initialize_networks(hidden_sizes: HiddenSizes) -> NetworkPair:
    """Initialize two monotonic neural networks.
    
    Implements INITIALIZE_NETWORKS from pseudo code.
    
    Parameters
    ----------
    hidden_sizes : List[int]
        Hidden layer sizes for each network
        
    Returns
    -------
    NetworkPair
        (v0_net, v1_net) - Two monotonic networks
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
    
    Implements SAMPLE_STATES from pseudo code:
        u = random_uniform(0, 1)
        s[i] = s_min + u * (s_max - s_min)
    
    Parameters
    ----------
    n : int
        Number of samples
    s_min : Scalar
        Minimum state
    s_max : Scalar
        Maximum state
        
    Returns
    -------
    TensorVector
        Sampled states of shape (N,)
    """
    u: torch.Tensor = torch.rand(n)
    s: TensorVector = s_min + u * (s_max - s_min)
    
    return s


def check_convergence(loss: torch.Tensor, tolerance: Scalar) -> bool:
    """Check if algorithm has converged.
    
    Implements CHECK_CONVERGENCE from pseudo code:
        rmse = sqrt(loss)
        converged = (rmse < tolerance)
    
    Parameters
    ----------
    loss : torch.Tensor
        Current loss value
    tolerance : Scalar
        Convergence tolerance
        
    Returns
    -------
    bool
        True if converged
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
) -> Tuple[nn.Module, nn.Module, List[float], int]:
    """Solve for value functions using neural network iteration.
    
    Implements SOLVE_VALUE_FUNCTION from pseudo code.
    
    Parameters
    ----------
    beta : Scalar
        Reward coefficient
    gamma : Scalar
        State decay rate
    delta : Scalar
        Discount factor
    s_min : Scalar
        Minimum state
    s_max : Scalar
        Maximum state
    hidden_sizes : List[int]
        Hidden layer sizes
    learning_rate : Scalar
        Learning rate for optimizer
    batch_size : int
        Number of states per batch
    tolerance : Scalar
        Convergence tolerance
    max_iterations : int
        Maximum number of iterations
        
    Returns
    -------
    Tuple[nn.Module, nn.Module, List[float], int]
        (v0_net, v1_net, losses, n_iterations)
    """
    # Step 1: Initialize networks
    v0_net: nn.Module
    v1_net: nn.Module
    v0_net, v1_net = initialize_networks(hidden_sizes)
    
    # Create optimizer for both networks
    optimizer: optim.Optimizer = optim.Adam(
        list(v0_net.parameters()) + list(v1_net.parameters()),
        lr=learning_rate,
    )
    
    # Track losses
    losses: List[float] = []
    
    # Step 2: Iterate
    for k in range(max_iterations):
        # Sample batch of states
        s_batch: TensorVector = sample_states(batch_size, s_min, s_max)
        
        # Forward pass: get predictions
        v0_pred: TensorVector = evaluate_network(v0_net, s_batch)
        v1_pred: TensorVector = evaluate_network(v1_net, s_batch)
        
        # Compute targets (with frozen weights - no gradient)
        with torch.no_grad():
            target0: TensorVector
            target1: TensorVector
            target0, target1 = compute_bellman_targets(
                v0_net, v1_net, s_batch,
                beta, gamma, delta, s_min, s_max,
            )
        
        # Compute loss
        loss: torch.Tensor = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        # Track loss
        losses.append(loss.item())
        
        # Update weights via gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check convergence
        if check_convergence(loss, tolerance):
            return (v0_net, v1_net, losses, k + 1)
    
    # Did not converge
    return (v0_net, v1_net, losses, max_iterations)

