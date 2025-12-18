"""Monotonic neural network implementation.

Based on pseudo code subroutines:
- BUILD_MONOTONIC_NETWORK
- APPLY_MONOTONIC_LINEAR
- APPLY_SOFTPLUS
- EVALUATE_NETWORK
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdp_solver.types import HiddenSizes, TensorVector


class MonotonicLinear(nn.Module):
    """Linear layer with non-negative weights via softplus parameterization.
    
    Implements APPLY_MONOTONIC_LINEAR from pseudo code:
        W = log(1 + exp(W_raw))
        y = W @ x + b
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize monotonic linear layer.
        
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        """
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
        """Apply monotonic linear transformation.
        
        Parameters
        ----------
        x : TensorVector
            Input tensor of shape (batch_size, in_features)
            
        Returns
        -------
        TensorVector
            Output tensor of shape (batch_size, out_features)
        """
        # Enforce non-negative weights via softplus
        weight: torch.Tensor = F.softplus(self.weight_raw)
        
        # Linear transformation: y = W @ x + b
        return F.linear(x, weight, self.bias)


class SoftplusActivation(nn.Module):
    """Softplus activation function.
    
    Implements APPLY_SOFTPLUS from pseudo code:
        y[i] = log(1 + exp(x[i]))
    """

    def forward(self, x: TensorVector) -> TensorVector:
        """Apply softplus activation.
        
        Parameters
        ----------
        x : TensorVector
            Input tensor
            
        Returns
        -------
        TensorVector
            Output tensor with softplus applied element-wise
        """
        return F.softplus(x)


def build_monotonic_network(hidden_sizes: HiddenSizes) -> nn.Sequential:
    """Build a monotonic neural network.
    
    Implements BUILD_MONOTONIC_NETWORK from pseudo code.
    
    Parameters
    ----------
    hidden_sizes : List[int]
        List of hidden layer sizes
        
    Returns
    -------
    nn.Sequential
        Monotonic neural network
    """
    layers: List[nn.Module] = []
    in_features: int = 1  # Single state input
    
    for i in range(len(hidden_sizes)):
        out_features: int = hidden_sizes[i]
        
        # Add monotonic linear layer
        layers.append(MonotonicLinear(in_features, out_features))
        
        # Add softplus activation
        layers.append(SoftplusActivation())
        
        in_features = out_features
    
    # Output layer (no activation)
    layers.append(MonotonicLinear(in_features, 1))
    
    return nn.Sequential(*layers)


def evaluate_network(network: nn.Module, s: TensorVector) -> TensorVector:
    """Evaluate network at given states.
    
    Implements EVALUATE_NETWORK from pseudo code.
    
    Parameters
    ----------
    network : nn.Module
        Monotonic neural network
    s : TensorVector
        Input states of shape (N,) or (N, 1)
        
    Returns
    -------
    TensorVector
        Output values of shape (N,)
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

