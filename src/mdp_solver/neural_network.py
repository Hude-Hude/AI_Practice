"""Monotonic neural network architecture for value function approximation."""

from typing import List, Tuple

import torch
import torch.nn as nn


class MonotonicLayer(nn.Module):
    """
    A linear layer with non-negative weights (monotonic in inputs).

    Weights are parameterized as softplus of unconstrained parameters
    to ensure W > 0 while allowing gradient-based optimization.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Unconstrained weight parameters
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features))

        # Bias (unconstrained)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with non-negative weights.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features)
        """
        # Apply softplus to get non-negative weights
        weight: torch.Tensor = nn.functional.softplus(self.weight_raw)
        return nn.functional.linear(input=x, weight=weight, bias=self.bias)


class MonotonicValueNetwork(nn.Module):
    """
    Monotonic neural network for approximating value functions.

    Ensures output is monotonically increasing in input state by using:
    - Non-negative weights (via softplus parameterization)
    - Monotonic activation functions (softplus)

    Parameters
    ----------
    hidden_sizes : List[int]
        Number of units in each hidden layer
    """

    def __init__(self, hidden_sizes: List[int]) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_size: int = 1  # Single state input

        for hidden_size in hidden_sizes:
            layers.append(
                MonotonicLayer(in_features=in_size, out_features=hidden_size)
            )
            layers.append(nn.Softplus())
            in_size = hidden_size

        # Output layer (single value output)
        layers.append(MonotonicLayer(in_features=in_size, out_features=1))

        self.network = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute value function at state(s).

        Parameters
        ----------
        s : torch.Tensor
            State(s), shape (batch_size,) or (batch_size, 1)

        Returns
        -------
        torch.Tensor
            Value(s), shape (batch_size,)
        """
        if s.dim() == 1:
            s = s.unsqueeze(dim=-1)
        return self.network(s).squeeze(dim=-1)


class ValueFunctionApproximator(nn.Module):
    """
    Approximates choice-specific value functions v(s, 0) and v(s, 1).

    Uses separate monotonic networks for each action.

    Parameters
    ----------
    hidden_sizes : List[int]
        Hidden layer sizes for each network
    """

    def __init__(self, hidden_sizes: List[int]) -> None:
        super().__init__()
        self.v0_net = MonotonicValueNetwork(hidden_sizes=hidden_sizes)
        self.v1_net = MonotonicValueNetwork(hidden_sizes=hidden_sizes)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute choice-specific values for both actions.

        Parameters
        ----------
        s : torch.Tensor
            State(s)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (v0, v1) - Values for action 0 and action 1
        """
        v0: torch.Tensor = self.v0_net(s)
        v1: torch.Tensor = self.v1_net(s)
        return v0, v1

    def compute_integrated_value(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute integrated value function (log-sum-exp).

        V(s) = log(exp(v0) + exp(v1))

        Parameters
        ----------
        s : torch.Tensor
            State(s)

        Returns
        -------
        torch.Tensor
            Integrated value V(s)
        """
        v0, v1 = self.forward(s)
        return torch.logsumexp(torch.stack([v0, v1], dim=-1), dim=-1)

    def compute_choice_probability(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute probability of choosing action 1.

        P(a=1|s) = sigmoid(v1 - v0)

        Parameters
        ----------
        s : torch.Tensor
            State(s)

        Returns
        -------
        torch.Tensor
            P(a=1|s)
        """
        v0, v1 = self.forward(s)
        return torch.sigmoid(v1 - v0)

