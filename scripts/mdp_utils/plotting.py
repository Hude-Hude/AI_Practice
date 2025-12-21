"""Plotting utilities for MDP solver and simulator.

This module provides reusable plotting functions for visualizing
MDP solutions, ensuring consistent output across scripts.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Color Maps
# =============================================================================

# Color gradients for comparative statics
CMAP_BLUE = mcolors.LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])
CMAP_RED = mcolors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])


# =============================================================================
# Basic Plots
# =============================================================================


def plot_convergence(
    losses: np.ndarray,
    figsize: Tuple[float, float] = (10, 4),
) -> None:
    """Plot training convergence (MSE loss over iterations).

    Parameters
    ----------
    losses : np.ndarray
        Array of loss values per iteration
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.semilogy(losses, "b-", linewidth=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_value_functions(
    s: np.ndarray,
    v0_values: np.ndarray,
    v1_values: np.ndarray,
    V_bar: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Plot choice-specific value functions.

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    v0_values : np.ndarray
        Value function for action 0
    v1_values : np.ndarray
        Value function for action 1
    V_bar : np.ndarray, optional
        Integrated value function
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.plot(s, v0_values, "b-", label="v(s, 0) - No investment", linewidth=2)
    plt.plot(s, v1_values, "r-", label="v(s, 1) - Invest", linewidth=2)
    if V_bar is not None:
        plt.plot(s, V_bar, "k--", label="V̄(s) - Integrated value", linewidth=2)
    plt.xlabel("State s")
    plt.ylabel("Value")
    plt.title("Choice-Specific and Integrated Value Functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_choice_probabilities(
    s: np.ndarray,
    prob_a0: np.ndarray,
    prob_a1: np.ndarray,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Plot choice probabilities as a function of state.

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    prob_a0 : np.ndarray
        Probability of action 0
    prob_a1 : np.ndarray
        Probability of action 1
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.plot(s, prob_a0, "b-", label="P(a=0|s) - No investment", linewidth=2)
    plt.plot(s, prob_a1, "r-", label="P(a=1|s) - Invest", linewidth=2)
    plt.xlabel("State s")
    plt.ylabel("Probability")
    plt.title("Choice Probabilities")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_value_difference(
    s: np.ndarray,
    v0_values: np.ndarray,
    v1_values: np.ndarray,
    figsize: Tuple[float, float] = (10, 5),
) -> None:
    """Plot value difference v(s,1) - v(s,0).

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    v0_values : np.ndarray
        Value function for action 0
    v1_values : np.ndarray
        Value function for action 1
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    value_diff = v1_values - v0_values
    plt.plot(s, value_diff, "m-", linewidth=2)
    plt.xlabel("State s")
    plt.ylabel("v(s, 1) - v(s, 0)")
    plt.title("Value Difference: Benefit of Investment")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_bellman_residuals(
    s: np.ndarray,
    residual0: np.ndarray,
    residual1: np.ndarray,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Plot Bellman residuals for both actions.

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    residual0 : np.ndarray
        Bellman residual for action 0
    residual1 : np.ndarray
        Bellman residual for action 1
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax0 = axes[0]
    ax0.plot(s, residual0, "b-", linewidth=1.5)
    ax0.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax0.set_xlabel("State s")
    ax0.set_ylabel("v(s,0) - target(s,0)")
    ax0.set_title("Bellman Residual: a=0 (No Investment)")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(s, residual1, "r-", linewidth=1.5)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("State s")
    ax1.set_ylabel("v(s,1) - target(s,1)")
    ax1.set_title("Bellman Residual: a=1 (Investment)")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("Bellman Residual Summary:")
    print(
        f"  a=0: Mean = {np.mean(residual0):.6f}, "
        f"Max |residual| = {np.max(np.abs(residual0)):.6f}"
    )
    print(
        f"  a=1: Mean = {np.mean(residual1):.6f}, "
        f"Max |residual| = {np.max(np.abs(residual1)):.6f}"
    )
    print(f"  Combined RMSE = {np.sqrt(np.mean(residual0**2 + residual1**2) / 2):.6f}")


# =============================================================================
# Comparative Statics Plots
# =============================================================================


def plot_comparative_statics_values(
    s: np.ndarray,
    results: Dict,
    param_values: np.ndarray,
    param_name: str,
    param_min: float,
    param_max: float,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Plot value functions for comparative statics (parameter sweep).

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    results : dict
        Dictionary mapping parameter values to {'v0': ..., 'v1': ...}
    param_values : np.ndarray
        Array of parameter values
    param_name : str
        Name of the parameter (for colorbar label)
    param_min : float
        Minimum parameter value (for colorbar)
    param_max : float
        Maximum parameter value (for colorbar)
    figsize : tuple
        Figure size (width, height)
    """
    n_params = len(param_values)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot v(s, 0) - No investment (black to blue)
    ax0 = axes[0]
    for i, p in enumerate(param_values):
        color = CMAP_BLUE(i / (n_params - 1))
        ax0.plot(s, results[p]["v0"], color=color, linewidth=1.5, alpha=0.8)
    ax0.set_xlabel("State s")
    ax0.set_ylabel("v(s, 0)")
    ax0.set_title("Value Function: No Investment (a=0)")
    ax0.grid(True, alpha=0.3)

    # Add colorbar
    sm0 = plt.cm.ScalarMappable(
        cmap=CMAP_BLUE, norm=plt.Normalize(vmin=param_min, vmax=param_max)
    )
    sm0.set_array([])
    cbar0 = plt.colorbar(sm0, ax=ax0)
    cbar0.set_label(param_name)

    # Plot v(s, 1) - Investment (black to red)
    ax1 = axes[1]
    for i, p in enumerate(param_values):
        color = CMAP_RED(i / (n_params - 1))
        ax1.plot(s, results[p]["v1"], color=color, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel("State s")
    ax1.set_ylabel("v(s, 1)")
    ax1.set_title("Value Function: Investment (a=1)")
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    sm1 = plt.cm.ScalarMappable(
        cmap=CMAP_RED, norm=plt.Normalize(vmin=param_min, vmax=param_max)
    )
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1)
    cbar1.set_label(param_name)

    plt.tight_layout()
    plt.show()


def plot_comparative_statics_probs(
    s: np.ndarray,
    results: Dict,
    param_values: np.ndarray,
    param_name: str,
    param_min: float,
    param_max: float,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Plot choice probabilities for comparative statics (parameter sweep).

    Parameters
    ----------
    s : np.ndarray
        State values (x-axis)
    results : dict
        Dictionary mapping parameter values to {'p0': ..., 'p1': ...}
    param_values : np.ndarray
        Array of parameter values
    param_name : str
        Name of the parameter (for colorbar label)
    param_min : float
        Minimum parameter value (for colorbar)
    param_max : float
        Maximum parameter value (for colorbar)
    figsize : tuple
        Figure size (width, height)
    """
    n_params = len(param_values)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot P(a=0|s) - No investment (black to blue)
    ax0 = axes[0]
    for i, p in enumerate(param_values):
        color = CMAP_BLUE(i / (n_params - 1))
        ax0.plot(s, results[p]["p0"], color=color, linewidth=1.5, alpha=0.8)
    ax0.set_xlabel("State s")
    ax0.set_ylabel("P(a=0|s)")
    ax0.set_title("Choice Probability: No Investment (a=0)")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    # Add colorbar
    sm0 = plt.cm.ScalarMappable(
        cmap=CMAP_BLUE, norm=plt.Normalize(vmin=param_min, vmax=param_max)
    )
    sm0.set_array([])
    cbar0 = plt.colorbar(sm0, ax=ax0)
    cbar0.set_label(param_name)

    # Plot P(a=1|s) - Investment (black to red)
    ax1 = axes[1]
    for i, p in enumerate(param_values):
        color = CMAP_RED(i / (n_params - 1))
        ax1.plot(s, results[p]["p1"], color=color, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel("State s")
    ax1.set_ylabel("P(a=1|s)")
    ax1.set_title("Choice Probability: Investment (a=1)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    sm1 = plt.cm.ScalarMappable(
        cmap=CMAP_RED, norm=plt.Normalize(vmin=param_min, vmax=param_max)
    )
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1)
    cbar1.set_label(param_name)

    plt.tight_layout()
    plt.show()

