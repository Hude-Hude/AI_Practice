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


# =============================================================================
# Simulation Plots
# =============================================================================


def plot_state_evolution(
    states: np.ndarray,
    actions: np.ndarray,
    s_init_mean: float,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Plot mean state and investment rate over time.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (n_agents, n_periods)
    actions : np.ndarray
        Action history, shape (n_agents, n_periods)
    s_init_mean : float
        Mean initial state (for reference line)
    figsize : tuple
        Figure size (width, height)
    """
    n_periods = states.shape[1]
    periods = np.arange(n_periods)
    mean_state = states.mean(axis=0)
    std_state = states.std(axis=0)
    action_rate = actions.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Mean state over time
    ax0 = axes[0]
    ax0.plot(periods, mean_state, "b-", linewidth=2, label="Mean state")
    ax0.fill_between(
        periods,
        mean_state - std_state,
        mean_state + std_state,
        alpha=0.3,
        color="blue",
        label="±1 std",
    )
    ax0.axhline(
        y=s_init_mean,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Initial mean: {s_init_mean:.1f}",
    )
    ax0.set_xlabel("Period")
    ax0.set_ylabel("State")
    ax0.set_title("Mean State Over Time")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Right: Investment rate over time
    ax1 = axes[1]
    ax1.plot(periods, action_rate, "r-", linewidth=2)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Period")
    ax1.set_ylabel("P(a=1|t)")
    ax1.set_title("Investment Rate Over Time")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_trajectories(
    states: np.ndarray,
    s_init_mean: float,
    n_samples: int = 10,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """Plot sample agent trajectories.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (n_agents, n_periods)
    s_init_mean : float
        Mean initial state (for reference line)
    n_samples : int
        Number of sample trajectories to plot
    figsize : tuple
        Figure size (width, height)
    """
    n_periods = states.shape[1]
    periods = np.arange(n_periods)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(min(n_samples, states.shape[0])):
        ax.plot(periods, states[i, :], alpha=0.6, linewidth=1)

    ax.axhline(
        y=s_init_mean,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Initial mean: {s_init_mean:.1f}",
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("State")
    ax.set_title(f"Sample Trajectories ({n_samples} agents)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_choice_validation(
    states: np.ndarray,
    actions: np.ndarray,
    s_theory: np.ndarray,
    prob_theory: np.ndarray,
    n_bins: int = 20,
    min_count: int = 50,
    figsize: Tuple[float, float] = (10, 5),
) -> float:
    """Plot theoretical vs empirical choice probabilities by state.

    Parameters
    ----------
    states : np.ndarray
        Flattened state observations
    actions : np.ndarray
        Flattened action observations
    s_theory : np.ndarray
        State values for theoretical curve
    prob_theory : np.ndarray
        Theoretical P(a=1|s) values
    n_bins : int
        Number of bins for empirical data
    min_count : int
        Minimum observations per bin to display
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    float
        RMSE between theoretical and empirical probabilities
    """
    # Create bins
    bin_edges = np.linspace(states.min(), states.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute empirical P(a=1|s) for each bin
    empirical_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (states >= bin_edges[i]) & (states < bin_edges[i + 1])
        if mask.sum() > 0:
            empirical_prob[i] = actions[mask].mean()
            bin_counts[i] = mask.sum()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(s_theory, prob_theory, "r-", linewidth=2, label="Theoretical P(a=1|s)")

    valid_bins = bin_counts > min_count
    ax.scatter(
        bin_centers[valid_bins],
        empirical_prob[valid_bins],
        s=50,
        c="blue",
        alpha=0.7,
        label="Simulated (binned)",
        zorder=5,
    )

    ax.set_xlabel("State s")
    ax.set_ylabel("P(a=1|s)")
    ax.set_title("Choice Probability: Theoretical vs Simulated")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Compute RMSE
    valid_theory = np.interp(bin_centers[valid_bins], s_theory, prob_theory)
    rmse = np.sqrt(np.mean((empirical_prob[valid_bins] - valid_theory) ** 2))
    print(f"RMSE between theoretical and simulated choice probabilities: {rmse:.4f}")

    return rmse


def plot_calibration(
    actions: np.ndarray,
    prob_theory: np.ndarray,
    n_bins: int = 50,
    figsize: Tuple[float, float] = (8, 8),
) -> float:
    """Plot calibration: empirical vs theoretical choice probabilities.

    Parameters
    ----------
    actions : np.ndarray
        Flattened action observations
    prob_theory : np.ndarray
        Theoretical P(a=1|s) for each observation
    n_bins : int
        Number of quantile bins
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    float
        Calibration RMSE
    """
    # Quantile-based binning
    bin_edges = np.percentile(prob_theory, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    n_bins_actual = len(bin_edges) - 1

    # Compute empirical P(a=1) for each bin
    empirical_by_theory = np.zeros(n_bins_actual)
    theory_mean = np.zeros(n_bins_actual)
    bin_counts = np.zeros(n_bins_actual)

    for i in range(n_bins_actual):
        if i == n_bins_actual - 1:
            mask = (prob_theory >= bin_edges[i]) & (prob_theory <= bin_edges[i + 1])
        else:
            mask = (prob_theory >= bin_edges[i]) & (prob_theory < bin_edges[i + 1])
        if mask.sum() > 0:
            empirical_by_theory[i] = actions[mask].mean()
            theory_mean[i] = prob_theory[mask].mean()
            bin_counts[i] = mask.sum()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration", alpha=0.7)

    valid_cal = bin_counts > 0
    ax.scatter(
        theory_mean[valid_cal],
        empirical_by_theory[valid_cal],
        s=50,
        c="blue",
        alpha=0.7,
        zorder=5,
        label=f"Binned observations (n={valid_cal.sum()})",
    )

    ax.set_xlabel("Theoretical P(a=1|s)")
    ax.set_ylabel("Empirical P(a=1|s)")
    ax.set_title("Calibration Plot: Empirical vs Theoretical Choice Probability")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calibration statistics
    cal_rmse = np.sqrt(
        np.mean((empirical_by_theory[valid_cal] - theory_mean[valid_cal]) ** 2)
    )
    print(f"Calibration RMSE: {cal_rmse:.4f}")
    print(f"Number of bins: {valid_cal.sum()}")
    print(
        f"Theoretical probability range: [{prob_theory.min():.3f}, {prob_theory.max():.3f}]"
    )

    return cal_rmse


def plot_state_transitions(
    states: np.ndarray,
    actions: np.ndarray,
    gamma: float,
    subsample: int = 5000,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Plot state transitions by action.

    Parameters
    ----------
    states : np.ndarray
        State history, shape (n_agents, n_periods)
    actions : np.ndarray
        Action history, shape (n_agents, n_periods)
    gamma : float
        State decay rate
    subsample : int
        Number of points to subsample for visibility
    figsize : tuple
        Figure size (width, height)
    """
    # Extract transitions
    s_current = states[:, :-1].flatten()
    s_next = states[:, 1:].flatten()
    a_current = actions[:, :-1].flatten()

    mask_a0 = a_current == 0
    mask_a1 = a_current == 1

    fig, ax = plt.subplots(figsize=figsize)

    # Subsample for visibility
    np.random.seed(42)
    idx = np.random.choice(
        len(s_current), size=min(subsample, len(s_current)), replace=False
    )

    ax.scatter(
        s_current[idx][mask_a0[idx]],
        s_next[idx][mask_a0[idx]],
        alpha=0.3,
        s=10,
        c="blue",
        label="a=0 (no investment)",
    )
    ax.scatter(
        s_current[idx][mask_a1[idx]],
        s_next[idx][mask_a1[idx]],
        alpha=0.3,
        s=10,
        c="red",
        label="a=1 (investment)",
    )

    # Theoretical transition lines
    s_line = np.linspace(0, states.max(), 100)
    ax.plot(
        s_line,
        (1 - gamma) * s_line,
        "b--",
        linewidth=2,
        label=f"s' = {1-gamma:.1f}s (a=0)",
    )
    ax.plot(
        s_line,
        (1 - gamma) * s_line + 1,
        "r--",
        linewidth=2,
        label=f"s' = {1-gamma:.1f}s + 1 (a=1)",
    )

    ax.set_xlabel("Current State s")
    ax.set_ylabel("Next State s'")
    ax.set_title("State Transitions by Action")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_state_distribution(
    s_init: np.ndarray,
    final_states: np.ndarray,
    s_min: float,
    s_max: float,
    n_periods: int,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Plot initial vs final state distributions.

    Parameters
    ----------
    s_init : np.ndarray
        Initial states, shape (n_agents,)
    final_states : np.ndarray
        Final states, shape (n_agents,)
    s_min : float
        Minimum state bound
    s_max : float
        Maximum state bound
    n_periods : int
        Number of periods (for title)
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Initial states
    ax0 = axes[0]
    ax0.hist(s_init, bins=20, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax0.axvline(
        x=s_init.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {s_init.mean():.2f}",
    )
    ax0.set_xlabel("State s")
    ax0.set_ylabel("Density")
    ax0.set_title(f"Initial States (t=0)\nUniform[{s_min}, {s_max}]")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Right: Final states
    ax1 = axes[1]
    ax1.hist(final_states, bins=20, density=True, alpha=0.7, color="teal", edgecolor="white")
    ax1.axvline(
        x=final_states.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {final_states.mean():.2f}",
    )
    ax1.set_xlabel("State s")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Final States (t={n_periods-1})\n(Stationary Distribution)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Initial States (t=0):")
    print(f"  Mean: {s_init.mean():.2f}, Std: {s_init.std():.2f}")
    print(f"\nFinal States (t={n_periods-1}):")
    print(f"  Mean: {final_states.mean():.2f}, Std: {final_states.std():.2f}")


def plot_reward_distribution(
    rewards: np.ndarray,
    delta: float,
    figsize: Tuple[float, float] = (14, 5),
) -> Tuple[np.ndarray, np.ndarray]:
    """Plot cumulative vs discounted reward distributions.

    Parameters
    ----------
    rewards : np.ndarray
        Reward history, shape (n_agents, n_periods)
    delta : float
        Discount factor
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    tuple
        (cumulative_rewards, discounted_rewards) arrays
    """
    n_periods = rewards.shape[1]

    # Compute rewards
    cumulative_rewards = rewards.sum(axis=1)
    discount_factors = delta ** np.arange(n_periods)
    discounted_rewards = (rewards * discount_factors).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Cumulative rewards
    ax0 = axes[0]
    ax0.hist(
        cumulative_rewards, bins=20, density=True, alpha=0.7, color="purple", edgecolor="white"
    )
    ax0.axvline(
        x=cumulative_rewards.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {cumulative_rewards.mean():.1f}",
    )
    ax0.set_xlabel("Cumulative Reward (Σ rₜ)")
    ax0.set_ylabel("Density")
    ax0.set_title("Cumulative Rewards (Undiscounted)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Right: Discounted rewards
    ax1 = axes[1]
    ax1.hist(
        discounted_rewards, bins=20, density=True, alpha=0.7, color="darkgreen", edgecolor="white"
    )
    ax1.axvline(
        x=discounted_rewards.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {discounted_rewards.mean():.1f}",
    )
    ax1.set_xlabel(f"Discounted Reward (Σ δᵗrₜ, δ={delta})")
    ax1.set_ylabel("Density")
    ax1.set_title("Discounted Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Cumulative Rewards (Σ rₜ):")
    print(f"  Mean: {cumulative_rewards.mean():.2f}, Std: {cumulative_rewards.std():.2f}")
    print(f"\nDiscounted Rewards (Σ δᵗrₜ, δ={delta}):")
    print(f"  Mean: {discounted_rewards.mean():.2f}, Std: {discounted_rewards.std():.2f}")

    return cumulative_rewards, discounted_rewards

