"""
OPM Simulator: Monte Carlo Simulation for Static Oligopoly Pricing Model

Implements the SIMULATE_OPM_MARKETS algorithm for generating a cross-section
of markets with demand and cost shocks.

Based on pseudocode from simulate_opm.qmd.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Import from opm.solver for shared functionality
from opm.solver_opm import (
    solve_equilibrium_prices,
    compute_delta,
)


# =============================================================================
# Type Definitions
# =============================================================================

Scalar = float
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SimulationResult:
    """Result of OPM Monte Carlo simulation.

    Attributes
    ----------
    prices : np.ndarray
        Equilibrium prices, shape (n_markets, J)
    shares : np.ndarray
        Market shares, shape (n_markets, J)
    markups : np.ndarray
        Markups (p - c), shape (n_markets, J)
    converged : np.ndarray
        Convergence flags, shape (n_markets,)
    foc_errors : np.ndarray
        FOC residual norms, shape (n_markets,)
    n_markets : int
        Number of markets simulated
    n_products : int
        Number of products (J)
    """

    prices: Matrix
    shares: Matrix
    markups: Matrix
    converged: NDArray[np.bool_]
    foc_errors: Vector
    n_markets: int
    n_products: int


# =============================================================================
# Main Algorithm
# =============================================================================


def simulate_opm_markets(
    delta_bar: Vector,
    costs_bar: Vector,
    alpha: Scalar,
    ownership: Matrix,
    n_markets: int,
    sigma_xi: Scalar,
    sigma_omega: Scalar,
    seed: Optional[int] = None,
    damping: Scalar = 1.0,
    tolerance: Scalar = 1e-10,
    max_iterations: int = 1000,
) -> SimulationResult:
    """Simulate OPM markets using Monte Carlo with demand/cost shocks.

    Implements SIMULATE_OPM_MARKETS from pseudocode.

    For each market m:
    1. Draw demand shocks: ξ_m ~ N(0, σ_ξ²)
    2. Draw cost shocks: ω_m ~ N(0, σ_ω²)
    3. Solve equilibrium: p_m = c_m + η_m
    4. Store outcomes and compute FOC residuals

    Parameters
    ----------
    delta_bar : np.ndarray
        Baseline mean utilities, shape (J,)
    costs_bar : np.ndarray
        Baseline marginal costs, shape (J,)
    alpha : float
        Price sensitivity coefficient (structural, fixed)
    ownership : np.ndarray
        Ownership matrix, shape (J, J)
    n_markets : int
        Number of market draws (M)
    sigma_xi : float
        Standard deviation of demand shocks
    sigma_omega : float
        Standard deviation of cost shocks
    seed : int, optional
        Random seed for reproducibility
    damping : float, optional
        Damping factor for solver (default 1.0)
    tolerance : float, optional
        Convergence tolerance for solver (default 1e-10)
    max_iterations : int, optional
        Maximum solver iterations (default 1000)

    Returns
    -------
    SimulationResult
        Contains prices, shares, markups, convergence info for all markets
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Get number of products
    J = len(delta_bar)

    # Initialize storage matrices
    prices = np.zeros((n_markets, J))
    shares = np.zeros((n_markets, J))
    markups = np.zeros((n_markets, J))
    converged = np.zeros(n_markets, dtype=bool)
    foc_errors = np.zeros(n_markets)

    # Simulate each market
    for m in range(n_markets):
        # Draw demand shocks
        xi_m = np.random.normal(loc=0, scale=sigma_xi, size=J)
        delta_m = delta_bar + xi_m

        # Draw cost shocks
        omega_m = np.random.normal(loc=0, scale=sigma_omega, size=J)
        costs_m = costs_bar + omega_m

        # Solve equilibrium for this market
        result_m = solve_equilibrium_prices(
            delta=delta_m,
            alpha=alpha,
            costs=costs_m,
            ownership=ownership,
            damping=damping,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

        # Store outcomes
        prices[m] = result_m.prices
        shares[m] = result_m.shares
        markups[m] = result_m.markups
        converged[m] = result_m.converged

        # Compute FOC residual
        Delta_m = compute_delta(alpha=alpha, shares=result_m.shares)
        A_m = ownership * Delta_m
        foc_residual = result_m.shares - A_m @ result_m.markups
        foc_errors[m] = np.max(np.abs(foc_residual))

    return SimulationResult(
        prices=prices,
        shares=shares,
        markups=markups,
        converged=converged,
        foc_errors=foc_errors,
        n_markets=n_markets,
        n_products=J,
    )

