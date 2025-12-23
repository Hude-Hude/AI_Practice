"""
OPM Solver: Static Oligopoly Pricing Model

Solves for Bertrand-Nash equilibrium prices with logit demand
and differentiated products.

Reference: Berry (1994), "Estimating Discrete-Choice Models of
Product Differentiation", RAND Journal of Economics.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class EquilibriumResult:
    """Result of equilibrium price solver."""

    prices: NDArray[np.float64]  # Equilibrium prices
    markups: NDArray[np.float64]  # Equilibrium markups (p - c)
    shares: NDArray[np.float64]  # Equilibrium market shares
    converged: bool  # Convergence flag
    n_iterations: int  # Number of iterations
    final_error: float  # Final convergence error


def compute_shares(
    delta: NDArray[np.float64],
    alpha: float,
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute market shares using logit demand.

    Parameters
    ----------
    delta : Vector[J]
        Mean utilities for each product.
    alpha : Scalar
        Price sensitivity coefficient (> 0).
    prices : Vector[J]
        Prices for each product.

    Returns
    -------
    shares : Vector[J]
        Market shares for each product.
    """
    # Compute mean utilities net of price
    v = delta - alpha * prices

    # Numerically stable softmax
    v_max = np.max(v)
    exp_v = np.exp(v - v_max)
    denom = np.exp(-v_max) + np.sum(exp_v)

    shares = exp_v / denom
    return shares


def compute_delta(
    alpha: float,
    shares: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute share derivative matrix.

    Parameters
    ----------
    alpha : Scalar
        Price sensitivity coefficient.
    shares : Vector[J]
        Market shares.

    Returns
    -------
    Delta : Matrix[J, J]
        Share derivative matrix where Delta[j,k] = -∂s_j/∂p_k.
    """
    J = len(shares)
    Delta = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                # Own-price effect: -∂s_j/∂p_j = α * s_j * (1 - s_j)
                Delta[j, k] = alpha * shares[j] * (1 - shares[j])
            else:
                # Cross-price effect: -∂s_j/∂p_k = -α * s_j * s_k
                Delta[j, k] = -alpha * shares[j] * shares[k]

    return Delta


def solve_markup_equation(
    ownership: NDArray[np.float64],
    Delta: NDArray[np.float64],
    shares: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Solve the markup equation for equilibrium markups.

    The FOC system is: s - (Ω ⊙ Δ)η = 0
    Solving: η = (Ω ⊙ Δ)^{-1} s

    Parameters
    ----------
    ownership : Matrix[J, J]
        Ownership matrix (Ω[j,k] = 1 if same firm, 0 otherwise).
    Delta : Matrix[J, J]
        Share derivative matrix where Δ[j,k] = -∂s_j/∂p_k.
    shares : Vector[J]
        Market shares.

    Returns
    -------
    markups : Vector[J]
        Equilibrium markups.
    """
    # Compute Ω ⊙ Δ (element-wise product)
    A = ownership * Delta

    # Solve linear system: A * η = s
    markups = np.linalg.solve(A, shares)

    return markups


def solve_equilibrium_prices(
    delta: NDArray[np.float64],
    alpha: float,
    costs: NDArray[np.float64],
    ownership: NDArray[np.float64],
    damping: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> EquilibriumResult:
    """
    Solve for Bertrand-Nash equilibrium prices.

    Uses fixed-point iteration on markups with optional damping.

    Parameters
    ----------
    delta : Vector[J]
        Mean utilities for each product.
    alpha : Scalar
        Price sensitivity coefficient (> 0).
    costs : Vector[J]
        Marginal costs for each product.
    ownership : Matrix[J, J]
        Ownership matrix (Ω[j,k] = 1 if products j,k owned by same firm).
    damping : Scalar
        Damping factor λ ∈ (0, 1]. Default 1.0 (no damping).
    tolerance : Scalar
        Convergence tolerance for markup change.
    max_iterations : Int
        Maximum number of iterations.

    Returns
    -------
    result : EquilibriumResult
        Contains equilibrium prices, markups, shares, and convergence info.
    """
    J = len(delta)

    # Initialize markups (start from marginal cost pricing)
    markups = np.zeros(J)

    for k in range(max_iterations):
        # Current prices
        prices = costs + markups

        # Compute market shares
        shares = compute_shares(delta=delta, alpha=alpha, prices=prices)

        # Compute share derivative matrix
        Delta = compute_delta(alpha=alpha, shares=shares)

        # Compute new markups from FOC
        markups_new = solve_markup_equation(
            ownership=ownership,
            Delta=Delta,
            shares=shares,
        )

        # Damped update
        markups_update = (1 - damping) * markups + damping * markups_new

        # Check convergence
        error = np.max(np.abs(markups_update - markups))
        if error < tolerance:
            markups = markups_update
            prices = costs + markups
            shares = compute_shares(delta=delta, alpha=alpha, prices=prices)
            return EquilibriumResult(
                prices=prices,
                markups=markups,
                shares=shares,
                converged=True,
                n_iterations=k + 1,
                final_error=error,
            )

        markups = markups_update

    # Did not converge
    prices = costs + markups
    shares = compute_shares(delta=delta, alpha=alpha, prices=prices)
    return EquilibriumResult(
        prices=prices,
        markups=markups,
        shares=shares,
        converged=False,
        n_iterations=max_iterations,
        final_error=error,
    )

