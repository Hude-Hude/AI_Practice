"""
OPM Estimator: Two-Step Estimation for Static Oligopoly Pricing Model

Implements:
1. Berry inversion: Recover mean utilities from shares
2. GMM estimation: Estimate price sensitivity (α) using instruments
3. Cost recovery: Back out marginal costs from FOC

Based on pseudocode from estimate_opm.qmd.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, minimize

# Type aliases
Scalar = float
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EstimationResult:
    """Result of OPM estimation.

    Attributes
    ----------
    alpha_hat : float
        Estimated price sensitivity
    alpha_se : float
        Standard error of alpha_hat
    xi_hat : np.ndarray
        Recovered demand shocks, shape (n_markets, J)
    omega_hat : np.ndarray
        Recovered cost shocks, shape (n_markets, J)
    costs_hat : np.ndarray
        Recovered marginal costs, shape (n_markets, J)
    markups_hat : np.ndarray
        Implied markups, shape (n_markets, J)
    gmm_objective : float
        GMM objective value at optimum
    n_markets : int
        Number of markets
    n_products : int
        Number of products
    """

    alpha_hat: Scalar
    alpha_se: Scalar
    xi_hat: Matrix
    omega_hat: Matrix
    costs_hat: Matrix
    markups_hat: Matrix
    gmm_objective: Scalar
    n_markets: int
    n_products: int


# =============================================================================
# Berry Inversion
# =============================================================================


def berry_inversion(
    shares: Matrix,
) -> Matrix:
    """Invert market shares to recover mean utilities.

    For logit demand without random coefficients:
        δ_jm = ln(s_jm) - ln(s_0m)

    where s_0m = 1 - Σ_j s_jm is the outside option share.

    Parameters
    ----------
    shares : np.ndarray
        Market shares, shape (n_markets, J)

    Returns
    -------
    np.ndarray
        Mean utilities δ_jm, shape (n_markets, J)
    """
    # Compute outside option share
    outside_share = 1.0 - np.sum(shares, axis=1, keepdims=True)

    # Berry inversion: δ = ln(s) - ln(s_0)
    delta = np.log(shares) - np.log(outside_share)

    return delta


# =============================================================================
# GMM Estimation
# =============================================================================


def compute_demand_shocks(
    delta_obs: Matrix,
    prices: Matrix,
    delta_bar: Vector,
    alpha: Scalar,
) -> Matrix:
    """Compute demand shocks given alpha.

    From the model:
        δ_jm^obs = δ̄_j - α*p_jm + ξ_jm

    Rearranging:
        ξ_jm = δ_jm^obs - δ̄_j + α*p_jm

    Parameters
    ----------
    delta_obs : np.ndarray
        Observed mean utilities from Berry inversion, shape (n_markets, J)
    prices : np.ndarray
        Observed prices, shape (n_markets, J)
    delta_bar : np.ndarray
        Baseline mean utilities, shape (J,)
    alpha : float
        Price sensitivity coefficient

    Returns
    -------
    np.ndarray
        Demand shocks ξ_jm, shape (n_markets, J)
    """
    xi = delta_obs - delta_bar + alpha * prices
    return xi


def gmm_objective(
    alpha: Scalar,
    delta_obs: Matrix,
    prices: Matrix,
    delta_bar: Vector,
    instruments: Matrix,
    weight_matrix: Optional[Matrix] = None,
) -> Scalar:
    """Compute GMM objective function.

    Q(α) = ξ(α)' Z W Z' ξ(α)

    where ξ(α) = δ^obs - δ̄ + α*p

    Parameters
    ----------
    alpha : float
        Price sensitivity coefficient
    delta_obs : np.ndarray
        Observed mean utilities, shape (n_markets, J)
    prices : np.ndarray
        Observed prices, shape (n_markets, J)
    delta_bar : np.ndarray
        Baseline mean utilities, shape (J,)
    instruments : np.ndarray
        Instruments, shape (n_markets * J, K)
    weight_matrix : np.ndarray, optional
        GMM weight matrix, shape (K, K). Default: (Z'Z)^{-1}

    Returns
    -------
    float
        GMM objective value
    """
    # Compute demand shocks
    xi = compute_demand_shocks(
        delta_obs=delta_obs,
        prices=prices,
        delta_bar=delta_bar,
        alpha=alpha,
    )
    xi_vec = xi.flatten()

    # Moment conditions: g = Z' * ξ
    moments = instruments.T @ xi_vec

    # Weight matrix
    if weight_matrix is None:
        weight_matrix = np.linalg.inv(instruments.T @ instruments)

    # GMM objective: Q = g' W g
    Q = moments.T @ weight_matrix @ moments

    return Q


def estimate_alpha_gmm(
    prices: Matrix,
    shares: Matrix,
    delta_bar: Vector,
    instruments: Matrix,
    alpha_bounds: Tuple[Scalar, Scalar] = (0.1, 5.0),
    tolerance: Scalar = 1e-8,
) -> Tuple[Scalar, Scalar, Scalar]:
    """Estimate price sensitivity α via GMM.

    Parameters
    ----------
    prices : np.ndarray
        Observed prices, shape (n_markets, J)
    shares : np.ndarray
        Observed market shares, shape (n_markets, J)
    delta_bar : np.ndarray
        Baseline mean utilities, shape (J,)
    instruments : np.ndarray
        Instruments, shape (n_markets * J, K)
    alpha_bounds : tuple
        Bounds for alpha search (min, max)
    tolerance : float
        Optimization tolerance

    Returns
    -------
    alpha_hat : float
        Estimated price sensitivity
    alpha_se : float
        Standard error (from numerical Hessian)
    gmm_value : float
        GMM objective at optimum
    """
    # Berry inversion
    delta_obs = berry_inversion(shares=shares)

    # Weight matrix (two-step efficient GMM)
    weight_matrix = np.linalg.inv(instruments.T @ instruments)

    # Objective function
    def objective(alpha):
        return gmm_objective(
            alpha=alpha,
            delta_obs=delta_obs,
            prices=prices,
            delta_bar=delta_bar,
            instruments=instruments,
            weight_matrix=weight_matrix,
        )

    # Minimize using bounded scalar optimization
    result = minimize_scalar(
        objective,
        bounds=alpha_bounds,
        method="bounded",
        options={"xatol": tolerance},
    )

    alpha_hat = result.x
    gmm_value = result.fun

    # Standard error from numerical Hessian
    eps = 1e-4
    f_plus = objective(alpha_hat + eps)
    f_minus = objective(alpha_hat - eps)
    f_center = gmm_value
    hessian = (f_plus - 2 * f_center + f_minus) / (eps ** 2)

    if hessian > 0:
        alpha_se = np.sqrt(1.0 / hessian)
    else:
        alpha_se = np.nan  # Hessian not positive definite

    return alpha_hat, alpha_se, gmm_value


# =============================================================================
# Cost Recovery
# =============================================================================


def recover_costs(
    prices: Matrix,
    shares: Matrix,
    alpha: Scalar,
    costs_bar: Vector,
    ownership: Matrix,
) -> Tuple[Matrix, Matrix, Matrix]:
    """Recover marginal costs from FOC.

    FOC: p_j = c_j + η_j
    where η_j = 1/(α(1-s_j)) for single-product firms

    Therefore: c_j = p_j - η_j
    Cost shock: ω_j = c_j - c̄_j

    Parameters
    ----------
    prices : np.ndarray
        Observed prices, shape (n_markets, J)
    shares : np.ndarray
        Observed market shares, shape (n_markets, J)
    alpha : float
        Estimated price sensitivity
    costs_bar : np.ndarray
        Baseline marginal costs, shape (J,)
    ownership : np.ndarray
        Ownership matrix, shape (J, J)

    Returns
    -------
    costs_hat : np.ndarray
        Recovered marginal costs, shape (n_markets, J)
    omega_hat : np.ndarray
        Recovered cost shocks, shape (n_markets, J)
    markups_hat : np.ndarray
        Implied markups, shape (n_markets, J)
    """
    n_markets, J = shares.shape

    # Initialize
    markups_hat = np.zeros((n_markets, J))
    costs_hat = np.zeros((n_markets, J))
    omega_hat = np.zeros((n_markets, J))

    for m in range(n_markets):
        s_m = shares[m]

        # Compute Delta matrix (share derivatives)
        # Δ_jk = -α * s_j * s_k for j ≠ k
        # Δ_jj = α * s_j * (1 - s_j)
        Delta_m = -alpha * np.outer(s_m, s_m)
        np.fill_diagonal(Delta_m, alpha * s_m * (1 - s_m))

        # A = Ω ⊙ Δ (element-wise product)
        A_m = ownership * Delta_m

        # Markup formula: η = A^{-1} @ s
        # But for single-product firms: η_j = 1/(α(1-s_j))
        if np.allclose(ownership, np.eye(J)):
            # Single-product firms: direct formula
            markups_hat[m] = 1.0 / (alpha * (1 - s_m))
        else:
            # Multi-product firms: solve system
            markups_hat[m] = np.linalg.solve(A_m, s_m)

        # Back out costs
        costs_hat[m] = prices[m] - markups_hat[m]

        # Cost shocks
        omega_hat[m] = costs_hat[m] - costs_bar

    return costs_hat, omega_hat, markups_hat


# =============================================================================
# Instruments
# =============================================================================


def construct_cost_instruments(
    omega_true: Matrix,
) -> Matrix:
    """Construct instruments from true cost shocks.

    In simulation, we know the true cost shocks ω, which are:
    - Correlated with prices (relevance)
    - Uncorrelated with demand shocks ξ (exogeneity)

    Parameters
    ----------
    omega_true : np.ndarray
        True cost shocks, shape (n_markets, J)

    Returns
    -------
    np.ndarray
        Instruments, shape (n_markets * J, K)
    """
    # Flatten and reshape as column vector instrument
    n_markets, J = omega_true.shape

    # Use omega directly as instrument (one instrument per product)
    instruments = omega_true.flatten().reshape(-1, 1)

    # Could add more instruments: omega^2, cross-products, etc.
    # For now, just use omega

    return instruments


def construct_blp_instruments(
    characteristics: Matrix,
) -> Matrix:
    """Construct BLP-style instruments from product characteristics.

    BLP instruments: sum of other products' characteristics.
    Z_jm = Σ_{k≠j} x_km

    Parameters
    ----------
    characteristics : np.ndarray
        Product characteristics, shape (n_markets, J) or (J,)

    Returns
    -------
    np.ndarray
        Instruments, shape (n_markets * J, K)
    """
    if characteristics.ndim == 1:
        # Same characteristics across markets
        J = len(characteristics)
        # Sum of others' characteristics
        total = np.sum(characteristics)
        blp_iv = total - characteristics  # Z_j = Σ_{k≠j} x_k
        # Replicate for consistency
        instruments = blp_iv.reshape(-1, 1)
    else:
        n_markets, J = characteristics.shape
        blp_iv = np.zeros((n_markets, J))
        for m in range(n_markets):
            total = np.sum(characteristics[m])
            blp_iv[m] = total - characteristics[m]
        instruments = blp_iv.flatten().reshape(-1, 1)

    return instruments


# =============================================================================
# Main Estimation Function
# =============================================================================


def estimate_opm(
    prices: Matrix,
    shares: Matrix,
    delta_bar: Vector,
    costs_bar: Vector,
    ownership: Matrix,
    instruments: Matrix,
    alpha_bounds: Tuple[Scalar, Scalar] = (0.1, 5.0),
    gmm_tolerance: Scalar = 1e-8,
) -> EstimationResult:
    """Two-step estimation for OPM.

    Step 1: Estimate α via GMM using Berry inversion
    Step 2: Recover costs and shocks using estimated α

    Parameters
    ----------
    prices : np.ndarray
        Observed prices, shape (n_markets, J)
    shares : np.ndarray
        Observed market shares, shape (n_markets, J)
    delta_bar : np.ndarray
        Baseline mean utilities, shape (J,)
    costs_bar : np.ndarray
        Baseline marginal costs, shape (J,)
    ownership : np.ndarray
        Ownership matrix, shape (J, J)
    instruments : np.ndarray
        Instruments for GMM, shape (n_markets * J, K)
    alpha_bounds : tuple
        Bounds for alpha search
    gmm_tolerance : float
        GMM optimization tolerance

    Returns
    -------
    EstimationResult
        Contains alpha_hat, recovered shocks, costs, diagnostics
    """
    n_markets, J = prices.shape

    # ─────────────────────────────────────────────────────
    # STEP 1: DEMAND ESTIMATION (GMM)
    # ─────────────────────────────────────────────────────

    alpha_hat, alpha_se, gmm_value = estimate_alpha_gmm(
        prices=prices,
        shares=shares,
        delta_bar=delta_bar,
        instruments=instruments,
        alpha_bounds=alpha_bounds,
        tolerance=gmm_tolerance,
    )

    # Recover demand shocks at estimated alpha
    delta_obs = berry_inversion(shares=shares)
    xi_hat = compute_demand_shocks(
        delta_obs=delta_obs,
        prices=prices,
        delta_bar=delta_bar,
        alpha=alpha_hat,
    )

    # ─────────────────────────────────────────────────────
    # STEP 2: COST RECOVERY
    # ─────────────────────────────────────────────────────

    costs_hat, omega_hat, markups_hat = recover_costs(
        prices=prices,
        shares=shares,
        alpha=alpha_hat,
        costs_bar=costs_bar,
        ownership=ownership,
    )

    return EstimationResult(
        alpha_hat=alpha_hat,
        alpha_se=alpha_se,
        xi_hat=xi_hat,
        omega_hat=omega_hat,
        costs_hat=costs_hat,
        markups_hat=markups_hat,
        gmm_objective=gmm_value,
        n_markets=n_markets,
        n_products=J,
    )

