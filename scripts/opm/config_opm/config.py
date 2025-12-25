"""
OPM Configuration: Baseline Parameters for Static Oligopoly Pricing Model

This module stores all primitives required by the OPM solver, simulator,
and estimator. Configuration is treated as data—no hard-coded values
inside solver/simulator/estimator notebooks.

Reference: Berry (1994), RAND Journal of Economics
"""

import numpy as np

# =============================================================================
# MARKET STRUCTURE
# =============================================================================

# Number of products
J = 3

# Number of firms (for baseline: each firm owns one product)
F = 3

# =============================================================================
# DEMAND PARAMETERS
# =============================================================================

# Mean utilities (product quality index)
# Higher delta = more attractive product
delta = np.ones(J) * 1.0

# Price sensitivity coefficient (alpha > 0)
# Higher alpha = more elastic demand = lower markups
alpha = 1.0

# =============================================================================
# COST PARAMETERS
# =============================================================================

# Marginal costs
costs = np.ones(J) * 0.5

# =============================================================================
# OWNERSHIP STRUCTURE
# =============================================================================

# Ownership matrix: Omega[j,k] = 1 if products j,k owned by same firm
# Baseline: single-product firms (identity matrix)
ownership = np.eye(J)

# =============================================================================
# SOLVER PARAMETERS
# =============================================================================

# Damping factor for fixed-point iteration (lambda in (0,1])
# 1.0 = no damping (faster but may not converge)
# <1.0 = damped (slower but more stable)
damping = 1.0

# Convergence tolerance
tolerance = 1e-10

# Maximum iterations
max_iterations = 1000

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Number of market draws for Monte Carlo simulation
n_markets = 1000

# Standard deviation of demand shocks (ξ ~ N(0, σ_ξ²))
# Higher σ_ξ → more variation in product quality across markets
sigma_xi = 0.5

# Standard deviation of cost shocks (ω ~ N(0, σ_ω²))
# Higher σ_ω → more variation in marginal costs across markets
sigma_omega = 0.2

# Random seed for reproducibility
simulation_seed = 42

# =============================================================================
# ANALYTICAL BENCHMARKS
# =============================================================================

def compute_expected_markup(share: float, alpha: float = alpha) -> float:
    """
    Compute expected markup for single-product firm.
    
    Formula: η = 1 / (α * (1 - s))
    """
    return 1.0 / (alpha * (1.0 - share))


def compute_pass_through(share: float) -> float:
    """
    Compute cost pass-through rate for single-product firm.
    
    Formula: ρ = 1 - s
    """
    return 1.0 - share


# =============================================================================
# SCENARIO CONFIGURATIONS
# =============================================================================

SCENARIOS = {
    "baseline": {
        "delta": np.array([1.0, 1.0, 1.0]),
        "costs": np.array([0.5, 0.5, 0.5]),
        "ownership": np.eye(3),
        "alpha": 1.0,
        "description": "Symmetric/homogeneous products",
    },
    "quality": {
        "delta": np.array([0.5, 1.0, 2.0]),
        "costs": np.array([0.5, 0.5, 0.5]),
        "ownership": np.eye(3),
        "alpha": 1.0,
        "description": "Quality differentiation only (δ varies, c constant)",
    },
    "cost": {
        "delta": np.array([1.0, 1.0, 1.0]),
        "costs": np.array([0.3, 0.5, 0.7]),
        "ownership": np.eye(3),
        "alpha": 1.0,
        "description": "Cost differentiation only (δ constant, c varies)",
    },
    "vertical": {
        "delta": np.array([0.5, 1.0, 2.0]),
        "costs": np.array([0.3, 0.5, 0.8]),
        "ownership": np.eye(3),
        "alpha": 1.0,
        "description": "Vertical differentiation (δ and c positively correlated)",
    },
    "general": {
        "delta": np.array([2.0, 0.5, 1.0]),
        "costs": np.array([0.3, 0.7, 0.5]),
        "ownership": np.eye(3),
        "alpha": 1.0,
        "description": "General heterogeneous products (δ and c independent)",
    },
}

# =============================================================================
# ALTERNATIVE OWNERSHIP STRUCTURES
# =============================================================================

def get_monopoly_ownership(J: int = J) -> np.ndarray:
    """Return ownership matrix for monopoly (single firm owns all products)."""
    return np.ones((J, J))


def get_duopoly_ownership() -> np.ndarray:
    """
    Return ownership matrix for duopoly.
    Firm 1 owns products 0,1; Firm 2 owns product 2.
    """
    return np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=float)


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    print("OPM Configuration")
    print("=" * 40)
    print(f"Products (J): {J}")
    print(f"Firms (F): {F}")
    print(f"Mean utilities (δ): {delta}")
    print(f"Price sensitivity (α): {alpha}")
    print(f"Marginal costs (c): {costs}")
    print(f"Ownership (Ω):\n{ownership}")
    print()
    print("Solver Parameters:")
    print(f"  Damping (λ): {damping}")
    print(f"  Tolerance (ε): {tolerance}")
    print(f"  Max iterations: {max_iterations}")
    print()
    print("Simulation Parameters:")
    print(f"  Number of markets (M): {n_markets}")
    print(f"  Demand shock std (σ_ξ): {sigma_xi}")
    print(f"  Cost shock std (σ_ω): {sigma_omega}")
    print(f"  Random seed: {simulation_seed}")

