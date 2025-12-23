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
# ALTERNATIVE CONFIGURATIONS
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
    print(f"Damping (λ): {damping}")
    print(f"Tolerance (ε): {tolerance}")
    print(f"Max iterations: {max_iterations}")

