"""OPM Solver: Static Oligopoly Pricing Model."""

from opm.solver.opm_solver import (
    EquilibriumResult,
    compute_delta,
    compute_shares,
    solve_equilibrium_prices,
    solve_markup_equation,
)

__all__ = [
    "EquilibriumResult",
    "compute_shares",
    "compute_delta",
    "solve_markup_equation",
    "solve_equilibrium_prices",
]
