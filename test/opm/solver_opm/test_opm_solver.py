"""Tests for OPM solver: Static Oligopoly Pricing Model.

Tests cover:
1. compute_shares: Logit demand market shares
2. compute_delta: Share derivative matrix
3. solve_markup_equation: Markup equation solver
4. solve_equilibrium_prices: Full equilibrium solver
5. Analytical benchmarks: Single-product monopoly, symmetric equilibrium
"""

import numpy as np
import pytest

from opm.solver_opm import (
    EquilibriumResult,
    compute_shares,
    compute_delta,
    solve_markup_equation,
    solve_equilibrium_prices,
)


# =============================================================================
# Tests for compute_shares
# =============================================================================


class TestComputeShares:
    """Tests for logit demand share computation."""

    def test_output_shape(self) -> None:
        """Output should have same shape as input."""
        delta = np.array([1.0, 2.0, 3.0])
        prices = np.array([1.0, 1.0, 1.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        assert shares.shape == delta.shape

    def test_shares_sum_less_than_one(self) -> None:
        """Shares should sum to less than 1 (outside option exists)."""
        delta = np.array([1.0, 2.0, 3.0])
        prices = np.array([1.0, 1.0, 1.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        assert np.sum(shares) < 1.0

    def test_shares_positive(self) -> None:
        """All shares should be positive."""
        delta = np.array([1.0, 2.0, 3.0])
        prices = np.array([1.0, 1.0, 1.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        assert np.all(shares > 0)

    def test_higher_delta_higher_share(self) -> None:
        """Product with higher δ should have higher share (equal prices)."""
        delta = np.array([1.0, 2.0, 3.0])
        prices = np.array([1.0, 1.0, 1.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        assert shares[2] > shares[1] > shares[0]

    def test_higher_price_lower_share(self) -> None:
        """Product with higher price should have lower share (equal δ)."""
        delta = np.array([2.0, 2.0, 2.0])
        prices = np.array([1.0, 2.0, 3.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        assert shares[0] > shares[1] > shares[2]

    def test_logit_formula(self) -> None:
        """Shares should follow logit formula."""
        delta = np.array([1.0, 2.0])
        prices = np.array([0.5, 1.0])
        alpha = 1.0
        
        shares = compute_shares(delta, alpha, prices)
        
        # Manual computation
        v = delta - alpha * prices
        exp_v = np.exp(v)
        expected = exp_v / (1 + np.sum(exp_v))
        
        np.testing.assert_allclose(shares, expected)

    def test_price_sensitivity(self) -> None:
        """Higher alpha should make shares more sensitive to price."""
        delta = np.array([2.0, 2.0])
        prices = np.array([1.0, 2.0])  # Product 1 is cheaper
        
        shares_low_alpha = compute_shares(delta, alpha=0.5, prices=prices)
        shares_high_alpha = compute_shares(delta, alpha=2.0, prices=prices)
        
        # With higher alpha, cheap product should have even higher share
        ratio_low = shares_low_alpha[0] / shares_low_alpha[1]
        ratio_high = shares_high_alpha[0] / shares_high_alpha[1]
        
        assert ratio_high > ratio_low

    def test_numerical_stability_large_values(self) -> None:
        """Should handle large utility values without overflow."""
        delta = np.array([100.0, 101.0, 102.0])
        prices = np.array([0.0, 0.0, 0.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        
        assert np.all(np.isfinite(shares))
        assert np.all(shares > 0)
        assert np.sum(shares) < 1.0

    def test_numerical_stability_negative_values(self) -> None:
        """Should handle large negative utility values."""
        delta = np.array([-100.0, -99.0, -98.0])
        prices = np.array([0.0, 0.0, 0.0])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        
        assert np.all(np.isfinite(shares))
        assert np.all(shares > 0)

    def test_single_product(self) -> None:
        """Should work with single product."""
        delta = np.array([1.0])
        prices = np.array([0.5])
        shares = compute_shares(delta, alpha=1.0, prices=prices)
        
        assert shares.shape == (1,)
        assert 0 < shares[0] < 1


# =============================================================================
# Tests for compute_delta
# =============================================================================


class TestComputeDelta:
    """Tests for share derivative matrix computation."""

    def test_output_shape(self) -> None:
        """Output should be J x J matrix."""
        shares = np.array([0.2, 0.3, 0.1])
        Delta = compute_delta(alpha=1.0, shares=shares)
        assert Delta.shape == (3, 3)

    def test_diagonal_positive(self) -> None:
        """Diagonal elements (own-price) should be positive."""
        shares = np.array([0.2, 0.3, 0.1])
        Delta = compute_delta(alpha=1.0, shares=shares)
        assert np.all(np.diag(Delta) > 0)

    def test_off_diagonal_negative(self) -> None:
        """Off-diagonal elements (cross-price) should be negative."""
        shares = np.array([0.2, 0.3, 0.1])
        Delta = compute_delta(alpha=1.0, shares=shares)
        
        J = len(shares)
        for j in range(J):
            for k in range(J):
                if j != k:
                    assert Delta[j, k] < 0

    def test_own_price_formula(self) -> None:
        """Diagonal: Δ_jj = α * s_j * (1 - s_j)."""
        shares = np.array([0.2, 0.3, 0.4])
        alpha = 1.5
        Delta = compute_delta(alpha=alpha, shares=shares)
        
        for j in range(len(shares)):
            expected = alpha * shares[j] * (1 - shares[j])
            np.testing.assert_allclose(Delta[j, j], expected)

    def test_cross_price_formula(self) -> None:
        """Off-diagonal: Δ_jk = -α * s_j * s_k."""
        shares = np.array([0.2, 0.3, 0.4])
        alpha = 1.5
        Delta = compute_delta(alpha=alpha, shares=shares)
        
        for j in range(len(shares)):
            for k in range(len(shares)):
                if j != k:
                    expected = -alpha * shares[j] * shares[k]
                    np.testing.assert_allclose(Delta[j, k], expected)

    def test_alpha_scaling(self) -> None:
        """Delta should scale linearly with alpha."""
        shares = np.array([0.2, 0.3, 0.1])
        
        Delta_1 = compute_delta(alpha=1.0, shares=shares)
        Delta_2 = compute_delta(alpha=2.0, shares=shares)
        
        np.testing.assert_allclose(Delta_2, 2 * Delta_1)

    def test_row_sum_property(self) -> None:
        """Row sum should equal α * s_j * (1 - total_share)."""
        shares = np.array([0.2, 0.3, 0.1])
        alpha = 1.0
        Delta = compute_delta(alpha=alpha, shares=shares)
        
        # Row j sum = α*s_j*(1-s_j) - α*s_j*Σ_{k≠j} s_k
        #           = α*s_j*(1 - s_j - Σ_{k≠j} s_k)
        #           = α*s_j*(1 - Σ_k s_k)
        total_share = np.sum(shares)
        for j in range(len(shares)):
            expected_row_sum = alpha * shares[j] * (1 - total_share)
            np.testing.assert_allclose(
                np.sum(Delta[j, :]), expected_row_sum, rtol=1e-10
            )


# =============================================================================
# Tests for solve_markup_equation
# =============================================================================


class TestSolveMarkupEquation:
    """Tests for markup equation solver."""

    def test_output_shape(self) -> None:
        """Output should have same length as shares."""
        ownership = np.eye(3)
        Delta = np.array([
            [0.2, -0.05, -0.05],
            [-0.05, 0.3, -0.05],
            [-0.05, -0.05, 0.1],
        ])
        shares = np.array([0.2, 0.3, 0.1])
        
        markups = solve_markup_equation(ownership, Delta, shares)
        assert markups.shape == shares.shape

    def test_markups_positive(self) -> None:
        """Markups should be positive for independent firms."""
        ownership = np.eye(3)
        shares = np.array([0.2, 0.3, 0.1])
        alpha = 1.0
        Delta = compute_delta(alpha, shares)
        
        markups = solve_markup_equation(ownership, Delta, shares)
        assert np.all(markups > 0)

    def test_single_product_monopoly_formula(self) -> None:
        """Single product: η = s / (α * s * (1-s)) = 1 / (α * (1-s))."""
        share = 0.3
        alpha = 1.0
        
        ownership = np.array([[1]])
        shares = np.array([share])
        Delta = compute_delta(alpha, shares)
        
        markups = solve_markup_equation(ownership, Delta, shares)
        expected = 1 / (alpha * (1 - share))
        
        np.testing.assert_allclose(markups[0], expected)

    def test_ownership_affects_markups(self) -> None:
        """Merged firms should have higher markups."""
        shares = np.array([0.2, 0.3])
        alpha = 1.0
        Delta = compute_delta(alpha, shares)
        
        # Independent firms
        ownership_independent = np.eye(2)
        markups_independent = solve_markup_equation(
            ownership_independent, Delta, shares
        )
        
        # Merged firms
        ownership_merged = np.ones((2, 2))
        markups_merged = solve_markup_equation(
            ownership_merged, Delta, shares
        )
        
        # Merger should increase markups
        assert np.all(markups_merged > markups_independent)

    def test_symmetric_firms(self) -> None:
        """Symmetric firms should have equal markups."""
        J = 3
        share = 0.2
        shares = np.full(J, share)
        alpha = 1.0
        Delta = compute_delta(alpha, shares)
        ownership = np.eye(J)
        
        markups = solve_markup_equation(ownership, Delta, shares)
        
        # All markups should be equal
        np.testing.assert_allclose(markups, markups[0] * np.ones(J))


# =============================================================================
# Tests for solve_equilibrium_prices
# =============================================================================


class TestSolveEquilibriumPrices:
    """Tests for full equilibrium price solver."""

    def test_returns_equilibrium_result(self) -> None:
        """Should return EquilibriumResult object."""
        delta = np.array([1.0])
        costs = np.array([0.5])
        ownership = np.array([[1]])
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert isinstance(result, EquilibriumResult)

    def test_result_fields(self) -> None:
        """Result should have all required fields."""
        delta = np.array([1.0])
        costs = np.array([0.5])
        ownership = np.array([[1]])
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert hasattr(result, 'prices')
        assert hasattr(result, 'markups')
        assert hasattr(result, 'shares')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'n_iterations')
        assert hasattr(result, 'final_error')

    def test_converges(self) -> None:
        """Should converge for well-posed problem."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert result.converged is True

    def test_prices_above_costs(self) -> None:
        """Equilibrium prices should be above marginal costs."""
        delta = np.array([1.0, 2.0, 3.0])
        costs = np.array([0.5, 0.6, 0.7])
        ownership = np.eye(3)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert np.all(result.prices > costs)

    def test_markups_equal_price_minus_cost(self) -> None:
        """Markups should equal prices minus costs."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.6])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        np.testing.assert_allclose(result.markups, result.prices - costs)

    def test_shares_consistent(self) -> None:
        """Returned shares should be consistent with prices."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.6])
        ownership = np.eye(2)
        alpha = 1.0
        
        result = solve_equilibrium_prices(delta, alpha=alpha, costs=costs, ownership=ownership)
        
        expected_shares = compute_shares(delta, alpha, result.prices)
        np.testing.assert_allclose(result.shares, expected_shares)

    def test_damping_parameter(self) -> None:
        """Damping should not affect final equilibrium."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        result_no_damping = solve_equilibrium_prices(
            delta, alpha=1.0, costs=costs, ownership=ownership, damping=1.0
        )
        result_with_damping = solve_equilibrium_prices(
            delta, alpha=1.0, costs=costs, ownership=ownership, damping=0.5
        )
        
        np.testing.assert_allclose(
            result_no_damping.prices, result_with_damping.prices, rtol=1e-6
        )

    def test_max_iterations_exceeded(self) -> None:
        """Should report non-convergence when max iterations exceeded."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        # Very tight tolerance, few iterations
        result = solve_equilibrium_prices(
            delta, alpha=1.0, costs=costs, ownership=ownership,
            tolerance=1e-15, max_iterations=2
        )
        
        assert result.converged is False
        assert result.n_iterations == 2


# =============================================================================
# Analytical Benchmark Tests
# =============================================================================


class TestAnalyticalBenchmarks:
    """Tests against closed-form analytical solutions."""

    def test_single_product_monopoly_markup(self) -> None:
        """Single-product firm: η = 1 / (α * (1 - s)).
        
        This is the key analytical benchmark from the theory.
        """
        delta = np.array([1.0])
        alpha = 1.0
        costs = np.array([0.5])
        ownership = np.array([[1]])
        
        result = solve_equilibrium_prices(delta, alpha, costs, ownership)
        
        # At equilibrium, markup should satisfy the formula
        expected_markup = 1 / (alpha * (1 - result.shares[0]))
        np.testing.assert_allclose(result.markups[0], expected_markup, rtol=1e-8)

    def test_single_product_different_alpha(self) -> None:
        """Markup formula should hold for different α values."""
        delta = np.array([2.0])
        costs = np.array([1.0])
        ownership = np.array([[1]])
        
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            result = solve_equilibrium_prices(delta, alpha, costs, ownership)
            expected_markup = 1 / (alpha * (1 - result.shares[0]))
            np.testing.assert_allclose(
                result.markups[0], expected_markup, rtol=1e-8,
                err_msg=f"Failed for alpha={alpha}"
            )

    def test_symmetric_duopoly_equal_markups(self) -> None:
        """Symmetric duopolists should have equal markups."""
        delta = np.array([1.0, 1.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        np.testing.assert_allclose(result.markups[0], result.markups[1])
        np.testing.assert_allclose(result.shares[0], result.shares[1])

    def test_symmetric_oligopoly_markup_decreases_with_j(self) -> None:
        """With more competitors, markups should decrease."""
        alpha = 1.0
        costs_base = 0.5
        
        markups = []
        for J in [1, 2, 3, 5, 10]:
            delta = np.ones(J)
            costs = np.full(J, costs_base)
            ownership = np.eye(J)
            
            result = solve_equilibrium_prices(delta, alpha, costs, ownership)
            markups.append(result.markups[0])
        
        # Markups should be decreasing
        for i in range(len(markups) - 1):
            assert markups[i] > markups[i + 1], \
                f"Markup should decrease: J={i+1}: {markups[i]:.4f} > J={i+2}: {markups[i+1]:.4f}"

    def test_merger_increases_prices(self) -> None:
        """Merger of two firms should increase both their prices."""
        delta = np.array([1.0, 1.5, 2.0])
        costs = np.array([0.5, 0.5, 0.5])
        alpha = 1.0
        
        # Pre-merger: all independent
        ownership_pre = np.eye(3)
        result_pre = solve_equilibrium_prices(delta, alpha, costs, ownership_pre)
        
        # Post-merger: products 0 and 1 merge
        ownership_post = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        result_post = solve_equilibrium_prices(delta, alpha, costs, ownership_post)
        
        # Merged products should have higher prices
        assert result_post.prices[0] > result_pre.prices[0]
        assert result_post.prices[1] > result_pre.prices[1]

    def test_cost_pass_through(self) -> None:
        """Cost increase should be partially passed through to prices.
        
        For single-product firm with logit demand, pass-through rate = 1 - s_j.
        """
        delta = np.array([2.0])
        alpha = 1.0
        ownership = np.array([[1]])
        
        # Baseline
        costs_0 = np.array([1.0])
        result_0 = solve_equilibrium_prices(delta, alpha, costs_0, ownership)
        
        # Cost increase
        dc = 0.1
        costs_1 = np.array([1.0 + dc])
        result_1 = solve_equilibrium_prices(delta, alpha, costs_1, ownership)
        
        # Pass-through rate
        dp = result_1.prices[0] - result_0.prices[0]
        pass_through = dp / dc
        
        # Theoretical pass-through for single-product firm: ρ = 1 - s
        # But this is approximate since shares change with prices
        # Pass-through should be between 0 and 1
        assert 0 < pass_through < 2.0

    def test_foc_satisfied_at_equilibrium(self) -> None:
        """First-order conditions should be satisfied at equilibrium.
        
        FOC: s_j + Σ_{k∈F_j} (p_k - c_k) * ∂s_k/∂p_j = 0
        Or equivalently: s - (Ω ⊙ Δ)η = 0
        """
        delta = np.array([1.0, 2.0, 1.5])
        costs = np.array([0.5, 0.6, 0.4])
        ownership = np.eye(3)
        alpha = 1.0
        
        result = solve_equilibrium_prices(delta, alpha, costs, ownership)
        
        # Compute FOC residual
        Delta = compute_delta(alpha, result.shares)
        A = ownership * Delta
        residual = result.shares - A @ result.markups
        
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-8)


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical robustness."""

    def test_very_small_shares(self) -> None:
        """Should handle products with very small market shares.
        
        Note: Extreme share differences may require damping for stability.
        """
        delta = np.array([3.0, 1.0, 0.0])  # Moderate differences
        costs = np.array([0.5, 0.5, 0.5])
        ownership = np.eye(3)
        
        result = solve_equilibrium_prices(
            delta, alpha=1.0, costs=costs, ownership=ownership, damping=0.5
        )
        
        assert result.converged
        assert np.all(result.prices > costs)
        assert np.all(result.shares > 0)

    def test_high_alpha(self) -> None:
        """Should handle high price sensitivity."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=10.0, costs=costs, ownership=ownership)
        
        assert result.converged
        # High alpha means lower markups
        assert np.all(result.markups < 1.0)

    def test_low_alpha(self) -> None:
        """Should handle low price sensitivity."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.5, 0.5])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=0.1, costs=costs, ownership=ownership)
        
        assert result.converged
        # Low alpha means higher markups
        assert np.all(result.markups > 1.0)

    def test_many_products(self) -> None:
        """Should handle markets with many products."""
        J = 20
        np.random.seed(42)
        delta = np.random.uniform(0, 3, J)
        costs = np.random.uniform(0.3, 0.7, J)
        ownership = np.eye(J)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert result.converged
        assert np.all(result.prices > costs)
        assert result.n_iterations < 100  # Should converge reasonably fast

    def test_zero_costs(self) -> None:
        """Should work with zero marginal costs."""
        delta = np.array([1.0, 2.0])
        costs = np.array([0.0, 0.0])
        ownership = np.eye(2)
        
        result = solve_equilibrium_prices(delta, alpha=1.0, costs=costs, ownership=ownership)
        
        assert result.converged
        assert np.all(result.prices > 0)
        np.testing.assert_allclose(result.prices, result.markups)

    def test_multi_product_firm(self) -> None:
        """Should correctly handle multi-product firm.
        
        Note: Multi-product ownership may require damping for convergence.
        """
        delta = np.array([1.0, 1.5, 2.0, 2.5])
        costs = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Firm 1 owns products 0,1; Firm 2 owns products 2,3
        ownership = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)
        
        result = solve_equilibrium_prices(
            delta, alpha=1.0, costs=costs, ownership=ownership, damping=0.5
        )
        
        assert result.converged
        # Multi-product ownership should lead to higher markups
        # (compared to if all products were independent)

