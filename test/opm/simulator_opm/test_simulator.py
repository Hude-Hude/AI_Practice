"""Tests for OPM simulator: Monte Carlo simulation with demand/cost shocks.

Tests cover:
1. Output shapes and types
2. Convergence across markets
3. FOC residuals at equilibrium
4. Reproducibility with seed
5. Economic sanity checks
6. Shock effects on variation
"""

import numpy as np
import pytest

from opm.simulator_opm import (
    SimulationResult,
    simulate_opm_markets,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def baseline_params():
    """Baseline simulation parameters."""
    return {
        "delta_bar": np.array([1.0, 1.0, 1.0]),
        "costs_bar": np.array([0.5, 0.5, 0.5]),
        "alpha": 1.0,
        "ownership": np.eye(3),
        "n_markets": 100,
        "sigma_xi": 0.5,
        "sigma_omega": 0.2,
        "seed": 42,
    }


# =============================================================================
# Tests for Output Structure
# =============================================================================


class TestOutputStructure:
    """Tests for simulation output shapes and types."""

    def test_returns_simulation_result(self, baseline_params) -> None:
        """Should return SimulationResult object."""
        result = simulate_opm_markets(**baseline_params)
        assert isinstance(result, SimulationResult)

    def test_prices_shape(self, baseline_params) -> None:
        """Prices should have shape (n_markets, J)."""
        result = simulate_opm_markets(**baseline_params)
        assert result.prices.shape == (100, 3)

    def test_shares_shape(self, baseline_params) -> None:
        """Shares should have shape (n_markets, J)."""
        result = simulate_opm_markets(**baseline_params)
        assert result.shares.shape == (100, 3)

    def test_markups_shape(self, baseline_params) -> None:
        """Markups should have shape (n_markets, J)."""
        result = simulate_opm_markets(**baseline_params)
        assert result.markups.shape == (100, 3)

    def test_converged_shape(self, baseline_params) -> None:
        """Converged should have shape (n_markets,)."""
        result = simulate_opm_markets(**baseline_params)
        assert result.converged.shape == (100,)

    def test_foc_errors_shape(self, baseline_params) -> None:
        """FOC errors should have shape (n_markets,)."""
        result = simulate_opm_markets(**baseline_params)
        assert result.foc_errors.shape == (100,)

    def test_n_markets_attribute(self, baseline_params) -> None:
        """Should store n_markets correctly."""
        result = simulate_opm_markets(**baseline_params)
        assert result.n_markets == 100

    def test_n_products_attribute(self, baseline_params) -> None:
        """Should store n_products correctly."""
        result = simulate_opm_markets(**baseline_params)
        assert result.n_products == 3


# =============================================================================
# Tests for Convergence
# =============================================================================


class TestConvergence:
    """Tests for equilibrium convergence across markets."""

    def test_all_markets_converge(self, baseline_params) -> None:
        """All markets should converge with reasonable parameters."""
        result = simulate_opm_markets(**baseline_params)
        assert np.all(result.converged)

    def test_convergence_rate_100_percent(self, baseline_params) -> None:
        """Convergence rate should be 100%."""
        result = simulate_opm_markets(**baseline_params)
        convergence_rate = np.mean(result.converged)
        assert convergence_rate == 1.0

    def test_convergence_with_large_shocks(self, baseline_params) -> None:
        """Should still converge with larger shocks (may need damping)."""
        params = baseline_params.copy()
        params["sigma_xi"] = 1.0
        params["sigma_omega"] = 0.5
        params["damping"] = 0.5  # Add damping for stability
        
        result = simulate_opm_markets(**params)
        convergence_rate = np.mean(result.converged)
        assert convergence_rate >= 0.95  # Allow small failure rate


# =============================================================================
# Tests for FOC Residuals
# =============================================================================


class TestFOCResiduals:
    """Tests for first-order condition satisfaction."""

    def test_foc_errors_small(self, baseline_params) -> None:
        """FOC errors should be very small for converged markets."""
        result = simulate_opm_markets(**baseline_params)
        assert np.max(result.foc_errors) < 1e-8

    def test_foc_errors_nonnegative(self, baseline_params) -> None:
        """FOC errors should be non-negative (they are norms)."""
        result = simulate_opm_markets(**baseline_params)
        assert np.all(result.foc_errors >= 0)


# =============================================================================
# Tests for Reproducibility
# =============================================================================


class TestReproducibility:
    """Tests for simulation reproducibility with seeds."""

    def test_same_seed_same_results(self, baseline_params) -> None:
        """Same seed should produce identical results."""
        result1 = simulate_opm_markets(**baseline_params)
        result2 = simulate_opm_markets(**baseline_params)
        
        np.testing.assert_array_equal(result1.prices, result2.prices)
        np.testing.assert_array_equal(result1.shares, result2.shares)
        np.testing.assert_array_equal(result1.markups, result2.markups)

    def test_different_seed_different_results(self, baseline_params) -> None:
        """Different seeds should produce different results."""
        params1 = baseline_params.copy()
        params2 = baseline_params.copy()
        params2["seed"] = 123
        
        result1 = simulate_opm_markets(**params1)
        result2 = simulate_opm_markets(**params2)
        
        assert not np.allclose(result1.prices, result2.prices)

    def test_no_seed_different_results(self, baseline_params) -> None:
        """No seed should produce different results on repeated calls."""
        params = baseline_params.copy()
        params["seed"] = None
        params["n_markets"] = 10  # Smaller for speed
        
        result1 = simulate_opm_markets(**params)
        result2 = simulate_opm_markets(**params)
        
        # Very unlikely to be identical without seed
        assert not np.allclose(result1.prices, result2.prices)


# =============================================================================
# Tests for Economic Sanity Checks
# =============================================================================


class TestEconomicSanity:
    """Tests for economic sanity of simulation outputs."""

    def test_prices_positive(self, baseline_params) -> None:
        """All prices should be positive."""
        result = simulate_opm_markets(**baseline_params)
        assert np.all(result.prices > 0)

    def test_markups_positive(self, baseline_params) -> None:
        """All markups should be positive (firms make profit)."""
        result = simulate_opm_markets(**baseline_params)
        assert np.all(result.markups > 0)

    def test_shares_positive(self, baseline_params) -> None:
        """All shares should be positive."""
        result = simulate_opm_markets(**baseline_params)
        assert np.all(result.shares > 0)

    def test_shares_sum_less_than_one(self, baseline_params) -> None:
        """Shares should sum to less than 1 (outside option exists)."""
        result = simulate_opm_markets(**baseline_params)
        total_shares = np.sum(result.shares, axis=1)
        assert np.all(total_shares < 1)

    def test_prices_equal_costs_plus_markups(self, baseline_params) -> None:
        """Prices should equal costs + markups for each market."""
        result = simulate_opm_markets(**baseline_params)
        
        # Reconstruct costs for each market
        np.random.seed(baseline_params["seed"])
        for m in range(result.n_markets):
            # Skip demand shocks (they don't affect this relationship)
            _ = np.random.normal(size=3)
            omega_m = np.random.normal(
                loc=0,
                scale=baseline_params["sigma_omega"],
                size=3,
            )
            costs_m = baseline_params["costs_bar"] + omega_m
            
            expected_prices = costs_m + result.markups[m]
            np.testing.assert_allclose(
                result.prices[m], expected_prices, rtol=1e-10
            )


# =============================================================================
# Tests for Shock Effects
# =============================================================================


class TestShockEffects:
    """Tests for effects of demand and cost shocks."""

    def test_zero_shocks_identical_markets(self, baseline_params) -> None:
        """Zero shock variance should produce identical markets."""
        params = baseline_params.copy()
        params["sigma_xi"] = 0.0
        params["sigma_omega"] = 0.0
        
        result = simulate_opm_markets(**params)
        
        # All markets should have identical prices
        for m in range(1, result.n_markets):
            np.testing.assert_allclose(
                result.prices[m], result.prices[0], rtol=1e-10
            )

    def test_higher_sigma_xi_more_price_variation(self, baseline_params) -> None:
        """Higher demand shock variance should increase price variation."""
        params_low = baseline_params.copy()
        params_low["sigma_xi"] = 0.1
        params_low["sigma_omega"] = 0.0
        
        params_high = baseline_params.copy()
        params_high["sigma_xi"] = 1.0
        params_high["sigma_omega"] = 0.0
        
        result_low = simulate_opm_markets(**params_low)
        result_high = simulate_opm_markets(**params_high)
        
        std_low = np.std(result_low.prices)
        std_high = np.std(result_high.prices)
        
        assert std_high > std_low

    def test_higher_sigma_omega_more_cost_variation(self, baseline_params) -> None:
        """Higher cost shock variance should increase markup variation."""
        params_low = baseline_params.copy()
        params_low["sigma_xi"] = 0.0
        params_low["sigma_omega"] = 0.1
        
        params_high = baseline_params.copy()
        params_high["sigma_xi"] = 0.0
        params_high["sigma_omega"] = 0.5
        
        result_low = simulate_opm_markets(**params_low)
        result_high = simulate_opm_markets(**params_high)
        
        std_low = np.std(result_low.prices)
        std_high = np.std(result_high.prices)
        
        assert std_high > std_low


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_single_market(self, baseline_params) -> None:
        """Should work with single market."""
        params = baseline_params.copy()
        params["n_markets"] = 1
        
        result = simulate_opm_markets(**params)
        
        assert result.prices.shape == (1, 3)
        assert result.converged[0] == True

    def test_single_product(self) -> None:
        """Should work with single product."""
        result = simulate_opm_markets(
            delta_bar=np.array([1.0]),
            costs_bar=np.array([0.5]),
            alpha=1.0,
            ownership=np.array([[1.0]]),
            n_markets=10,
            sigma_xi=0.5,
            sigma_omega=0.2,
            seed=42,
        )
        
        assert result.prices.shape == (10, 1)
        assert np.all(result.converged)

    def test_many_products(self) -> None:
        """Should work with many products."""
        J = 10
        result = simulate_opm_markets(
            delta_bar=np.ones(J),
            costs_bar=np.ones(J) * 0.5,
            alpha=1.0,
            ownership=np.eye(J),
            n_markets=20,
            sigma_xi=0.3,
            sigma_omega=0.1,
            seed=42,
        )
        
        assert result.prices.shape == (20, 10)
        assert np.all(result.converged)

    def test_asymmetric_baseline(self) -> None:
        """Should work with asymmetric baseline parameters."""
        result = simulate_opm_markets(
            delta_bar=np.array([0.5, 1.0, 2.0]),
            costs_bar=np.array([0.3, 0.5, 0.7]),
            alpha=1.5,
            ownership=np.eye(3),
            n_markets=50,
            sigma_xi=0.3,
            sigma_omega=0.1,
            seed=42,
        )
        
        assert np.all(result.converged)
        assert np.all(result.foc_errors < 1e-8)

