"""Tests for OPM estimator: Two-step estimation (GMM + cost recovery).

Tests cover:
1. Berry inversion correctness
2. GMM estimation of alpha
3. Cost recovery from FOC
4. Shock recovery accuracy
5. Instrument construction
6. Full estimation pipeline
7. Edge cases
"""

import numpy as np
import pytest

from opm.estimator_opm import (
    EstimationResult,
    berry_inversion,
    compute_demand_shocks,
    gmm_objective,
    estimate_alpha_gmm,
    recover_costs,
    construct_cost_instruments,
    construct_blp_instruments,
    estimate_opm,
)
from opm.simulator_opm import simulate_opm_markets


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_data():
    """Simple test data with known properties."""
    return {
        "prices": np.array([[1.5, 1.5, 1.5], [1.6, 1.4, 1.5]]),
        "shares": np.array([[0.2, 0.2, 0.2], [0.18, 0.22, 0.2]]),
        "delta_bar": np.array([1.0, 1.0, 1.0]),
        "costs_bar": np.array([0.5, 0.5, 0.5]),
        "ownership": np.eye(3),
    }


@pytest.fixture
def simulated_data():
    """Generate simulated data for estimation tests."""
    result = simulate_opm_markets(
        delta_bar=np.array([1.0, 1.0, 1.0]),
        costs_bar=np.array([0.5, 0.5, 0.5]),
        alpha=1.0,
        ownership=np.eye(3),
        n_markets=500,
        sigma_xi=0.3,
        sigma_omega=0.15,
        seed=42,
    )
    return {
        "prices": result.prices,
        "shares": result.shares,
        "xi_true": result.xi,
        "omega_true": result.omega,
        "delta_bar": np.array([1.0, 1.0, 1.0]),
        "costs_bar": np.array([0.5, 0.5, 0.5]),
        "ownership": np.eye(3),
        "true_alpha": 1.0,
    }


# =============================================================================
# Tests for Berry Inversion
# =============================================================================


class TestBerryInversion:
    """Tests for Berry inversion function."""

    def test_output_shape(self, simple_data) -> None:
        """Berry inversion should return same shape as input."""
        delta = berry_inversion(shares=simple_data["shares"])
        assert delta.shape == simple_data["shares"].shape

    def test_symmetric_shares_equal_delta(self) -> None:
        """Symmetric shares should give equal mean utilities."""
        shares = np.array([[0.2, 0.2, 0.2]])
        delta = berry_inversion(shares=shares)
        np.testing.assert_allclose(delta[0, 0], delta[0, 1], rtol=1e-10)
        np.testing.assert_allclose(delta[0, 1], delta[0, 2], rtol=1e-10)

    def test_higher_share_higher_delta(self) -> None:
        """Higher share should correspond to higher mean utility."""
        shares = np.array([[0.1, 0.2, 0.3]])
        delta = berry_inversion(shares=shares)
        assert delta[0, 0] < delta[0, 1] < delta[0, 2]

    def test_inversion_formula(self) -> None:
        """Verify Berry inversion formula: δ = ln(s) - ln(s_0)."""
        shares = np.array([[0.15, 0.25, 0.20]])
        s0 = 1.0 - np.sum(shares, axis=1, keepdims=True)
        expected = np.log(shares) - np.log(s0)
        delta = berry_inversion(shares=shares)
        np.testing.assert_allclose(delta, expected, rtol=1e-10)

    def test_multiple_markets(self) -> None:
        """Should handle multiple markets correctly."""
        shares = np.array([
            [0.2, 0.2, 0.2],
            [0.1, 0.3, 0.2],
            [0.25, 0.15, 0.20],
        ])
        delta = berry_inversion(shares=shares)
        assert delta.shape == (3, 3)
        # Each market should have internally consistent deltas
        for m in range(3):
            s0 = 1.0 - np.sum(shares[m])
            expected = np.log(shares[m]) - np.log(s0)
            np.testing.assert_allclose(delta[m], expected, rtol=1e-10)


# =============================================================================
# Tests for Demand Shock Computation
# =============================================================================


class TestDemandShocks:
    """Tests for demand shock computation."""

    def test_output_shape(self, simple_data) -> None:
        """Should return same shape as input."""
        delta_obs = berry_inversion(shares=simple_data["shares"])
        xi = compute_demand_shocks(
            delta_obs=delta_obs,
            prices=simple_data["prices"],
            delta_bar=simple_data["delta_bar"],
            alpha=1.0,
        )
        assert xi.shape == simple_data["prices"].shape

    def test_formula(self) -> None:
        """Verify ξ = δ_obs - δ̄ + α*p."""
        delta_obs = np.array([[0.5, 0.6, 0.7]])
        prices = np.array([[1.0, 1.2, 0.8]])
        delta_bar = np.array([1.0, 1.0, 1.0])
        alpha = 1.5

        xi = compute_demand_shocks(
            delta_obs=delta_obs,
            prices=prices,
            delta_bar=delta_bar,
            alpha=alpha,
        )

        expected = delta_obs - delta_bar + alpha * prices
        np.testing.assert_allclose(xi, expected, rtol=1e-10)

    def test_zero_alpha(self) -> None:
        """With α=0, ξ = δ_obs - δ̄."""
        delta_obs = np.array([[0.5, 0.6, 0.7]])
        prices = np.array([[1.0, 1.2, 0.8]])
        delta_bar = np.array([0.4, 0.5, 0.6])

        xi = compute_demand_shocks(
            delta_obs=delta_obs,
            prices=prices,
            delta_bar=delta_bar,
            alpha=0.0,
        )

        expected = delta_obs - delta_bar
        np.testing.assert_allclose(xi, expected, rtol=1e-10)


# =============================================================================
# Tests for GMM Objective
# =============================================================================


class TestGMMObjective:
    """Tests for GMM objective function."""

    def test_nonnegative(self, simulated_data) -> None:
        """GMM objective should be non-negative."""
        delta_obs = berry_inversion(shares=simulated_data["shares"])
        instruments = construct_cost_instruments(simulated_data["omega_true"])

        for alpha in [0.5, 1.0, 1.5, 2.0]:
            Q = gmm_objective(
                alpha=alpha,
                delta_obs=delta_obs,
                prices=simulated_data["prices"],
                delta_bar=simulated_data["delta_bar"],
                instruments=instruments,
            )
            assert Q >= 0

    def test_minimum_near_true_alpha(self, simulated_data) -> None:
        """GMM objective should be minimized near true alpha."""
        delta_obs = berry_inversion(shares=simulated_data["shares"])
        instruments = construct_cost_instruments(simulated_data["omega_true"])

        Q_true = gmm_objective(
            alpha=1.0,
            delta_obs=delta_obs,
            prices=simulated_data["prices"],
            delta_bar=simulated_data["delta_bar"],
            instruments=instruments,
        )

        # Check that objective is higher away from true value
        for alpha in [0.5, 2.0]:
            Q_other = gmm_objective(
                alpha=alpha,
                delta_obs=delta_obs,
                prices=simulated_data["prices"],
                delta_bar=simulated_data["delta_bar"],
                instruments=instruments,
            )
            assert Q_other > Q_true


# =============================================================================
# Tests for Alpha Estimation
# =============================================================================


class TestAlphaEstimation:
    """Tests for GMM estimation of alpha."""

    def test_returns_tuple(self, simulated_data) -> None:
        """Should return (alpha_hat, alpha_se, gmm_value)."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_alpha_gmm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            instruments=instruments,
        )
        assert len(result) == 3

    def test_recovers_true_alpha(self, simulated_data) -> None:
        """Should recover true alpha within reasonable tolerance."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        alpha_hat, alpha_se, _ = estimate_alpha_gmm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            instruments=instruments,
        )

        true_alpha = simulated_data["true_alpha"]
        bias = abs(alpha_hat - true_alpha)

        # Should be within 10% of true value
        assert bias < 0.1 * true_alpha

    def test_se_positive(self, simulated_data) -> None:
        """Standard error should be positive."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        _, alpha_se, _ = estimate_alpha_gmm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            instruments=instruments,
        )

        assert alpha_se > 0 or np.isnan(alpha_se)

    def test_gmm_value_small(self, simulated_data) -> None:
        """GMM objective at optimum should be small."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        _, _, gmm_value = estimate_alpha_gmm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            instruments=instruments,
        )

        # Should be very small at optimum
        assert gmm_value < 0.01


# =============================================================================
# Tests for Cost Recovery
# =============================================================================


class TestCostRecovery:
    """Tests for cost recovery from FOC."""

    def test_output_shapes(self, simple_data) -> None:
        """Should return correct shapes."""
        costs, omega, markups = recover_costs(
            prices=simple_data["prices"],
            shares=simple_data["shares"],
            alpha=1.0,
            costs_bar=simple_data["costs_bar"],
            ownership=simple_data["ownership"],
        )

        n_markets, J = simple_data["prices"].shape
        assert costs.shape == (n_markets, J)
        assert omega.shape == (n_markets, J)
        assert markups.shape == (n_markets, J)

    def test_markups_positive(self, simulated_data) -> None:
        """Recovered markups should be positive."""
        costs, omega, markups = recover_costs(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            alpha=1.0,
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
        )

        assert np.all(markups > 0)

    def test_price_equals_cost_plus_markup(self, simulated_data) -> None:
        """Should satisfy p = c + η."""
        costs, omega, markups = recover_costs(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            alpha=1.0,
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
        )

        np.testing.assert_allclose(
            simulated_data["prices"],
            costs + markups,
            rtol=1e-8,
        )

    def test_markup_formula_single_product(self, simulated_data) -> None:
        """For single-product firms, η = 1/(α(1-s))."""
        alpha = 1.0
        costs, omega, markups = recover_costs(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            alpha=alpha,
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
        )

        expected_markups = 1.0 / (alpha * (1 - simulated_data["shares"]))
        np.testing.assert_allclose(markups, expected_markups, rtol=1e-8)

    def test_omega_recovery(self, simulated_data) -> None:
        """Recovered cost shocks should match true shocks."""
        costs, omega, markups = recover_costs(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            alpha=simulated_data["true_alpha"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
        )

        # Should be perfectly correlated with true omega
        corr = np.corrcoef(omega.flatten(), simulated_data["omega_true"].flatten())[0, 1]
        assert corr > 0.999


# =============================================================================
# Tests for Instrument Construction
# =============================================================================


class TestInstruments:
    """Tests for instrument construction."""

    def test_cost_instruments_shape(self) -> None:
        """Cost instruments should have correct shape."""
        omega = np.random.randn(100, 3)
        instruments = construct_cost_instruments(omega)
        assert instruments.shape == (300, 1)

    def test_blp_instruments_shape(self) -> None:
        """BLP instruments should have correct shape."""
        characteristics = np.array([1.0, 2.0, 3.0])
        instruments = construct_blp_instruments(characteristics)
        assert instruments.shape == (3, 1)

    def test_blp_instruments_formula(self) -> None:
        """BLP instruments should be sum of others' characteristics."""
        characteristics = np.array([1.0, 2.0, 3.0])
        instruments = construct_blp_instruments(characteristics)

        # Z_j = Σ_{k≠j} x_k
        expected = np.array([5.0, 4.0, 3.0])  # [2+3, 1+3, 1+2]
        np.testing.assert_allclose(instruments.flatten(), expected)


# =============================================================================
# Tests for Full Estimation Pipeline
# =============================================================================


class TestFullEstimation:
    """Tests for complete estimation pipeline."""

    def test_returns_estimation_result(self, simulated_data) -> None:
        """Should return EstimationResult object."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_opm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
            instruments=instruments,
        )
        assert isinstance(result, EstimationResult)

    def test_alpha_recovery(self, simulated_data) -> None:
        """Should recover true alpha accurately."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_opm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
            instruments=instruments,
        )

        bias = abs(result.alpha_hat - simulated_data["true_alpha"])
        assert bias < 0.1  # Within 10%

    def test_xi_recovery(self, simulated_data) -> None:
        """Should recover demand shocks accurately."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_opm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
            instruments=instruments,
        )

        corr = np.corrcoef(
            result.xi_hat.flatten(),
            simulated_data["xi_true"].flatten(),
        )[0, 1]
        assert corr > 0.99

    def test_omega_recovery(self, simulated_data) -> None:
        """Should recover cost shocks accurately."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_opm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
            instruments=instruments,
        )

        corr = np.corrcoef(
            result.omega_hat.flatten(),
            simulated_data["omega_true"].flatten(),
        )[0, 1]
        assert corr > 0.99

    def test_output_shapes(self, simulated_data) -> None:
        """All outputs should have correct shapes."""
        instruments = construct_cost_instruments(simulated_data["omega_true"])
        result = estimate_opm(
            prices=simulated_data["prices"],
            shares=simulated_data["shares"],
            delta_bar=simulated_data["delta_bar"],
            costs_bar=simulated_data["costs_bar"],
            ownership=simulated_data["ownership"],
            instruments=instruments,
        )

        n_markets, J = simulated_data["prices"].shape
        assert result.xi_hat.shape == (n_markets, J)
        assert result.omega_hat.shape == (n_markets, J)
        assert result.costs_hat.shape == (n_markets, J)
        assert result.markups_hat.shape == (n_markets, J)
        assert result.n_markets == n_markets
        assert result.n_products == J


# =============================================================================
# Tests for Different Scenarios
# =============================================================================


class TestDifferentScenarios:
    """Tests for estimation across different scenarios."""

    def test_quality_differentiation(self) -> None:
        """Should work with quality differentiation."""
        result = simulate_opm_markets(
            delta_bar=np.array([0.5, 1.0, 2.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            alpha=1.0,
            ownership=np.eye(3),
            n_markets=500,
            sigma_xi=0.3,
            sigma_omega=0.15,
            seed=123,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([0.5, 1.0, 2.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            ownership=np.eye(3),
            instruments=instruments,
        )

        assert abs(est_result.alpha_hat - 1.0) < 0.15

    def test_cost_differentiation(self) -> None:
        """Should work with cost differentiation."""
        result = simulate_opm_markets(
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.3, 0.5, 0.7]),
            alpha=1.0,
            ownership=np.eye(3),
            n_markets=500,
            sigma_xi=0.3,
            sigma_omega=0.15,
            seed=456,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.3, 0.5, 0.7]),
            ownership=np.eye(3),
            instruments=instruments,
        )

        # Allow 20% tolerance for cost differentiation scenario
        assert abs(est_result.alpha_hat - 1.0) < 0.20

    def test_different_alpha(self) -> None:
        """Should recover different alpha values."""
        true_alpha = 2.0
        result = simulate_opm_markets(
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            alpha=true_alpha,
            ownership=np.eye(3),
            n_markets=500,
            sigma_xi=0.3,
            sigma_omega=0.15,
            seed=789,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            ownership=np.eye(3),
            instruments=instruments,
            alpha_bounds=(0.5, 5.0),
        )

        assert abs(est_result.alpha_hat - true_alpha) < 0.2


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_market(self) -> None:
        """Should handle single market."""
        result = simulate_opm_markets(
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            alpha=1.0,
            ownership=np.eye(3),
            n_markets=1,
            sigma_xi=0.1,
            sigma_omega=0.05,
            seed=42,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            ownership=np.eye(3),
            instruments=instruments,
        )

        # Should complete without error
        assert est_result.n_markets == 1

    def test_many_markets(self) -> None:
        """Should handle many markets."""
        result = simulate_opm_markets(
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            alpha=1.0,
            ownership=np.eye(3),
            n_markets=2000,
            sigma_xi=0.3,
            sigma_omega=0.15,
            seed=42,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([1.0, 1.0, 1.0]),
            costs_bar=np.array([0.5, 0.5, 0.5]),
            ownership=np.eye(3),
            instruments=instruments,
        )

        # More data should give better estimates
        assert abs(est_result.alpha_hat - 1.0) < 0.05

    def test_single_product(self) -> None:
        """Should handle single product."""
        result = simulate_opm_markets(
            delta_bar=np.array([1.0]),
            costs_bar=np.array([0.5]),
            alpha=1.0,
            ownership=np.eye(1),
            n_markets=500,
            sigma_xi=0.3,
            sigma_omega=0.15,
            seed=42,
        )

        instruments = construct_cost_instruments(result.omega)
        est_result = estimate_opm(
            prices=result.prices,
            shares=result.shares,
            delta_bar=np.array([1.0]),
            costs_bar=np.array([0.5]),
            ownership=np.eye(1),
            instruments=instruments,
        )

        assert est_result.n_products == 1

