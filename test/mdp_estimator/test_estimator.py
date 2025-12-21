"""Unit and integration tests for MDP Estimator.

Unit Tests (fast, no solver needed):
- Parameter validation (returns -inf for invalid params)
- Data structure checks

Integration Tests (slow, requires solver):
- Full estimation workflow
- Parameter recovery
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mdp_estimator import (
    EstimationResult,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)
from mdp_simulator import PanelData


# =============================================================================
# Fixtures for Unit Tests (no solver needed)
# =============================================================================

@pytest.fixture
def fake_panel_data():
    """Create fake panel data for unit tests (no solver needed)."""
    n_agents, n_periods = 10, 5
    return PanelData(
        states=np.random.uniform(0, 10, size=(n_agents, n_periods)),
        actions=np.random.randint(0, 2, size=(n_agents, n_periods)),
        rewards=np.random.randn(n_agents, n_periods),
        n_agents=n_agents,
        n_periods=n_periods,
    )


@pytest.fixture
def solver_params():
    """Fast solver hyperparameters for integration tests.
    
    Uses smaller network, fewer iterations, and looser tolerance
    to speed up tests while still verifying code correctness.
    """
    return {
        's_min': 0.0,
        's_max': 10.0,
        'hidden_sizes': [8],           # Small network (vs [16])
        'learning_rate': 0.1,
        'batch_size': 64,
        'tolerance': 0.1,              # Loose tolerance (vs 0.01)
        'max_iterations': 1000,        # Few iterations (vs 20000)
        'target_update_freq': 10,
    }


# =============================================================================
# Fixtures for Integration Tests (small data for speed)
# =============================================================================

@pytest.fixture
def true_params():
    """True structural parameters (from saved simulation config)."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'output', 'simulate_mdp', 'config.json'
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'delta': config['delta'],
    }


@pytest.fixture
def simulated_data():
    """Load small subset of saved panel data for fast integration tests.
    
    Uses only first 10 agents and 10 periods (100 observations)
    instead of full 100x100 (10,000 observations).
    """
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'output', 'simulate_mdp'
    )
    
    states = np.load(os.path.join(data_dir, 'states.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    
    # Use small subset for fast tests
    n_agents_test = 10
    n_periods_test = 10
    
    return PanelData(
        states=states[:n_agents_test, :n_periods_test],
        actions=actions[:n_agents_test, :n_periods_test],
        rewards=rewards[:n_agents_test, :n_periods_test],
        n_agents=n_agents_test,
        n_periods=n_periods_test,
    )


# =============================================================================
# UNIT TESTS (Fast - no solver needed)
# =============================================================================

class TestParameterValidation:
    """Unit tests for parameter validation (fast, no solver needed)."""
    
    def test_invalid_beta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([0.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_beta_negative_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta < 0 should return -inf immediately (no solver call)."""
        theta = np.array([-1.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 1.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 0.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 1.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf


class TestEstimationResultStructure:
    """Unit tests for EstimationResult data structure."""
    
    def test_estimation_result_fields(self):
        """EstimationResult should have all required fields."""
        result = EstimationResult(
            theta_hat=np.array([1.0, 0.1, 0.9]),
            std_errors=np.array([0.1, 0.01, 0.05]),
            cov_matrix=np.eye(3) * 0.01,
            log_likelihood=-100.0,
            n_iterations=50,
            converged=True,
            optimization_result=None,
        )
        
        assert result.theta_hat.shape == (3,)
        assert result.std_errors.shape == (3,)
        assert result.cov_matrix.shape == (3, 3)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)


# =============================================================================
# INTEGRATION TESTS (Slow - requires solver)
# Mark with pytest.mark.slow so they can be skipped
# =============================================================================

@pytest.mark.slow
class TestComputeLogLikelihood:
    """Integration tests for compute_log_likelihood (requires solver)."""
    
    def test_returns_finite_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should return a finite value for valid parameters."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert np.isfinite(ll), "Log-likelihood should be finite"
    
    def test_returns_negative_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should be negative (log of probabilities < 1)."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll < 0, "Log-likelihood should be negative"
    
    def test_true_params_higher_likelihood(self, simulated_data, true_params, solver_params):
        """True parameters should have higher likelihood than wrong parameters."""
        theta_true = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        theta_wrong = np.array([2.5, 0.3, 0.7])  # Significantly different
        
        ll_true = compute_log_likelihood(theta_true, simulated_data, solver_params)
        ll_wrong = compute_log_likelihood(theta_wrong, simulated_data, solver_params)
        
        assert ll_true > ll_wrong, \
            f"True params LL ({ll_true:.2f}) should be > wrong params LL ({ll_wrong:.2f})"


@pytest.mark.slow
class TestComputeStandardErrors:
    """Integration tests for compute_standard_errors (requires solver)."""
    
    def test_returns_positive_std_errors(self, simulated_data, true_params, solver_params):
        """Standard errors should be positive."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check std_errors are positive (or NaN if singular)
        if not np.any(np.isnan(std_errors)):
            assert np.all(std_errors > 0), "Standard errors should be positive"
    
    def test_cov_matrix_shape(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be 3x3."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        assert cov_matrix.shape == (3, 3), "Covariance matrix should be 3x3"
    
    def test_cov_matrix_symmetric(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be symmetric."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check symmetry (allowing for NaN)
        if not np.any(np.isnan(cov_matrix)):
            np.testing.assert_allclose(
                cov_matrix, cov_matrix.T, rtol=1e-5,
                err_msg="Covariance matrix should be symmetric"
            )


@pytest.mark.slow
class TestEstimateMLE:
    """Integration tests for estimate_mle (requires solver)."""
    
    def test_returns_estimation_result(self, simulated_data, true_params, solver_params):
        """estimate_mle should return EstimationResult object."""
        theta_init = (true_params['beta'], true_params['gamma'], true_params['delta'])
        
        # Run with very few iterations just to test interface
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=5, verbose=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert result.std_errors.shape == (3,), "std_errors should have 3 elements"
        assert result.cov_matrix.shape == (3, 3), "cov_matrix should be 3x3"
        assert isinstance(result.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(result.n_iterations, int), "n_iterations should be int"
        assert isinstance(result.converged, bool), "converged should be bool"


@pytest.mark.slow
class TestIntegration:
    """Integration tests for full estimation workflow (requires solver)."""
    
    def test_full_workflow_runs(self, simulated_data, true_params, solver_params):
        """Full estimation workflow should run without error."""
        # Start from perturbed initial values
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        # Run estimation with limited iterations
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        # Basic sanity checks
        assert np.all(np.isfinite(result.theta_hat)), "Estimates should be finite"
        assert np.isfinite(result.log_likelihood), "Log-likelihood should be finite"
        assert result.n_iterations > 0, "Should have run at least one iteration"
    
    def test_estimates_in_valid_range(self, simulated_data, true_params, solver_params):
        """Estimates should be in valid parameter range."""
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        beta_hat, gamma_hat, delta_hat = result.theta_hat
        
        assert beta_hat > 0, "beta should be positive"
        assert 0 < gamma_hat < 1, "gamma should be in (0, 1)"
        assert 0 < delta_hat < 1, "delta should be in (0, 1)"


@pytest.mark.slow
class TestGridSearch:
    """Integration tests for grid search estimation (requires solver)."""
    
    def test_grid_search_returns_result(self, simulated_data, true_params, solver_params):
        """grid_search_mle should return EstimationResult object."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert np.isfinite(result.log_likelihood), "log_likelihood should be finite"
        assert result.n_iterations == 8, "Should have 2^3 = 8 evaluations"
        assert result.converged is True, "Grid search always converges"
    
    def test_grid_search_finds_best_in_grid(self, simulated_data, true_params, solver_params):
        """Grid search should return the grid point with highest likelihood."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        # Verify theta_hat is in the grid
        grid_results = result.optimization_result
        beta_grid = grid_results['grid']['beta']
        gamma_grid = grid_results['grid']['gamma']
        delta_grid = grid_results['grid']['delta']
        
        assert result.theta_hat[0] in beta_grid, "beta should be in grid"
        assert result.theta_hat[1] in gamma_grid, "gamma should be in grid"
        assert result.theta_hat[2] in delta_grid, "delta should be in grid"
        
        # Verify it has the highest likelihood
        all_lls = [e['log_likelihood'] for e in grid_results['evaluations']]
        assert result.log_likelihood == max(all_lls), "Should return max likelihood point"
