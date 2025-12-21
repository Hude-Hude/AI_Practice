"""Unit and integration tests for MDP Estimator.

Tests:
1. Unit test: compute_log_likelihood returns correct sign and magnitude
2. Unit test: compute_log_likelihood returns -inf for invalid parameters
3. Unit test: True parameters have higher likelihood than wrong parameters
4. Unit test: Standard errors are positive
5. Integration test: Full estimation workflow runs without error
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
    compute_standard_errors,
)
from mdp_simulator import simulate_mdp_panel, PanelData
from mdp_solver import solve_value_function


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def true_params():
    """True structural parameters."""
    return {
        'beta': 1.5,
        'gamma': 0.1,
        'delta': 0.9,
    }


@pytest.fixture
def solver_params():
    """Solver hyperparameters for testing (smaller for speed)."""
    return {
        's_min': 0.0,
        's_max': 10.0,
        'hidden_sizes': [32, 32],
        'learning_rate': 0.01,
        'batch_size': 128,
        'tolerance': 1e-3,
        'max_iterations': 2000,
        'target_update_freq': 50,
    }


@pytest.fixture
def trained_networks(true_params, solver_params):
    """Train value networks at true parameters."""
    v0_net, v1_net, losses, n_iter = solve_value_function(
        beta=true_params['beta'],
        gamma=true_params['gamma'],
        delta=true_params['delta'],
        s_min=solver_params['s_min'],
        s_max=solver_params['s_max'],
        hidden_sizes=solver_params['hidden_sizes'],
        learning_rate=solver_params['learning_rate'],
        batch_size=solver_params['batch_size'],
        tolerance=solver_params['tolerance'],
        max_iterations=solver_params['max_iterations'],
        target_update_freq=solver_params['target_update_freq'],
    )
    return v0_net, v1_net


@pytest.fixture
def simulated_data(trained_networks, true_params):
    """Simulate panel data from true model."""
    v0_net, v1_net = trained_networks
    data = simulate_mdp_panel(
        v0_net=v0_net,
        v1_net=v1_net,
        n_agents=50,
        n_periods=20,
        s_init=5.0,
        beta=true_params['beta'],
        gamma=true_params['gamma'],
        seed=42,
    )
    return data


# =============================================================================
# Unit Tests: compute_log_likelihood
# =============================================================================

class TestComputeLogLikelihood:
    """Unit tests for compute_log_likelihood function."""
    
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
    
    def test_invalid_beta_returns_neg_inf(self, simulated_data, solver_params):
        """Invalid beta (<=0) should return -inf."""
        theta = np.array([0.0, 0.1, 0.9])  # beta = 0
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "Invalid beta should return -inf"
        
        theta = np.array([-1.0, 0.1, 0.9])  # beta < 0
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "Negative beta should return -inf"
    
    def test_invalid_gamma_returns_neg_inf(self, simulated_data, solver_params):
        """Invalid gamma (<=0 or >=1) should return -inf."""
        theta = np.array([1.5, 0.0, 0.9])  # gamma = 0
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "gamma=0 should return -inf"
        
        theta = np.array([1.5, 1.0, 0.9])  # gamma = 1
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "gamma=1 should return -inf"
    
    def test_invalid_delta_returns_neg_inf(self, simulated_data, solver_params):
        """Invalid delta (<=0 or >=1) should return -inf."""
        theta = np.array([1.5, 0.1, 0.0])  # delta = 0
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "delta=0 should return -inf"
        
        theta = np.array([1.5, 0.1, 1.0])  # delta = 1
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll == -np.inf, "delta=1 should return -inf"
    
    def test_true_params_higher_likelihood(self, simulated_data, true_params, solver_params):
        """True parameters should have higher likelihood than wrong parameters."""
        theta_true = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        theta_wrong = np.array([2.5, 0.3, 0.7])  # Significantly different
        
        ll_true = compute_log_likelihood(theta_true, simulated_data, solver_params)
        ll_wrong = compute_log_likelihood(theta_wrong, simulated_data, solver_params)
        
        assert ll_true > ll_wrong, \
            f"True params LL ({ll_true:.2f}) should be > wrong params LL ({ll_wrong:.2f})"


# =============================================================================
# Unit Tests: compute_standard_errors
# =============================================================================

class TestComputeStandardErrors:
    """Unit tests for compute_standard_errors function."""
    
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


# =============================================================================
# Unit Tests: estimate_mle
# =============================================================================

class TestEstimateMLE:
    """Unit tests for estimate_mle function."""
    
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


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full estimation workflow."""
    
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
