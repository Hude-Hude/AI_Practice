"""Tests for value iteration algorithm."""

import numpy as np
import pytest

from mdp_solver.value_iteration import (
    check_convergence,
    compute_choice_probability,
    compute_integrated_value,
    initialize_value_function,
    interpolate_value,
    solve_value_function,
)


class TestInitializeValueFunction:
    """Tests for initialize_value_function."""

    def test_returns_zeros(self) -> None:
        """Should return array of zeros."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        V = initialize_value_function(state_grid=state_grid)
        np.testing.assert_array_equal(V, np.zeros(50))

    def test_correct_length(self) -> None:
        """Should match length of state grid."""
        state_grid = np.linspace(start=0.0, stop=5.0, num=100)
        V = initialize_value_function(state_grid=state_grid)
        assert len(V) == 100


class TestInterpolateValue:
    """Tests for interpolate_value."""

    def test_on_grid_points(self) -> None:
        """Interpolation at grid points should return exact values."""
        state_grid = np.array([0.0, 1.0, 2.0, 3.0])
        V = np.array([0.0, 1.0, 4.0, 9.0])
        result = interpolate_value(V=V, state_grid=state_grid, query_states=state_grid)
        np.testing.assert_array_almost_equal(result, V)

    def test_midpoint_interpolation(self) -> None:
        """Should linearly interpolate between grid points."""
        state_grid = np.array([0.0, 2.0])
        V = np.array([0.0, 2.0])
        query = np.array([1.0])
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        np.testing.assert_array_almost_equal(result, np.array([1.0]))


class TestComputeIntegratedValue:
    """Tests for compute_integrated_value."""

    def test_logsumexp_formula(self) -> None:
        """Should compute log(exp(v0) + exp(v1))."""
        v0 = np.array([0.0, 1.0])
        v1 = np.array([0.0, 1.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        expected = np.log(2 * np.exp(v0))  # Since v0 == v1
        np.testing.assert_array_almost_equal(result, expected)

    def test_dominance(self) -> None:
        """When one value dominates, result should be close to max."""
        v0 = np.array([0.0])
        v1 = np.array([100.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        assert result[0] > 99.0  # Should be very close to 100


class TestComputeChoiceProbability:
    """Tests for compute_choice_probability."""

    def test_equal_values_half_probability(self) -> None:
        """Equal values should give 50% probability."""
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = np.array([1.0, 2.0, 3.0])
        prob = compute_choice_probability(v0=v0, v1=v1)
        np.testing.assert_array_almost_equal(prob, np.array([0.5, 0.5, 0.5]))

    def test_probability_bounds(self) -> None:
        """Probabilities should be in [0, 1]."""
        v0 = np.array([-10.0, 0.0, 10.0])
        v1 = np.array([10.0, 0.0, -10.0])
        prob = compute_choice_probability(v0=v0, v1=v1)
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)

    def test_higher_v1_higher_probability(self) -> None:
        """Higher v1 should give higher probability of action 1."""
        v0 = np.array([0.0])
        v1 = np.array([1.0])
        prob = compute_choice_probability(v0=v0, v1=v1)
        assert prob[0] > 0.5


class TestCheckConvergence:
    """Tests for check_convergence."""

    def test_identical_converged(self) -> None:
        """Identical arrays should be converged."""
        V = np.array([1.0, 2.0, 3.0])
        assert check_convergence(V_old=V, V_new=V, tolerance=1e-10)

    def test_large_diff_not_converged(self) -> None:
        """Large differences should not be converged."""
        V_old = np.array([1.0, 2.0, 3.0])
        V_new = np.array([1.0, 2.0, 4.0])
        assert not check_convergence(V_old=V_old, V_new=V_new, tolerance=0.5)


class TestSolveValueFunction:
    """Tests for solve_value_function."""

    def test_converges(self) -> None:
        """Should converge within max iterations."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        V, v0, v1, n_iter = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.9,
            tolerance=1e-6,
            max_iterations=1000,
        )
        assert n_iter < 1000

    def test_value_monotonicity(self) -> None:
        """Value function should be monotonically increasing."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        V, v0, v1, _ = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.9,
        )
        # V should be increasing
        assert np.all(np.diff(V) >= -1e-6)

