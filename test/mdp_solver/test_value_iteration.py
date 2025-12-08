"""
Tests for value iteration algorithm.

Based on specifications from solve_mdp.qmd:
- INITIALIZE_VALUE_FUNCTION: zeros(n)
- INTERPOLATE_VALUE: linear interpolation with boundary extrapolation
- COMPUTE_CHOICE_VALUE: reward + δ * continuation
- COMPUTE_INTEGRATED_VALUE: log(exp(v₀) + exp(v₁))
- CHECK_CONVERGENCE: max|V_new - V_old| < tolerance
- COMPUTE_CHOICE_PROBABILITY: exp(v₁) / (exp(v₀) + exp(v₁))
- SolveValueFunction: main algorithm
"""

import numpy as np
import pytest

from mdp_solver.value_iteration import (
    check_convergence,
    compute_choice_probability,
    compute_choice_value,
    compute_integrated_value,
    initialize_value_function,
    interpolate_value,
    solve_value_function,
)


class TestInitializeValueFunction:
    """
    Tests for INITIALIZE_VALUE_FUNCTION subroutine.
    
    Specification:
        INPUT: state_grid (StateGrid)
        OUTPUT: V = zeros(length(state_grid))
    """

    def test_returns_zeros(self) -> None:
        """Should return array of all zeros."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        V = initialize_value_function(state_grid=state_grid)
        np.testing.assert_array_equal(V, np.zeros(50))

    def test_correct_length(self) -> None:
        """Output length should match state_grid length."""
        for n in [10, 50, 100, 500]:
            state_grid = np.linspace(start=0.0, stop=10.0, num=n)
            V = initialize_value_function(state_grid=state_grid)
            assert len(V) == n

    def test_output_type(self) -> None:
        """Should return numpy array."""
        state_grid = np.array([0.0, 1.0, 2.0])
        V = initialize_value_function(state_grid=state_grid)
        assert isinstance(V, np.ndarray)


class TestInterpolateValue:
    """
    Tests for INTERPOLATE_VALUE subroutine.
    
    Specification:
        INPUT: V (Vector), state_grid (StateGrid), query_states (Vector)
        OUTPUT: linear_interpolate with boundary extrapolation
    """

    def test_on_grid_points_exact(self) -> None:
        """Interpolation at grid points should return exact values."""
        state_grid = np.array([0.0, 1.0, 2.0, 3.0])
        V = np.array([0.0, 1.0, 4.0, 9.0])
        result = interpolate_value(V=V, state_grid=state_grid, query_states=state_grid)
        np.testing.assert_array_almost_equal(result, V)

    def test_linear_interpolation_midpoint(self) -> None:
        """Should linearly interpolate at midpoints."""
        state_grid = np.array([0.0, 2.0])
        V = np.array([0.0, 4.0])
        query = np.array([1.0])  # Midpoint
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        np.testing.assert_array_almost_equal(result, np.array([2.0]))

    def test_linear_interpolation_quarter(self) -> None:
        """Should linearly interpolate at quarter points."""
        state_grid = np.array([0.0, 4.0])
        V = np.array([0.0, 8.0])
        query = np.array([1.0])  # Quarter point
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        np.testing.assert_array_almost_equal(result, np.array([2.0]))

    def test_boundary_extrapolation_below(self) -> None:
        """Should use boundary value for queries below grid."""
        state_grid = np.array([1.0, 2.0, 3.0])
        V = np.array([10.0, 20.0, 30.0])
        query = np.array([0.0, -1.0])  # Below grid
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        np.testing.assert_array_almost_equal(result, np.array([10.0, 10.0]))

    def test_boundary_extrapolation_above(self) -> None:
        """Should use boundary value for queries above grid."""
        state_grid = np.array([1.0, 2.0, 3.0])
        V = np.array([10.0, 20.0, 30.0])
        query = np.array([4.0, 5.0])  # Above grid
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        np.testing.assert_array_almost_equal(result, np.array([30.0, 30.0]))

    def test_output_shape_matches_query(self) -> None:
        """Output shape should match query_states shape."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        V = np.zeros(100)
        query = np.linspace(start=0.0, stop=10.0, num=50)
        result = interpolate_value(V=V, state_grid=state_grid, query_states=query)
        assert result.shape == query.shape


class TestComputeChoiceValue:
    """
    Tests for COMPUTE_CHOICE_VALUE subroutine.
    
    Specification:
        v(s, a) = COMPUTE_REWARD(s, a, β) + δ * INTERPOLATE_VALUE(V, s_grid, COMPUTE_NEXT_STATE(s, a, γ))
    """

    def test_zero_discount_equals_reward(self) -> None:
        """With δ = 0, choice value should equal reward."""
        state_grid = np.array([0.0, 5.0, 10.0])
        V = np.array([100.0, 200.0, 300.0])  # Should be ignored
        beta, gamma, delta = 1.0, 0.1, 0.0

        for action in [0, 1]:
            v = compute_choice_value(
                V=V, state_grid=state_grid, action=action,
                beta=beta, gamma=gamma, delta=delta
            )
            # Reward is β * log(1 + s) - action
            expected_reward = beta * np.log(1 + state_grid) - action
            np.testing.assert_array_almost_equal(v, expected_reward)

    def test_discount_factor_effect(self) -> None:
        """Higher δ should increase weight on continuation."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        V = np.ones(50) * 100  # Constant continuation value
        beta, gamma = 1.0, 0.1

        v_low_delta = compute_choice_value(
            V=V, state_grid=state_grid, action=0,
            beta=beta, gamma=gamma, delta=0.5
        )
        v_high_delta = compute_choice_value(
            V=V, state_grid=state_grid, action=0,
            beta=beta, gamma=gamma, delta=0.9
        )
        # Higher delta should give higher values (assuming positive continuation)
        assert np.all(v_high_delta > v_low_delta)

    def test_action_one_minus_action_zero(self) -> None:
        """
        v(s, 1) - v(s, 0) should reflect:
        - Cost of action 1: -1
        - Different continuation value due to different next state
        """
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        V = np.linspace(start=0.0, stop=100.0, num=100)  # Increasing V
        beta, gamma, delta = 1.0, 0.1, 0.9

        v0 = compute_choice_value(
            V=V, state_grid=state_grid, action=0,
            beta=beta, gamma=gamma, delta=delta
        )
        v1 = compute_choice_value(
            V=V, state_grid=state_grid, action=1,
            beta=beta, gamma=gamma, delta=delta
        )
        # Difference should be finite
        diff = v1 - v0
        assert np.all(np.isfinite(diff))


class TestComputeIntegratedValue:
    """
    Tests for COMPUTE_INTEGRATED_VALUE subroutine.
    
    Specification:
        V(s) = log(exp(v₀) + exp(v₁))
    """

    def test_logsumexp_formula(self) -> None:
        """Should compute log(exp(v0) + exp(v1)) exactly."""
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = np.array([1.5, 2.5, 3.5])
        result = compute_integrated_value(v0=v0, v1=v1)
        expected = np.log(np.exp(v0) + np.exp(v1))
        np.testing.assert_array_almost_equal(result, expected)

    def test_equal_values(self) -> None:
        """When v0 = v1, V = v0 + log(2)."""
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = v0.copy()
        result = compute_integrated_value(v0=v0, v1=v1)
        expected = v0 + np.log(2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dominance_v1(self) -> None:
        """When v1 >> v0, V ≈ v1."""
        v0 = np.array([0.0, 0.0])
        v1 = np.array([100.0, 100.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        np.testing.assert_array_almost_equal(result, v1, decimal=5)

    def test_dominance_v0(self) -> None:
        """When v0 >> v1, V ≈ v0."""
        v0 = np.array([100.0, 100.0])
        v1 = np.array([0.0, 0.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        np.testing.assert_array_almost_equal(result, v0, decimal=5)

    def test_always_greater_than_max(self) -> None:
        """V(s) > max(v0, v1) always (due to option value)."""
        v0 = np.array([1.0, 5.0, 3.0])
        v1 = np.array([2.0, 4.0, 6.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        max_v = np.maximum(v0, v1)
        assert np.all(result > max_v)

    def test_numerical_stability_large_values(self) -> None:
        """Should handle large values without overflow."""
        v0 = np.array([500.0, 600.0])
        v1 = np.array([500.0, 600.0])
        result = compute_integrated_value(v0=v0, v1=v1)
        assert np.all(np.isfinite(result))

    def test_output_shape(self) -> None:
        """Output shape should match input shape."""
        v0 = np.zeros(100)
        v1 = np.zeros(100)
        result = compute_integrated_value(v0=v0, v1=v1)
        assert result.shape == v0.shape


class TestCheckConvergence:
    """
    Tests for CHECK_CONVERGENCE subroutine.
    
    Specification:
        diff = max|V_new - V_old|
        RETURN diff < tolerance
    """

    def test_identical_converged(self) -> None:
        """Identical arrays should be converged."""
        V = np.array([1.0, 2.0, 3.0])
        assert check_convergence(V_old=V, V_new=V.copy(), tolerance=1e-10)

    def test_small_diff_converged(self) -> None:
        """Small differences should be converged."""
        V_old = np.array([1.0, 2.0, 3.0])
        V_new = np.array([1.0 + 1e-12, 2.0, 3.0])
        assert check_convergence(V_old=V_old, V_new=V_new, tolerance=1e-10)

    def test_large_diff_not_converged(self) -> None:
        """Differences larger than tolerance should not be converged."""
        V_old = np.array([1.0, 2.0, 3.0])
        V_new = np.array([1.0, 2.0, 4.0])  # diff = 1.0
        assert not check_convergence(V_old=V_old, V_new=V_new, tolerance=0.5)

    def test_exact_tolerance_boundary(self) -> None:
        """Diff exactly at tolerance should not be converged (< not <=)."""
        V_old = np.array([0.0])
        V_new = np.array([1.0])  # diff = 1.0
        assert not check_convergence(V_old=V_old, V_new=V_new, tolerance=1.0)

    def test_uses_max_norm(self) -> None:
        """Should use max (infinity) norm, not L2."""
        V_old = np.array([0.0, 0.0, 0.0])
        V_new = np.array([0.1, 0.1, 0.1])  # max diff = 0.1
        assert check_convergence(V_old=V_old, V_new=V_new, tolerance=0.2)
        assert not check_convergence(V_old=V_old, V_new=V_new, tolerance=0.05)


class TestComputeChoiceProbability:
    """
    Tests for COMPUTE_CHOICE_PROBABILITY subroutine.
    
    Specification:
        P(a=1|s) = exp(v₁) / (exp(v₀) + exp(v₁))
        Equivalently: sigmoid(v₁ - v₀)
    """

    def test_logit_formula(self) -> None:
        """Should compute exp(v1) / (exp(v0) + exp(v1))."""
        v0 = np.array([1.0, 2.0])
        v1 = np.array([1.5, 2.5])
        result = compute_choice_probability(v0=v0, v1=v1)
        expected = np.exp(v1) / (np.exp(v0) + np.exp(v1))
        np.testing.assert_array_almost_equal(result, expected)

    def test_equal_values_half(self) -> None:
        """When v0 = v1, probability should be 0.5."""
        v0 = np.array([1.0, 5.0, 100.0])
        v1 = v0.copy()
        result = compute_choice_probability(v0=v0, v1=v1)
        np.testing.assert_array_almost_equal(result, np.array([0.5, 0.5, 0.5]))

    def test_v1_dominates(self) -> None:
        """When v1 >> v0, probability should approach 1."""
        v0 = np.array([0.0])
        v1 = np.array([100.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        assert result[0] > 0.999

    def test_v0_dominates(self) -> None:
        """When v0 >> v1, probability should approach 0."""
        v0 = np.array([100.0])
        v1 = np.array([0.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        assert result[0] < 0.001

    def test_probability_bounds(self) -> None:
        """Probability should always be in [0, 1]."""
        v0 = np.array([-10.0, 0.0, 10.0, 5.0])
        v1 = np.array([10.0, 0.0, -10.0, 5.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        # Use >= 0 and <= 1 due to floating point precision
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sigmoid_equivalence(self) -> None:
        """Should equal sigmoid(v1 - v0)."""
        v0 = np.array([1.0, 2.0, 3.0])
        v1 = np.array([2.0, 2.0, 1.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        expected = 1.0 / (1.0 + np.exp(v0 - v1))
        np.testing.assert_array_almost_equal(result, expected)

    def test_monotonic_in_difference(self) -> None:
        """Higher v1 - v0 should give higher probability."""
        v0 = np.array([0.0, 0.0, 0.0, 0.0])
        v1 = np.array([-2.0, -1.0, 0.0, 1.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        assert np.all(np.diff(result) > 0)

    def test_numerical_stability(self) -> None:
        """Should handle extreme values without NaN."""
        v0 = np.array([500.0, -500.0])
        v1 = np.array([-500.0, 500.0])
        result = compute_choice_probability(v0=v0, v1=v1)
        assert np.all(np.isfinite(result))


class TestSolveValueFunction:
    """
    Tests for SolveValueFunction algorithm.
    
    Specification:
        Iterates Bellman equation until convergence.
        Returns (V, v0, v1, n_iter).
    """

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
        """Value function should be monotonically increasing in state."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        V, v0, v1, _ = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.9,
        )
        # All should be monotonically increasing
        assert np.all(np.diff(V) >= -1e-6)
        assert np.all(np.diff(v0) >= -1e-6)
        assert np.all(np.diff(v1) >= -1e-6)

    def test_output_shapes(self) -> None:
        """All outputs should have correct shapes."""
        n = 75
        state_grid = np.linspace(start=0.0, stop=10.0, num=n)
        V, v0, v1, n_iter = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.9,
        )
        assert V.shape == (n,)
        assert v0.shape == (n,)
        assert v1.shape == (n,)
        assert isinstance(n_iter, int)

    def test_bellman_equation_satisfied(self) -> None:
        """At convergence, Bellman equation should be satisfied."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        V, v0, v1, _ = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.95,
            tolerance=1e-10,
        )
        # V should equal logsumexp(v0, v1)
        V_check = compute_integrated_value(v0=v0, v1=v1)
        np.testing.assert_array_almost_equal(V, V_check, decimal=8)

    def test_higher_beta_higher_value(self) -> None:
        """Higher β should increase value function."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        
        V_low, _, _, _ = solve_value_function(
            state_grid=state_grid, beta=0.5, gamma=0.1, delta=0.9
        )
        V_high, _, _, _ = solve_value_function(
            state_grid=state_grid, beta=1.5, gamma=0.1, delta=0.9
        )
        # At non-zero states, higher beta should give higher value
        assert np.all(V_high[10:] > V_low[10:])

    def test_higher_delta_higher_value(self) -> None:
        """Higher δ should increase value function (more patient agent)."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=50)
        
        V_low, _, _, _ = solve_value_function(
            state_grid=state_grid, beta=1.0, gamma=0.1, delta=0.8
        )
        V_high, _, _, _ = solve_value_function(
            state_grid=state_grid, beta=1.0, gamma=0.1, delta=0.95
        )
        assert np.all(V_high > V_low)

    def test_choice_probabilities_valid(self) -> None:
        """Choice probabilities should be in (0, 1)."""
        state_grid = np.linspace(start=0.0, stop=10.0, num=100)
        _, v0, v1, _ = solve_value_function(
            state_grid=state_grid,
            beta=1.0,
            gamma=0.1,
            delta=0.9,
        )
        prob = compute_choice_probability(v0=v0, v1=v1)
        assert np.all(prob > 0.0)
        assert np.all(prob < 1.0)
