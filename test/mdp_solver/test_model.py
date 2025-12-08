"""
Tests for model primitives.

Based on specifications from solve_mdp.qmd:
- COMPUTE_REWARD: u(s, a) = β * s - a
- COMPUTE_NEXT_STATE: s' = (1 - γ) * s + a
"""

import numpy as np
import pytest

from mdp_solver.model import compute_next_state, compute_reward


class TestComputeReward:
    """
    Tests for COMPUTE_REWARD subroutine.
    
    Specification:
        INPUT: s (Vector), action (0 or 1), β (Scalar)
        OUTPUT: reward = β * s - action
    """

    def test_formula_action_zero(self) -> None:
        """u(s, 0) = β * s - 0 = β * s."""
        s = np.array([1.0, 2.0, 3.0])
        beta = 2.0
        result = compute_reward(s=s, action=0, beta=beta)
        expected = beta * s - 0
        np.testing.assert_array_almost_equal(result, expected)

    def test_formula_action_one(self) -> None:
        """u(s, 1) = β * s - 1."""
        s = np.array([1.0, 2.0, 3.0])
        beta = 2.0
        result = compute_reward(s=s, action=1, beta=beta)
        expected = beta * s - 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_action_one_costs_one_unit(self) -> None:
        """Action 1 should cost exactly 1 unit more than action 0."""
        s = np.array([5.0, 10.0])
        beta = 1.5
        reward_0 = compute_reward(s=s, action=0, beta=beta)
        reward_1 = compute_reward(s=s, action=1, beta=beta)
        np.testing.assert_array_almost_equal(reward_0 - reward_1, np.ones_like(s))

    def test_higher_state_higher_reward(self) -> None:
        """With β > 0, higher state should yield higher reward."""
        s = np.array([1.0, 5.0, 10.0])
        beta = 1.0
        reward = compute_reward(s=s, action=0, beta=beta)
        assert np.all(np.diff(reward) > 0)

    def test_beta_scales_linearly(self) -> None:
        """Doubling β should double the state contribution."""
        s = np.array([2.0, 4.0])
        reward_beta1 = compute_reward(s=s, action=0, beta=1.0)
        reward_beta2 = compute_reward(s=s, action=0, beta=2.0)
        np.testing.assert_array_almost_equal(reward_beta2, 2 * reward_beta1)

    def test_zero_state_action_zero(self) -> None:
        """u(0, 0) = 0."""
        s = np.array([0.0])
        result = compute_reward(s=s, action=0, beta=1.0)
        np.testing.assert_array_almost_equal(result, np.array([0.0]))

    def test_zero_state_action_one(self) -> None:
        """u(0, 1) = -1."""
        s = np.array([0.0])
        result = compute_reward(s=s, action=1, beta=1.0)
        np.testing.assert_array_almost_equal(result, np.array([-1.0]))

    def test_output_shape_matches_input(self) -> None:
        """Output should have same shape as input state."""
        s = np.linspace(start=0.0, stop=10.0, num=100)
        result = compute_reward(s=s, action=0, beta=1.0)
        assert result.shape == s.shape


class TestComputeNextState:
    """
    Tests for COMPUTE_NEXT_STATE subroutine.
    
    Specification:
        INPUT: s (Vector), action (0 or 1), γ (Scalar)
        OUTPUT: s' = (1 - γ) * s + action
    """

    def test_formula_action_zero(self) -> None:
        """s' = (1 - γ) * s when a = 0."""
        s = np.array([10.0, 20.0])
        gamma = 0.2
        result = compute_next_state(s=s, action=0, gamma=gamma)
        expected = (1 - gamma) * s
        np.testing.assert_array_almost_equal(result, expected)

    def test_formula_action_one(self) -> None:
        """s' = (1 - γ) * s + 1 when a = 1."""
        s = np.array([10.0, 20.0])
        gamma = 0.2
        result = compute_next_state(s=s, action=1, gamma=gamma)
        expected = (1 - gamma) * s + 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_action_one_boosts_by_one(self) -> None:
        """Action 1 should boost next state by exactly 1."""
        s = np.array([5.0, 10.0])
        gamma = 0.1
        next_0 = compute_next_state(s=s, action=0, gamma=gamma)
        next_1 = compute_next_state(s=s, action=1, gamma=gamma)
        np.testing.assert_array_almost_equal(next_1 - next_0, np.ones_like(s))

    def test_decay_toward_zero(self) -> None:
        """With action 0, state should decay toward zero."""
        s = np.array([10.0])
        gamma = 0.1
        next_s = compute_next_state(s=s, action=0, gamma=gamma)
        assert next_s[0] < s[0]
        assert next_s[0] > 0

    def test_zero_gamma_no_decay(self) -> None:
        """With γ = 0, state should not decay."""
        s = np.array([5.0])
        next_s = compute_next_state(s=s, action=0, gamma=0.0)
        np.testing.assert_array_almost_equal(next_s, s)

    def test_from_zero_action_one(self) -> None:
        """From s = 0 with action 1, next state should be 1."""
        s = np.array([0.0])
        gamma = 0.5  # Any gamma
        next_s = compute_next_state(s=s, action=1, gamma=gamma)
        np.testing.assert_array_almost_equal(next_s, np.array([1.0]))

    def test_steady_state_action_one(self) -> None:
        """
        Steady state with action 1: s* = (1-γ)*s* + 1 => s* = 1/γ.
        """
        gamma = 0.1
        steady_state = 1.0 / gamma  # = 10.0
        s = np.array([steady_state])
        next_s = compute_next_state(s=s, action=1, gamma=gamma)
        np.testing.assert_array_almost_equal(next_s, s, decimal=10)

    def test_output_shape_matches_input(self) -> None:
        """Output should have same shape as input state."""
        s = np.linspace(start=0.0, stop=10.0, num=100)
        result = compute_next_state(s=s, action=0, gamma=0.1)
        assert result.shape == s.shape

    def test_gamma_range(self) -> None:
        """Should work for γ in (0, 1)."""
        s = np.array([5.0])
        for gamma in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = compute_next_state(s=s, action=0, gamma=gamma)
            assert 0 < result[0] < s[0]
