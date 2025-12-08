"""Tests for model primitives."""

import numpy as np
import pytest

from mdp_solver.model import compute_next_state, compute_reward


class TestComputeReward:
    """Tests for compute_reward function."""

    def test_action_zero_no_cost(self) -> None:
        """Action 0 should have no cost component."""
        s = np.array([1.0, 2.0, 3.0])
        beta = 1.0
        reward = compute_reward(s=s, action=0, beta=beta)
        expected = beta * s  # No -a term when a=0
        np.testing.assert_array_almost_equal(reward, expected)

    def test_action_one_has_cost(self) -> None:
        """Action 1 should subtract 1 from reward."""
        s = np.array([1.0, 2.0, 3.0])
        beta = 1.0
        reward = compute_reward(s=s, action=1, beta=beta)
        expected = beta * s - 1
        np.testing.assert_array_almost_equal(reward, expected)

    def test_beta_scales_state(self) -> None:
        """Beta parameter should scale the state contribution."""
        s = np.array([1.0, 2.0])
        beta = 2.0
        reward = compute_reward(s=s, action=0, beta=beta)
        expected = 2.0 * s
        np.testing.assert_array_almost_equal(reward, expected)


class TestComputeNextState:
    """Tests for compute_next_state function."""

    def test_action_zero_decay(self) -> None:
        """Action 0 should decay state toward zero."""
        s = np.array([10.0])
        gamma = 0.1
        next_s = compute_next_state(s=s, action=0, gamma=gamma)
        expected = (1 - gamma) * s
        np.testing.assert_array_almost_equal(next_s, expected)

    def test_action_one_boost(self) -> None:
        """Action 1 should boost state by 1."""
        s = np.array([5.0])
        gamma = 0.1
        next_s = compute_next_state(s=s, action=1, gamma=gamma)
        expected = (1 - gamma) * s + 1
        np.testing.assert_array_almost_equal(next_s, expected)

    def test_zero_state_action_one(self) -> None:
        """From state 0 with action 1, next state should be 1."""
        s = np.array([0.0])
        gamma = 0.1
        next_s = compute_next_state(s=s, action=1, gamma=gamma)
        np.testing.assert_array_almost_equal(next_s, np.array([1.0]))

