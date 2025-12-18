"""Tests for model primitives."""

import torch

from mdp_solver import clamp_states, compute_next_state, compute_reward


class TestComputeReward:
    """Tests for compute_reward."""

    def test_formula_action_zero(self) -> None:
        """u(s, 0) = β * log(1 + s)."""
        s = torch.tensor([0.0, 1.0, 2.0, 5.0])
        beta = 2.0
        r = compute_reward(s, action=0, beta=beta)
        expected = beta * torch.log(1 + s)
        torch.testing.assert_close(r, expected)

    def test_formula_action_one(self) -> None:
        """u(s, 1) = β * log(1 + s) - 1."""
        s = torch.tensor([0.0, 1.0, 2.0, 5.0])
        beta = 2.0
        r = compute_reward(s, action=1, beta=beta)
        expected = beta * torch.log(1 + s) - 1
        torch.testing.assert_close(r, expected)

    def test_action_cost(self) -> None:
        """Action 1 should cost exactly 1 more than action 0."""
        s = torch.tensor([0.0, 5.0, 10.0])
        beta = 1.5
        r0 = compute_reward(s, action=0, beta=beta)
        r1 = compute_reward(s, action=1, beta=beta)
        diff = r0 - r1
        torch.testing.assert_close(diff, torch.ones_like(diff))

    def test_zero_state(self) -> None:
        """At s=0: u(0,0)=0, u(0,1)=-1."""
        s = torch.tensor([0.0])
        r0 = compute_reward(s, action=0, beta=1.0)
        r1 = compute_reward(s, action=1, beta=1.0)
        torch.testing.assert_close(r0, torch.tensor([0.0]))
        torch.testing.assert_close(r1, torch.tensor([-1.0]))

    def test_monotonicity(self) -> None:
        """Reward should increase with state."""
        s = torch.linspace(0, 10, 100)
        r = compute_reward(s, action=0, beta=1.0)
        diffs = torch.diff(r)
        assert torch.all(diffs > 0)


class TestComputeNextState:
    """Tests for compute_next_state."""

    def test_formula_action_zero(self) -> None:
        """s' = (1 - γ) * s when a = 0."""
        s = torch.tensor([10.0, 20.0])
        gamma = 0.2
        s_next = compute_next_state(s, action=0, gamma=gamma)
        expected = (1 - gamma) * s
        torch.testing.assert_close(s_next, expected)

    def test_formula_action_one(self) -> None:
        """s' = (1 - γ) * s + 1 when a = 1."""
        s = torch.tensor([10.0, 20.0])
        gamma = 0.2
        s_next = compute_next_state(s, action=1, gamma=gamma)
        expected = (1 - gamma) * s + 1
        torch.testing.assert_close(s_next, expected)

    def test_action_boost(self) -> None:
        """Action 1 should boost state by 1."""
        s = torch.tensor([5.0, 10.0])
        gamma = 0.1
        s_next0 = compute_next_state(s, action=0, gamma=gamma)
        s_next1 = compute_next_state(s, action=1, gamma=gamma)
        diff = s_next1 - s_next0
        torch.testing.assert_close(diff, torch.ones_like(diff))

    def test_decay(self) -> None:
        """With action 0, state should decay."""
        s = torch.tensor([10.0])
        gamma = 0.1
        s_next = compute_next_state(s, action=0, gamma=gamma)
        assert s_next.item() < s.item()
        assert s_next.item() > 0

    def test_from_zero(self) -> None:
        """From s=0 with action 1, next state should be 1."""
        s = torch.tensor([0.0])
        s_next = compute_next_state(s, action=1, gamma=0.5)
        torch.testing.assert_close(s_next, torch.tensor([1.0]))


class TestClampStates:
    """Tests for clamp_states."""

    def test_within_bounds(self) -> None:
        """States within bounds should be unchanged."""
        s = torch.tensor([2.0, 5.0, 8.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        torch.testing.assert_close(s_clamped, s)

    def test_below_min(self) -> None:
        """States below min should be clamped to min."""
        s = torch.tensor([-5.0, -1.0, 0.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(s_clamped, expected)

    def test_above_max(self) -> None:
        """States above max should be clamped to max."""
        s = torch.tensor([10.0, 15.0, 100.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([10.0, 10.0, 10.0])
        torch.testing.assert_close(s_clamped, expected)

    def test_mixed(self) -> None:
        """Mixed states should be clamped correctly."""
        s = torch.tensor([-2.0, 5.0, 15.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([0.0, 5.0, 10.0])
        torch.testing.assert_close(s_clamped, expected)
