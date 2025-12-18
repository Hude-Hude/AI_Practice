"""Tests for value_function module."""

import torch

from mdp_solver.network import build_monotonic_network
from mdp_solver.value_function import (
    compute_bellman_loss,
    compute_bellman_targets,
    compute_choice_probability,
    compute_integrated_value,
)


class TestComputeIntegratedValue:
    """Tests for compute_integrated_value."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        V = compute_integrated_value(v0_net, v1_net, s)
        assert V.shape == (50,)

    def test_greater_than_max(self) -> None:
        """Integrated value should be >= max(v0, v1)."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        
        from mdp_solver.network import evaluate_network
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        V = compute_integrated_value(v0_net, v1_net, s)
        
        v_max = torch.maximum(v0, v1)
        assert torch.all(V >= v_max - 1e-6)

    def test_logsumexp_formula(self) -> None:
        """Should equal log(exp(v0) + exp(v1))."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        
        from mdp_solver.network import evaluate_network
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        V = compute_integrated_value(v0_net, v1_net, s)
        
        expected = torch.logsumexp(torch.stack([v0, v1], dim=1), dim=1)
        torch.testing.assert_close(V, expected)


class TestComputeBellmanTargets:
    """Tests for compute_bellman_targets."""

    def test_output_shapes(self) -> None:
        """Should return two tensors of correct shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        
        target0, target1 = compute_bellman_targets(
            v0_net, v1_net, s,
            beta=1.0, gamma=0.1, delta=0.95,
            s_min=0.0, s_max=10.0,
        )
        
        assert target0.shape == (50,)
        assert target1.shape == (50,)

    def test_discount_zero(self) -> None:
        """With delta=0, targets should equal rewards."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        beta = 2.0
        
        target0, target1 = compute_bellman_targets(
            v0_net, v1_net, s,
            beta=beta, gamma=0.1, delta=0.0,
            s_min=0.0, s_max=15.0,
        )
        
        expected0 = beta * torch.log(1 + s)  # action 0
        expected1 = beta * torch.log(1 + s) - 1  # action 1
        
        torch.testing.assert_close(target0, expected0)
        torch.testing.assert_close(target1, expected1)


class TestComputeBellmanLoss:
    """Tests for compute_bellman_loss."""

    def test_zero_error(self) -> None:
        """Loss should be zero when predictions equal targets."""
        v0_pred = torch.tensor([1.0, 2.0, 3.0])
        v1_pred = torch.tensor([1.5, 2.5, 3.5])
        target0 = v0_pred.clone()
        target1 = v1_pred.clone()
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        assert loss.item() < 1e-10

    def test_positive_loss(self) -> None:
        """Loss should be positive when there's error."""
        v0_pred = torch.tensor([1.0, 2.0, 3.0])
        v1_pred = torch.tensor([1.5, 2.5, 3.5])
        target0 = torch.tensor([2.0, 3.0, 4.0])
        target1 = torch.tensor([2.5, 3.5, 4.5])
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        assert loss.item() > 0

    def test_formula(self) -> None:
        """Should compute mean squared error correctly."""
        v0_pred = torch.tensor([1.0, 2.0])
        v1_pred = torch.tensor([1.0, 2.0])
        target0 = torch.tensor([2.0, 3.0])  # error = 1, 1
        target1 = torch.tensor([2.0, 3.0])  # error = 1, 1
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        # (1^2 + 1^2 + 1^2 + 1^2) / (2 * 2) = 4 / 4 = 1.0
        expected = 1.0
        assert abs(loss.item() - expected) < 1e-6


class TestComputeChoiceProbability:
    """Tests for compute_choice_probability."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        prob = compute_choice_probability(v0_net, v1_net, s)
        assert prob.shape == (50,)

    def test_bounds(self) -> None:
        """Probability should be in [0, 1]."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 100)
        prob = compute_choice_probability(v0_net, v1_net, s)
        assert torch.all(prob >= 0)
        assert torch.all(prob <= 1)

    def test_sigmoid_formula(self) -> None:
        """Should equal sigmoid(v1 - v0)."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        
        from mdp_solver.network import evaluate_network
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        prob = compute_choice_probability(v0_net, v1_net, s)
        
        expected = torch.sigmoid(v1 - v0)
        torch.testing.assert_close(prob, expected)

