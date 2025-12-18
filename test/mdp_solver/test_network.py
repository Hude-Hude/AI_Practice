"""Tests for network module."""

import torch
import torch.nn.functional as F

from mdp_solver.network import (
    MonotonicLinear,
    SoftplusActivation,
    build_monotonic_network,
    evaluate_network,
)


class TestMonotonicLinear:
    """Tests for MonotonicLinear layer."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        layer = MonotonicLinear(in_features=1, out_features=8)
        x = torch.randn(10, 1)
        y = layer(x)
        assert y.shape == (10, 8)

    def test_weights_positive(self) -> None:
        """Effective weights should be positive via softplus."""
        layer = MonotonicLinear(in_features=4, out_features=8)
        # Force some raw weights to be negative
        layer.weight_raw.data = torch.randn(8, 4) * 2 - 1
        
        # Compute effective weights
        effective_weights = F.softplus(layer.weight_raw)
        
        assert torch.all(effective_weights > 0)

    def test_monotonicity_single_input(self) -> None:
        """Output should increase with input for single feature."""
        layer = MonotonicLinear(in_features=1, out_features=1)
        
        # Create increasing inputs
        x = torch.linspace(0, 10, 100).unsqueeze(1)
        y = layer(x).squeeze()
        
        # Check monotonicity
        diffs = torch.diff(y)
        assert torch.all(diffs >= -1e-6)  # Allow small numerical errors


class TestSoftplusActivation:
    """Tests for SoftplusActivation."""

    def test_output_positive(self) -> None:
        """Softplus output should always be positive."""
        activation = SoftplusActivation()
        x = torch.randn(100) * 10  # Wide range of inputs
        y = activation(x)
        assert torch.all(y > 0)

    def test_monotonicity(self) -> None:
        """Softplus should be monotonically increasing."""
        activation = SoftplusActivation()
        x = torch.linspace(-10, 10, 100)
        y = activation(x)
        diffs = torch.diff(y)
        assert torch.all(diffs > 0)

    def test_formula(self) -> None:
        """Should match log(1 + exp(x))."""
        activation = SoftplusActivation()
        x = torch.tensor([0.0, 1.0, -1.0, 5.0])
        y = activation(x)
        expected = torch.log(1 + torch.exp(x))
        torch.testing.assert_close(y, expected)


class TestBuildMonotonicNetwork:
    """Tests for build_monotonic_network."""

    def test_creates_sequential(self) -> None:
        """Should return nn.Sequential."""
        network = build_monotonic_network([8, 8])
        assert isinstance(network, torch.nn.Sequential)

    def test_correct_layer_count(self) -> None:
        """Should have correct number of layers."""
        # [8, 8] -> MonotonicLinear + Softplus + MonotonicLinear + Softplus + MonotonicLinear
        network = build_monotonic_network([8, 8])
        # 2 hidden * 2 (linear + activation) + 1 output = 5
        assert len(network) == 5

    def test_output_shape(self) -> None:
        """Output should be scalar per input."""
        network = build_monotonic_network([16, 16])
        x = torch.randn(32, 1)
        y = network(x)
        assert y.shape == (32, 1)


class TestEvaluateNetwork:
    """Tests for evaluate_network."""

    def test_1d_input(self) -> None:
        """Should handle 1D input."""
        network = build_monotonic_network([8])
        s = torch.randn(50)
        values = evaluate_network(network, s)
        assert values.shape == (50,)

    def test_2d_input(self) -> None:
        """Should handle 2D input."""
        network = build_monotonic_network([8])
        s = torch.randn(50, 1)
        values = evaluate_network(network, s)
        assert values.shape == (50,)

    def test_monotonicity(self) -> None:
        """Network output should be monotonic in input."""
        torch.manual_seed(42)
        network = build_monotonic_network([16, 16])
        
        s = torch.linspace(0, 10, 100)
        values = evaluate_network(network, s)
        
        diffs = torch.diff(values)
        assert torch.all(diffs >= -1e-5)  # Allow small numerical errors

