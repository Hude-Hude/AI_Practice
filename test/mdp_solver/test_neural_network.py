"""Tests for neural network architecture."""

import pytest
import torch

from mdp_solver.neural_network import (
    MonotonicLayer,
    MonotonicValueNetwork,
    ValueFunctionApproximator,
)


class TestMonotonicLayer:
    """Tests for MonotonicLayer."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        layer = MonotonicLayer(in_features=5, out_features=10)
        x = torch.randn(32, 5)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_weights_positive(self) -> None:
        """Effective weights should be positive (softplus)."""
        layer = MonotonicLayer(in_features=3, out_features=4)
        weight = torch.nn.functional.softplus(layer.weight_raw)
        assert torch.all(weight > 0)


class TestMonotonicValueNetwork:
    """Tests for MonotonicValueNetwork."""

    def test_output_shape(self) -> None:
        """Output should match batch size."""
        net = MonotonicValueNetwork(hidden_sizes=[16, 16])
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        out = net(s)
        assert out.shape == (50,)

    def test_monotonicity(self) -> None:
        """Output should be monotonically increasing in input."""
        torch.manual_seed(42)
        net = MonotonicValueNetwork(hidden_sizes=[32, 32])
        s = torch.linspace(start=0.0, end=10.0, steps=100)
        with torch.no_grad():
            out = net(s)
        diffs = torch.diff(out)
        # All differences should be non-negative (monotonic)
        assert torch.all(diffs >= -1e-6)


class TestValueFunctionApproximator:
    """Tests for ValueFunctionApproximator."""

    def test_forward_returns_tuple(self) -> None:
        """Forward should return (v0, v1) tuple."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.tensor([1.0, 2.0, 3.0])
        v0, v1 = model(s)
        assert v0.shape == (3,)
        assert v1.shape == (3,)

    def test_integrated_value_shape(self) -> None:
        """Integrated value should have correct shape."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=5.0, steps=20)
        V = model.compute_integrated_value(s)
        assert V.shape == (20,)

    def test_choice_probability_bounds(self) -> None:
        """Choice probability should be in [0, 1]."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        with torch.no_grad():
            prob = model.compute_choice_probability(s)
        assert torch.all(prob >= 0.0)
        assert torch.all(prob <= 1.0)

    def test_both_networks_monotonic(self) -> None:
        """Both v0 and v1 networks should be monotonic."""
        torch.manual_seed(123)
        model = ValueFunctionApproximator(hidden_sizes=[32, 32])
        s = torch.linspace(start=0.0, end=10.0, steps=100)
        with torch.no_grad():
            v0, v1 = model(s)
        
        v0_diffs = torch.diff(v0)
        v1_diffs = torch.diff(v1)
        
        assert torch.all(v0_diffs >= -1e-6), "v0 should be monotonic"
        assert torch.all(v1_diffs >= -1e-6), "v1 should be monotonic"

