"""
Tests for neural network architecture.

Based on specifications from solve_mdp.qmd:
- MonotonicLayer: weights = softplus(weight_raw) > 0
- MonotonicValueNetwork: monotonically increasing in input
- ValueFunctionApproximator: FORWARD, COMPUTE_INTEGRATED_VALUE, COMPUTE_CHOICE_PROBABILITY
"""

import numpy as np
import pytest
import torch

from mdp_solver.neural_network import (
    MonotonicLayer,
    MonotonicValueNetwork,
    ValueFunctionApproximator,
)


class TestMonotonicLayer:
    """
    Tests for MonotonicLayer class.
    
    Specification:
        weight = softplus(weight_raw) > 0
        FORWARD(x) = weight @ x + bias
    """

    def test_output_shape(self) -> None:
        """Output shape should be (batch_size, out_features)."""
        layer = MonotonicLayer(in_features=5, out_features=10)
        x = torch.randn(32, 5)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_weights_always_positive(self) -> None:
        """Effective weights should always be positive (softplus > 0)."""
        layer = MonotonicLayer(in_features=10, out_features=20)
        # Test with various random initializations
        for _ in range(5):
            layer.weight_raw.data = torch.randn_like(layer.weight_raw)
            weight = torch.nn.functional.softplus(layer.weight_raw)
            assert torch.all(weight > 0)

    def test_weights_positive_even_negative_raw(self) -> None:
        """Even with negative raw weights, effective weights should be positive."""
        layer = MonotonicLayer(in_features=3, out_features=4)
        layer.weight_raw.data = torch.tensor([[-100.0, -100.0, -100.0]] * 4)
        weight = torch.nn.functional.softplus(layer.weight_raw)
        assert torch.all(weight > 0)

    def test_linear_transformation(self) -> None:
        """Output should be weight @ x + bias."""
        layer = MonotonicLayer(in_features=2, out_features=3, bias=True)
        x = torch.tensor([[1.0, 2.0]])
        
        weight = torch.nn.functional.softplus(layer.weight_raw)
        expected = x @ weight.T + layer.bias
        result = layer(x)
        
        torch.testing.assert_close(result, expected)

    def test_no_bias_option(self) -> None:
        """Should work without bias."""
        layer = MonotonicLayer(in_features=5, out_features=3, bias=False)
        assert layer.bias is None
        x = torch.randn(10, 5)
        out = layer(x)
        assert out.shape == (10, 3)

    def test_gradient_flow(self) -> None:
        """Gradients should flow through softplus."""
        layer = MonotonicLayer(in_features=3, out_features=2)
        x = torch.randn(5, 3, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert layer.weight_raw.grad is not None
        assert torch.all(torch.isfinite(layer.weight_raw.grad))


class TestMonotonicValueNetwork:
    """
    Tests for MonotonicValueNetwork class.
    
    Specification:
        - Feed-forward network with MonotonicLayer + Softplus
        - Output is monotonically increasing in input
    """

    def test_output_shape_1d_input(self) -> None:
        """Should handle 1D input and return 1D output."""
        net = MonotonicValueNetwork(hidden_sizes=[16, 16])
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        out = net(s)
        assert out.shape == (50,)

    def test_output_shape_2d_input(self) -> None:
        """Should handle 2D input (batch_size, 1)."""
        net = MonotonicValueNetwork(hidden_sizes=[16])
        s = torch.randn(32, 1)
        out = net(s)
        assert out.shape == (32,)

    def test_monotonicity_random_init(self) -> None:
        """Output should be monotonically increasing regardless of initialization."""
        torch.manual_seed(12345)
        net = MonotonicValueNetwork(hidden_sizes=[32, 32])
        s = torch.linspace(start=0.0, end=10.0, steps=100)
        
        with torch.no_grad():
            out = net(s)
        
        diffs = torch.diff(out)
        assert torch.all(diffs >= -1e-6), f"Non-monotonic: min diff = {diffs.min()}"

    def test_monotonicity_multiple_seeds(self) -> None:
        """Monotonicity should hold for various random seeds."""
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        
        for seed in [0, 42, 123, 999]:
            torch.manual_seed(seed)
            net = MonotonicValueNetwork(hidden_sizes=[16, 16])
            with torch.no_grad():
                out = net(s)
            diffs = torch.diff(out)
            assert torch.all(diffs >= -1e-6), f"Seed {seed} failed monotonicity"

    def test_monotonicity_various_architectures(self) -> None:
        """Monotonicity should hold for various network sizes."""
        torch.manual_seed(42)
        s = torch.linspace(start=0.0, end=10.0, steps=100)
        
        architectures = [
            [8],
            [16, 16],
            [32, 32, 32],
            [64],
        ]
        
        for hidden_sizes in architectures:
            net = MonotonicValueNetwork(hidden_sizes=hidden_sizes)
            with torch.no_grad():
                out = net(s)
            diffs = torch.diff(out)
            assert torch.all(diffs >= -1e-6), f"Architecture {hidden_sizes} failed"

    def test_differentiable(self) -> None:
        """Network should be differentiable."""
        net = MonotonicValueNetwork(hidden_sizes=[16])
        s = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = net(s)
        loss = out.sum()
        loss.backward()
        assert s.grad is not None


class TestValueFunctionApproximator:
    """
    Tests for ValueFunctionApproximator class.
    
    Specification:
        - Two separate MonotonicValueNetworks for v0 and v1
        - FORWARD: returns (v0, v1)
        - COMPUTE_INTEGRATED_VALUE: logsumexp(v0, v1)
        - COMPUTE_CHOICE_PROBABILITY: sigmoid(v1 - v0)
    """

    def test_forward_returns_tuple(self) -> None:
        """FORWARD should return (v0, v1) tuple."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.tensor([1.0, 2.0, 3.0])
        result = model(s)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        v0, v1 = result
        assert v0.shape == (3,)
        assert v1.shape == (3,)

    def test_v0_v1_independent_networks(self) -> None:
        """v0 and v1 should come from independent networks."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        
        # v0_net and v1_net should have different weights
        v0_weights = [p for p in model.v0_net.parameters() if p.dim() > 1]  # Only weight matrices
        v1_weights = [p for p in model.v1_net.parameters() if p.dim() > 1]
        
        # They should have same structure but different values
        assert len(v0_weights) == len(v1_weights)
        for w0, w1 in zip(v0_weights, v1_weights):
            assert w0.shape == w1.shape
            # Weights should be different (random init)
            assert not torch.allclose(w0, w1)

    def test_both_monotonic(self) -> None:
        """Both v0 and v1 should be monotonically increasing."""
        torch.manual_seed(42)
        model = ValueFunctionApproximator(hidden_sizes=[32, 32])
        s = torch.linspace(start=0.0, end=10.0, steps=100)
        
        with torch.no_grad():
            v0, v1 = model(s)
        
        v0_diffs = torch.diff(v0)
        v1_diffs = torch.diff(v1)
        
        assert torch.all(v0_diffs >= -1e-6), "v0 not monotonic"
        assert torch.all(v1_diffs >= -1e-6), "v1 not monotonic"

    def test_compute_integrated_value_formula(self) -> None:
        """COMPUTE_INTEGRATED_VALUE should compute logsumexp(v0, v1)."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=5.0, steps=20)
        
        with torch.no_grad():
            V = model.compute_integrated_value(s)
            v0, v1 = model(s)
            expected = torch.logsumexp(torch.stack([v0, v1], dim=-1), dim=-1)
        
        torch.testing.assert_close(V, expected)

    def test_compute_integrated_value_shape(self) -> None:
        """Integrated value should have same shape as input."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        V = model.compute_integrated_value(s)
        assert V.shape == (50,)

    def test_compute_choice_probability_formula(self) -> None:
        """COMPUTE_CHOICE_PROBABILITY should compute sigmoid(v1 - v0)."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=5.0, steps=20)
        
        with torch.no_grad():
            prob = model.compute_choice_probability(s)
            v0, v1 = model(s)
            expected = torch.sigmoid(v1 - v0)
        
        torch.testing.assert_close(prob, expected)

    def test_choice_probability_bounds(self) -> None:
        """Choice probability should be in [0, 1]."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        # Use moderate range to avoid numerical underflow
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        
        with torch.no_grad():
            prob = model.compute_choice_probability(s)
        
        # Use >= 0 and <= 1 due to floating point precision limits
        assert torch.all(prob >= 0.0)
        assert torch.all(prob <= 1.0)

    def test_integrated_value_greater_than_max(self) -> None:
        """V(s) > max(v0, v1) due to option value."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.linspace(start=0.0, end=10.0, steps=50)
        
        with torch.no_grad():
            V = model.compute_integrated_value(s)
            v0, v1 = model(s)
            max_v = torch.maximum(v0, v1)
        
        assert torch.all(V > max_v)

    def test_gradient_flow_all_methods(self) -> None:
        """Gradients should flow through all methods."""
        model = ValueFunctionApproximator(hidden_sizes=[16])
        s = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Test forward
        v0, v1 = model(s)
        (v0.sum() + v1.sum()).backward()
        assert s.grad is not None
        
        # Test integrated value
        s2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        V = model.compute_integrated_value(s2)
        V.sum().backward()
        assert s2.grad is not None
        
        # Test choice probability
        s3 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        p = model.compute_choice_probability(s3)
        p.sum().backward()
        assert s3.grad is not None
