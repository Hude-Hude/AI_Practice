"""Tests for solver module."""

import torch

from mdp.solver import (
    check_convergence,
    copy_network,
    evaluate_network,
    generate_state_grid,
    initialize_networks,
    sample_states,
    solve_value_function,
)


class TestInitializeNetworks:
    """Tests for initialize_networks."""

    def test_returns_two_networks(self) -> None:
        """Should return a tuple of two networks."""
        v0_net, v1_net = initialize_networks([8, 8])
        assert v0_net is not None
        assert v1_net is not None

    def test_networks_independent(self) -> None:
        """Networks should have different weights."""
        v0_net, v1_net = initialize_networks([8])
        
        # Get first layer weights
        w0 = list(v0_net.parameters())[0]
        w1 = list(v1_net.parameters())[0]
        
        # Should not be identical (random initialization)
        assert not torch.allclose(w0, w1)

    def test_correct_architecture(self) -> None:
        """Networks should have correct number of layers."""
        v0_net, v1_net = initialize_networks([16, 16])
        # 2 hidden * 2 + 1 output = 5 layers
        assert len(v0_net) == 5
        assert len(v1_net) == 5


class TestCopyNetwork:
    """Tests for copy_network."""

    def test_returns_copy(self) -> None:
        """Should return a copy of the network."""
        v0_net, _ = initialize_networks([8])
        v0_copy = copy_network(v0_net)
        assert v0_copy is not None
        assert v0_copy is not v0_net

    def test_same_weights(self) -> None:
        """Copied network should have identical weights."""
        v0_net, _ = initialize_networks([8])
        v0_copy = copy_network(v0_net)
        
        # Check all parameters are equal
        for p1, p2 in zip(v0_net.parameters(), v0_copy.parameters()):
            assert torch.allclose(p1, p2)

    def test_frozen_gradients(self) -> None:
        """Copied network should have requires_grad=False."""
        v0_net, _ = initialize_networks([8])
        v0_copy = copy_network(v0_net)
        
        for param in v0_copy.parameters():
            assert not param.requires_grad

    def test_independent_weights(self) -> None:
        """Modifying original should not affect copy."""
        v0_net, _ = initialize_networks([8])
        v0_copy = copy_network(v0_net)
        
        # Store original copy weights
        original_weight = list(v0_copy.parameters())[0].clone()
        
        # Modify original network
        with torch.no_grad():
            for param in v0_net.parameters():
                param.add_(1.0)
        
        # Copy should be unchanged
        assert torch.allclose(list(v0_copy.parameters())[0], original_weight)

    def test_same_output(self) -> None:
        """Copied network should produce same outputs."""
        v0_net, _ = initialize_networks([8, 8])
        v0_copy = copy_network(v0_net)
        
        s = torch.linspace(0, 10, 50)
        
        with torch.no_grad():
            out_original = evaluate_network(v0_net, s)
            out_copy = evaluate_network(v0_copy, s)
        
        assert torch.allclose(out_original, out_copy)


class TestSampleStates:
    """Tests for sample_states."""

    def test_correct_count(self) -> None:
        """Should return correct number of samples."""
        s = sample_states(n=100, s_min=0.0, s_max=10.0)
        assert s.shape == (100,)

    def test_within_bounds(self) -> None:
        """All samples should be within [s_min, s_max]."""
        s = sample_states(n=1000, s_min=2.0, s_max=8.0)
        assert torch.all(s >= 2.0)
        assert torch.all(s <= 8.0)

    def test_uniform_distribution(self) -> None:
        """Samples should cover the range reasonably uniformly."""
        s = sample_states(n=10000, s_min=0.0, s_max=10.0)
        
        # Check mean is close to midpoint
        mean = s.mean().item()
        assert 4.5 < mean < 5.5

        # Check standard deviation is reasonable
        std = s.std().item()
        expected_std = 10.0 / (12 ** 0.5)  # Uniform distribution std
        assert abs(std - expected_std) < 0.5


class TestGenerateStateGrid:
    """Tests for generate_state_grid."""

    def test_correct_count(self) -> None:
        """Should return correct number of grid points."""
        s = generate_state_grid(n=100, s_min=0.0, s_max=10.0)
        assert s.shape == (100,)

    def test_within_bounds(self) -> None:
        """All grid points should be within [s_min, s_max]."""
        s = generate_state_grid(n=50, s_min=2.0, s_max=8.0)
        assert torch.all(s >= 2.0)
        assert torch.all(s <= 8.0)

    def test_evenly_spaced(self) -> None:
        """Grid points should be evenly spaced."""
        s = generate_state_grid(n=11, s_min=0.0, s_max=10.0)
        
        # Check spacing is uniform
        diffs = torch.diff(s)
        assert torch.allclose(diffs, torch.ones_like(diffs))

    def test_includes_endpoints(self) -> None:
        """Grid should include both s_min and s_max."""
        s = generate_state_grid(n=5, s_min=2.0, s_max=8.0)
        
        assert s[0].item() == 2.0
        assert s[-1].item() == 8.0

    def test_deterministic(self) -> None:
        """Same inputs should produce same grid."""
        s1 = generate_state_grid(n=100, s_min=0.0, s_max=10.0)
        s2 = generate_state_grid(n=100, s_min=0.0, s_max=10.0)
        
        assert torch.allclose(s1, s2)


class TestCheckConvergence:
    """Tests for check_convergence."""

    def test_converged(self) -> None:
        """Should return True when loss is below tolerance."""
        loss = torch.tensor(1e-8)  # sqrt = 1e-4
        assert check_convergence(loss, tolerance=1e-3)

    def test_not_converged(self) -> None:
        """Should return False when loss is above tolerance."""
        loss = torch.tensor(1.0)  # sqrt = 1.0
        assert not check_convergence(loss, tolerance=0.1)

    def test_boundary(self) -> None:
        """Should handle boundary case correctly."""
        loss = torch.tensor(1e-6)  # sqrt = 1e-3
        assert check_convergence(loss, tolerance=1e-3 + 1e-10)
        assert not check_convergence(loss, tolerance=1e-3 - 1e-10)


class TestSolveValueFunction:
    """Tests for solve_value_function."""

    def test_returns_networks(self) -> None:
        """Should return trained networks."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.9,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[8],
            learning_rate=0.01,
            batch_size=32,
            tolerance=0.1,
            max_iterations=10,
        )
        
        assert v0_net is not None
        assert v1_net is not None

    def test_returns_losses(self) -> None:
        """Should return list of losses."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.9,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[8],
            learning_rate=0.01,
            batch_size=32,
            tolerance=1e-10,  # Won't converge
            max_iterations=50,
        )
        
        assert len(losses) == 50
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self) -> None:
        """Loss should generally decrease over iterations."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.9,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[16, 16],
            learning_rate=0.01,
            batch_size=64,
            tolerance=1e-10,
            max_iterations=100,
        )
        
        # Final loss should be less than initial loss
        assert losses[-1] < losses[0]

    def test_convergence(self) -> None:
        """Loss should decrease significantly over training."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.5,  # Low discount for faster convergence
            s_min=0.0, s_max=5.0,
            hidden_sizes=[32, 32],
            learning_rate=0.01,
            batch_size=128,
            tolerance=1e-10,  # Very tight tolerance (won't converge)
            max_iterations=200,
        )
        
        # Loss should decrease by at least 50%
        assert losses[-1] < losses[0] * 0.5

    def test_networks_are_monotonic(self) -> None:
        """Trained networks should maintain monotonicity."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.9,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[16],
            learning_rate=0.01,
            batch_size=32,
            tolerance=1e-10,
            max_iterations=50,
        )
        
        # Check monotonicity
        s = torch.linspace(0, 10, 100)
        
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        
        assert torch.all(torch.diff(v0) >= -1e-5)
        assert torch.all(torch.diff(v1) >= -1e-5)

    def test_target_update_freq_parameter(self) -> None:
        """Should accept target_update_freq parameter."""
        torch.manual_seed(42)
        v0_net, v1_net, losses, n_iter = solve_value_function(
            beta=1.0, gamma=0.1, delta=0.9,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[8],
            learning_rate=0.01,
            batch_size=32,
            tolerance=1e-10,
            max_iterations=50,
            target_update_freq=10,
        )
        
        assert v0_net is not None
        assert v1_net is not None
        assert len(losses) == 50

    def test_target_network_improves_stability(self) -> None:
        """Target networks should help with training stability."""
        torch.manual_seed(42)
        
        # With frequent target updates (like no target network)
        # delta is the discount factor (should be < 1)
        _, _, losses_freq, _ = solve_value_function(
            beta=0.1, gamma=0.1, delta=0.95,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[16, 16],
            learning_rate=0.05,
            batch_size=64,
            tolerance=1e-10,
            max_iterations=100,
            target_update_freq=1,  # Update every iteration (no stable target)
        )
        
        torch.manual_seed(42)
        
        # With infrequent target updates (stable targets)
        _, _, losses_stable, _ = solve_value_function(
            beta=0.1, gamma=0.1, delta=0.95,
            s_min=0.0, s_max=10.0,
            hidden_sizes=[16, 16],
            learning_rate=0.05,
            batch_size=64,
            tolerance=1e-10,
            max_iterations=100,
            target_update_freq=50,  # Stable targets for 50 iterations
        )
        
        # Both should train (loss should decrease)
        assert losses_freq[-1] < losses_freq[0]
        assert losses_stable[-1] < losses_stable[0]
