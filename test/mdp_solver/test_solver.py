"""Tests for solver module."""

import torch

from mdp_solver.solver import (
    check_convergence,
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
        from mdp_solver.network import evaluate_network
        s = torch.linspace(0, 10, 100)
        
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        
        assert torch.all(torch.diff(v0) >= -1e-5)
        assert torch.all(torch.diff(v1) >= -1e-5)

