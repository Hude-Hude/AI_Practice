"""Tests for MDP simulator module."""

import numpy as np
import torch

from mdp.solver import (
    build_monotonic_network,
    compute_choice_probability,
    solve_value_function,
)
from mdp.simulator import (
    simulate_mdp_panel,
    PanelData,
)
from mdp.simulator.mdp_simulator import (
    compute_reward,
    compute_next_state,
)


class TestComputeReward:
    """Tests for compute_reward."""

    def test_basic_computation(self) -> None:
        """Should compute r = β * log(1 + s) - a."""
        s = np.array([0.0, 1.0, 10.0])
        a = np.array([0, 1, 0])
        beta = 1.0

        r = compute_reward(s=s, a=a, beta=beta)

        expected = beta * np.log(1 + s) - a
        np.testing.assert_allclose(r, expected)

    def test_zero_state(self) -> None:
        """Should handle s=0 correctly (log(1) = 0)."""
        s = np.array([0.0])
        a = np.array([0])
        beta = 1.0

        r = compute_reward(s=s, a=a, beta=beta)

        assert r[0] == 0.0

    def test_action_cost(self) -> None:
        """Action a=1 should cost 1 unit."""
        s = np.array([5.0, 5.0])
        a = np.array([0, 1])
        beta = 1.0

        r = compute_reward(s=s, a=a, beta=beta)

        assert r[0] - r[1] == 1.0

    def test_beta_scaling(self) -> None:
        """Reward should scale with beta."""
        s = np.array([10.0])
        a = np.array([0])

        r1 = compute_reward(s=s, a=a, beta=1.0)
        r2 = compute_reward(s=s, a=a, beta=2.0)

        assert r2[0] == 2 * r1[0]


class TestComputeNextState:
    """Tests for compute_next_state."""

    def test_basic_computation(self) -> None:
        """Should compute s' = (1 - γ) * s + a."""
        s = np.array([10.0, 10.0])
        a = np.array([0, 1])
        gamma = 0.1

        s_next = compute_next_state(s=s, a=a, gamma=gamma)

        expected = (1 - gamma) * s + a
        np.testing.assert_allclose(s_next, expected)

    def test_no_decay(self) -> None:
        """With gamma=0, state should persist."""
        s = np.array([5.0])
        a = np.array([0])
        gamma = 0.0

        s_next = compute_next_state(s=s, a=a, gamma=gamma)

        assert s_next[0] == 5.0

    def test_full_decay(self) -> None:
        """With gamma=1, next state should equal action."""
        s = np.array([100.0])
        a = np.array([1])
        gamma = 1.0

        s_next = compute_next_state(s=s, a=a, gamma=gamma)

        assert s_next[0] == 1.0

    def test_investment_increases_state(self) -> None:
        """Action a=1 should increase state by 1."""
        s = np.array([5.0, 5.0])
        a = np.array([0, 1])
        gamma = 0.0

        s_next = compute_next_state(s=s, a=a, gamma=gamma)

        assert s_next[1] - s_next[0] == 1.0


class TestSimulateMdpPanel:
    """Tests for simulate_mdp_panel."""

    def test_returns_panel_data(self) -> None:
        """Should return PanelData object."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=10,
            n_periods=5,
            s_init=1.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        assert isinstance(panel, PanelData)

    def test_correct_shapes(self) -> None:
        """Output arrays should have correct shapes."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        n_agents = 100
        n_periods = 20

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=n_periods,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        assert panel.states.shape == (n_agents, n_periods)
        assert panel.actions.shape == (n_agents, n_periods)
        assert panel.rewards.shape == (n_agents, n_periods)
        assert panel.n_agents == n_agents
        assert panel.n_periods == n_periods

    def test_initial_states_scalar(self) -> None:
        """Scalar s_init should set all agents to same initial state."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        s_init = 5.0
        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=50,
            n_periods=10,
            s_init=s_init,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        np.testing.assert_array_equal(panel.states[:, 0], s_init)

    def test_initial_states_array(self) -> None:
        """Array s_init should set individual initial states."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        n_agents = 50
        s_init = np.linspace(0, 10, n_agents)

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=10,
            s_init=s_init,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        np.testing.assert_array_equal(panel.states[:, 0], s_init)

    def test_actions_binary(self) -> None:
        """Actions should be 0 or 1."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=100,
            n_periods=50,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        assert np.all((panel.actions == 0) | (panel.actions == 1))

    def test_state_transitions(self) -> None:
        """States should follow transition equation."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        gamma = 0.1
        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=100,
            n_periods=50,
            s_init=5.0,
            beta=1.0,
            gamma=gamma,
            seed=42,
        )

        # Check transition for all periods except last
        for t in range(panel.n_periods - 1):
            expected_next = (1 - gamma) * panel.states[:, t] + panel.actions[:, t]
            np.testing.assert_allclose(
                panel.states[:, t + 1], 
                expected_next,
                rtol=1e-6,  # Relaxed due to float32/float64 conversion
            )

    def test_rewards_computed_correctly(self) -> None:
        """Rewards should match reward function."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        beta = 1.5
        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=100,
            n_periods=20,
            s_init=5.0,
            beta=beta,
            gamma=0.1,
            seed=42,
        )

        expected_rewards = beta * np.log(1 + panel.states) - panel.actions
        np.testing.assert_allclose(
            panel.rewards, 
            expected_rewards,
            rtol=1e-6,  # Relaxed due to float32/float64 conversion
        )

    def test_reproducibility(self) -> None:
        """Same seed should produce identical results."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        panel1 = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=50,
            n_periods=20,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=123,
        )

        panel2 = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=50,
            n_periods=20,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=123,
        )

        np.testing.assert_array_equal(panel1.states, panel2.states)
        np.testing.assert_array_equal(panel1.actions, panel2.actions)
        np.testing.assert_array_equal(panel1.rewards, panel2.rewards)

    def test_different_seeds_different_results(self) -> None:
        """Different seeds should produce different results."""
        v0_net = build_monotonic_network(hidden_sizes=[8])
        v1_net = build_monotonic_network(hidden_sizes=[8])

        panel1 = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=100,
            n_periods=50,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=1,
        )

        panel2 = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=100,
            n_periods=50,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=2,
        )

        # Actions should differ
        assert not np.array_equal(panel1.actions, panel2.actions)


class TestChoiceProbabilityConsistency:
    """Tests that simulated choices match theoretical probabilities."""

    def test_empirical_matches_theoretical(self) -> None:
        """Empirical choice frequencies should match logit probabilities."""
        # Train a simple model
        torch.manual_seed(42)
        v0_net, v1_net, _, _ = solve_value_function(
            beta=1.0,
            gamma=0.1,
            delta=0.9,
            s_min=0.0,
            s_max=10.0,
            hidden_sizes=[16],
            learning_rate=0.1,
            batch_size=64,
            tolerance=0.01,
            max_iterations=500,
        )

        # Simulate many agents at a fixed state
        n_agents = 10000
        s_fixed = 5.0

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=1,  # Only first period matters
            s_init=s_fixed,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        # Compute empirical P(a=1)
        empirical_p1 = panel.actions[:, 0].mean()

        # Compute theoretical P(a=1)
        s_tensor = torch.tensor([s_fixed])
        with torch.no_grad():
            _, p1_theory = compute_choice_probability(v0_net, v1_net, s_tensor)
            theoretical_p1 = p1_theory.item()

        # Should be close (within sampling error)
        # For n=10000, standard error ≈ sqrt(p(1-p)/n) ≈ 0.005
        assert abs(empirical_p1 - theoretical_p1) < 0.02

    def test_choice_varies_with_state(self) -> None:
        """Choice probability should depend on state."""
        torch.manual_seed(42)
        v0_net, v1_net, _, _ = solve_value_function(
            beta=1.0,
            gamma=0.1,
            delta=0.9,
            s_min=0.0,
            s_max=10.0,
            hidden_sizes=[16],
            learning_rate=0.1,
            batch_size=64,
            tolerance=0.01,
            max_iterations=500,
        )

        n_agents = 5000

        # Simulate at low state
        panel_low = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=1,
            s_init=1.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        # Simulate at high state
        panel_high = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=1,
            s_init=15.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        p1_low = panel_low.actions[:, 0].mean()
        p1_high = panel_high.actions[:, 0].mean()

        # At low state, should invest more (to build up state)
        # At high state, should invest less (already have high state)
        assert p1_low > p1_high


class TestStationaryDistribution:
    """Tests for ergodic properties of simulated data."""

    def test_state_stabilizes(self) -> None:
        """State distribution should stabilize over time."""
        torch.manual_seed(42)
        v0_net, v1_net, _, _ = solve_value_function(
            beta=1.0,
            gamma=0.1,
            delta=0.9,
            s_min=0.0,
            s_max=10.0,
            hidden_sizes=[16],
            learning_rate=0.1,
            batch_size=64,
            tolerance=0.01,
            max_iterations=500,
        )

        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=1000,
            n_periods=100,
            s_init=5.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )

        # Compare early and late state distributions
        early_mean = panel.states[:, 10:20].mean()
        late_mean = panel.states[:, 80:100].mean()

        early_std = panel.states[:, 10:20].std()
        late_std = panel.states[:, 80:100].std()

        # Means should be similar (within some tolerance)
        # This is a weak test - mainly checking it doesn't diverge
        assert abs(early_mean - late_mean) < 2.0
        assert abs(early_std - late_std) < 1.0


class TestIntegration:
    """Integration tests for full simulation pipeline."""

    def test_simulate_with_trained_policy_verify_transitions(self) -> None:
        """Simulate with trained policy, verify all state transitions are correct.
        
        This integration test:
        1. Trains a policy (value functions)
        2. Simulates many agents over many periods
        3. Verifies every state transition follows s' = (1-γ)s + a
        """
        # Train a policy
        torch.manual_seed(42)
        beta = 1.0
        gamma = 0.1
        delta = 0.9
        
        v0_net, v1_net, losses, _ = solve_value_function(
            beta=beta,
            gamma=gamma,
            delta=delta,
            s_min=0.0,
            s_max=20.0,
            hidden_sizes=[16],
            learning_rate=0.1,
            batch_size=64,
            tolerance=0.01,
            max_iterations=1000,
        )
        
        # Verify training converged
        assert losses[-1] < losses[0], "Training should reduce loss"
        
        # Simulate with trained policy
        n_agents = 500
        n_periods = 100
        
        panel = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=n_agents,
            n_periods=n_periods,
            s_init=10.0,
            beta=beta,
            gamma=gamma,
            seed=123,
        )
        
        # Verify all state transitions
        for t in range(n_periods - 1):
            s_t = panel.states[:, t]
            a_t = panel.actions[:, t]
            s_next_actual = panel.states[:, t + 1]
            s_next_expected = (1 - gamma) * s_t + a_t
            
            np.testing.assert_allclose(
                s_next_actual,
                s_next_expected,
                rtol=1e-6,
                err_msg=f"State transition failed at period {t}",
            )
        
        # Verify all rewards
        for t in range(n_periods):
            s_t = panel.states[:, t]
            a_t = panel.actions[:, t]
            r_actual = panel.rewards[:, t]
            r_expected = beta * np.log(1 + s_t) - a_t
            
            np.testing.assert_allclose(
                r_actual,
                r_expected,
                rtol=1e-5,  # Relaxed due to float32/float64 conversion
                err_msg=f"Reward computation failed at period {t}",
            )
        
        # Verify actions are consistent with choice probabilities
        # At each state, empirical choice frequency should match theoretical
        # (We check this statistically over all observations)
        s_flat = panel.states.flatten()
        a_flat = panel.actions.flatten()
        
        # Bin states and check empirical vs theoretical probabilities
        n_bins = 10
        bin_edges = np.linspace(s_flat.min(), s_flat.max(), n_bins + 1)
        
        for i in range(n_bins):
            mask = (s_flat >= bin_edges[i]) & (s_flat < bin_edges[i + 1])
            if mask.sum() < 100:
                continue  # Skip bins with too few observations
            
            # Empirical P(a=1|s in bin)
            empirical_p1 = a_flat[mask].mean()
            
            # Theoretical P(a=1|s) at bin center
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            s_tensor = torch.tensor([bin_center], dtype=torch.float32)
            with torch.no_grad():
                _, p1_theory = compute_choice_probability(v0_net, v1_net, s_tensor)
                theoretical_p1 = p1_theory.item()
            
            # Should be within sampling error (generous tolerance)
            assert abs(empirical_p1 - theoretical_p1) < 0.1, (
                f"Choice probability mismatch at s≈{bin_center:.1f}: "
                f"empirical={empirical_p1:.3f}, theoretical={theoretical_p1:.3f}"
            )

    def test_different_initial_states_converge(self) -> None:
        """Agents starting at different states should converge to similar distribution."""
        torch.manual_seed(42)
        
        v0_net, v1_net, _, _ = solve_value_function(
            beta=1.0,
            gamma=0.1,
            delta=0.9,
            s_min=0.0,
            s_max=20.0,
            hidden_sizes=[16],
            learning_rate=0.1,
            batch_size=64,
            tolerance=0.01,
            max_iterations=500,
        )
        
        # Simulate from low initial state
        panel_low = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=500,
            n_periods=100,
            s_init=1.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )
        
        # Simulate from high initial state
        panel_high = simulate_mdp_panel(
            v0_net=v0_net,
            v1_net=v1_net,
            n_agents=500,
            n_periods=100,
            s_init=15.0,
            beta=1.0,
            gamma=0.1,
            seed=42,
        )
        
        # Initial states are very different
        assert abs(panel_low.states[:, 0].mean() - panel_high.states[:, 0].mean()) > 10
        
        # Late-period states should be more similar (convergence to stationary dist)
        late_mean_low = panel_low.states[:, 80:].mean()
        late_mean_high = panel_high.states[:, 80:].mean()
        
        # Should have converged closer together
        assert abs(late_mean_low - late_mean_high) < 3.0

