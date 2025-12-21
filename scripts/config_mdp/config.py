"""Configuration for MDP solver and simulator.

This module defines all parameters used by solve_mdp and simulate_mdp.
Both scripts import from here to ensure consistent configurations.
"""

# =============================================================================
# Model Parameters
# =============================================================================

# Reward function: u(s, a) = beta * log(1 + s) - a
beta = 1.0  # Utility coefficient

# State transition: s' = (1 - gamma) * s + a
gamma = 0.1  # State decay rate

# Discount factor
delta = 0.95

# State space bounds
s_min = 0.0
s_max = 20.0

# =============================================================================
# Neural Network Architecture
# =============================================================================

# Hidden layer sizes for monotonic networks
hidden_sizes = [16]

# =============================================================================
# Training Parameters
# =============================================================================

learning_rate = 0.1  # Higher LR works well with target networks
batch_size = 256
tolerance = 0.01  # Convergence tolerance (RMSE)
max_iterations = 20000
target_update_freq = 10  # Target network update frequency

