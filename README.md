# MDP Practice: Dynamic Discrete Choice Models

A complete Python implementation for **solving**, **simulating**, and **estimating** Dynamic Discrete Choice (DDC) Models using neural network value iteration.

## Project Overview

This project implements a full structural estimation pipeline for Markov Decision Processes (MDPs):

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MDP Solver    │────▶│   MDP Simulator  │────▶│   MDP Estimator  │
│  (Neural Net    │     │  (Monte Carlo    │     │  (Structural     │
│   Value Iter.)  │     │   Panel Data)    │     │   MLE/NFXP)      │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

### The Model

Agents make binary choices (invest or not) to maximize lifetime utility:

- **State transition** (deterministic): $s_{t+1} = (1 - \gamma)s_t + a_t$
- **Flow utility**: $u(s, a) = \beta \log(1 + s) - a$
- **Type-I Extreme Value shocks**: Leading to logit choice probabilities
- **Infinite horizon**: With discount factor $\delta$

### Key Features

| Component | Method | Key Innovation |
|-----------|--------|----------------|
| **Solver** | Neural Network Value Iteration | Monotonic networks for economic consistency |
| **Simulator** | Monte Carlo Panel Data | Logit choice probabilities from solved values |
| **Estimator** | Two-Step NFXP | OLS for γ, calibrated δ, MLE for β |

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/Hude-Hude/MDP_Practice.git
cd MDP_Practice
uv sync
```

### Run the Full Pipeline

```bash
# 1. Solve the MDP (trains neural network value functions)
cd scripts/solve_mdp && quarto render solve_mdp.qmd

# 2. Simulate panel data (Monte Carlo simulation)
cd ../simulate_mdp && quarto render simulate_mdp.qmd

# 3. Estimate structural parameters (recover true values)
cd ../estimate_mdp && quarto render estimate_mdp.qmd
```

### Interactive Development

```bash
# Start live preview servers
quarto preview scripts/solve_mdp/solve_mdp.qmd --port 7629 &
quarto preview scripts/simulate_mdp/simulate_mdp.qmd --port 7630 &
quarto preview scripts/estimate_mdp/estimate_mdp.qmd --port 7628 &
```

---

## Project Structure

```
MDP_Practice/
├── src/
│   ├── mdp_solver/           # Neural network value iteration
│   │   └── mdp_solver.py     # solve_value_function(), compute_choice_probability()
│   ├── mdp_simulator/        # Monte Carlo simulation
│   │   └── mdp_simulator.py  # simulate_mdp_panel(), PanelData
│   └── mdp_estimator/        # Structural estimation
│       └── mdp_estimator.py  # estimate_two_step(), compute_log_likelihood()
├── scripts/
│   ├── config_mdp/           # Centralized parameters
│   │   └── config.py         # β, γ, δ, network architecture
│   ├── mdp_utils/            # Shared plotting utilities
│   │   └── plotting.py       # Consistent visualization
│   ├── solve_mdp/            # Solver report
│   │   └── solve_mdp.qmd
│   ├── simulate_mdp/         # Simulation report
│   │   └── simulate_mdp.qmd
│   └── estimate_mdp/         # Estimation report
│       └── estimate_mdp.qmd
├── output/
│   ├── solve_mdp/            # Trained networks, losses
│   └── simulate_mdp/         # Panel data (states, actions, rewards)
└── test/                     # Unit and integration tests
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp_solver`)

Solves the Bellman equation using neural network approximation:

```python
from mdp_solver import solve_value_function, compute_choice_probability

v0_net, v1_net, losses, n_iter = solve_value_function(
    beta=1.0, gamma=0.1, delta=0.95,
    s_min=0.0, s_max=10.0,
    hidden_sizes=[16],
)
```

**Key innovations:**
- Monotonic neural networks (non-negative weights via softplus)
- Target networks for training stability
- Fixed state grid for consistent optimization

### 2. MDP Simulator (`src/mdp_simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp_simulator import simulate_mdp_panel

panel = simulate_mdp_panel(
    v0_net, v1_net,
    beta=1.0, gamma=0.1, delta=0.95,
    n_agents=100, n_periods=100,
    s_init=np.random.uniform(0, 10, 100),
)
# panel.states: (100, 100), panel.actions: (100, 100), panel.rewards: (100, 100)
```

### 3. MDP Estimator (`src/mdp_estimator`)

Recovers structural parameters from simulated data:

```python
from mdp_estimator import estimate_two_step

result = estimate_two_step(
    data=panel,
    delta_calibrated=0.95,  # Calibrated
    solver_params=solver_params,
    beta_bounds=(0.5, 1.5),
    n_points=20,
)
# result.gamma_hat: 0.1 (exact from OLS)
# result.beta_hat: ~1.03 (within 2 SE of true)
```

**Two-step approach addresses weak identification:**
1. **Step 1**: Estimate γ directly from transitions (OLS, exact recovery)
2. **Step 2**: Estimate β via 1D grid search (δ calibrated)

---

## Key Results

### Parameter Recovery

| Parameter | True | Estimated | Method |
|-----------|------|-----------|--------|
| β (reward) | 1.0 | ~1.03 | MLE (1D grid) |
| γ (decay) | 0.1 | 0.1 (exact) | OLS |
| δ (discount) | 0.95 | 0.95 | Calibrated |

### Estimation Challenge Solved

**Problem**: Joint 3-parameter estimation suffered from weak identification (flat likelihood surface).

**Solution**: Two-step estimation exploiting model structure:
- γ identified from deterministic transitions alone
- δ typically calibrated in structural estimation
- β well-identified when γ and δ are fixed

---

## Running Tests

```bash
# All tests
uv run pytest test/ -v

# Fast unit tests only
uv run pytest test/ -v -m "not slow"

# Slow integration tests
uv run pytest test/ -v -m slow
```

---

## Configuration

All model parameters are centralized in `scripts/config_mdp/config.py`:

```python
# Model parameters
beta = 1.0      # Reward coefficient
gamma = 0.1     # State decay rate
delta = 0.95    # Discount factor
s_min = 0.0     # State lower bound
s_max = 10.0    # State upper bound

# Network architecture
hidden_sizes = [16]        # Single hidden layer
learning_rate = 0.1        # Higher LR works well with target networks
batch_size = 256
tolerance = 0.01           # Convergence tolerance (RMSE)
max_iterations = 20000
target_update_freq = 10    # Target network update frequency
```

---

## Technical Notes

### Solver Implementation

- **Monotonic constraint**: Non-negative weights via softplus parameterization
- **Activation**: Tanh (bounded, zero-centered)
- **Target networks**: Updated every `target_update_freq` iterations
- **Convergence**: ~3000 iterations, RMSE < 0.001

### Estimation Implementation

- **NFXP structure**: Outer loop (parameter search) + Inner loop (DP solution)
- **Warm-starting**: Pre-trained networks speed up inner loop convergence
- **Standard errors**: Numerical Hessian at the optimum

---

## License

MIT License
