# MDP Practice: Dynamic Discrete Choice Models

A complete Python implementation for **solving**, **simulating**, and **estimating** Dynamic Discrete Choice (DDC) Models using neural network value iteration.

## Project Overview

This project implements a full structural estimation pipeline for Markov Decision Processes (MDPs):

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MDP Solver    │────▶│   MDP Simulator  │────▶│   MDP Estimator  │
│  (Neural Net    │     │  (Monte Carlo    │     │  (Two-Step       │
│   Value Iter.)  │     │   Panel Data)    │     │   NFXP-MLE)      │
└─────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   Value Functions          Panel Data             Recovered θ̂
   (v₀, v₁ networks)    (100×100 obs)         (β̂, γ̂, δ calibrated)
```

### The Model

Agents make binary choices (invest or not) to maximize lifetime utility:

- **State transition** (deterministic): $s_{t+1} = (1 - \gamma)s_t + a_t$
- **Flow utility**: $u(s, a) = \beta \log(1 + s) - a$
- **Type-I Extreme Value shocks**: Leading to logit choice probabilities
- **Infinite horizon**: With discount factor $\delta$

---

## Key Results

### Stage 1: Solver
- Converges in ~3,000 iterations with RMSE < 0.01
- Monotonic neural networks enforce economic structure (value increasing in state)
- Comparative statics show sensible responses to β and γ

### Stage 2: Simulator
- Generates 100 agents × 100 periods = 10,000 observations
- All validation checks pass:
  - ✓ Choice probabilities match theoretical logit
  - ✓ State transitions follow $s' = (1-\gamma)s + a$
  - ✓ State distribution converges to stationarity

### Stage 3: Estimator
Successfully recovers structural parameters using two-step estimation:

| Parameter | True | Estimated | Method | Recovery |
|-----------|------|-----------|--------|----------|
| **β** (reward) | 1.0 | ~1.03 | MLE (1D grid) | ✓ Within 2 SE |
| **γ** (decay) | 0.1 | 0.1 | OLS | ✓ Exact (R²=1.0) |
| **δ** (discount) | 0.95 | 0.95 | Calibrated | — |

**Key insight**: Joint 3-parameter estimation suffers from weak identification (flat likelihood surface). The two-step approach exploits model structure: γ is identified from transitions alone, δ is typically calibrated, leaving only β for MLE.

---

## Project Structure

```
MDP_Practice/
├── src/
│   ├── mdp_solver/           # Neural network value iteration
│   ├── mdp_simulator/        # Monte Carlo simulation
│   └── mdp_estimator/        # Two-step structural estimation
├── scripts/
│   ├── config_mdp/           # Centralized parameters
│   │   └── config.py         # β=1.0, γ=0.1, δ=0.95, etc.
│   ├── mdp_utils/            # Shared plotting utilities
│   ├── solve_mdp/            # Solver report with comparative statics
│   ├── simulate_mdp/         # Simulation report with diagnostics
│   └── estimate_mdp/         # Estimation report with identification analysis
├── output/
│   ├── solve_mdp/            # Trained networks (v0_net.pt, v1_net.pt)
│   └── simulate_mdp/         # Panel data (states.npy, actions.npy)
├── test/                     # Unit and integration tests
└── docs/conversation/        # Development session logs
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp_solver`)

Solves the Bellman equation using neural network approximation with **target networks** for stability:

```python
from mdp_solver import solve_value_function

v0_net, v1_net, losses, n_iter = solve_value_function(
    beta=1.0, gamma=0.1, delta=0.95,
    s_min=0.0, s_max=10.0,
    hidden_sizes=[16],  # Single hidden layer
)
```

**Key features:**
- **Monotonic networks**: Non-negative weights via softplus ensure $v(s,a)$ increasing in $s$
- **Target networks**: Frozen copies updated periodically for stable Bellman targets
- **Tanh activation**: Bounded outputs prevent explosive growth
- **Comparative statics**: Reports analyze effects of varying β (0→2) and γ (0→0.1)

### 2. MDP Simulator (`src/mdp_simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp_simulator import simulate_mdp_panel

panel = simulate_mdp_panel(
    v0_net, v1_net,
    n_agents=100, n_periods=100,
    s_init=np.random.uniform(0, 10, 100),
    beta=1.0, gamma=0.1,
)
```

**Validation checks:**
- Choice probability RMSE < 0.05
- Calibration RMSE < 0.03
- State convergence to stationarity
- Exact transition verification
- Reward consistency

### 3. MDP Estimator (`src/mdp_estimator`)

Recovers structural parameters using **two-step estimation**:

```python
from mdp_estimator import estimate_two_step

result = estimate_two_step(
    data=panel,
    delta_calibrated=0.95,
    solver_params=solver_params,
    beta_bounds=(0.5, 1.5),
    n_points=20,
)
# result.gamma_hat: 0.1 (exact from OLS)
# result.beta_hat: ~1.03 (within 2 SE)
```

**Two-step approach:**
1. **Step 1**: Estimate γ from transitions via OLS (exact, R²=1.0)
2. **Step 2**: Estimate β via 1D grid search (δ calibrated, ~20 evaluations)

This decomposition addresses weak identification between β and δ.

---

## Configuration

All parameters are centralized in `scripts/config_mdp/config.py`:

```python
# Model parameters
beta = 1.0      # Reward coefficient
gamma = 0.1     # State decay rate  
delta = 0.95    # Discount factor
s_min = 0.0     # State lower bound
s_max = 10.0    # State upper bound

# Network architecture
hidden_sizes = [16]        # Single hidden layer
learning_rate = 0.1        # Higher LR works with target networks
batch_size = 256
tolerance = 0.01           # Convergence tolerance (RMSE)
max_iterations = 20000
target_update_freq = 10    # Target network sync frequency
```

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

## Technical Notes

### Why Two-Step Estimation?

Joint estimation of (β, γ, δ) reveals **weak identification**:
- Likelihood at true θ: -6875
- Likelihood at estimated θ: -6874
- Difference: <2 log-units (essentially indistinguishable)

The two-step approach exploits model structure:
- **γ enters only transitions**: $s' = (1-\gamma)s + a$ → OLS gives exact recovery
- **δ is typically calibrated**: Standard practice in structural IO
- **β becomes 1D**: Grid search is robust to flat surfaces

### NFXP Structure

Each likelihood evaluation requires solving the DP (inner loop):
```
FOR each candidate β:
    1. Solve DP: v0, v1 = solve_value_function(β, γ_ols, δ_calib)
    2. Compute choice probabilities from (v0, v1)  
    3. Evaluate log-likelihood
RETURN β with highest likelihood
```

**Optimization**: Warm-starting from pre-trained networks speeds convergence.

---

## License

MIT License
