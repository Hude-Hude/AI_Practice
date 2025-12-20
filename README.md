# MDP Solver with Neural Network Value Iteration

A Python implementation for solving **Dynamic Discrete Choice (DDC) Models** with Type-I Extreme Value shocks using monotonic neural networks.

## Project Overview

This project implements a neural network-based approach to solve Markov Decision Processes (MDPs) where agents make binary choices under uncertainty. The model captures:

- **State-dependent utility**: $u(s, a) = \beta \log(1 + s) - a$
- **Deterministic transitions**: $s' = (1 - \gamma)s + a$
- **Type-I Extreme Value shocks**: Leading to logit choice probabilities
- **Infinite horizon**: With discount factor $\delta$

## Features

- **Monotonic Neural Networks**: Enforces economically sensible behavior (value increasing in state)
- **Target Networks**: Stabilizes training by providing fixed Bellman targets
- **Fixed State Grid**: Eliminates sampling variance for consistent optimization
- **Bellman Residual Minimization**: Trains networks to satisfy the fixed-point equation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd MDP_Practice

# Install dependencies with uv
uv sync
```

## Usage

### Quick Start

```python
from mdp_solver import solve_value_function, compute_choice_probability

# Solve the MDP
v0_net, v1_net, losses, n_iter = solve_value_function(
    beta=0.5,           # Utility coefficient
    gamma=0.1,          # Depreciation rate
    delta=0.95,         # Discount factor
    s_min=0.0,
    s_max=20.0,
    hidden_sizes=[16],  # Single hidden layer
    learning_rate=0.1,
    batch_size=256,
    tolerance=0.01,
    max_iterations=5000,
    target_update_freq=10,
)

# Compute choice probabilities
p0, p1 = compute_choice_probability(v0_net, v1_net, states)
```

### Running the Quarto Report

```bash
cd scripts/solve_mdp
quarto preview solve_mdp.qmd
```

## Project Structure

```
MDP_Practice/
├── src/mdp_solver/          # Main implementation
│   ├── mdp_solver.py        # Neural network solver
│   └── __init__.py          # Package exports
├── test/mdp_solver/         # Unit tests
│   ├── test_network.py      # Network component tests
│   ├── test_solver.py       # Solver tests
│   ├── test_model.py        # Model primitive tests
│   └── test_value_function.py
├── scripts/solve_mdp/       # Quarto report
│   └── solve_mdp.qmd        # Interactive analysis
└── README.md
```

## Running Tests

```bash
uv run pytest test/ -v
```

## Model Details

### Bellman Equation

The choice-specific value function satisfies:
$$v(s, a) = \beta \log(1 + s) - a + \delta \bar{V}((1-\gamma)s + a)$$

where $\bar{V}(s) = \log(\exp(v(s,0)) + \exp(v(s,1)))$ is the integrated value function.

### Choice Probabilities

With Type-I Extreme Value shocks, optimal choice probabilities follow the logit formula:
$$P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$

### Neural Network Architecture

- **Monotonic constraint**: Non-negative weights via softplus parameterization
- **Activation**: Tanh (bounded, zero-centered)
- **Architecture**: Single hidden layer with 16 units

---

## Development Findings & Technical Notes

### Key Issues Discovered and Resolved

#### 1. Discount Factor Parameter Confusion

**Problem**: The Bellman target computation used `delta * V(s')` but `delta` was incorrectly set to 10.0 (investment cost) instead of 0.95 (discount factor).

**Symptom**: Loss exploded exponentially during training.

**Solution**: Corrected parameter usage - `delta` is the discount factor (0.95), `beta` is the utility coefficient.

#### 2. Activation Function Choice

**Problem**: Original implementation used Softplus activation, which is unbounded (0, ∞).

**Symptom**: Value functions collapsed to constant outputs regardless of state.

**Analysis**:
- Softplus outputs are always positive
- With positive weights (monotonicity constraint), signals compound and grow
- Network saturates to near-constant values

**Solution**: Switched to Tanh activation:
- Bounded to [-1, 1]
- Zero-centered
- Better gradient flow through layers

#### 3. Network Architecture Depth

**Problem**: Deep networks [64, 64] failed to learn state-dependent value functions.

**Symptom**: Flat value functions, constant choice probabilities (~0.27).

**Analysis**:
- Too many parameters (4,225) for a simple 1D monotonic function
- Signal dilution through multiple layers
- Tanh saturation causing vanishing gradients

**Solution**: Use shallow networks [16] (single hidden layer):
- Converges in ~600 iterations (vs 5000+ for deep)
- RMSE ~0.01 (vs 0.36 for deep)
- Correct state-dependent behavior

#### 4. Moving Target Problem

**Problem**: Computing Bellman targets from the same networks being updated creates instability.

**Solution**: Implemented target networks (frozen copies updated periodically):
- Policy networks: Updated every iteration
- Target networks: Updated every `target_update_freq` iterations
- Provides stable regression targets

#### 5. Sampling Variance

**Problem**: Random state sampling each iteration introduces noise in the loss signal.

**Solution**: Fixed state grid using `torch.linspace`:
- Same states evaluated every iteration
- Consistent loss signal
- Faster, more stable convergence

### Results Summary

With correct implementation:

| Parameter | Value |
|-----------|-------|
| β (utility) | 0.5 |
| γ (decay) | 0.1 |
| δ (discount) | 0.95 |
| Architecture | [16] |

| Metric | Result |
|--------|--------|
| Convergence | ~600 iterations |
| Final RMSE | ~0.01 |
| P(a=1) at s=0 | ~0.58 (invest) |
| P(a=1) at s=20 | ~0.32 (don't invest) |

**Economic Interpretation**: The model correctly learns an (S,s)-type inventory policy where investment is attractive at low states (high marginal utility) and unattractive at high states (diminishing returns).

### Known Limitations

1. **Boundary smoothness**: Value functions show slight irregularities at low state boundaries
2. **Extrapolation**: Network behavior beyond training range [s_min, s_max] not guaranteed

### Future Improvements

- Investigate boundary smoothness issues
- Add gradient clipping for additional stability
- Explore alternative monotonic architectures
- Add support for continuous action spaces
