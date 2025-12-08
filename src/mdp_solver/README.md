# MDP Solver

A Python package for solving Markov Decision Processes with Type-I Extreme Value shocks.

## Features

- **Continuous state space** with grid discretization
- **Binary action choice** (a ∈ {0, 1})
- **Type-I extreme value shocks** with closed-form solutions
- **Monotonic neural network** approximation for value functions

## Installation

```bash
cd src/mdp_solver
uv sync
```

## Usage

### Value Function Iteration

```python
import numpy as np
from mdp_solver import solve_value_function, compute_choice_probability

# Define state grid
state_grid = np.linspace(start=0.0, stop=10.0, num=200)

# Solve the model
V, v0, v1, n_iter = solve_value_function(
    state_grid=state_grid,
    beta=1.0,      # Reward coefficient
    gamma=0.1,     # State decay rate
    delta=0.95,    # Discount factor
)

# Compute choice probabilities
prob_a1 = compute_choice_probability(v0=v0, v1=v1)
```

### Neural Network Approximation

```python
import torch
from mdp_solver import ValueFunctionApproximator

# Create monotonic network
model = ValueFunctionApproximator(hidden_sizes=[32, 32])

# Evaluate at states
s = torch.linspace(start=0.0, end=10.0, steps=100)
v0, v1 = model(s)
prob = model.compute_choice_probability(s)
```

## Model

- **Reward**: u(s, a) = βs - a
- **Transition**: s' = (1 - γ)s + a
- **Shocks**: εₐ ~ Type-I Extreme Value (iid)

## License

MIT

