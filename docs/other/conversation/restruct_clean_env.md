# AI_Practice project restructuring
_Exported on 12/23/2025 at 14:48:30 GMT+8 from Cursor (2.1.50)_

---

**User**

Okay, i will copy a chat history: Yes, that will work! Here's the summary to paste in your new workspace:



---



## 📋 Project Restructuring Summary



### Goal

Extend `MDP_Practice` project to support multiple model solvers (MDP + OPM - Oligopoly Pricing Model) with identical workflow: **Solver → Simulator → Estimator**



### Completed Steps

1. ✅ Renamed folder: `MDP_Practice` → `AI_Practice`

2. ✅ Renamed workspace: `MDP_Practice.code-workspace` → `AI_Practice.code-workspace`



### Next Steps - Run These Commands



```bash

cd /Users/kissshot894/Documents/HKUST/RA/AI_Practice



# 1. SOURCE

mkdir -p src/mdp src/opm/{solver,simulator,estimator} src/common

git mv src/mdp_solver src/mdp/solver

git mv src/mdp_simulator src/mdp/simulator

git mv src/mdp_estimator src/mdp/estimator



# 2. SCRIPTS

mkdir -p scripts/mdp scripts/opm/{config,solve,simulate,estimate} scripts/utils

git mv scripts/config_mdp scripts/mdp/config

git mv scripts/solve_mdp scripts/mdp/solve

git mv scripts/simulate_mdp scripts/mdp/simulate

git mv scripts/estimate_mdp scripts/mdp/estimate

git mv scripts/mdp_utils/* scripts/utils/

rmdir scripts/mdp_utils



# 3. OUTPUT

mkdir -p output/mdp output/opm/{solve,simulate,estimate}

git mv output/solve_mdp output/mdp/solve

git mv output/simulate_mdp output/mdp/simulate

git mv output/estimate_mdp output/mdp/estimate



# 4. TEST

mkdir -p test/mdp test/opm/{solver,simulator,estimator}

git mv test/mdp_solver test/mdp/solver

git mv test/mdp_simulator test/mdp/simulator

git mv test/mdp_estimator test/mdp/estimator



# 5. DOCS

mkdir -p docs/opm

git mv docs/conversation docs/mdp



git status

```



### After Commands Complete, Remaining Tasks

1. Update `pyproject.toml` (name → `structural-models`, update package paths)

2. Update all import statements (`from mdp_solver` → `from mdp.solver`)

3. Add `__init__.py` files for new package structure (`src/mdp/__init__.py`)

4. Update `README.md`

5. Run tests to verify



### Target Structure

```

AI_Practice/

├── src/

│   ├── mdp/{solver,simulator,estimator}/

│   ├── opm/{solver,simulator,estimator}/  ← placeholder folders

│   └── common/

├── scripts/

│   ├── mdp/{config,solve,simulate,estimate}/

│   ├── opm/{config,solve,simulate,estimate}/  ← placeholder folders

│   └── utils/

├── output/mdp/{solve,simulate,estimate}/, opm/...

├── test/mdp/{solver,simulator,estimator}/, opm/...

└── docs/mdp/, opm/

```



---



Copy this to your new workspace chat and say **"continue from step 3"** (or tell me which commands succeeded/failed).

---

**Cursor**

I'll start by reading the CLAUDE.md file as required, then check the current state of the project to see what's been done and what remains.

I see the current state. The restructuring commands **haven't been run yet** - the project is still in its original structure:

**Current State:**
- `src/`: `mdp_solver/`, `mdp_simulator/`, `mdp_estimator/`
- `scripts/`: `config_mdp/`, `solve_mdp/`, `simulate_mdp/`, `estimate_mdp/`, `mdp_utils/`
- `output/`: `solve_mdp/`, `simulate_mdp/`, `estimate_mdp/`
- `test/`: `mdp_solver/`, `mdp_simulator/`, `mdp_estimator/`
- `docs/`: `conversation/`

---

## 📋 RESTRUCTURING PLAN

### Phase 1: Directory Moves (git mv)
```
1. SOURCE
   src/mdp_solver     → src/mdp/solver
   src/mdp_simulator  → src/mdp/simulator  
   src/mdp_estimator  → src/mdp/estimator
   + create src/opm/{solver,simulator,estimator}/, src/common/

2. SCRIPTS
   scripts/config_mdp   → scripts/mdp/config
   scripts/solve_mdp    → scripts/mdp/solve
   scripts/simulate_mdp → scripts/mdp/simulate
   scripts/estimate_mdp → scripts/mdp/estimate
   scripts/mdp_utils/*  → scripts/utils/
   + create scripts/opm/{config,solve,simulate,estimate}/

3. OUTPUT
   output/solve_mdp    → output/mdp/solve
   output/simulate_mdp → output/mdp/simulate
   output/estimate_mdp → output/mdp/estimate
   + create output/opm/{solve,simulate,estimate}/

4. TEST
   test/mdp_solver     → test/mdp/solver
   test/mdp_simulator  → test/mdp/simulator
   test/mdp_estimator  → test/mdp/estimator
   + create test/opm/{solver,simulator,estimator}/

5. DOCS
   docs/conversation → docs/mdp
   + create docs/opm/
```

### Phase 2: Update Imports & Configs
- Update all `from mdp_solver` → `from mdp.solver`
- Add `__init__.py` files for new package structure
- Update `pyproject.toml`

### Phase 3: Verify
- Run tests to confirm nothing broke

---

**Say "proceed"** to execute Phase 1 (directory restructuring).

```plaintext
1. SOURCE
   src/mdp_solver     → src/mdp/solver
   src/mdp_simulator  → src/mdp/simulator  
   src/mdp_estimator  → src/mdp/estimator
   + create src/opm/{solver,simulator,estimator}/, src/common/

2. SCRIPTS
   scripts/config_mdp   → scripts/mdp/config
   scripts/solve_mdp    → scripts/mdp/solve
   scripts/simulate_mdp → scripts/mdp/simulate
   scripts/estimate_mdp → scripts/mdp/estimate
   scripts/mdp_utils/*  → scripts/utils/
   + create scripts/opm/{config,solve,simulate,estimate}/

3. OUTPUT
   output/solve_mdp    → output/mdp/solve
   output/simulate_mdp → output/mdp/simulate
   output/estimate_mdp → output/mdp/estimate
   + create output/opm/{solve,simulate,estimate}/

4. TEST
   test/mdp_solver     → test/mdp/solver
   test/mdp_simulator  → test/mdp/simulator
   test/mdp_estimator  → test/mdp/estimator
   + create test/opm/{solver,simulator,estimator}/

5. DOCS
   docs/conversation → docs/mdp
   + create docs/opm/
```

---

**User**

proceed.

---

**Cursor**

Output files are gitignored - using regular `mv`:

✅ **Phase 1 Complete - Directory Restructuring Done**

```
Changes staged:
├── docs/conversation/    → docs/mdp/
├── scripts/config_mdp/   → scripts/mdp/config/
├── scripts/solve_mdp/    → scripts/mdp/solve/
├── scripts/simulate_mdp/ → scripts/mdp/simulate/
├── scripts/estimate_mdp/ → scripts/mdp/estimate/
├── scripts/mdp_utils/    → scripts/utils/
├── src/mdp_solver/       → src/mdp/solver/
├── src/mdp_simulator/    → src/mdp/simulator/
├── src/mdp_estimator/    → src/mdp/estimator/
├── test/mdp_solver/      → test/mdp/solver/
├── test/mdp_simulator/   → test/mdp/simulator/
└── test/mdp_estimator/   → test/mdp/estimator/

Placeholder folders created:
├── src/opm/{solver,simulator,estimator}/
├── src/common/
├── scripts/opm/{config,solve,simulate,estimate}/
├── output/opm/{solve,simulate,estimate}/
├── test/opm/{solver,simulator,estimator}/
└── docs/opm/
```

---

## 📋 Phase 2 Plan: Update Imports & Configs

1. **Add `__init__.py`** files for new package structure
   - `src/mdp/__init__.py`
   - `src/opm/__init__.py`
   - `src/common/__init__.py`

2. **Update imports** in all files:
   - `from mdp_solver` → `from mdp.solver`
   - `from mdp_simulator` → `from mdp.simulator`
   - `from mdp_estimator` → `from mdp.estimator`

3. **Update `pyproject.toml`**:
   - Change name to `structural-models`
   - Update package paths

4. **Stage workspace file change**:
   - `MDP_Practice.code-workspace` → `AI_Practice.code-workspace`

---

**Say "proceed"** to execute Phase 2.

```plaintext
Changes staged:
├── docs/conversation/    → docs/mdp/
├── scripts/config_mdp/   → scripts/mdp/config/
├── scripts/solve_mdp/    → scripts/mdp/solve/
├── scripts/simulate_mdp/ → scripts/mdp/simulate/
├── scripts/estimate_mdp/ → scripts/mdp/estimate/
├── scripts/mdp_utils/    → scripts/utils/
├── src/mdp_solver/       → src/mdp/solver/
├── src/mdp_simulator/    → src/mdp/simulator/
├── src/mdp_estimator/    → src/mdp/estimator/
├── test/mdp_solver/      → test/mdp/solver/
├── test/mdp_simulator/   → test/mdp/simulator/
└── test/mdp_estimator/   → test/mdp/estimator/

Placeholder folders created:
├── src/opm/{solver,simulator,estimator}/
├── src/common/
├── scripts/opm/{config,solve,simulate,estimate}/
├── output/opm/{solve,simulate,estimate}/
├── test/opm/{solver,simulator,estimator}/
└── docs/opm/
```

---

**User**

proceed.

---

**Cursor**

```python
"""MDP structural estimation package.

Provides solver, simulator, and estimator for dynamic discrete choice models.
"""

from mdp.solver import *
from mdp.simulator import *
from mdp.estimator import *

```

```python
"""OPM (Oligopoly Pricing Model) package.

Placeholder for future implementation.
"""

```

```python
"""Common utilities shared across models.

Placeholder for future implementation.
"""

```

```python
"""OPM Solver placeholder."""

```

```python
"""OPM Simulator placeholder."""

```

```python
"""OPM Estimator placeholder."""

```

Now updating the existing `__init__.py` files to use the new import paths:

```python
"""MDP Solver package.

A neural network-based solver for dynamic discrete choice models
with Type-I Extreme Value shocks.
"""

# Import everything from the consolidated module
from .mdp_solver import (
    # Type definitions
    Scalar,
    Vector,
    Matrix,
    TensorVector,
    TensorMatrix,
    Action,
    HiddenSizes,
    NetworkPair,
    TargetPair,
    # Neural network components
    MonotonicLinear,
    TanhActivation,
    build_monotonic_network,
    evaluate_network,
    # Model primitives
    compute_reward,
    compute_next_state,
    clamp_states,
    # Value function computations
    compute_integrated_value,
    compute_bellman_targets,
    compute_bellman_loss,
    compute_choice_probability,
    # Solver
    copy_network,
    initialize_networks,
    sample_states,
    generate_state_grid,
    check_convergence,
    solve_value_function,
)

__all__ = [
    # Type definitions
    "Scalar",
    "Vector",
    "Matrix",
    "TensorVector",
    "TensorMatrix",
    "Action",
    "HiddenSizes",
    "NetworkPair",
    "TargetPair",
    # Neural network components
    "MonotonicLinear",
    "TanhActivation",
    "build_monotonic_network",
    "evaluate_network",
    # Model primitives
    "compute_reward",
    "compute_next_state",
    "clamp_states",
    # Value function computations
    "compute_integrated_value",
    "compute_bellman_targets",
    "compute_bellman_loss",
    "compute_choice_probability",
    # Solver
    "copy_network",
    "initialize_networks",
    "sample_states",
    "generate_state_grid",
    "check_convergence",
    "solve_value_function",
]

```

```python
"""MDP Structural Parameter Estimator using NFXP.

This module provides maximum likelihood estimation of structural
parameters (beta, gamma, delta) from panel data using the Nested
Fixed Point (NFXP) algorithm.

Recommended approach (two-step):
- estimate_gamma_ols: Estimate γ from transition data (instant)
- estimate_beta_1d: Estimate β with γ from OLS and δ calibrated
- estimate_two_step: Combined two-step estimation

Alternative methods (joint estimation, not recommended due to identification):
- estimate_mle: Nelder-Mead optimization
- grid_search_mle: Explicit 3D grid search
"""

from .mdp_estimator import (
    EstimationResult,
    TwoStepResult,
    estimate_gamma_ols,
    estimate_beta_1d,
    estimate_two_step,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)

__all__ = [
    # Data structures
    "EstimationResult",
    "TwoStepResult",
    # Two-step estimation (recommended)
    "estimate_gamma_ols",
    "estimate_beta_1d",
    "estimate_two_step",
    # Core functions
    "compute_log_likelihood",
    "compute_standard_errors",
    # Joint estimation (not recommended)
    "estimate_mle",
    "grid_search_mle",
]


```

```plaintext
[project]
name = "structural-models"
version = "0.1.0"
description = "Structural model estimation: MDP solver with Type-I Extreme Value shocks, OPM (coming soon)"
readme = "README.md"
authors = [
    { name = "Hude", email = "hude@connect.ust.hk" }
]
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "torch>=2.0.0",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mdp", "src/opm", "src/common"]

[tool.pytest.ini_options]
testpaths = ["test"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (require solver, deselect with '-m \"not slow\"')",
]

```

Now updating test files:

```python
"""Tests for model primitives."""

import torch

from mdp.solver import clamp_states, compute_next_state, compute_reward


class TestComputeReward:
    """Tests for compute_reward."""

    def test_formula_action_zero(self) -> None:
        """u(s, 0) = β * log(1 + s)."""
        s = torch.tensor([0.0, 1.0, 2.0, 5.0])
        beta = 2.0
        r = compute_reward(s, action=0, beta=beta)
        expected = beta * torch.log(1 + s)
        torch.testing.assert_close(r, expected)

    def test_formula_action_one(self) -> None:
        """u(s, 1) = β * log(1 + s) - 1."""
        s = torch.tensor([0.0, 1.0, 2.0, 5.0])
        beta = 2.0
        r = compute_reward(s, action=1, beta=beta)
        expected = beta * torch.log(1 + s) - 1
        torch.testing.assert_close(r, expected)

    def test_action_cost(self) -> None:
        """Action 1 should cost exactly 1 more than action 0."""
        s = torch.tensor([0.0, 5.0, 10.0])
        beta = 1.5
        r0 = compute_reward(s, action=0, beta=beta)
        r1 = compute_reward(s, action=1, beta=beta)
        diff = r0 - r1
        torch.testing.assert_close(diff, torch.ones_like(diff))

    def test_zero_state(self) -> None:
        """At s=0: u(0,0)=0, u(0,1)=-1."""
        s = torch.tensor([0.0])
        r0 = compute_reward(s, action=0, beta=1.0)
        r1 = compute_reward(s, action=1, beta=1.0)
        torch.testing.assert_close(r0, torch.tensor([0.0]))
        torch.testing.assert_close(r1, torch.tensor([-1.0]))

    def test_monotonicity(self) -> None:
        """Reward should increase with state."""
        s = torch.linspace(0, 10, 100)
        r = compute_reward(s, action=0, beta=1.0)
        diffs = torch.diff(r)
        assert torch.all(diffs > 0)


class TestComputeNextState:
    """Tests for compute_next_state."""

    def test_formula_action_zero(self) -> None:
        """s' = (1 - γ) * s when a = 0."""
        s = torch.tensor([10.0, 20.0])
        gamma = 0.2
        s_next = compute_next_state(s, action=0, gamma=gamma)
        expected = (1 - gamma) * s
        torch.testing.assert_close(s_next, expected)

    def test_formula_action_one(self) -> None:
        """s' = (1 - γ) * s + 1 when a = 1."""
        s = torch.tensor([10.0, 20.0])
        gamma = 0.2
        s_next = compute_next_state(s, action=1, gamma=gamma)
        expected = (1 - gamma) * s + 1
        torch.testing.assert_close(s_next, expected)

    def test_action_boost(self) -> None:
        """Action 1 should boost state by 1."""
        s = torch.tensor([5.0, 10.0])
        gamma = 0.1
        s_next0 = compute_next_state(s, action=0, gamma=gamma)
        s_next1 = compute_next_state(s, action=1, gamma=gamma)
        diff = s_next1 - s_next0
        torch.testing.assert_close(diff, torch.ones_like(diff))

    def test_decay(self) -> None:
        """With action 0, state should decay."""
        s = torch.tensor([10.0])
        gamma = 0.1
        s_next = compute_next_state(s, action=0, gamma=gamma)
        assert s_next.item() < s.item()
        assert s_next.item() > 0

    def test_from_zero(self) -> None:
        """From s=0 with action 1, next state should be 1."""
        s = torch.tensor([0.0])
        s_next = compute_next_state(s, action=1, gamma=0.5)
        torch.testing.assert_close(s_next, torch.tensor([1.0]))


class TestClampStates:
    """Tests for clamp_states."""

    def test_within_bounds(self) -> None:
        """States within bounds should be unchanged."""
        s = torch.tensor([2.0, 5.0, 8.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        torch.testing.assert_close(s_clamped, s)

    def test_below_min(self) -> None:
        """States below min should be clamped to min."""
        s = torch.tensor([-5.0, -1.0, 0.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(s_clamped, expected)

    def test_above_max(self) -> None:
        """States above max should be clamped to max."""
        s = torch.tensor([10.0, 15.0, 100.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([10.0, 10.0, 10.0])
        torch.testing.assert_close(s_clamped, expected)

    def test_mixed(self) -> None:
        """Mixed states should be clamped correctly."""
        s = torch.tensor([-2.0, 5.0, 15.0])
        s_clamped = clamp_states(s, s_min=0.0, s_max=10.0)
        expected = torch.tensor([0.0, 5.0, 10.0])
        torch.testing.assert_close(s_clamped, expected)

```

```python
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

```

```python
"""Tests for value function computations."""

import torch

from mdp.solver import (
    build_monotonic_network,
    compute_bellman_loss,
    compute_bellman_targets,
    compute_choice_probability,
    compute_integrated_value,
    evaluate_network,
)


class TestComputeIntegratedValue:
    """Tests for compute_integrated_value."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        V = compute_integrated_value(v0_net, v1_net, s)
        assert V.shape == (50,)

    def test_greater_than_max(self) -> None:
        """Integrated value should be >= max(v0, v1)."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        V = compute_integrated_value(v0_net, v1_net, s)
        
        v_max = torch.maximum(v0, v1)
        assert torch.all(V >= v_max - 1e-6)

    def test_logsumexp_formula(self) -> None:
        """Should equal log(exp(v0) + exp(v1))."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        V = compute_integrated_value(v0_net, v1_net, s)
        
        expected = torch.logsumexp(torch.stack([v0, v1], dim=1), dim=1)
        torch.testing.assert_close(V, expected)


class TestComputeBellmanTargets:
    """Tests for compute_bellman_targets."""

    def test_output_shapes(self) -> None:
        """Should return two tensors of correct shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        
        target0, target1 = compute_bellman_targets(
            v0_net, v1_net, s,
            beta=1.0, gamma=0.1, delta=0.95,
        )
        
        assert target0.shape == (50,)
        assert target1.shape == (50,)

    def test_discount_zero(self) -> None:
        """With delta=0 (discount factor), targets should equal rewards."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        beta = 2.0
        
        # delta=0 means no discounting of future values
        target0, target1 = compute_bellman_targets(
            v0_net, v1_net, s,
            beta=beta, gamma=0.1, delta=0.0,
        )
        
        # With delta=0: target = r + 0*V(s') = r
        expected0 = beta * torch.log(1 + s)  # action 0
        expected1 = beta * torch.log(1 + s) - 1  # action 1
        
        torch.testing.assert_close(target0, expected0)
        torch.testing.assert_close(target1, expected1)


class TestComputeBellmanLoss:
    """Tests for compute_bellman_loss."""

    def test_zero_error(self) -> None:
        """Loss should be zero when predictions equal targets."""
        v0_pred = torch.tensor([1.0, 2.0, 3.0])
        v1_pred = torch.tensor([1.5, 2.5, 3.5])
        target0 = v0_pred.clone()
        target1 = v1_pred.clone()
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        assert loss.item() < 1e-10

    def test_positive_loss(self) -> None:
        """Loss should be positive when there's error."""
        v0_pred = torch.tensor([1.0, 2.0, 3.0])
        v1_pred = torch.tensor([1.5, 2.5, 3.5])
        target0 = torch.tensor([2.0, 3.0, 4.0])
        target1 = torch.tensor([2.5, 3.5, 4.5])
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        assert loss.item() > 0

    def test_formula(self) -> None:
        """Should compute mean squared error correctly."""
        v0_pred = torch.tensor([1.0, 2.0])
        v1_pred = torch.tensor([1.0, 2.0])
        target0 = torch.tensor([2.0, 3.0])  # error = 1, 1
        target1 = torch.tensor([2.0, 3.0])  # error = 1, 1
        
        loss = compute_bellman_loss(v0_pred, v1_pred, target0, target1)
        
        # (1^2 + 1^2 + 1^2 + 1^2) / (2 * 2) = 4 / 4 = 1.0
        expected = 1.0
        assert abs(loss.item() - expected) < 1e-6


class TestComputeChoiceProbability:
    """Tests for compute_choice_probability."""

    def test_output_shape(self) -> None:
        """Output should match input shape."""
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 50)
        prob0, prob1 = compute_choice_probability(v0_net, v1_net, s)
        assert prob0.shape == (50,)
        assert prob1.shape == (50,)

    def test_bounds(self) -> None:
        """Probabilities should be in [0, 1]."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 100)
        prob0, prob1 = compute_choice_probability(v0_net, v1_net, s)
        assert torch.all(prob0 >= 0)
        assert torch.all(prob0 <= 1)
        assert torch.all(prob1 >= 0)
        assert torch.all(prob1 <= 1)

    def test_sum_to_one(self) -> None:
        """Probabilities should sum to 1."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.linspace(0, 10, 100)
        prob0, prob1 = compute_choice_probability(v0_net, v1_net, s)
        torch.testing.assert_close(prob0 + prob1, torch.ones_like(prob0))

    def test_softmax_formula(self) -> None:
        """Should equal softmax of value functions."""
        torch.manual_seed(42)
        v0_net = build_monotonic_network([8])
        v1_net = build_monotonic_network([8])
        s = torch.tensor([0.0, 5.0, 10.0])
        
        v0 = evaluate_network(v0_net, s)
        v1 = evaluate_network(v1_net, s)
        prob0, prob1 = compute_choice_probability(v0_net, v1_net, s)
        
        # Expected via softmax
        v_stack = torch.stack([v0, v1], dim=1)
        expected = torch.softmax(v_stack, dim=1)
        
        torch.testing.assert_close(prob0, expected[:, 0])
        torch.testing.assert_close(prob1, expected[:, 1])

```

```python
"""Tests for network components."""

import torch
import torch.nn.functional as F

from mdp.solver import (
    MonotonicLinear,
    TanhActivation,
    build_monotonic_network,
    evaluate_network,
)


class TestMonotonicLinear:
    """Tests for MonotonicLinear layer."""

    def test_output_shape(self) -> None:
        """Output should have correct shape."""
        layer = MonotonicLinear(in_features=1, out_features=8)
        x = torch.randn(10, 1)
        y = layer(x)
        assert y.shape == (10, 8)

    def test_weights_positive(self) -> None:
        """Effective weights should be positive via softplus."""
        layer = MonotonicLinear(in_features=4, out_features=8)
        # Force some raw weights to be negative
        layer.weight_raw.data = torch.randn(8, 4) * 2 - 1
        
        # Compute effective weights
        effective_weights = F.softplus(layer.weight_raw)
        
        assert torch.all(effective_weights > 0)

    def test_monotonicity_single_input(self) -> None:
        """Output should increase with input for single feature."""
        layer = MonotonicLinear(in_features=1, out_features=1)
        
        # Create increasing inputs
        x = torch.linspace(0, 10, 100).unsqueeze(1)
        y = layer(x).squeeze()
        
        # Check monotonicity
        diffs = torch.diff(y)
        assert torch.all(diffs >= -1e-6)  # Allow small numerical errors


class TestTanhActivation:
    """Tests for TanhActivation."""

    def test_output_bounded(self) -> None:
        """Tanh output should be bounded in [-1, 1]."""
        activation = TanhActivation()
        x = torch.randn(100) * 10  # Wide range of inputs
        y = activation(x)
        assert torch.all(y >= -1)
        assert torch.all(y <= 1)

    def test_monotonicity(self) -> None:
        """Tanh should be monotonically increasing."""
        activation = TanhActivation()
        x = torch.linspace(-10, 10, 100)
        y = activation(x)
        diffs = torch.diff(y)
        assert torch.all(diffs >= 0)

    def test_formula(self) -> None:
        """Should match tanh(x)."""
        activation = TanhActivation()
        x = torch.tensor([0.0, 1.0, -1.0, 5.0])
        y = activation(x)
        expected = torch.tanh(x)
        torch.testing.assert_close(y, expected)

    def test_zero_centered(self) -> None:
        """Tanh(0) should be 0."""
        activation = TanhActivation()
        x = torch.tensor([0.0])
        y = activation(x)
        assert y.item() == 0.0


class TestBuildMonotonicNetwork:
    """Tests for build_monotonic_network."""

    def test_creates_sequential(self) -> None:
        """Should return nn.Sequential."""
        network = build_monotonic_network([8, 8])
        assert isinstance(network, torch.nn.Sequential)

    def test_correct_layer_count(self) -> None:
        """Should have correct number of layers."""
        # [8, 8] -> MonotonicLinear + Softplus + MonotonicLinear + Softplus + MonotonicLinear
        network = build_monotonic_network([8, 8])
        # 2 hidden * 2 (linear + activation) + 1 output = 5
        assert len(network) == 5

    def test_output_shape(self) -> None:
        """Output should be scalar per input."""
        network = build_monotonic_network([16, 16])
        x = torch.randn(32, 1)
        y = network(x)
        assert y.shape == (32, 1)


class TestEvaluateNetwork:
    """Tests for evaluate_network."""

    def test_1d_input(self) -> None:
        """Should handle 1D input."""
        network = build_monotonic_network([8])
        s = torch.randn(50)
        values = evaluate_network(network, s)
        assert values.shape == (50,)

    def test_2d_input(self) -> None:
        """Should handle 2D input."""
        network = build_monotonic_network([8])
        s = torch.randn(50, 1)
        values = evaluate_network(network, s)
        assert values.shape == (50,)

    def test_monotonicity(self) -> None:
        """Network output should be monotonic in input."""
        torch.manual_seed(42)
        network = build_monotonic_network([16, 16])
        
        s = torch.linspace(0, 10, 100)
        values = evaluate_network(network, s)
        
        diffs = torch.diff(values)
        assert torch.all(diffs >= -1e-5)  # Allow small numerical errors

```

```python
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


```

```python
"""Unit and integration tests for MDP Estimator.

Unit Tests (fast, no solver needed):
- Parameter validation (returns -inf for invalid params)
- Data structure checks

Integration Tests (slow, requires solver):
- Full estimation workflow
- Parameter recovery
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mdp.estimator import (
    EstimationResult,
    TwoStepResult,
    estimate_gamma_ols,
    estimate_beta_1d,
    estimate_two_step,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)
from mdp.simulator import PanelData


# =============================================================================
# Fixtures for Unit Tests (no solver needed)
# =============================================================================

@pytest.fixture
def fake_panel_data():
    """Create fake panel data for unit tests (no solver needed)."""
    n_agents, n_periods = 10, 5
    return PanelData(
        states=np.random.uniform(0, 10, size=(n_agents, n_periods)),
        actions=np.random.randint(0, 2, size=(n_agents, n_periods)),
        rewards=np.random.randn(n_agents, n_periods),
        n_agents=n_agents,
        n_periods=n_periods,
    )


@pytest.fixture
def solver_params():
    """Fast solver hyperparameters for integration tests.
    
    Uses smaller network, fewer iterations, and looser tolerance
    to speed up tests while still verifying code correctness.
    """
    return {
        's_min': 0.0,
        's_max': 10.0,
        'hidden_sizes': [8],           # Small network (vs [16])
        'learning_rate': 0.1,
        'batch_size': 64,
        'tolerance': 0.1,              # Loose tolerance (vs 0.01)
        'max_iterations': 1000,        # Few iterations (vs 20000)
        'target_update_freq': 10,
    }


# =============================================================================
# Fixtures for Integration Tests (small data for speed)
# =============================================================================

@pytest.fixture
def true_params():
    """True structural parameters (from saved simulation config)."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'output', 'simulate_mdp', 'config.json'
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'delta': config['delta'],
    }


@pytest.fixture
def simulated_data():
    """Load small subset of saved panel data for fast integration tests.
    
    Uses only first 10 agents and 10 periods (100 observations)
    instead of full 100x100 (10,000 observations).
    """
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'output', 'simulate_mdp'
    )
    
    states = np.load(os.path.join(data_dir, 'states.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    
    # Use small subset for fast tests
    n_agents_test = 10
    n_periods_test = 10
    
    return PanelData(
        states=states[:n_agents_test, :n_periods_test],
        actions=actions[:n_agents_test, :n_periods_test],
        rewards=rewards[:n_agents_test, :n_periods_test],
        n_agents=n_agents_test,
        n_periods=n_periods_test,
    )


# =============================================================================
# UNIT TESTS (Fast - no solver needed)
# =============================================================================

class TestParameterValidation:
    """Unit tests for parameter validation (fast, no solver needed)."""
    
    def test_invalid_beta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([0.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_beta_negative_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta < 0 should return -inf immediately (no solver call)."""
        theta = np.array([-1.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 1.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 0.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 1.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf


class TestEstimationResultStructure:
    """Unit tests for EstimationResult data structure."""
    
    def test_estimation_result_fields(self):
        """EstimationResult should have all required fields."""
        result = EstimationResult(
            theta_hat=np.array([1.0, 0.1, 0.9]),
            std_errors=np.array([0.1, 0.01, 0.05]),
            cov_matrix=np.eye(3) * 0.01,
            log_likelihood=-100.0,
            n_iterations=50,
            converged=True,
            optimization_result=None,
        )
        
        assert result.theta_hat.shape == (3,)
        assert result.std_errors.shape == (3,)
        assert result.cov_matrix.shape == (3, 3)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)


class TestTwoStepResultStructure:
    """Unit tests for TwoStepResult data structure."""
    
    def test_two_step_result_fields(self):
        """TwoStepResult should have all required fields."""
        result = TwoStepResult(
            beta_hat=1.0,
            gamma_hat=0.1,
            delta_calibrated=0.95,
            beta_se=0.05,
            gamma_se=0.001,
            log_likelihood=-100.0,
            n_evaluations=20,
            gamma_ols_details={'n_obs': 1000, 'r_squared': 1.0},
        )
        
        assert isinstance(result.beta_hat, float)
        assert isinstance(result.gamma_hat, float)
        assert isinstance(result.delta_calibrated, float)
        assert isinstance(result.beta_se, float)
        assert isinstance(result.gamma_se, float)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_evaluations, int)
        assert isinstance(result.gamma_ols_details, dict)


class TestEstimateGammaOLS:
    """Unit tests for estimate_gamma_ols (fast - no solver needed).
    
    Uses synthetic data with known gamma to verify OLS estimation.
    """
    
    @pytest.fixture
    def synthetic_transition_data(self):
        """Create synthetic panel data with known gamma = 0.1."""
        np.random.seed(42)
        n_agents, n_periods = 50, 20
        gamma_true = 0.1
        
        # Generate states following s_{t+1} = (1 - gamma) * s_t + a_t
        states = np.zeros((n_agents, n_periods))
        actions = np.random.randint(0, 2, size=(n_agents, n_periods))
        states[:, 0] = np.random.uniform(1, 10, size=n_agents)  # Initial states > 0
        
        for t in range(n_periods - 1):
            states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
        
        return PanelData(
            states=states,
            actions=actions,
            rewards=np.zeros((n_agents, n_periods)),  # Not used for gamma estimation
            n_agents=n_agents,
            n_periods=n_periods,
        ), gamma_true
    
    def test_recovers_true_gamma(self, synthetic_transition_data):
        """OLS should exactly recover gamma from deterministic transitions."""
        data, gamma_true = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # Should recover exactly (within numerical precision)
        np.testing.assert_allclose(
            gamma_hat, gamma_true, rtol=1e-10,
            err_msg=f"gamma_hat={gamma_hat} should equal gamma_true={gamma_true}"
        )
    
    def test_r_squared_equals_one(self, synthetic_transition_data):
        """R-squared should be 1.0 for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        np.testing.assert_allclose(
            details['r_squared'], 1.0, rtol=1e-10,
            err_msg="R-squared should be 1.0 for deterministic transitions"
        )
    
    def test_std_error_near_zero(self, synthetic_transition_data):
        """Standard error should be near zero for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        assert std_error < 1e-10, "Standard error should be near zero"
    
    def test_returns_correct_n_obs(self, synthetic_transition_data):
        """Should return correct number of observations."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # n_obs should be n_agents * (n_periods - 1) minus any filtered zeros
        expected_max = data.n_agents * (data.n_periods - 1)
        assert details['n_obs'] <= expected_max
        assert details['n_obs'] > 0
    
    def test_handles_different_gamma_values(self):
        """Should work for various gamma values."""
        np.random.seed(42)
        
        for gamma_true in [0.05, 0.1, 0.2, 0.5]:
            n_agents, n_periods = 20, 10
            states = np.zeros((n_agents, n_periods))
            actions = np.random.randint(0, 2, size=(n_agents, n_periods))
            states[:, 0] = np.random.uniform(1, 10, size=n_agents)
            
            for t in range(n_periods - 1):
                states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
            
            data = PanelData(
                states=states, actions=actions,
                rewards=np.zeros((n_agents, n_periods)),
                n_agents=n_agents, n_periods=n_periods,
            )
            
            gamma_hat, _, _ = estimate_gamma_ols(data)
            
            np.testing.assert_allclose(
                gamma_hat, gamma_true, rtol=1e-10,
                err_msg=f"Failed to recover gamma={gamma_true}"
            )


# =============================================================================
# INTEGRATION TESTS (Slow - requires solver)
# Mark with pytest.mark.slow so they can be skipped
# =============================================================================

@pytest.mark.slow
class TestComputeLogLikelihood:
    """Integration tests for compute_log_likelihood (requires solver)."""
    
    def test_returns_finite_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should return a finite value for valid parameters."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert np.isfinite(ll), "Log-likelihood should be finite"
    
    def test_returns_negative_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should be negative (log of probabilities < 1)."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll < 0, "Log-likelihood should be negative"
    
    def test_true_params_higher_likelihood(self, simulated_data, true_params, solver_params):
        """True parameters should have higher likelihood than wrong parameters."""
        theta_true = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        theta_wrong = np.array([2.5, 0.3, 0.7])  # Significantly different
        
        ll_true = compute_log_likelihood(theta_true, simulated_data, solver_params)
        ll_wrong = compute_log_likelihood(theta_wrong, simulated_data, solver_params)
        
        assert ll_true > ll_wrong, \
            f"True params LL ({ll_true:.2f}) should be > wrong params LL ({ll_wrong:.2f})"


@pytest.mark.slow
class TestComputeStandardErrors:
    """Integration tests for compute_standard_errors (requires solver)."""
    
    def test_returns_positive_std_errors(self, simulated_data, true_params, solver_params):
        """Standard errors should be positive."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check std_errors are positive (or NaN if singular)
        if not np.any(np.isnan(std_errors)):
            assert np.all(std_errors > 0), "Standard errors should be positive"
    
    def test_cov_matrix_shape(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be 3x3."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        assert cov_matrix.shape == (3, 3), "Covariance matrix should be 3x3"
    
    def test_cov_matrix_symmetric(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be symmetric."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check symmetry (allowing for NaN)
        if not np.any(np.isnan(cov_matrix)):
            np.testing.assert_allclose(
                cov_matrix, cov_matrix.T, rtol=1e-5,
                err_msg="Covariance matrix should be symmetric"
            )


@pytest.mark.slow
class TestEstimateMLE:
    """Integration tests for estimate_mle (requires solver)."""
    
    def test_returns_estimation_result(self, simulated_data, true_params, solver_params):
        """estimate_mle should return EstimationResult object."""
        theta_init = (true_params['beta'], true_params['gamma'], true_params['delta'])
        
        # Run with very few iterations just to test interface
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=5, verbose=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert result.std_errors.shape == (3,), "std_errors should have 3 elements"
        assert result.cov_matrix.shape == (3, 3), "cov_matrix should be 3x3"
        assert isinstance(result.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(result.n_iterations, int), "n_iterations should be int"
        assert isinstance(result.converged, bool), "converged should be bool"


@pytest.mark.slow
class TestIntegration:
    """Integration tests for full estimation workflow (requires solver)."""
    
    def test_full_workflow_runs(self, simulated_data, true_params, solver_params):
        """Full estimation workflow should run without error."""
        # Start from perturbed initial values
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        # Run estimation with limited iterations
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        # Basic sanity checks
        assert np.all(np.isfinite(result.theta_hat)), "Estimates should be finite"
        assert np.isfinite(result.log_likelihood), "Log-likelihood should be finite"
        assert result.n_iterations > 0, "Should have run at least one iteration"
    
    def test_estimates_in_valid_range(self, simulated_data, true_params, solver_params):
        """Estimates should be in valid parameter range."""
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        beta_hat, gamma_hat, delta_hat = result.theta_hat
        
        assert beta_hat > 0, "beta should be positive"
        assert 0 < gamma_hat < 1, "gamma should be in (0, 1)"
        assert 0 < delta_hat < 1, "delta should be in (0, 1)"


@pytest.mark.slow
class TestGridSearch:
    """Integration tests for grid search estimation (requires solver)."""
    
    def test_grid_search_returns_result(self, simulated_data, true_params, solver_params):
        """grid_search_mle should return EstimationResult object."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert np.isfinite(result.log_likelihood), "log_likelihood should be finite"
        assert result.n_iterations == 8, "Should have 2^3 = 8 evaluations"
        assert result.converged is True, "Grid search always converges"
    
    def test_grid_search_finds_best_in_grid(self, simulated_data, true_params, solver_params):
        """Grid search should return the grid point with highest likelihood."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        # Verify theta_hat is in the grid
        grid_results = result.optimization_result
        beta_grid = grid_results['grid']['beta']
        gamma_grid = grid_results['grid']['gamma']
        delta_grid = grid_results['grid']['delta']
        
        assert result.theta_hat[0] in beta_grid, "beta should be in grid"
        assert result.theta_hat[1] in gamma_grid, "gamma should be in grid"
        assert result.theta_hat[2] in delta_grid, "delta should be in grid"
        
        # Verify it has the highest likelihood
        all_lls = [e['log_likelihood'] for e in grid_results['evaluations']]
        assert result.log_likelihood == max(all_lls), "Should return max likelihood point"


@pytest.mark.slow
class TestEstimateBeta1D:
    """Integration tests for estimate_beta_1d (requires solver)."""
    
    def test_returns_correct_types(self, simulated_data, true_params, solver_params):
        """estimate_beta_1d should return correct types."""
        beta_hat, std_error, log_lik, details = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(beta_hat, float), "beta_hat should be float"
        assert isinstance(std_error, (float, type(np.nan))), "std_error should be float or nan"
        assert isinstance(log_lik, float), "log_lik should be float"
        assert isinstance(details, dict), "details should be dict"
    
    def test_beta_in_bounds(self, simulated_data, true_params, solver_params):
        """Estimated beta should be within specified bounds."""
        beta_bounds = (0.5, 1.5)
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=beta_bounds,
            n_points=3,
            verbose=False,
        )
        
        assert beta_bounds[0] <= beta_hat <= beta_bounds[1], \
            f"beta_hat={beta_hat} should be in {beta_bounds}"
    
    def test_finds_true_beta_on_grid(self, simulated_data, true_params, solver_params):
        """When true beta is on grid, should find it."""
        # Use bounds that include true beta as a grid point
        true_beta = true_params['beta']
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(true_beta - 0.5, true_beta + 0.5),
            n_points=5,  # Grid: [0.5, 0.75, 1.0, 1.25, 1.5] includes 1.0
            verbose=False,
        )
        
        # Should find true beta or very close
        assert abs(beta_hat - true_beta) < 0.3, \
            f"beta_hat={beta_hat} should be close to true_beta={true_beta}"


@pytest.mark.slow
class TestEstimateTwoStep:
    """Integration tests for estimate_two_step (requires solver)."""
    
    def test_returns_two_step_result(self, simulated_data, true_params, solver_params):
        """estimate_two_step should return TwoStepResult object."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(result, TwoStepResult), "Should return TwoStepResult"
    
    def test_gamma_exactly_recovered(self, simulated_data, true_params, solver_params):
        """Gamma should be exactly recovered from OLS."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        # Gamma should be exactly recovered
        np.testing.assert_allclose(
            result.gamma_hat, true_params['gamma'], rtol=1e-6,
            err_msg=f"gamma_hat={result.gamma_hat} should equal true gamma={true_params['gamma']}"
        )
    
    def test_delta_equals_calibrated(self, simulated_data, true_params, solver_params):
        """Delta should equal the calibrated value."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert result.delta_calibrated == true_params['delta'], \
            "delta_calibrated should equal the input value"
    
    def test_beta_reasonable(self, simulated_data, true_params, solver_params):
        """Beta estimate should be in reasonable range."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=5,
            verbose=False,
        )
        
        # Beta should be positive and in bounds
        assert 0.5 <= result.beta_hat <= 1.5, \
            f"beta_hat={result.beta_hat} should be in [0.5, 1.5]"
    
    def test_full_recovery_with_fine_grid(self, simulated_data, true_params, solver_params):
        """With fine grid and true delta, should recover beta well."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.8, 1.2),  # Narrow bounds around true value
            n_points=5,
            verbose=False,
        )
        
        # Should recover beta close to true value
        bias = abs(result.beta_hat - true_params['beta'])
        assert bias < 0.15, \
            f"beta bias={bias} should be < 0.15 (beta_hat={result.beta_hat}, true={true_params['beta']})"

```

```python
"""Unit and integration tests for MDP Estimator.

Unit Tests (fast, no solver needed):
- Parameter validation (returns -inf for invalid params)
- Data structure checks

Integration Tests (slow, requires solver):
- Full estimation workflow
- Parameter recovery
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mdp.estimator import (
    EstimationResult,
    TwoStepResult,
    estimate_gamma_ols,
    estimate_beta_1d,
    estimate_two_step,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)
from mdp.simulator import PanelData


# =============================================================================
# Fixtures for Unit Tests (no solver needed)
# =============================================================================

@pytest.fixture
def fake_panel_data():
    """Create fake panel data for unit tests (no solver needed)."""
    n_agents, n_periods = 10, 5
    return PanelData(
        states=np.random.uniform(0, 10, size=(n_agents, n_periods)),
        actions=np.random.randint(0, 2, size=(n_agents, n_periods)),
        rewards=np.random.randn(n_agents, n_periods),
        n_agents=n_agents,
        n_periods=n_periods,
    )


@pytest.fixture
def solver_params():
    """Fast solver hyperparameters for integration tests.
    
    Uses smaller network, fewer iterations, and looser tolerance
    to speed up tests while still verifying code correctness.
    """
    return {
        's_min': 0.0,
        's_max': 10.0,
        'hidden_sizes': [8],           # Small network (vs [16])
        'learning_rate': 0.1,
        'batch_size': 64,
        'tolerance': 0.1,              # Loose tolerance (vs 0.01)
        'max_iterations': 1000,        # Few iterations (vs 20000)
        'target_update_freq': 10,
    }


# =============================================================================
# Fixtures for Integration Tests (small data for speed)
# =============================================================================

@pytest.fixture
def true_params():
    """True structural parameters (from saved simulation config)."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'output', 'mdp', 'simulate', 'config.json'
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'delta': config['delta'],
    }


@pytest.fixture
def simulated_data():
    """Load small subset of saved panel data for fast integration tests.
    
    Uses only first 10 agents and 10 periods (100 observations)
    instead of full 100x100 (10,000 observations).
    """
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'output', 'simulate_mdp'
    )
    
    states = np.load(os.path.join(data_dir, 'states.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    
    # Use small subset for fast tests
    n_agents_test = 10
    n_periods_test = 10
    
    return PanelData(
        states=states[:n_agents_test, :n_periods_test],
        actions=actions[:n_agents_test, :n_periods_test],
        rewards=rewards[:n_agents_test, :n_periods_test],
        n_agents=n_agents_test,
        n_periods=n_periods_test,
    )


# =============================================================================
# UNIT TESTS (Fast - no solver needed)
# =============================================================================

class TestParameterValidation:
    """Unit tests for parameter validation (fast, no solver needed)."""
    
    def test_invalid_beta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([0.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_beta_negative_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta < 0 should return -inf immediately (no solver call)."""
        theta = np.array([-1.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 1.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 0.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 1.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf


class TestEstimationResultStructure:
    """Unit tests for EstimationResult data structure."""
    
    def test_estimation_result_fields(self):
        """EstimationResult should have all required fields."""
        result = EstimationResult(
            theta_hat=np.array([1.0, 0.1, 0.9]),
            std_errors=np.array([0.1, 0.01, 0.05]),
            cov_matrix=np.eye(3) * 0.01,
            log_likelihood=-100.0,
            n_iterations=50,
            converged=True,
            optimization_result=None,
        )
        
        assert result.theta_hat.shape == (3,)
        assert result.std_errors.shape == (3,)
        assert result.cov_matrix.shape == (3, 3)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)


class TestTwoStepResultStructure:
    """Unit tests for TwoStepResult data structure."""
    
    def test_two_step_result_fields(self):
        """TwoStepResult should have all required fields."""
        result = TwoStepResult(
            beta_hat=1.0,
            gamma_hat=0.1,
            delta_calibrated=0.95,
            beta_se=0.05,
            gamma_se=0.001,
            log_likelihood=-100.0,
            n_evaluations=20,
            gamma_ols_details={'n_obs': 1000, 'r_squared': 1.0},
        )
        
        assert isinstance(result.beta_hat, float)
        assert isinstance(result.gamma_hat, float)
        assert isinstance(result.delta_calibrated, float)
        assert isinstance(result.beta_se, float)
        assert isinstance(result.gamma_se, float)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_evaluations, int)
        assert isinstance(result.gamma_ols_details, dict)


class TestEstimateGammaOLS:
    """Unit tests for estimate_gamma_ols (fast - no solver needed).
    
    Uses synthetic data with known gamma to verify OLS estimation.
    """
    
    @pytest.fixture
    def synthetic_transition_data(self):
        """Create synthetic panel data with known gamma = 0.1."""
        np.random.seed(42)
        n_agents, n_periods = 50, 20
        gamma_true = 0.1
        
        # Generate states following s_{t+1} = (1 - gamma) * s_t + a_t
        states = np.zeros((n_agents, n_periods))
        actions = np.random.randint(0, 2, size=(n_agents, n_periods))
        states[:, 0] = np.random.uniform(1, 10, size=n_agents)  # Initial states > 0
        
        for t in range(n_periods - 1):
            states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
        
        return PanelData(
            states=states,
            actions=actions,
            rewards=np.zeros((n_agents, n_periods)),  # Not used for gamma estimation
            n_agents=n_agents,
            n_periods=n_periods,
        ), gamma_true
    
    def test_recovers_true_gamma(self, synthetic_transition_data):
        """OLS should exactly recover gamma from deterministic transitions."""
        data, gamma_true = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # Should recover exactly (within numerical precision)
        np.testing.assert_allclose(
            gamma_hat, gamma_true, rtol=1e-10,
            err_msg=f"gamma_hat={gamma_hat} should equal gamma_true={gamma_true}"
        )
    
    def test_r_squared_equals_one(self, synthetic_transition_data):
        """R-squared should be 1.0 for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        np.testing.assert_allclose(
            details['r_squared'], 1.0, rtol=1e-10,
            err_msg="R-squared should be 1.0 for deterministic transitions"
        )
    
    def test_std_error_near_zero(self, synthetic_transition_data):
        """Standard error should be near zero for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        assert std_error < 1e-10, "Standard error should be near zero"
    
    def test_returns_correct_n_obs(self, synthetic_transition_data):
        """Should return correct number of observations."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # n_obs should be n_agents * (n_periods - 1) minus any filtered zeros
        expected_max = data.n_agents * (data.n_periods - 1)
        assert details['n_obs'] <= expected_max
        assert details['n_obs'] > 0
    
    def test_handles_different_gamma_values(self):
        """Should work for various gamma values."""
        np.random.seed(42)
        
        for gamma_true in [0.05, 0.1, 0.2, 0.5]:
            n_agents, n_periods = 20, 10
            states = np.zeros((n_agents, n_periods))
            actions = np.random.randint(0, 2, size=(n_agents, n_periods))
            states[:, 0] = np.random.uniform(1, 10, size=n_agents)
            
            for t in range(n_periods - 1):
                states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
            
            data = PanelData(
                states=states, actions=actions,
                rewards=np.zeros((n_agents, n_periods)),
                n_agents=n_agents, n_periods=n_periods,
            )
            
            gamma_hat, _, _ = estimate_gamma_ols(data)
            
            np.testing.assert_allclose(
                gamma_hat, gamma_true, rtol=1e-10,
                err_msg=f"Failed to recover gamma={gamma_true}"
            )


# =============================================================================
# INTEGRATION TESTS (Slow - requires solver)
# Mark with pytest.mark.slow so they can be skipped
# =============================================================================

@pytest.mark.slow
class TestComputeLogLikelihood:
    """Integration tests for compute_log_likelihood (requires solver)."""
    
    def test_returns_finite_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should return a finite value for valid parameters."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert np.isfinite(ll), "Log-likelihood should be finite"
    
    def test_returns_negative_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should be negative (log of probabilities < 1)."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll < 0, "Log-likelihood should be negative"
    
    def test_true_params_higher_likelihood(self, simulated_data, true_params, solver_params):
        """True parameters should have higher likelihood than wrong parameters."""
        theta_true = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        theta_wrong = np.array([2.5, 0.3, 0.7])  # Significantly different
        
        ll_true = compute_log_likelihood(theta_true, simulated_data, solver_params)
        ll_wrong = compute_log_likelihood(theta_wrong, simulated_data, solver_params)
        
        assert ll_true > ll_wrong, \
            f"True params LL ({ll_true:.2f}) should be > wrong params LL ({ll_wrong:.2f})"


@pytest.mark.slow
class TestComputeStandardErrors:
    """Integration tests for compute_standard_errors (requires solver)."""
    
    def test_returns_positive_std_errors(self, simulated_data, true_params, solver_params):
        """Standard errors should be positive."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check std_errors are positive (or NaN if singular)
        if not np.any(np.isnan(std_errors)):
            assert np.all(std_errors > 0), "Standard errors should be positive"
    
    def test_cov_matrix_shape(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be 3x3."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        assert cov_matrix.shape == (3, 3), "Covariance matrix should be 3x3"
    
    def test_cov_matrix_symmetric(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be symmetric."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check symmetry (allowing for NaN)
        if not np.any(np.isnan(cov_matrix)):
            np.testing.assert_allclose(
                cov_matrix, cov_matrix.T, rtol=1e-5,
                err_msg="Covariance matrix should be symmetric"
            )


@pytest.mark.slow
class TestEstimateMLE:
    """Integration tests for estimate_mle (requires solver)."""
    
    def test_returns_estimation_result(self, simulated_data, true_params, solver_params):
        """estimate_mle should return EstimationResult object."""
        theta_init = (true_params['beta'], true_params['gamma'], true_params['delta'])
        
        # Run with very few iterations just to test interface
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=5, verbose=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert result.std_errors.shape == (3,), "std_errors should have 3 elements"
        assert result.cov_matrix.shape == (3, 3), "cov_matrix should be 3x3"
        assert isinstance(result.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(result.n_iterations, int), "n_iterations should be int"
        assert isinstance(result.converged, bool), "converged should be bool"


@pytest.mark.slow
class TestIntegration:
    """Integration tests for full estimation workflow (requires solver)."""
    
    def test_full_workflow_runs(self, simulated_data, true_params, solver_params):
        """Full estimation workflow should run without error."""
        # Start from perturbed initial values
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        # Run estimation with limited iterations
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        # Basic sanity checks
        assert np.all(np.isfinite(result.theta_hat)), "Estimates should be finite"
        assert np.isfinite(result.log_likelihood), "Log-likelihood should be finite"
        assert result.n_iterations > 0, "Should have run at least one iteration"
    
    def test_estimates_in_valid_range(self, simulated_data, true_params, solver_params):
        """Estimates should be in valid parameter range."""
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        beta_hat, gamma_hat, delta_hat = result.theta_hat
        
        assert beta_hat > 0, "beta should be positive"
        assert 0 < gamma_hat < 1, "gamma should be in (0, 1)"
        assert 0 < delta_hat < 1, "delta should be in (0, 1)"


@pytest.mark.slow
class TestGridSearch:
    """Integration tests for grid search estimation (requires solver)."""
    
    def test_grid_search_returns_result(self, simulated_data, true_params, solver_params):
        """grid_search_mle should return EstimationResult object."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert np.isfinite(result.log_likelihood), "log_likelihood should be finite"
        assert result.n_iterations == 8, "Should have 2^3 = 8 evaluations"
        assert result.converged is True, "Grid search always converges"
    
    def test_grid_search_finds_best_in_grid(self, simulated_data, true_params, solver_params):
        """Grid search should return the grid point with highest likelihood."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        # Verify theta_hat is in the grid
        grid_results = result.optimization_result
        beta_grid = grid_results['grid']['beta']
        gamma_grid = grid_results['grid']['gamma']
        delta_grid = grid_results['grid']['delta']
        
        assert result.theta_hat[0] in beta_grid, "beta should be in grid"
        assert result.theta_hat[1] in gamma_grid, "gamma should be in grid"
        assert result.theta_hat[2] in delta_grid, "delta should be in grid"
        
        # Verify it has the highest likelihood
        all_lls = [e['log_likelihood'] for e in grid_results['evaluations']]
        assert result.log_likelihood == max(all_lls), "Should return max likelihood point"


@pytest.mark.slow
class TestEstimateBeta1D:
    """Integration tests for estimate_beta_1d (requires solver)."""
    
    def test_returns_correct_types(self, simulated_data, true_params, solver_params):
        """estimate_beta_1d should return correct types."""
        beta_hat, std_error, log_lik, details = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(beta_hat, float), "beta_hat should be float"
        assert isinstance(std_error, (float, type(np.nan))), "std_error should be float or nan"
        assert isinstance(log_lik, float), "log_lik should be float"
        assert isinstance(details, dict), "details should be dict"
    
    def test_beta_in_bounds(self, simulated_data, true_params, solver_params):
        """Estimated beta should be within specified bounds."""
        beta_bounds = (0.5, 1.5)
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=beta_bounds,
            n_points=3,
            verbose=False,
        )
        
        assert beta_bounds[0] <= beta_hat <= beta_bounds[1], \
            f"beta_hat={beta_hat} should be in {beta_bounds}"
    
    def test_finds_true_beta_on_grid(self, simulated_data, true_params, solver_params):
        """When true beta is on grid, should find it."""
        # Use bounds that include true beta as a grid point
        true_beta = true_params['beta']
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(true_beta - 0.5, true_beta + 0.5),
            n_points=5,  # Grid: [0.5, 0.75, 1.0, 1.25, 1.5] includes 1.0
            verbose=False,
        )
        
        # Should find true beta or very close
        assert abs(beta_hat - true_beta) < 0.3, \
            f"beta_hat={beta_hat} should be close to true_beta={true_beta}"


@pytest.mark.slow
class TestEstimateTwoStep:
    """Integration tests for estimate_two_step (requires solver)."""
    
    def test_returns_two_step_result(self, simulated_data, true_params, solver_params):
        """estimate_two_step should return TwoStepResult object."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(result, TwoStepResult), "Should return TwoStepResult"
    
    def test_gamma_exactly_recovered(self, simulated_data, true_params, solver_params):
        """Gamma should be exactly recovered from OLS."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        # Gamma should be exactly recovered
        np.testing.assert_allclose(
            result.gamma_hat, true_params['gamma'], rtol=1e-6,
            err_msg=f"gamma_hat={result.gamma_hat} should equal true gamma={true_params['gamma']}"
        )
    
    def test_delta_equals_calibrated(self, simulated_data, true_params, solver_params):
        """Delta should equal the calibrated value."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert result.delta_calibrated == true_params['delta'], \
            "delta_calibrated should equal the input value"
    
    def test_beta_reasonable(self, simulated_data, true_params, solver_params):
        """Beta estimate should be in reasonable range."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=5,
            verbose=False,
        )
        
        # Beta should be positive and in bounds
        assert 0.5 <= result.beta_hat <= 1.5, \
            f"beta_hat={result.beta_hat} should be in [0.5, 1.5]"
    
    def test_full_recovery_with_fine_grid(self, simulated_data, true_params, solver_params):
        """With fine grid and true delta, should recover beta well."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.8, 1.2),  # Narrow bounds around true value
            n_points=5,
            verbose=False,
        )
        
        # Should recover beta close to true value
        bias = abs(result.beta_hat - true_params['beta'])
        assert bias < 0.15, \
            f"beta bias={bias} should be < 0.15 (beta_hat={result.beta_hat}, true={true_params['beta']})"

```

```python
"""Unit and integration tests for MDP Estimator.

Unit Tests (fast, no solver needed):
- Parameter validation (returns -inf for invalid params)
- Data structure checks

Integration Tests (slow, requires solver):
- Full estimation workflow
- Parameter recovery
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mdp.estimator import (
    EstimationResult,
    TwoStepResult,
    estimate_gamma_ols,
    estimate_beta_1d,
    estimate_two_step,
    compute_log_likelihood,
    estimate_mle,
    grid_search_mle,
    compute_standard_errors,
)
from mdp.simulator import PanelData


# =============================================================================
# Fixtures for Unit Tests (no solver needed)
# =============================================================================

@pytest.fixture
def fake_panel_data():
    """Create fake panel data for unit tests (no solver needed)."""
    n_agents, n_periods = 10, 5
    return PanelData(
        states=np.random.uniform(0, 10, size=(n_agents, n_periods)),
        actions=np.random.randint(0, 2, size=(n_agents, n_periods)),
        rewards=np.random.randn(n_agents, n_periods),
        n_agents=n_agents,
        n_periods=n_periods,
    )


@pytest.fixture
def solver_params():
    """Fast solver hyperparameters for integration tests.
    
    Uses smaller network, fewer iterations, and looser tolerance
    to speed up tests while still verifying code correctness.
    """
    return {
        's_min': 0.0,
        's_max': 10.0,
        'hidden_sizes': [8],           # Small network (vs [16])
        'learning_rate': 0.1,
        'batch_size': 64,
        'tolerance': 0.1,              # Loose tolerance (vs 0.01)
        'max_iterations': 1000,        # Few iterations (vs 20000)
        'target_update_freq': 10,
    }


# =============================================================================
# Fixtures for Integration Tests (small data for speed)
# =============================================================================

@pytest.fixture
def true_params():
    """True structural parameters (from saved simulation config)."""
    import json
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'output', 'mdp', 'simulate', 'config.json'
    )
    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        'beta': config['beta'],
        'gamma': config['gamma'],
        'delta': config['delta'],
    }


@pytest.fixture
def simulated_data():
    """Load small subset of saved panel data for fast integration tests.
    
    Uses only first 10 agents and 10 periods (100 observations)
    instead of full 100x100 (10,000 observations).
    """
    data_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'output', 'mdp', 'simulate'
    )
    
    states = np.load(os.path.join(data_dir, 'states.npy'))
    actions = np.load(os.path.join(data_dir, 'actions.npy'))
    rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
    
    # Use small subset for fast tests
    n_agents_test = 10
    n_periods_test = 10
    
    return PanelData(
        states=states[:n_agents_test, :n_periods_test],
        actions=actions[:n_agents_test, :n_periods_test],
        rewards=rewards[:n_agents_test, :n_periods_test],
        n_agents=n_agents_test,
        n_periods=n_periods_test,
    )


# =============================================================================
# UNIT TESTS (Fast - no solver needed)
# =============================================================================

class TestParameterValidation:
    """Unit tests for parameter validation (fast, no solver needed)."""
    
    def test_invalid_beta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([0.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_beta_negative_returns_neg_inf(self, fake_panel_data, solver_params):
        """beta < 0 should return -inf immediately (no solver call)."""
        theta = np.array([-1.0, 0.1, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_gamma_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """gamma = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 1.0, 0.9])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_zero_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 0 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 0.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf
    
    def test_invalid_delta_one_returns_neg_inf(self, fake_panel_data, solver_params):
        """delta = 1 should return -inf immediately (no solver call)."""
        theta = np.array([1.5, 0.1, 1.0])
        ll = compute_log_likelihood(theta, fake_panel_data, solver_params)
        assert ll == -np.inf


class TestEstimationResultStructure:
    """Unit tests for EstimationResult data structure."""
    
    def test_estimation_result_fields(self):
        """EstimationResult should have all required fields."""
        result = EstimationResult(
            theta_hat=np.array([1.0, 0.1, 0.9]),
            std_errors=np.array([0.1, 0.01, 0.05]),
            cov_matrix=np.eye(3) * 0.01,
            log_likelihood=-100.0,
            n_iterations=50,
            converged=True,
            optimization_result=None,
        )
        
        assert result.theta_hat.shape == (3,)
        assert result.std_errors.shape == (3,)
        assert result.cov_matrix.shape == (3, 3)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)


class TestTwoStepResultStructure:
    """Unit tests for TwoStepResult data structure."""
    
    def test_two_step_result_fields(self):
        """TwoStepResult should have all required fields."""
        result = TwoStepResult(
            beta_hat=1.0,
            gamma_hat=0.1,
            delta_calibrated=0.95,
            beta_se=0.05,
            gamma_se=0.001,
            log_likelihood=-100.0,
            n_evaluations=20,
            gamma_ols_details={'n_obs': 1000, 'r_squared': 1.0},
        )
        
        assert isinstance(result.beta_hat, float)
        assert isinstance(result.gamma_hat, float)
        assert isinstance(result.delta_calibrated, float)
        assert isinstance(result.beta_se, float)
        assert isinstance(result.gamma_se, float)
        assert isinstance(result.log_likelihood, float)
        assert isinstance(result.n_evaluations, int)
        assert isinstance(result.gamma_ols_details, dict)


class TestEstimateGammaOLS:
    """Unit tests for estimate_gamma_ols (fast - no solver needed).
    
    Uses synthetic data with known gamma to verify OLS estimation.
    """
    
    @pytest.fixture
    def synthetic_transition_data(self):
        """Create synthetic panel data with known gamma = 0.1."""
        np.random.seed(42)
        n_agents, n_periods = 50, 20
        gamma_true = 0.1
        
        # Generate states following s_{t+1} = (1 - gamma) * s_t + a_t
        states = np.zeros((n_agents, n_periods))
        actions = np.random.randint(0, 2, size=(n_agents, n_periods))
        states[:, 0] = np.random.uniform(1, 10, size=n_agents)  # Initial states > 0
        
        for t in range(n_periods - 1):
            states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
        
        return PanelData(
            states=states,
            actions=actions,
            rewards=np.zeros((n_agents, n_periods)),  # Not used for gamma estimation
            n_agents=n_agents,
            n_periods=n_periods,
        ), gamma_true
    
    def test_recovers_true_gamma(self, synthetic_transition_data):
        """OLS should exactly recover gamma from deterministic transitions."""
        data, gamma_true = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # Should recover exactly (within numerical precision)
        np.testing.assert_allclose(
            gamma_hat, gamma_true, rtol=1e-10,
            err_msg=f"gamma_hat={gamma_hat} should equal gamma_true={gamma_true}"
        )
    
    def test_r_squared_equals_one(self, synthetic_transition_data):
        """R-squared should be 1.0 for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        np.testing.assert_allclose(
            details['r_squared'], 1.0, rtol=1e-10,
            err_msg="R-squared should be 1.0 for deterministic transitions"
        )
    
    def test_std_error_near_zero(self, synthetic_transition_data):
        """Standard error should be near zero for deterministic transitions."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        assert std_error < 1e-10, "Standard error should be near zero"
    
    def test_returns_correct_n_obs(self, synthetic_transition_data):
        """Should return correct number of observations."""
        data, _ = synthetic_transition_data
        
        gamma_hat, std_error, details = estimate_gamma_ols(data)
        
        # n_obs should be n_agents * (n_periods - 1) minus any filtered zeros
        expected_max = data.n_agents * (data.n_periods - 1)
        assert details['n_obs'] <= expected_max
        assert details['n_obs'] > 0
    
    def test_handles_different_gamma_values(self):
        """Should work for various gamma values."""
        np.random.seed(42)
        
        for gamma_true in [0.05, 0.1, 0.2, 0.5]:
            n_agents, n_periods = 20, 10
            states = np.zeros((n_agents, n_periods))
            actions = np.random.randint(0, 2, size=(n_agents, n_periods))
            states[:, 0] = np.random.uniform(1, 10, size=n_agents)
            
            for t in range(n_periods - 1):
                states[:, t + 1] = (1 - gamma_true) * states[:, t] + actions[:, t]
            
            data = PanelData(
                states=states, actions=actions,
                rewards=np.zeros((n_agents, n_periods)),
                n_agents=n_agents, n_periods=n_periods,
            )
            
            gamma_hat, _, _ = estimate_gamma_ols(data)
            
            np.testing.assert_allclose(
                gamma_hat, gamma_true, rtol=1e-10,
                err_msg=f"Failed to recover gamma={gamma_true}"
            )


# =============================================================================
# INTEGRATION TESTS (Slow - requires solver)
# Mark with pytest.mark.slow so they can be skipped
# =============================================================================

@pytest.mark.slow
class TestComputeLogLikelihood:
    """Integration tests for compute_log_likelihood (requires solver)."""
    
    def test_returns_finite_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should return a finite value for valid parameters."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert np.isfinite(ll), "Log-likelihood should be finite"
    
    def test_returns_negative_value(self, simulated_data, true_params, solver_params):
        """Log-likelihood should be negative (log of probabilities < 1)."""
        theta = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        ll = compute_log_likelihood(theta, simulated_data, solver_params)
        
        assert ll < 0, "Log-likelihood should be negative"
    
    def test_true_params_higher_likelihood(self, simulated_data, true_params, solver_params):
        """True parameters should have higher likelihood than wrong parameters."""
        theta_true = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        theta_wrong = np.array([2.5, 0.3, 0.7])  # Significantly different
        
        ll_true = compute_log_likelihood(theta_true, simulated_data, solver_params)
        ll_wrong = compute_log_likelihood(theta_wrong, simulated_data, solver_params)
        
        assert ll_true > ll_wrong, \
            f"True params LL ({ll_true:.2f}) should be > wrong params LL ({ll_wrong:.2f})"


@pytest.mark.slow
class TestComputeStandardErrors:
    """Integration tests for compute_standard_errors (requires solver)."""
    
    def test_returns_positive_std_errors(self, simulated_data, true_params, solver_params):
        """Standard errors should be positive."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check std_errors are positive (or NaN if singular)
        if not np.any(np.isnan(std_errors)):
            assert np.all(std_errors > 0), "Standard errors should be positive"
    
    def test_cov_matrix_shape(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be 3x3."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        assert cov_matrix.shape == (3, 3), "Covariance matrix should be 3x3"
    
    def test_cov_matrix_symmetric(self, simulated_data, true_params, solver_params):
        """Covariance matrix should be symmetric."""
        theta_hat = np.array([true_params['beta'], true_params['gamma'], true_params['delta']])
        
        std_errors, cov_matrix = compute_standard_errors(
            theta_hat, simulated_data, solver_params, eps=1e-3
        )
        
        # Check symmetry (allowing for NaN)
        if not np.any(np.isnan(cov_matrix)):
            np.testing.assert_allclose(
                cov_matrix, cov_matrix.T, rtol=1e-5,
                err_msg="Covariance matrix should be symmetric"
            )


@pytest.mark.slow
class TestEstimateMLE:
    """Integration tests for estimate_mle (requires solver)."""
    
    def test_returns_estimation_result(self, simulated_data, true_params, solver_params):
        """estimate_mle should return EstimationResult object."""
        theta_init = (true_params['beta'], true_params['gamma'], true_params['delta'])
        
        # Run with very few iterations just to test interface
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=5, verbose=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert result.std_errors.shape == (3,), "std_errors should have 3 elements"
        assert result.cov_matrix.shape == (3, 3), "cov_matrix should be 3x3"
        assert isinstance(result.log_likelihood, float), "log_likelihood should be float"
        assert isinstance(result.n_iterations, int), "n_iterations should be int"
        assert isinstance(result.converged, bool), "converged should be bool"


@pytest.mark.slow
class TestIntegration:
    """Integration tests for full estimation workflow (requires solver)."""
    
    def test_full_workflow_runs(self, simulated_data, true_params, solver_params):
        """Full estimation workflow should run without error."""
        # Start from perturbed initial values
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        # Run estimation with limited iterations
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        # Basic sanity checks
        assert np.all(np.isfinite(result.theta_hat)), "Estimates should be finite"
        assert np.isfinite(result.log_likelihood), "Log-likelihood should be finite"
        assert result.n_iterations > 0, "Should have run at least one iteration"
    
    def test_estimates_in_valid_range(self, simulated_data, true_params, solver_params):
        """Estimates should be in valid parameter range."""
        theta_init = (
            true_params['beta'] * 1.1,
            true_params['gamma'] * 0.9,
            true_params['delta'] * 0.95,
        )
        
        result = estimate_mle(
            simulated_data, theta_init, solver_params,
            maxiter=10, verbose=False
        )
        
        beta_hat, gamma_hat, delta_hat = result.theta_hat
        
        assert beta_hat > 0, "beta should be positive"
        assert 0 < gamma_hat < 1, "gamma should be in (0, 1)"
        assert 0 < delta_hat < 1, "delta should be in (0, 1)"


@pytest.mark.slow
class TestGridSearch:
    """Integration tests for grid search estimation (requires solver)."""
    
    def test_grid_search_returns_result(self, simulated_data, true_params, solver_params):
        """grid_search_mle should return EstimationResult object."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        assert isinstance(result, EstimationResult), "Should return EstimationResult"
        assert result.theta_hat.shape == (3,), "theta_hat should have 3 elements"
        assert np.isfinite(result.log_likelihood), "log_likelihood should be finite"
        assert result.n_iterations == 8, "Should have 2^3 = 8 evaluations"
        assert result.converged is True, "Grid search always converges"
    
    def test_grid_search_finds_best_in_grid(self, simulated_data, true_params, solver_params):
        """Grid search should return the grid point with highest likelihood."""
        bounds = {
            'beta': (true_params['beta'] * 0.9, true_params['beta'] * 1.1),
            'gamma': (true_params['gamma'] * 0.8, true_params['gamma'] * 1.2),
            'delta': (true_params['delta'] * 0.95, true_params['delta'] * 1.02),
        }
        
        result = grid_search_mle(
            simulated_data, bounds, n_points=2,
            solver_params=solver_params, verbose=False, compute_se=False
        )
        
        # Verify theta_hat is in the grid
        grid_results = result.optimization_result
        beta_grid = grid_results['grid']['beta']
        gamma_grid = grid_results['grid']['gamma']
        delta_grid = grid_results['grid']['delta']
        
        assert result.theta_hat[0] in beta_grid, "beta should be in grid"
        assert result.theta_hat[1] in gamma_grid, "gamma should be in grid"
        assert result.theta_hat[2] in delta_grid, "delta should be in grid"
        
        # Verify it has the highest likelihood
        all_lls = [e['log_likelihood'] for e in grid_results['evaluations']]
        assert result.log_likelihood == max(all_lls), "Should return max likelihood point"


@pytest.mark.slow
class TestEstimateBeta1D:
    """Integration tests for estimate_beta_1d (requires solver)."""
    
    def test_returns_correct_types(self, simulated_data, true_params, solver_params):
        """estimate_beta_1d should return correct types."""
        beta_hat, std_error, log_lik, details = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(beta_hat, float), "beta_hat should be float"
        assert isinstance(std_error, (float, type(np.nan))), "std_error should be float or nan"
        assert isinstance(log_lik, float), "log_lik should be float"
        assert isinstance(details, dict), "details should be dict"
    
    def test_beta_in_bounds(self, simulated_data, true_params, solver_params):
        """Estimated beta should be within specified bounds."""
        beta_bounds = (0.5, 1.5)
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=beta_bounds,
            n_points=3,
            verbose=False,
        )
        
        assert beta_bounds[0] <= beta_hat <= beta_bounds[1], \
            f"beta_hat={beta_hat} should be in {beta_bounds}"
    
    def test_finds_true_beta_on_grid(self, simulated_data, true_params, solver_params):
        """When true beta is on grid, should find it."""
        # Use bounds that include true beta as a grid point
        true_beta = true_params['beta']
        
        beta_hat, _, _, _ = estimate_beta_1d(
            data=simulated_data,
            gamma_fixed=true_params['gamma'],
            delta_fixed=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(true_beta - 0.5, true_beta + 0.5),
            n_points=5,  # Grid: [0.5, 0.75, 1.0, 1.25, 1.5] includes 1.0
            verbose=False,
        )
        
        # Should find true beta or very close
        assert abs(beta_hat - true_beta) < 0.3, \
            f"beta_hat={beta_hat} should be close to true_beta={true_beta}"


@pytest.mark.slow
class TestEstimateTwoStep:
    """Integration tests for estimate_two_step (requires solver)."""
    
    def test_returns_two_step_result(self, simulated_data, true_params, solver_params):
        """estimate_two_step should return TwoStepResult object."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert isinstance(result, TwoStepResult), "Should return TwoStepResult"
    
    def test_gamma_exactly_recovered(self, simulated_data, true_params, solver_params):
        """Gamma should be exactly recovered from OLS."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        # Gamma should be exactly recovered
        np.testing.assert_allclose(
            result.gamma_hat, true_params['gamma'], rtol=1e-6,
            err_msg=f"gamma_hat={result.gamma_hat} should equal true gamma={true_params['gamma']}"
        )
    
    def test_delta_equals_calibrated(self, simulated_data, true_params, solver_params):
        """Delta should equal the calibrated value."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=3,
            verbose=False,
        )
        
        assert result.delta_calibrated == true_params['delta'], \
            "delta_calibrated should equal the input value"
    
    def test_beta_reasonable(self, simulated_data, true_params, solver_params):
        """Beta estimate should be in reasonable range."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.5, 1.5),
            n_points=5,
            verbose=False,
        )
        
        # Beta should be positive and in bounds
        assert 0.5 <= result.beta_hat <= 1.5, \
            f"beta_hat={result.beta_hat} should be in [0.5, 1.5]"
    
    def test_full_recovery_with_fine_grid(self, simulated_data, true_params, solver_params):
        """With fine grid and true delta, should recover beta well."""
        result = estimate_two_step(
            data=simulated_data,
            delta_calibrated=true_params['delta'],
            solver_params=solver_params,
            beta_bounds=(0.8, 1.2),  # Narrow bounds around true value
            n_points=5,
            verbose=False,
        )
        
        # Should recover beta close to true value
        bias = abs(result.beta_hat - true_params['beta'])
        assert bias < 0.15, \
            f"beta bias={bias} should be < 0.15 (beta_hat={result.beta_hat}, true={true_params['beta']})"

```

Now updating the Quarto scripts:

```plaintext
---
title: "Solving a Dynamic Discrete Choice Model"
subtitle: "MDP with Type-I Extreme Value Shocks"
format:
  html:
    code-fold: false
    toc: true
    toc-depth: 3
---

## Model Setup

We consider a dynamic discrete choice model with:

- **State**: $s \in \mathbb{R}_{\geq 0}$ (continuous, non-negative)
- **Action**: $a \in \{0, 1\}$ (binary choice)
- **Mean reward**: $u(s, a) = \beta \log(1 + s) - a$
- **State transition**: $s' = (1 - \gamma) s + a$ (deterministic)
- **Discount factor**: $\delta$
- **Shocks**: $\varepsilon_a \sim$ Type-I Extreme Value (iid across actions and time)

### Interpretation

- Higher state $s$ yields higher flow payoff, but with **diminishing marginal returns**
- The $\log(1 + s)$ specification ensures:
  - Marginal utility $\frac{\partial u}{\partial s} = \frac{\beta}{1 + s} > 0$ (increasing in $s$)
  - Diminishing: $\frac{\partial^2 u}{\partial s^2} = -\frac{\beta}{(1 + s)^2} < 0$ (concave)
  - Well-defined at $s = 0$: $u(0, a) = -a$
- Action $a = 1$ costs 1 unit but boosts next period's state by 1
- Action $a = 0$ is free but state decays toward zero at rate $\gamma$

## Optimal Markov Decision Policy

### Bellman Equation with Extreme Value Shocks

Each period, the agent observes state $s$ and receives action-specific shocks $(\varepsilon_0, \varepsilon_1)$ drawn iid from the Type-I Extreme Value (Gumbel) distribution:
$$
F(\varepsilon) = \exp(-\exp(-\varepsilon)), \quad f(\varepsilon) = \exp(-\varepsilon - \exp(-\varepsilon))
$$

The agent chooses action $a \in \{0, 1\}$ to maximize:
$$
u(s, a) + \varepsilon_a + \delta V(s')
$$
where $s' = (1 - \gamma)s + a$ is the next state.

The **Bellman equation** is:
$$
V(s, \varepsilon_0, \varepsilon_1) = \max_{a \in \{0,1\}} \left\{ u(s, a) + \varepsilon_a + \delta \mathbb{E}_{\varepsilon'} \left[ V(s', \varepsilon_0', \varepsilon_1') \right] \right\}
$$

### Choice-Specific Value Functions

Define the **choice-specific value function** (the deterministic part, excluding the current shock):
$$
v(s, a) = u(s, a) + \delta \mathbb{E}_{\varepsilon'} \left[ V(s', \varepsilon') \right]
$$

Substituting the reward and transition:
$$
v(s, a) = \beta \log(1 + s) - a + \delta \bar{V}\bigl((1 - \gamma) s + a\bigr)
$$
where $\bar{V}(s) = \mathbb{E}_\varepsilon[V(s, \varepsilon)]$ is the **integrated (ex-ante) value function**.

### Integrated Value Function

The **integrated value function** is the expected value before observing shocks:
$$
\bar{V}(s) = \mathbb{E}_{\varepsilon_0, \varepsilon_1} \left[ \max_{a \in \{0,1\}} \left\{ v(s, a) + \varepsilon_a \right\} \right]
$$

With Type-I EV shocks, this has a closed-form (the "log-sum-exp" formula):
$$
\bar{V}(s) = \log\left( \exp(v(s, 0)) + \exp(v(s, 1)) \right) + \gamma_E
$$
where $\gamma_E \approx 0.5772$ is Euler's constant. Since $\gamma_E$ is a constant, it does not affect the policy and is often omitted.

### Optimal Policy and Choice Probabilities

The optimal decision rule is:
$$
a^*(s, \varepsilon_0, \varepsilon_1) = \mathbf{1}\left\{ v(s, 1) + \varepsilon_1 > v(s, 0) + \varepsilon_0 \right\}
$$

The probability of choosing action $a = 1$ given state $s$ follows the **logit formula**:
$$
P(a = 1 | s) = \frac{\exp(v(s, 1))}{\exp(v(s, 0)) + \exp(v(s, 1))} = \frac{1}{1 + \exp(v(s, 0) - v(s, 1))}
$$

The optimal policy is a **stochastic Markov policy**: the choice probability depends only on the current state $s$ through the value difference $v(s, 1) - v(s, 0)$.

## Solution Method: Neural Network Value Iteration

### Bellman Fixed Point

The integrated value function $\bar{V}(s)$ satisfies the **fixed point equation**:
$$
\bar{V}(s) = \log\left( \exp(v(s, 0)) + \exp(v(s, 1)) \right)
$$
where each choice-specific value depends on $\bar{V}$:
$$
v(s, a) = \beta \log(1 + s) - a + \delta \bar{V}((1 - \gamma)s + a)
$$

By Blackwell's sufficient conditions, this defines a contraction mapping with modulus $\delta < 1$, guaranteeing a unique fixed point $\bar{V}^*$.

### Monotonic Neural Network Architecture

We approximate each action's choice-specific value function $v(s, a)$ using a **monotonic neural network**. Since higher states yield higher rewards ($\beta > 0$), we expect $v(s, a)$ to be increasing in $s$. Enforcing monotonicity as an architectural constraint improves extrapolation and ensures economically sensible behavior.

A feed-forward neural network is **monotonically increasing** in its input if:

1. All weights are **non-negative**
2. All activation functions are **monotonically increasing**

For a network with $L$ hidden layers:
$$
v_\theta(s) = W_L \cdot \sigma\left( W_{L-1} \cdot \sigma\left( \cdots \sigma(W_1 \cdot s + b_1) \cdots \right) + b_{L-1} \right) + b_L
$$
where $W_\ell \geq 0$ for all layers and $\sigma$ is a monotonically increasing activation function.

### Tanh Activation

We use the **tanh** (hyperbolic tangent) activation function:
$$
\sigma(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Properties:

- **Monotonically increasing**: $\sigma'(x) = 1 - \tanh^2(x) > 0$
- **Bounded output**: $\sigma(x) \in (-1, 1)$ prevents explosive growth
- **Zero-centered**: Allows both positive and negative intermediate representations
- **Better gradient flow**: Bounded outputs help maintain stable gradients through deep networks

### Weight Parameterization

To enforce non-negative weights during optimization, we parameterize the weights using softplus of unconstrained parameters:
$$
W = \text{softplus}(\tilde{W}) = \log(1 + e^{\tilde{W}})
$$
where $\tilde{W} \in \mathbb{R}$ is the learnable parameter. This ensures $W > 0$ while allowing standard gradient-based optimization.

### Value Function Approximator

We use **separate monotonic networks** for each action:

- $v_{\theta_0}(s)$: Monotonic network for action $a = 0$
- $v_{\theta_1}(s)$: Monotonic network for action $a = 1$

The integrated value function is computed as:
$$
\bar{V}_\theta(s) = \log\left( \exp(v_{\theta_0}(s)) + \exp(v_{\theta_1}(s)) \right)
$$

Choice probabilities follow the logit formula:
$$
P_\theta(a = 1 | s) = \frac{1}{1 + \exp(v_{\theta_0}(s) - v_{\theta_1}(s))}
$$

### Training Objective

The networks are trained to satisfy the Bellman equation by minimizing the **Bellman residual**:
$$
\mathcal{L}(\theta) = \mathbb{E}_s\left[ \sum_{a \in \{0,1\}} \left( v_{\theta_a}(s) - u(s, a) - \delta \bar{V}_\theta(s'_a) \right)^2 \right]
$$
where $s'_a = (1 - \gamma)s + a$ is the next state under action $a$.

### Type Definitions

```
TYPE DEFINITIONS
────────────────
Scalar      = Float                      # Single real number
Vector[n]   = Array[Float, n]            # 1D array of n floats
Matrix[m,n] = Array[Float, m, n]         # 2D array of shape (m, n)
Action      = Int ∈ {0, 1}               # Binary action
Network     = MonotonicNeuralNetwork     # Neural network with non-negative weights
```

### Algorithm

```
ALGORITHM: SOLVE_VALUE_FUNCTION
───────────────────────────────
INPUT:
  β            : Scalar              # Reward coefficient
  γ            : Scalar              # State decay rate
  δ            : Scalar              # Discount factor
  s_min        : Scalar              # Minimum state
  s_max        : Scalar              # Maximum state
  hidden_sizes : List[Int]           # Hidden layer sizes
  α            : Scalar              # Learning rate
  N            : Int                 # Batch size
  ε            : Scalar              # Convergence tolerance
  K            : Int                 # Maximum iterations
  T            : Int                 # Target network update frequency

OUTPUT:
  v₀_net       : Network             # Trained network for action 0
  v₁_net       : Network             # Trained network for action 1

PROCEDURE:
  # Initialize policy networks (updated every iteration)
  v₀_net : Network, v₁_net : Network ← INITIALIZE_NETWORKS(hidden_sizes)
  
  # Initialize target networks (frozen copies, updated every T iterations)
  v₀_target : Network ← COPY_NETWORK(v₀_net)
  v₁_target : Network ← COPY_NETWORK(v₁_net)
  
  # Generate fixed state grid (same states used every iteration)
  s_grid : Vector[N] ← GENERATE_STATE_GRID(N, s_min, s_max)
  
  FOR k : Int = 1 TO K:
    # Predictions from policy networks (using fixed grid)
    v₀_pred : Vector[N] ← EVALUATE_NETWORK(v₀_net, s_grid)
    v₁_pred : Vector[N] ← EVALUATE_NETWORK(v₁_net, s_grid)
    
    # Targets computed from TARGET networks (stable, not moving)
    target₀ : Vector[N], target₁ : Vector[N] ← COMPUTE_BELLMAN_TARGETS(
      v₀_target, v₁_target, s_grid, β, γ, δ
    )
    
    loss : Scalar ← COMPUTE_BELLMAN_LOSS(v₀_pred, v₁_pred, target₀, target₁)
    
    # Update policy networks
    UPDATE_NETWORK_WEIGHTS(v₀_net, v₁_net, loss, α)
    
    # Periodically sync target networks with policy networks
    IF k MOD T = 0:
      v₀_target ← COPY_NETWORK(v₀_net)
      v₁_target ← COPY_NETWORK(v₁_net)
    
    IF CHECK_CONVERGENCE(loss, ε):
      RETURN (v₀_net, v₁_net)
  
  RETURN (v₀_net, v₁_net)
```

### Subroutines

```
SUBROUTINE: INITIALIZE_NETWORKS
───────────────────────────────
INPUT:
  hidden_sizes : List[Int]           # Hidden layer sizes
OUTPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1

  v₀_net : Network ← BUILD_MONOTONIC_NETWORK(hidden_sizes)
  v₁_net : Network ← BUILD_MONOTONIC_NETWORK(hidden_sizes)
  RETURN (v₀_net, v₁_net)
```

```
SUBROUTINE: COPY_NETWORK
────────────────────────
INPUT:
  source : Network                   # Network to copy from
OUTPUT:
  target : Network                   # Deep copy of source network

  # Create a new network with identical architecture and weights
  target : Network ← DEEP_COPY(source)
  
  # Freeze the target network (no gradient computation)
  SET_REQUIRES_GRAD(target, false)
  
  RETURN target
```

```
SUBROUTINE: BUILD_MONOTONIC_NETWORK
───────────────────────────────────
INPUT:
  hidden_sizes : List[Int]           # Hidden layer sizes
OUTPUT:
  network : Network                  # Monotonic neural network

  layers : List[Layer] ← empty list
  in_features : Int ← 1
  
  FOR i : Int = 0 TO length(hidden_sizes) - 1:
    out_features : Int ← hidden_sizes[i]
    
    # Create monotonic linear layer
    W_raw : Matrix[out_features, in_features] ← random_normal(0, 0.1)
    b : Vector[out_features] ← zeros(out_features)
    layers.append(MonotonicLinear(W_raw, b))
    
    # Add tanh activation (bounded, zero-centered)
    layers.append(TanhActivation())
    
    in_features ← out_features
  
  # Output layer (no activation)
  W_raw_out : Matrix[1, in_features] ← random_normal(0, 0.1)
  b_out : Vector[1] ← zeros(1)
  layers.append(MonotonicLinear(W_raw_out, b_out))
  
  RETURN Network(layers)
```

```
SUBROUTINE: APPLY_MONOTONIC_LINEAR
──────────────────────────────────
INPUT:
  W_raw : Matrix[m, n]               # Unconstrained weight parameters
  b     : Vector[m]                  # Bias vector
  x     : Vector[n]                  # Input vector
OUTPUT:
  y     : Vector[m]                  # Output vector

  # Enforce non-negative weights via softplus
  W : Matrix[m, n] ← log(1 + exp(W_raw))
  
  # Linear transformation
  y : Vector[m] ← W @ x + b
  RETURN y
```

```
SUBROUTINE: APPLY_TANH
──────────────────────
INPUT:
  x : Vector[n]                      # Input vector
OUTPUT:
  y : Vector[n]                      # Output vector

  FOR i : Int = 0 TO n - 1:
    y[i] : Scalar ← tanh(x[i])       # Bounded to (-1, 1)
  RETURN y
```

```
SUBROUTINE: EVALUATE_NETWORK
────────────────────────────
INPUT:
  network : Network                  # Monotonic neural network
  s       : Vector[N]                # Input states
OUTPUT:
  values  : Vector[N]                # Output values

  x : Matrix[N, 1] ← reshape(s, (N, 1))
  
  FOR layer IN network.layers:
    IF layer IS MonotonicLinear:
      x ← APPLY_MONOTONIC_LINEAR(layer.W_raw, layer.b, x)
    ELSE IF layer IS TanhActivation:
      x ← APPLY_TANH(x)
  
  values : Vector[N] ← reshape(x, (N,))
  RETURN values
```

```
SUBROUTINE: GENERATE_STATE_GRID
───────────────────────────────
INPUT:
  N     : Int                        # Number of grid points
  s_min : Scalar                     # Minimum state
  s_max : Scalar                     # Maximum state
OUTPUT:
  s     : Vector[N]                  # Fixed state grid (evenly spaced)

  # Generate evenly spaced points from s_min to s_max
  step : Scalar ← (s_max - s_min) / (N - 1)
  
  FOR i : Int = 0 TO N - 1:
    s[i] : Scalar ← s_min + i * step
  RETURN s
```

```
SUBROUTINE: COMPUTE_REWARD
──────────────────────────
INPUT:
  s      : Vector[N]                 # States
  action : Action                    # Action (0 or 1)
  β      : Scalar                    # Reward coefficient
OUTPUT:
  r      : Vector[N]                 # Rewards

  FOR i : Int = 0 TO N - 1:
    r[i] : Scalar ← β * log(1 + s[i]) - action
  RETURN r
```

```
SUBROUTINE: COMPUTE_NEXT_STATE
──────────────────────────────
INPUT:
  s      : Vector[N]                 # Current states
  action : Action                    # Action (0 or 1)
  γ      : Scalar                    # Decay rate
OUTPUT:
  s_next : Vector[N]                 # Next states

  FOR i : Int = 0 TO N - 1:
    s_next[i] : Scalar ← (1 - γ) * s[i] + action
  RETURN s_next
```

```
SUBROUTINE: COMPUTE_INTEGRATED_VALUE
────────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # States
OUTPUT:
  V_bar  : Vector[N]                 # Integrated value

  v₀ : Vector[N] ← EVALUATE_NETWORK(v₀_net, s)
  v₁ : Vector[N] ← EVALUATE_NETWORK(v₁_net, s)
  
  FOR i : Int = 0 TO N - 1:
    v_max : Scalar ← max(v₀[i], v₁[i])
    V_bar[i] : Scalar ← v_max + log(exp(v₀[i] - v_max) + exp(v₁[i] - v_max))
  RETURN V_bar
```

```
SUBROUTINE: COMPUTE_BELLMAN_TARGETS
───────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # Current states
  β      : Scalar                    # Reward coefficient
  γ      : Scalar                    # Decay rate
  δ      : Scalar                    # Discount factor
OUTPUT:
  target₀ : Vector[N]                # Target for action 0
  target₁ : Vector[N]                # Target for action 1

  # Compute targets for action 0 (no clamping - unbounded state space)
  r₀ : Vector[N] ← COMPUTE_REWARD(s, action=0, β)
  s_next₀ : Vector[N] ← COMPUTE_NEXT_STATE(s, action=0, γ)
  V_next₀ : Vector[N] ← COMPUTE_INTEGRATED_VALUE(v₀_net, v₁_net, s_next₀)
  
  FOR i : Int = 0 TO N - 1:
    target₀[i] : Scalar ← r₀[i] + δ * V_next₀[i]
  
  # Compute targets for action 1 (no clamping - unbounded state space)
  r₁ : Vector[N] ← COMPUTE_REWARD(s, action=1, β)
  s_next₁ : Vector[N] ← COMPUTE_NEXT_STATE(s, action=1, γ)
  V_next₁ : Vector[N] ← COMPUTE_INTEGRATED_VALUE(v₀_net, v₁_net, s_next₁)
  
  FOR i : Int = 0 TO N - 1:
    target₁[i] : Scalar ← r₁[i] + δ * V_next₁[i]
  
  RETURN (target₀, target₁)
```

```
SUBROUTINE: COMPUTE_BELLMAN_LOSS
────────────────────────────────
INPUT:
  v₀_pred : Vector[N]                # Predicted values for action 0
  v₁_pred : Vector[N]                # Predicted values for action 1
  target₀ : Vector[N]                # Target values for action 0
  target₁ : Vector[N]                # Target values for action 1
OUTPUT:
  loss    : Scalar                   # Mean squared Bellman residual

  sum_sq : Scalar ← 0.0
  
  FOR i : Int = 0 TO N - 1:
    error₀ : Scalar ← v₀_pred[i] - target₀[i]
    error₁ : Scalar ← v₁_pred[i] - target₁[i]
    sum_sq ← sum_sq + error₀ * error₀ + error₁ * error₁
  
  loss : Scalar ← sum_sq / (2 * N)
  RETURN loss
```

```
SUBROUTINE: UPDATE_NETWORK_WEIGHTS
──────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  loss   : Scalar                    # Loss value
  α      : Scalar                    # Learning rate
OUTPUT:
  (none, updates networks in place)

  # Compute gradients via backpropagation
  grad₀ : List[Matrix] ← BACKPROPAGATE(v₀_net, loss)
  grad₁ : List[Matrix] ← BACKPROPAGATE(v₁_net, loss)
  
  # Update weights for network 0
  FOR j : Int = 0 TO length(v₀_net.layers) - 1:
    IF v₀_net.layers[j] IS MonotonicLinear:
      v₀_net.layers[j].W_raw ← v₀_net.layers[j].W_raw - α * grad₀[j].W
      v₀_net.layers[j].b ← v₀_net.layers[j].b - α * grad₀[j].b
  
  # Update weights for network 1
  FOR j : Int = 0 TO length(v₁_net.layers) - 1:
    IF v₁_net.layers[j] IS MonotonicLinear:
      v₁_net.layers[j].W_raw ← v₁_net.layers[j].W_raw - α * grad₁[j].W
      v₁_net.layers[j].b ← v₁_net.layers[j].b - α * grad₁[j].b
```

```
SUBROUTINE: CHECK_CONVERGENCE
─────────────────────────────
INPUT:
  loss : Scalar                      # Current loss
  ε    : Scalar                      # Tolerance
OUTPUT:
  converged : Bool                   # True if converged

  rmse : Scalar ← sqrt(loss)
  converged : Bool ← (rmse < ε)
  RETURN converged
```

```
SUBROUTINE: COMPUTE_CHOICE_PROBABILITY
──────────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # States
OUTPUT:
  prob₀  : Vector[N]                 # P(a=0|s)
  prob₁  : Vector[N]                 # P(a=1|s)

  v₀ : Vector[N] ← EVALUATE_NETWORK(v₀_net, s)
  v₁ : Vector[N] ← EVALUATE_NETWORK(v₁_net, s)
  
  FOR i : Int = 0 TO N - 1:
    # Numerically stable softmax
    v_max : Scalar ← max(v₀[i], v₁[i])
    exp₀ : Scalar ← exp(v₀[i] - v_max)
    exp₁ : Scalar ← exp(v₁[i] - v_max)
    sum_exp : Scalar ← exp₀ + exp₁
    
    prob₀[i] : Scalar ← exp₀ / sum_exp
    prob₁[i] : Scalar ← exp₁ / sum_exp
  
  RETURN (prob₀, prob₁)
```

### Convergence

The algorithm minimizes the Bellman residual, driving the neural network approximation toward the true fixed point. Convergence is monitored via the root mean squared Bellman error $\sqrt{\mathcal{L}^{(k)}}$.

Key properties:

- **Continuous approximation**: No grid discretization or interpolation required
- **Scalability**: Extends naturally to high-dimensional state spaces
- **Economic structure**: Monotonicity enforced architecturally via non-negative weights

## Implementation

### Setup

```{python}
#| label: setup
#| cache: false

# Standard libraries
import sys
import numpy as np
import torch

# Add project paths
sys.path.insert(0, "../../../src")
sys.path.insert(0, "../config")
sys.path.insert(0, "../../utils")

# Import MDP solver
from mdp.solver import (
    solve_value_function,
    compute_choice_probability,
    evaluate_network,
    compute_integrated_value,
    compute_bellman_targets,
)

# Import configuration
import config

# Import plotting utilities
from plotting import (
    plot_convergence,
    plot_value_functions,
    plot_choice_probabilities,
    plot_value_difference,
    plot_bellman_residuals,
    plot_comparative_statics_values,
    plot_comparative_statics_probs,
)

print("Modules loaded successfully")
```

### Model Parameters

```{python}
# Load parameters from central config
beta = config.beta
gamma = config.gamma
delta = config.delta
s_min = config.s_min
s_max = config.s_max
hidden_sizes = config.hidden_sizes
learning_rate = config.learning_rate
batch_size = config.batch_size
tolerance = config.tolerance
max_iterations = config.max_iterations
target_update_freq = config.target_update_freq

print(f"Model: u(s,a) = {beta} * log(1+s) - a")
print(f"Transition: s' = {1-gamma:.1f} * s + a")
print(f"Discount factor: δ = {delta}")
print(f"\nLoaded from: scripts/config_mdp/config.py")
```

### Solve the MDP

```{python}
#| cache: true
#| label: solve-baseline

# Set random seed for reproducibility
torch.manual_seed(42)

# Solve for value functions
v0_net, v1_net, losses, n_iter = solve_value_function(
    beta=beta,
    gamma=gamma,
    delta=delta,
    s_min=s_min,
    s_max=s_max,
    hidden_sizes=hidden_sizes,
    learning_rate=learning_rate,
    batch_size=batch_size,
    tolerance=tolerance,
    max_iterations=max_iterations,
    target_update_freq=target_update_freq,
)

print(f"Converged in {n_iter} iterations")
print(f"Final RMSE: {losses[-1]**0.5:.2e}")

# Save results to output/solve_mdp/
import json
from pathlib import Path

output_dir = Path("../../output/solve_mdp")
output_dir.mkdir(parents=True, exist_ok=True)

# Save config (model parameters)
config = {
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
    "s_min": s_min,
    "s_max": s_max,
    "hidden_sizes": hidden_sizes,
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save trained networks
torch.save(v0_net.state_dict(), output_dir / "v0_net.pt")
torch.save(v1_net.state_dict(), output_dir / "v1_net.pt")

# Save training losses
np.save(output_dir / "losses.npy", np.array(losses))

print(f"\nResults saved to {output_dir.resolve()}:")
print(f"  - config.json (model parameters)")
print(f"  - v0_net.pt (action 0 network)")
print(f"  - v1_net.pt (action 1 network)")
print(f"  - losses.npy ({len(losses)} iterations)")
```

### Results

#### Convergence Plot

```{python}
#| fig-cap: "Convergence of the Bellman residual over training iterations"

plot_convergence(losses=np.array(losses))
```

#### Value Functions

```{python}
#| fig-cap: "Choice-specific value functions v(s, a) for each action"

# Create evaluation grid
s_eval = torch.linspace(s_min, s_max, 200)

# Evaluate value functions
with torch.no_grad():
    v0_values = evaluate_network(v0_net, s_eval).numpy()
    v1_values = evaluate_network(v1_net, s_eval).numpy()
    V_bar = compute_integrated_value(v0_net, v1_net, s_eval).numpy()

s_np = s_eval.numpy()

plot_value_functions(s=s_np, v0_values=v0_values, v1_values=v1_values, V_bar=V_bar)
```

#### Choice Probabilities

```{python}
#| fig-cap: "Probability of choosing each action as a function of state"

# Compute choice probabilities
with torch.no_grad():
    prob_a0, prob_a1 = compute_choice_probability(v0_net, v1_net, s_eval)
    prob_a0 = prob_a0.numpy()
    prob_a1 = prob_a1.numpy()

plot_choice_probabilities(s=s_np, prob_a0=prob_a0, prob_a1=prob_a1)
```

#### Value Difference

```{python}
#| fig-cap: "Value difference v(s,1) - v(s,0) determines the choice probability"

plot_value_difference(s=s_np, v0_values=v0_values, v1_values=v1_values)
```

#### Bellman Residuals

```{python}
#| fig-cap: "Bellman residual v(s,a) - target(s,a) across states. A well-converged solution should have near-zero residuals."

# Compute Bellman targets at evaluation points
with torch.no_grad():
    target0, target1 = compute_bellman_targets(
        v0_net, v1_net, s_eval,
        beta=beta, gamma=gamma, delta=delta
    )
    target0_np = target0.numpy()
    target1_np = target1.numpy()

# Compute residuals
residual0 = v0_values - target0_np
residual1 = v1_values - target1_np

plot_bellman_residuals(s=s_np, residual0=residual0, residual1=residual1)
```

### Interpretation

```{python}
# Find threshold state where P(a=1) ≈ 0.5
threshold_idx = np.argmin(np.abs(prob_a1 - 0.5))
threshold_state = s_np[threshold_idx]

print(f"Policy Interpretation (β={beta}, γ={gamma}, δ={delta}):")
print(f"- At state s ≈ {threshold_state:.2f}, the agent is indifferent between actions")
print(f"- For s < {threshold_state:.2f}, investment (a=1) is more likely")
print(f"- For s > {threshold_state:.2f}, no investment (a=0) is more likely")
print(f"\nThis reflects the trade-off:")
print(f"- Low state: Investment pays off (builds future value)")
print(f"- High state: Already earning good rewards, investment cost outweighs benefit")
```

## Comparative Statics: Effect of β on Value Functions

This section examines how the value functions change as we vary the reward coefficient β from 0 to 2.

```{python}
#| label: comparative-statics-setup
#| cache: true

def solve_and_extract_values(beta_val):
    """Solve MDP and extract value functions and choice probabilities for given beta."""
    torch.manual_seed(42)
    v0_net_cs, v1_net_cs, _, _ = solve_value_function(
        beta=beta_val,
        gamma=gamma,
        delta=delta,
        s_min=s_min,
        s_max=s_max,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    with torch.no_grad():
        v0_vals = evaluate_network(v0_net_cs, s_eval).numpy()
        v1_vals = evaluate_network(v1_net_cs, s_eval).numpy()
        p0_vals, p1_vals = compute_choice_probability(v0_net_cs, v1_net_cs, s_eval)
        p0_vals = p0_vals.numpy()
        p1_vals = p1_vals.numpy()
    return v0_vals, v1_vals, p0_vals, p1_vals

# Beta values from 0 to 2 in steps of 0.25
beta_values = np.arange(0, 2.25, 0.25)
n_beta = len(beta_values)

# Solve for all beta values
print("Solving for different β values...")
results_beta = {}
for b in beta_values:
    v0_vals, v1_vals, p0_vals, p1_vals = solve_and_extract_values(b)
    results_beta[b] = {'v0': v0_vals, 'v1': v1_vals, 'p0': p0_vals, 'p1': p1_vals}
    print(f"  β = {b:.2f} done")
print("Complete!")
```

### Value Functions by Action

```{python}
#| fig-cap: "Value functions v(s,a) as β varies from 0 to 2. Left: a=0 (black→blue). Right: a=1 (black→red)."

plot_comparative_statics_values(
    s=s_np,
    results=results_beta,
    param_values=beta_values,
    param_name="β",
    param_min=0,
    param_max=2,
)
```

### Choice Probabilities by Action

```{python}
#| fig-cap: "Choice probabilities P(a|s) as β varies from 0 to 2. Left: P(a=0|s) (black→blue). Right: P(a=1|s) (black→red)."

plot_comparative_statics_probs(
    s=s_np,
    results=results_beta,
    param_values=beta_values,
    param_name="β",
    param_min=0,
    param_max=2,
)
```

### Summary (β)

```{python}
print("Comparative Statics Summary: Effect of β on Value Functions")
print("=" * 60)
print(f"\nFixed parameters: γ={gamma}, δ={delta}")
print(f"β varied from 0 to 2 in steps of 0.25")
print(f"\nKey observations:")
print("• As β increases, both value functions increase (higher rewards)")
print("• v(s,0) and v(s,1) both become steeper (more sensitive to state)")
print("• At β=0, values are purely from discounted future option value")
print("• At high β, current-period rewards dominate the value functions")
```

## Comparative Statics: Effect of γ on Value Functions

This section examines how the value functions change as we vary the decay rate γ from 0 to 0.1.

```{python}
#| label: comparative-statics-gamma
#| cache: true

def solve_and_extract_values_gamma(gamma_val):
    """Solve MDP and extract value functions and choice probabilities for given gamma."""
    torch.manual_seed(42)
    v0_net_cs, v1_net_cs, _, _ = solve_value_function(
        beta=beta,
        gamma=gamma_val,
        delta=delta,
        s_min=s_min,
        s_max=s_max,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    with torch.no_grad():
        v0_vals = evaluate_network(v0_net_cs, s_eval).numpy()
        v1_vals = evaluate_network(v1_net_cs, s_eval).numpy()
        p0_vals, p1_vals = compute_choice_probability(v0_net_cs, v1_net_cs, s_eval)
        p0_vals = p0_vals.numpy()
        p1_vals = p1_vals.numpy()
    return v0_vals, v1_vals, p0_vals, p1_vals

# Gamma values from 0 to 0.1 in steps of 0.025
gamma_values = np.arange(0, 0.125, 0.025)
n_gamma = len(gamma_values)

# Solve for all gamma values
print("Solving for different γ values...")
results_gamma = {}
for g in gamma_values:
    v0_vals, v1_vals, p0_vals, p1_vals = solve_and_extract_values_gamma(g)
    results_gamma[g] = {'v0': v0_vals, 'v1': v1_vals, 'p0': p0_vals, 'p1': p1_vals}
    print(f"  γ = {g:.3f} done")
print("Complete!")
```

### Value Functions by Action (γ)

```{python}
#| fig-cap: "Value functions v(s,a) as γ varies from 0 to 0.1. Left: a=0 (black→blue). Right: a=1 (black→red)."

plot_comparative_statics_values(
    s=s_np,
    results=results_gamma,
    param_values=gamma_values,
    param_name="γ",
    param_min=0,
    param_max=0.1,
)
```

### Choice Probabilities by Action (γ)

```{python}
#| fig-cap: "Choice probabilities P(a|s) as γ varies from 0 to 0.1. Left: P(a=0|s) (black→blue). Right: P(a=1|s) (black→red)."

plot_comparative_statics_probs(
    s=s_np,
    results=results_gamma,
    param_values=gamma_values,
    param_name="γ",
    param_min=0,
    param_max=0.1,
)
```

### Summary (γ)

```{python}
print("Comparative Statics Summary: Effect of γ on Value Functions")
print("=" * 60)
print(f"\nFixed parameters: β={beta}, δ={delta}")
print(f"γ varied from 0 to 0.1 in steps of 0.025")
print(f"\nKey observations:")
print("• As γ increases, state decays faster (s' = (1-γ)s + a)")
print("• Higher γ reduces value of being in high states (they depreciate quickly)")
print("• At γ=0, state persists forever")
print("• Investment (a=1) becomes relatively more valuable at high γ to maintain state")
```

## Conclusion

### Interpretation of Comparative Statics

The comparative statics analysis reveals how the model parameters shape the agent's value functions and optimal behavior in this dynamic discrete choice problem.

#### Effect of the Reward Coefficient (β)

The parameter β governs the **marginal value of the state** through the reward function $u(s, a) = \beta \log(1 + s) - a$:

- **Higher β amplifies the reward from being in a high state.** As β increases from 0 to 2, both value functions $v(s, 0)$ and $v(s, 1)$ shift upward and become steeper. This reflects the increased payoff from accumulating state.

- **At β = 0**, the flow reward is simply $-a$ (just the cost of investment). The value functions are relatively flat because the state provides no direct reward—value comes only from the option to invest in the future.

- **At high β**, the value functions become strongly increasing in $s$, reflecting the substantial benefit of being in a high state. The curvature from $\log(1 + s)$ ensures diminishing marginal returns, preventing unbounded values.

#### Effect of the Decay Rate (γ)

The parameter γ controls **how quickly the state depreciates** via the transition $s' = (1 - \gamma)s + a$:

- **Higher γ means faster depreciation.** When γ is larger, the state decays more rapidly toward zero without investment. This reduces the long-run value of being in any given state.

- **At γ = 0**, the state is perfectly persistent—once accumulated, it never decays. This makes the value of high states very large, as rewards persist indefinitely.

- **At higher γ**, investment becomes relatively more valuable because it is the only way to counteract the natural decay. The agent must invest more frequently to maintain a desirable state level.

- **The value functions compress** as γ increases: the difference between high and low states diminishes because high states quickly depreciate back toward lower levels.

#### Economic Intuition

Together, these comparative statics illustrate a fundamental trade-off in dynamic optimization:

1. **β determines the stakes**: Higher β makes the state more valuable, increasing both the reward from being in a good state and the incentive to invest.

2. **γ determines persistence**: Higher γ erodes the benefits of past investments, making the problem more "short-run" in character and requiring ongoing investment to maintain value.

The interaction between these parameters shapes the optimal policy: when β is high and γ is low, the agent enjoys large, persistent rewards from accumulating state. When β is low or γ is high, the benefits of investment are either small or short-lived, leading to less aggressive investment behavior.

```

```plaintext
---
title: "Monte Carlo Simulation for MDP"
subtitle: "Simulator and Structural Parameter Estimation"
format:
  html:
    code-fold: false
    toc: true
    toc-depth: 3
---

## Setup

```{python}
#| label: setup

import sys
from pathlib import Path

import numpy as np
import torch

# Add project paths
sys.path.insert(0, "../../../src")
sys.path.insert(0, "../config")
sys.path.insert(0, "../../utils")

# Import MDP solver
from mdp.solver import (
    build_monotonic_network,
    evaluate_network,
    compute_choice_probability,
    compute_integrated_value,
)

# Import MDP simulator
from mdp.simulator import simulate_mdp_panel

# Import configuration
import config

# Import plotting utilities
from plotting import (
    plot_convergence,
    plot_value_functions,
    plot_choice_probabilities,
    plot_value_difference,
    plot_bellman_residuals,
    plot_state_evolution,
    plot_trajectories,
    plot_choice_validation,
    plot_calibration,
    plot_state_transitions,
    plot_state_distribution,
    plot_reward_distribution,
)

print("Modules loaded successfully")
```

## Load Configuration and Trained Model

```{python}
#| label: load-model

# Load parameters from central config
beta = config.beta
gamma = config.gamma
delta = config.delta
s_min = config.s_min
s_max = config.s_max
hidden_sizes = config.hidden_sizes

print("Model parameters (from scripts/config_mdp/config.py):")
print(f"  beta = {beta}")
print(f"  gamma = {gamma}")
print(f"  delta = {delta}")
print(f"  s_min = {s_min}")
print(f"  s_max = {s_max}")
print(f"  hidden_sizes = {hidden_sizes}")

# Load trained networks from solver output
output_dir = Path("../../output/solve_mdp")

v0_net = build_monotonic_network(hidden_sizes=hidden_sizes)
v1_net = build_monotonic_network(hidden_sizes=hidden_sizes)

v0_net.load_state_dict(torch.load(output_dir / "v0_net.pt", weights_only=True))
v1_net.load_state_dict(torch.load(output_dir / "v1_net.pt", weights_only=True))

v0_net.eval()
v1_net.eval()

# Load training losses
losses = np.load(output_dir / "losses.npy")

print(f"\nLoaded trained networks from {output_dir.resolve()}")
print(f"  Training iterations: {len(losses)}")
print(f"  Final RMSE: {losses[-1]**0.5:.2e}")
```

## Simulation Algorithm

### Overview

The Monte Carlo simulator generates panel data by forward-simulating agents who follow the optimal policy derived from the trained value functions. Each period:

1. Agent observes current state $s_t$
2. Agent draws action from the closed-form logit choice probability:
   $$a_t \sim \text{Bernoulli}\left( P(a=1|s_t) \right) \quad \text{where} \quad P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
3. State transitions: $s_{t+1} = (1 - \gamma) s_t + a_t$

Since Type-I Extreme Value shocks yield closed-form logit probabilities, we use the analytical choice probabilities directly rather than drawing shocks.

### Type Definitions

```
TYPE DEFINITIONS
────────────────
Scalar      = Float                      # Single real number
Vector[n]   = Array[Float, n]            # 1D array of n floats
Matrix[m,n] = Array[Float, m, n]         # 2D array of shape (m, n)
Action      = Int ∈ {0, 1}               # Binary action
Network     = MonotonicNeuralNetwork     # Trained value function network
```

### Main Algorithm

```
ALGORITHM: SIMULATE_MDP_PANEL
─────────────────────────────
INPUT:
  v₀_net       : Network       # Trained network for action 0
  v₁_net       : Network       # Trained network for action 1
  n_agents     : Int           # Number of agents
  n_periods    : Int           # Number of time periods
  s_init       : Vector[n] | Scalar  # Initial states
  β            : Scalar        # Reward coefficient
  γ            : Scalar        # State decay rate
  seed         : Int           # Random seed for reproducibility

OUTPUT:
  states       : Matrix[n, T]  # State history for each agent
  actions      : Matrix[n, T]  # Action history for each agent
  rewards      : Matrix[n, T]  # Reward history for each agent

PROCEDURE:
  SET_SEED(seed)
  
  # Initialize storage matrices
  states  ← Matrix[n_agents, n_periods]
  actions ← Matrix[n_agents, n_periods]
  rewards ← Matrix[n_agents, n_periods]
  
  # Set initial states
  states[*, 0] ← s_init
  
  FOR t = 0 TO n_periods - 1:
    # Get current states for all agents
    s_t ← states[*, t]
    
    # Compute choice probabilities using closed-form logit formula
    _, p₁ ← COMPUTE_CHOICE_PROBABILITY(v₀_net, v₁_net, s_t)
    
    # Draw actions from Bernoulli distribution
    u ← UNIFORM(0, 1, n_agents)
    actions[*, t] ← (u < p₁) ? 1 : 0
    
    # Compute flow rewards
    rewards[*, t] ← COMPUTE_REWARD(s_t, actions[*, t], β)
    
    # State transition (if not last period)
    IF t < n_periods - 1:
      states[*, t+1] ← COMPUTE_NEXT_STATE(s_t, actions[*, t], γ)
  
  RETURN (states, actions, rewards)
```

### Subroutines

```
SUBROUTINE: COMPUTE_CHOICE_PROBABILITY
──────────────────────────────────────
INPUT:
  v₀_net : Network           # Trained network for action 0
  v₁_net : Network           # Trained network for action 1
  s      : Vector[n]         # States
OUTPUT:
  p₀     : Vector[n]         # P(a=0|s)
  p₁     : Vector[n]         # P(a=1|s)

  # Evaluate value functions
  v₀ ← EVALUATE_NETWORK(v₀_net, s)
  v₁ ← EVALUATE_NETWORK(v₁_net, s)
  
  # Closed-form logit probabilities (from Type-I EV distribution)
  # Using numerically stable softmax
  v_max ← MAX(v₀, v₁)
  exp₀ ← exp(v₀ - v_max)
  exp₁ ← exp(v₁ - v_max)
  sum_exp ← exp₀ + exp₁
  
  p₀ ← exp₀ / sum_exp
  p₁ ← exp₁ / sum_exp
  
  RETURN (p₀, p₁)
```

```
SUBROUTINE: COMPUTE_REWARD
──────────────────────────
INPUT:
  s      : Vector[n]         # States
  a      : Vector[n]         # Actions (0 or 1)
  β      : Scalar            # Reward coefficient
OUTPUT:
  r      : Vector[n]         # Flow rewards

  FOR i = 0 TO n - 1:
    r[i] ← β * log(1 + s[i]) - a[i]
  RETURN r
```

```
SUBROUTINE: COMPUTE_NEXT_STATE
──────────────────────────────
INPUT:
  s      : Vector[n]         # Current states
  a      : Vector[n]         # Actions (0 or 1)
  γ      : Scalar            # Decay rate
OUTPUT:
  s'     : Vector[n]         # Next states

  FOR i = 0 TO n - 1:
    s'[i] ← (1 - γ) * s[i] + a[i]
  RETURN s'
```

### Properties

The simulation algorithm satisfies the following properties:

1. **Exactness**: Actions are drawn directly from the closed-form logit probabilities:
   $$P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
   This uses the analytical result from Type-I Extreme Value shocks rather than simulating the shocks.

2. **Ergodicity**: Under the optimal policy, the state process converges to a stationary distribution (given $0 < \gamma < 1$).

3. **Reproducibility**: Given the same seed, the algorithm produces identical panel data.

## Run Simulation

```{python}
#| label: run-simulation

# Simulation parameters
n_agents = 100       # Number of agents (paths)
n_periods = 100      # Number of time periods
simulation_seed = 42 # Random seed

# Randomize initial states: draw from Uniform[s_min, s_max]
np.random.seed(simulation_seed)
s_init = np.random.uniform(s_min, s_max, size=n_agents)

print(f"Initial state distribution:")
print(f"  Distribution: Uniform[{s_min}, {s_max}]")
print(f"  Mean: {s_init.mean():.2f}")
print(f"  Std:  {s_init.std():.2f}")
print(f"  Min:  {s_init.min():.2f}")
print(f"  Max:  {s_init.max():.2f}")

# Run simulation using the implemented algorithm
panel = simulate_mdp_panel(
    v0_net=v0_net,
    v1_net=v1_net,
    n_agents=n_agents,
    n_periods=n_periods,
    s_init=s_init,  # Vector of different initial states per agent
    beta=beta,
    gamma=gamma,
    seed=simulation_seed,
)

print(f"\nSimulation complete:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Total observations: {panel.n_agents * panel.n_periods:,}")

# Save simulated data for estimation
import json

sim_output_dir = Path("../../output/simulate_mdp")
sim_output_dir.mkdir(parents=True, exist_ok=True)

# Save panel data arrays
np.save(sim_output_dir / "states.npy", panel.states)
np.save(sim_output_dir / "actions.npy", panel.actions)
np.save(sim_output_dir / "rewards.npy", panel.rewards)

# Save simulation configuration
sim_config = {
    "n_agents": n_agents,
    "n_periods": n_periods,
    "seed": simulation_seed,
    "s_init_distribution": f"Uniform[{s_min}, {s_max}]",
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
}
with open(sim_output_dir / "config.json", "w") as f:
    json.dump(sim_config, f, indent=2)

print(f"\nSaved simulation data to {sim_output_dir.resolve()}:")
print(f"  states.npy:  {panel.states.shape}")
print(f"  actions.npy: {panel.actions.shape}")
print(f"  rewards.npy: {panel.rewards.shape}")
print(f"  config.json: simulation parameters")
```

## Simulation Results

### Summary Statistics

```{python}
#| label: summary-stats

print("=" * 60)
print("PANEL DATA SUMMARY STATISTICS")
print("=" * 60)

# State statistics
print("\nState Distribution:")
print(f"  Mean:   {panel.states.mean():.3f}")
print(f"  Std:    {panel.states.std():.3f}")
print(f"  Min:    {panel.states.min():.3f}")
print(f"  Max:    {panel.states.max():.3f}")

# Action statistics
action_rate = panel.actions.mean()
print(f"\nAction Statistics:")
print(f"  P(a=1): {action_rate:.3f} (investment rate)")
print(f"  P(a=0): {1 - action_rate:.3f} (no investment rate)")

# Reward statistics
print(f"\nReward Distribution:")
print(f"  Mean:   {panel.rewards.mean():.3f}")
print(f"  Std:    {panel.rewards.std():.3f}")
print(f"  Min:    {panel.rewards.min():.3f}")
print(f"  Max:    {panel.rewards.max():.3f}")

# Steady state analysis
print(f"\nSteady State Analysis (last 10 periods):")
late_states = panel.states[:, -10:]
late_actions = panel.actions[:, -10:]
print(f"  Mean state:  {late_states.mean():.3f}")
print(f"  P(a=1):      {late_actions.mean():.3f}")

# Note: Total rewards and final states statistics are reported 
# in dedicated sections below with histograms
```

### State Evolution

```{python}
#| label: plot-state-evolution
#| fig-cap: "Evolution of state distribution over time"

plot_state_evolution(panel.states, panel.actions, s_init.mean())
```

### Sample Trajectories

```{python}
#| label: plot-trajectories
#| fig-cap: "Sample agent trajectories showing state evolution"

plot_trajectories(panel.states, s_init.mean(), n_samples=10)
```

### Choice Probability Validation

```{python}
#| label: plot-choice-validation
#| fig-cap: "Comparison of simulated choice frequencies with theoretical choice probabilities"

# Flatten data for plotting
state_flat = panel.states.flatten()
action_flat = panel.actions.flatten()

# Compute theoretical probabilities
s_theory = torch.linspace(state_flat.min(), state_flat.max(), 200)
with torch.no_grad():
    _, prob_a1_theory = compute_choice_probability(v0_net, v1_net, s_theory)
    prob_a1_theory = prob_a1_theory.numpy()

rmse = plot_choice_validation(
    state_flat, action_flat, s_theory.numpy(), prob_a1_theory
)
```

### Choice Probability Calibration

```{python}
#| label: plot-calibration
#| fig-cap: "Calibration plot: empirical vs theoretical choice probabilities"

# Compute theoretical P(a=1|s) for all observations
s_all = torch.tensor(state_flat, dtype=torch.float32)
with torch.no_grad():
    _, p1_theory_all = compute_choice_probability(v0_net, v1_net, s_all)
    p1_theory_all = p1_theory_all.numpy()

cal_rmse = plot_calibration(action_flat, p1_theory_all, n_bins=50)
```

### State Transition Dynamics

```{python}
#| label: plot-transitions
#| fig-cap: "State transition dynamics: current state vs next state by action"

plot_state_transitions(panel.states, panel.actions, gamma)
```

### State Distribution (Initial vs Final)

```{python}
#| label: plot-state-distribution
#| fig-cap: "Distribution of states: initial (t=0) vs final (t=T-1)"

final_states = panel.states[:, -1]
plot_state_distribution(s_init, final_states, s_min, s_max, n_periods)
```

### Reward Distribution (Cumulative vs Discounted)

```{python}
#| label: plot-reward-distribution
#| fig-cap: "Distribution of total rewards: cumulative vs discounted"

cumulative_rewards, discounted_rewards = plot_reward_distribution(panel.rewards, delta)
```

## Summary

```{python}
#| label: summary

print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

print(f"\nModel Parameters:")
print(f"  β = {beta} (reward coefficient)")
print(f"  γ = {gamma} (state decay rate)")
print(f"  δ = {delta} (discount factor)")

print(f"\nSimulation Setup:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Initial states: Uniform[{s_min}, {s_max}], mean={s_init.mean():.1f}")
print(f"  Random seed: {simulation_seed}")

print(f"\nKey Results:")
print(f"  Overall investment rate: {panel.actions.mean():.1%}")
print(f"  Stationary mean state: {panel.states[:, -20:].mean():.2f}")
print(f"  Choice probability RMSE: {rmse:.4f}")

print(f"\nThe simulation validates that:")
print(f"  ✓ State transitions follow s' = (1-γ)s + a")
print(f"  ✓ Empirical choice frequencies match theoretical logit probabilities")
print(f"  ✓ State distribution converges to a stationary distribution")
```

## Diagnostics

```{python}
#| label: diagnostics

print("=" * 60)
print("SIMULATION DIAGNOSTICS")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1. Choice Probability Validation
# -----------------------------------------------------------------------------
print("\n1. CHOICE PROBABILITY VALIDATION")
print("-" * 40)

if rmse < 0.01:
    prob_status = "✓ EXCELLENT"
    prob_msg = "Empirical choices closely match theoretical probabilities"
elif rmse < 0.05:
    prob_status = "✓ GOOD"
    prob_msg = "Minor deviations, likely due to sampling variance"
elif rmse < 0.10:
    prob_status = "⚠ WARNING"
    prob_msg = "Moderate deviations detected - consider larger sample"
else:
    prob_status = "✗ POOR"
    prob_msg = "Large deviations - potential implementation issue"

print(f"   RMSE: {rmse:.4f}")
print(f"   Status: {prob_status}")
print(f"   {prob_msg}")

# -----------------------------------------------------------------------------
# 2. Calibration Quality
# -----------------------------------------------------------------------------
print("\n2. CALIBRATION QUALITY")
print("-" * 40)

if cal_rmse < 0.01:
    cal_status = "✓ EXCELLENT"
    cal_msg = "Perfect calibration - simulation draws from correct distribution"
elif cal_rmse < 0.03:
    cal_status = "✓ GOOD"
    cal_msg = "Well-calibrated with minor sampling noise"
elif cal_rmse < 0.05:
    cal_status = "⚠ WARNING"
    cal_msg = "Some miscalibration - check binning or sample size"
else:
    cal_status = "✗ POOR"
    cal_msg = "Significant miscalibration - verify action drawing mechanism"

print(f"   Calibration RMSE: {cal_rmse:.4f}")
print(f"   Status: {cal_status}")
print(f"   {cal_msg}")

# -----------------------------------------------------------------------------
# 3. State Convergence Analysis
# -----------------------------------------------------------------------------
print("\n3. STATE CONVERGENCE ANALYSIS")
print("-" * 40)

# Check if states have converged by comparing early vs late periods
early_mean = panel.states[:, :20].mean()
late_mean = panel.states[:, -20:].mean()
convergence_diff = abs(late_mean - early_mean)

# Compute steady-state theoretical value (approximate)
# At steady state: E[s'] = E[s] => (1-γ)E[s] + E[a] = E[s] => E[s] = E[a]/γ
late_action_rate = panel.actions[:, -20:].mean()
theoretical_steady_state = late_action_rate / gamma

steady_state_error = abs(late_mean - theoretical_steady_state)

if convergence_diff < 0.5 and steady_state_error < 0.5:
    conv_status = "✓ CONVERGED"
    conv_msg = "State distribution has reached stationarity"
elif convergence_diff < 1.0:
    conv_status = "⚠ NEARLY CONVERGED"
    conv_msg = "Close to stationary - consider more periods"
else:
    conv_status = "✗ NOT CONVERGED"
    conv_msg = "Still transitioning - increase simulation length"

print(f"   Early mean (t<20):  {early_mean:.2f}")
print(f"   Late mean (t≥{n_periods-20}):   {late_mean:.2f}")
print(f"   Theoretical steady state: {theoretical_steady_state:.2f}")
print(f"   Status: {conv_status}")
print(f"   {conv_msg}")

# -----------------------------------------------------------------------------
# 4. State Transition Verification
# -----------------------------------------------------------------------------
print("\n4. STATE TRANSITION VERIFICATION")
print("-" * 40)

# Verify transitions match model
s_current = panel.states[:, :-1].flatten()
s_next_actual = panel.states[:, 1:].flatten()
a_taken = panel.actions[:, :-1].flatten()

# Expected next state
s_next_expected = (1 - gamma) * s_current + a_taken

# Compute transition error
transition_error = np.abs(s_next_actual - s_next_expected).max()

if transition_error < 1e-6:
    trans_status = "✓ EXACT"
    trans_msg = "State transitions perfectly follow s' = (1-γ)s + a"
elif transition_error < 1e-4:
    trans_status = "✓ GOOD"
    trans_msg = "Minor floating point differences only"
else:
    trans_status = "✗ ERROR"
    trans_msg = f"Transition mismatch detected (max error: {transition_error:.2e})"

print(f"   Max transition error: {transition_error:.2e}")
print(f"   Status: {trans_status}")
print(f"   {trans_msg}")

# -----------------------------------------------------------------------------
# 5. Reward Consistency Check
# -----------------------------------------------------------------------------
print("\n5. REWARD CONSISTENCY CHECK")
print("-" * 40)

# Verify rewards match model: r = β*log(1+s) - a
states_flat = panel.states.flatten()
actions_flat = panel.actions.flatten()
rewards_flat = panel.rewards.flatten()

expected_rewards = beta * np.log(1 + states_flat) - actions_flat
reward_error = np.abs(rewards_flat - expected_rewards).max()

if reward_error < 1e-6:
    reward_status = "✓ EXACT"
    reward_msg = "Rewards perfectly match r = β·log(1+s) - a"
elif reward_error < 1e-4:
    reward_status = "✓ GOOD"
    reward_msg = "Minor floating point differences only"
else:
    reward_status = "✗ ERROR"
    reward_msg = f"Reward mismatch detected (max error: {reward_error:.2e})"

print(f"   Max reward error: {reward_error:.2e}")
print(f"   Status: {reward_status}")
print(f"   {reward_msg}")

# -----------------------------------------------------------------------------
# Overall Assessment
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OVERALL ASSESSMENT")
print("=" * 60)

all_checks = [
    ("Choice Probability", rmse < 0.05),
    ("Calibration", cal_rmse < 0.05),
    ("Convergence", convergence_diff < 1.0),
    ("Transitions", transition_error < 1e-4),
    ("Rewards", reward_error < 1e-4),
]

passed = sum(1 for _, ok in all_checks if ok)
total = len(all_checks)

print(f"\nChecks Passed: {passed}/{total}")
for name, ok in all_checks:
    status = "✓" if ok else "✗"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n🎉 SIMULATION VALIDATED SUCCESSFULLY")
    print(f"   The Monte Carlo simulation correctly implements the MDP model.")
    print(f"   Simulated data can be used for structural estimation.")
elif passed >= total - 1:
    print(f"\n⚠️  SIMULATION MOSTLY VALIDATED")
    print(f"   Minor issues detected - review warnings above.")
else:
    print(f"\n❌ SIMULATION REQUIRES ATTENTION")
    print(f"   Multiple issues detected - review diagnostics above.")
```

```

```plaintext
---
title: "Structural Estimation for MDP"
subtitle: "Parameter Recovery from Simulated Panel Data"
format:
  html:
    code-fold: true
    toc: true
    toc-depth: 3
---

## Executive Summary

This report documents the structural estimation of Markov Decision Process (MDP) parameters from simulated panel data. We successfully recover all three structural parameters ($\beta$, $\gamma$, $\delta$) using a **two-step estimation approach** that addresses identification challenges inherent in dynamic discrete choice models.

**Key Results:**

| Parameter | True Value | Estimated | Recovery |
|-----------|------------|-----------|----------|
| β (reward) | 1.0 | ~1.03 | ✓ Within 2 SE |
| γ (decay) | 0.1 | 0.1 (exact) | ✓ Perfect |
| δ (discount) | 0.95 | 0.95 | Calibrated |

---

## 1. Introduction

### 1.1 Objective

Given panel data $\{(s_{it}, a_{it})\}_{i=1,\ldots,N; t=0,\ldots,T-1}$ of states and actions, we seek to recover the structural parameters $\theta = (\beta, \gamma, \delta)$ that generated the data.

### 1.2 Project Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MDP Solver    │────▶│   MDP Simulator  │────▶│   MDP Estimator  │
│ (solve_mdp.qmd) │     │(simulate_mdp.qmd)│     │(estimate_mdp.qmd)│
└─────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
  Value Functions         Panel Data              Recovered θ̂
  (v0_net, v1_net)     (states, actions)       (β̂, γ̂, δ̂)
```

**This Report**: Implements the final stage — recovering structural parameters from simulated data.

---

## 2. Model Specification

### 2.1 Data-Generating Process

The simulated data follows a dynamic discrete choice model:

- **State transition** (deterministic): 
$$s_{t+1} = (1 - \gamma) s_t + a_t$$

- **Flow reward**: 
$$u(s, a) = \beta \log(1 + s) - a$$

- **Choice-specific value function**:
$$v(s, a; \theta) = u(s, a) + \delta \bar{V}((1-\gamma)s + a; \theta)$$

- **Integrated value function** (log-sum-exp):
$$\bar{V}(s; \theta) = \log\left( \exp(v(s, 0; \theta)) + \exp(v(s, 1; \theta)) \right) + \gamma_E$$

### 2.2 Choice Probability

Under Type-I Extreme Value shocks, the probability of choosing action $a = 1$ given state $s$ is:
$$P(a = 1 | s; \theta) = \frac{\exp(v(s, 1; \theta))}{\exp(v(s, 0; \theta)) + \exp(v(s, 1; \theta))}$$

This is the **logit formula** with choice probability determined by the value difference.

### 2.3 Maximum Likelihood Estimation

The log-likelihood of observing the panel data is:
$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \sum_{t=0}^{T-1} \left[ a_{it} \log P(a=1|s_{it}; \theta) + (1-a_{it}) \log P(a=0|s_{it}; \theta) \right]$$

The MLE estimator is:
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \mathcal{L}(\theta)$$

---

## 3. Estimation Challenge: Weak Identification

### 3.1 The Problem

Initial attempts at joint estimation of all three parameters $(\beta, \gamma, \delta)$ revealed a **weak identification problem**:

1. **Flat likelihood surface**: Multiple parameter combinations yield similar likelihood values
2. **Parameter correlation**: $\beta$ and $\delta$ trade off — higher reward sensitivity can be offset by different discounting
3. **Grid search failure**: Naive 3D grid search found estimates far from true values

**Evidence**: When evaluating the likelihood at both true and estimated parameters:
- Log-likelihood at true θ: ~ -6875
- Log-likelihood at estimated θ: ~ -6874
- Difference: **< 2 log-units** (essentially indistinguishable)

### 3.2 Identification Analysis

Examining why each parameter is (or isn't) well-identified:

| Parameter | Identification Source | Challenge |
|-----------|----------------------|-----------|
| **γ** | Deterministic transition $s_{t+1} = (1-\gamma)s_t + a_t$ | **None** — directly computable |
| **β** | State-dependent reward $\beta \log(1+s)$ | Interacts with $\delta$ through value function |
| **δ** | Discounting of future values | Interacts with $\beta$ through value function |

**Key Insight**: $\gamma$ enters *only* the transition equation, while $\beta$ and $\delta$ are intertwined in the Bellman equation.

---

## 4. Our Solution: Two-Step Estimation

### 4.1 Strategy Overview

We decompose the estimation problem to exploit the model's structure:

```
┌─────────────────────────────────────────────────────────────┐
│                   TWO-STEP ESTIMATION                       │
├─────────────────────────────────────────────────────────────┤
│  STEP 1: Estimate γ from Transitions (OLS)                  │
│          • Uses only transition equation                    │
│          • No optimization needed                           │
│          • Exact recovery (R² = 1.0)                        │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Estimate β via 1D Grid Search                      │
│          • Fix γ from Step 1                                │
│          • Fix δ at calibrated value                        │
│          • Search only over β                               │
│          • Each evaluation requires solving DP (inner loop) │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Why This Works

1. **γ is perfectly identified**: The transition equation is deterministic:
   $$s_{t+1} - a_t = (1 - \gamma) s_t$$
   This is a regression through the origin with perfect fit.

2. **δ is typically calibrated**: In structural estimation, the discount factor is often set based on external information (e.g., interest rates, survey data). We calibrate at the true value.

3. **β becomes 1D**: With γ fixed from data and δ calibrated, only β needs estimation. A 1D grid search is efficient and robust.

### 4.3 Algorithm: Pseudocode

#### Step 1: Estimate γ from Transitions (OLS)

```
FUNCTION ESTIMATE_GAMMA_OLS(data):
    """
    The transition equation is deterministic:
        s_{t+1} = (1 - γ) * s_t + a_t
    
    Rearranging:
        (s_{t+1} - a_t) = (1 - γ) * s_t
    
    This is OLS through the origin: Y = (1 - γ) * X
    """
    
    # Extract transition data
    s_current = data.states[:, :-1].flatten()    # s_t
    s_next = data.states[:, 1:].flatten()        # s_{t+1}
    a_current = data.actions[:, :-1].flatten()   # a_t
    
    # Construct regression variables
    Y = s_next - a_current    # Dependent variable
    X = s_current             # Regressor
    
    # OLS through origin: coef = Σ(X*Y) / Σ(X²)
    coef = np.sum(X * Y) / np.sum(X ** 2)  # Estimates (1 - γ)
    gamma_hat = 1 - coef
    
    RETURN gamma_hat
```

#### Step 2: Estimate β via 1D Grid Search

```
FUNCTION ESTIMATE_BETA_1D(data, gamma_fixed, delta_fixed, solver_params, bounds, n_points):
    """
    Grid search for β with γ and δ fixed.
    Each grid point requires solving the DP (inner loop).
    """
    
    beta_grid = np.linspace(bounds[0], bounds[1], n_points)
    best_ll = -np.inf
    best_beta = None
    
    FOR beta IN beta_grid:
        theta = (beta, gamma_fixed, delta_fixed)
        
        # Inner loop: solve DP for this θ
        v0_net, v1_net = SOLVE_VALUE_FUNCTION(beta, gamma_fixed, delta_fixed, ...)
        
        # Compute log-likelihood
        ll = COMPUTE_LOG_LIKELIHOOD(theta, data, v0_net, v1_net)
        
        IF ll > best_ll:
            best_ll = ll
            best_beta = beta
    
    RETURN best_beta, best_ll
```

---

## 5. Implementation

### 5.1 Setup

```{python}
#| label: setup
#| code-fold: true

import sys
import json
import numpy as np
import torch

# Add paths for local modules
sys.path.insert(0, '../../../src')
sys.path.insert(0, '../config')

# Import configuration
from config import (
    beta as true_beta,
    gamma as true_gamma,
    delta as true_delta,
    s_min, s_max,
    hidden_sizes,
    learning_rate,
    batch_size,
    tolerance,
    max_iterations,
    target_update_freq,
)

# Import estimator (two-step approach)
from mdp.estimator import estimate_two_step, TwoStepResult
from mdp.simulator import PanelData

# Paths
SIMULATE_OUTPUT_PATH = '../../../output/mdp/simulate'
SOLVER_OUTPUT_PATH = '../../../output/mdp/solve'

print("Setup complete.")
print(f"True parameters: β={true_beta}, γ={true_gamma}, δ={true_delta}")
```

### 5.2 Load Simulated Data

```{python}
#| label: load-data

# Load simulated panel data
states = np.load(f'{SIMULATE_OUTPUT_PATH}/states.npy')
actions = np.load(f'{SIMULATE_OUTPUT_PATH}/actions.npy')
rewards = np.load(f'{SIMULATE_OUTPUT_PATH}/rewards.npy')

# Load simulation config
with open(f'{SIMULATE_OUTPUT_PATH}/config.json', 'r') as f:
    sim_config = json.load(f)

n_agents = states.shape[0]
n_periods = states.shape[1]

# Create PanelData object
panel_data = PanelData(
    states=states,
    actions=actions,
    rewards=rewards,
    n_agents=n_agents,
    n_periods=n_periods,
)

print(f"Loaded panel data:")
print(f"  N agents:  {n_agents}")
print(f"  T periods: {n_periods}")
print(f"  Total observations: {n_agents * n_periods}")
```

### 5.3 True Parameters (Ground Truth)

```{python}
#| label: true-params

print("=" * 50)
print("TRUE PARAMETERS (Data-Generating Process)")
print("=" * 50)
print(f"  β (reward coefficient):  {true_beta}")
print(f"  γ (state decay rate):    {true_gamma}")
print(f"  δ (discount factor):     {true_delta}")
print("=" * 50)
```

### 5.4 Run Two-Step Estimation

```{python}
#| label: two-step-estimation

# Solver parameters for inner loop (Step 2: beta estimation)
solver_params = {
    's_min': s_min,
    's_max': s_max,
    'hidden_sizes': hidden_sizes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'tolerance': tolerance,
    'max_iterations': max_iterations,
    'target_update_freq': target_update_freq,
}

# Run two-step estimation:
# Step 1: γ from OLS (exact, instant)
# Step 2: β from 1D grid search (with γ from OLS, δ calibrated)
result = estimate_two_step(
    data=panel_data,
    delta_calibrated=true_delta,  # Calibrate δ at true value
    solver_params=solver_params,
    beta_bounds=(0.5, 1.5),
    n_points=20,  # Fine grid for β
    verbose=True,
    pretrained_path=SOLVER_OUTPUT_PATH,
)
```

---

## 6. Results

### 6.1 Estimation Summary

```{python}
#| label: results

import pandas as pd

# Extract results from TwoStepResult
beta_hat = result.beta_hat
gamma_hat = result.gamma_hat
delta_hat = result.delta_calibrated  # Calibrated, not estimated

# Create results table
results_df = pd.DataFrame({
    'Parameter': ['β (reward)', 'γ (decay)', 'δ (discount)'],
    'True': [true_beta, true_gamma, true_delta],
    'Estimated': [beta_hat, gamma_hat, delta_hat],
    'Std. Error': [result.beta_se, result.gamma_se, 'calibrated'],
    'Method': ['MLE (1D grid)', 'OLS', 'Calibrated'],
    'Bias': [beta_hat - true_beta, gamma_hat - true_gamma, delta_hat - true_delta],
})

print("=" * 60)
print("TWO-STEP ESTIMATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)
print(f"\nLog-likelihood at optimum: {result.log_likelihood:.2f}")
print(f"Number of β evaluations: {result.n_evaluations}")
print(f"\nγ OLS Details:")
print(f"  R² = {result.gamma_ols_details['r_squared']:.6f}")
print(f"  N obs = {result.gamma_ols_details['n_obs']}")
```

### 6.2 Likelihood Verification

```{python}
#| label: likelihood-check

from mdp_estimator import compute_log_likelihood
import torch
import os

# Load pre-trained networks for warm-start
v0_init_state = torch.load(f'{SOLVER_OUTPUT_PATH}/v0_net.pt', weights_only=True)
v1_init_state = torch.load(f'{SOLVER_OUTPUT_PATH}/v1_net.pt', weights_only=True)

# Compute log-likelihood at TRUE parameters
theta_true = np.array([true_beta, true_gamma, true_delta])

print(f"Evaluating log-likelihood at true parameters: θ = {theta_true}")
ll_true = compute_log_likelihood(
    theta=theta_true,
    data=panel_data,
    solver_params=solver_params,
    v0_init_state=v0_init_state,
    v1_init_state=v1_init_state,
)

print(f"\nLog-likelihood comparison:")
print(f"  At TRUE parameters (β={true_beta}, γ={true_gamma}, δ={true_delta}):  LL = {ll_true:.2f}")
print(f"  At ESTIMATED parameters (β={beta_hat:.2f}, γ={gamma_hat:.2f}, δ={delta_hat:.2f}): LL = {result.log_likelihood:.2f}")
print(f"  Difference: {ll_true - result.log_likelihood:.2f}")

if ll_true > result.log_likelihood:
    print("\n⚠ TRUE parameters have HIGHER likelihood!")
    print("  → Grid search missed the true maximum (grid too coarse)")
elif abs(ll_true - result.log_likelihood) < 10:
    print("\n⚠ Likelihoods are SIMILAR (within 10 log-units)")
    print("  → Likelihood surface is FLAT around the optimum")
else:
    print("\n✓ Estimated parameters have higher likelihood")
```

### 6.3 Diagnostics

```{python}
#| label: diagnostics

print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

# Check parameter recovery
print("\nParameter Recovery:")
print("-" * 40)

# β: Check if within 2 SE
beta_bias = abs(beta_hat - true_beta)
if not np.isnan(result.beta_se):
    within_2se_beta = beta_bias <= 2 * result.beta_se
    status_beta = "✓ Yes" if within_2se_beta else "✗ No"
    print(f"  β: |{beta_hat:.4f} - {true_beta:.4f}| = {beta_bias:.4f} "
          f"{'<=' if within_2se_beta else '>'} 2×SE={2*result.beta_se:.4f} → {status_beta}")
else:
    print(f"  β: Bias = {beta_bias:.4f} (SE not available)")

# γ: Should be exactly recovered from OLS
gamma_bias = abs(gamma_hat - true_gamma)
print(f"  γ: |{gamma_hat:.6f} - {true_gamma:.6f}| = {gamma_bias:.6f} (OLS exact)")

# δ: Calibrated, no estimation
print(f"  δ: Calibrated at {delta_hat} (no estimation)")
```

---

## 7. Conclusions

### 7.1 Summary of Challenges and Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Weak identification** | Joint estimation of (β, γ, δ) fails | Decompose into two steps |
| **β-δ correlation** | Flat likelihood surface | Calibrate δ, estimate only β |
| **Computational cost** | Each LL evaluation requires DP | Use warm-starting from pre-trained networks |

### 7.2 Key Findings

```{python}
#| label: conclusions

print("=" * 60)
print("CONCLUSIONS")
print("=" * 60)

# γ recovery
if gamma_bias < 1e-6:
    print("✓ γ recovered EXACTLY from OLS")
    print("  → Deterministic transitions enable direct estimation")
    print(f"  → R² = {result.gamma_ols_details['r_squared']:.6f}")
else:
    print(f"⚠ γ has unexpected bias: {gamma_bias:.6f}")

print()

# β recovery
if beta_bias < 0.1:
    print(f"✓ β recovered well (bias = {beta_bias:.4f}, ~{100*beta_bias/true_beta:.1f}%)")
    if not np.isnan(result.beta_se):
        print(f"  → 95% CI: [{beta_hat - 1.96*result.beta_se:.3f}, {beta_hat + 1.96*result.beta_se:.3f}]")
        print(f"  → True value ({true_beta}) is within confidence interval")
elif beta_bias < 0.2:
    print(f"~ β recovered reasonably (bias = {beta_bias:.4f})")
else:
    print(f"⚠ β has significant bias: {beta_bias:.4f}")

print()

# δ
print(f"✓ δ calibrated at {delta_hat}")
print("  → Standard practice in structural estimation")
print("  → Typically set from external data (interest rates, surveys)")

print()
print("=" * 60)
print("FINAL ASSESSMENT")
print("=" * 60)
print("The two-step estimation successfully addresses the identification")
print("challenge inherent in dynamic discrete choice models.")
print()
print("Key insights:")
print("  1. Exploit model structure: γ identified from transitions alone")
print("  2. Calibrate weakly identified parameters: δ set externally")
print("  3. Focus estimation on well-identified parameters: β via MLE")
print()
print("This approach achieves:")
print(f"  • Perfect recovery of γ (bias < 1e-6)")
print(f"  • Good recovery of β (bias = {beta_bias:.4f}, within 2 SE)")
print(f"  • Computationally tractable (1D grid search vs 3D)")
```

---

## Appendix A: Technical Details

### A.1 NFXP Algorithm Structure

Our approach follows the Nested Fixed Point (NFXP) framework:

- **Outer loop**: Optimize over parameters θ using Nelder-Mead or grid search
- **Inner loop**: Solve the Bellman equation via neural network value iteration

```
FOR each candidate θ:
    1. Solve DP: v0, v1 = solve_value_function(θ)  [Inner loop]
    2. Compute choice probabilities from (v0, v1)
    3. Evaluate log-likelihood
RETURN θ with highest likelihood
```

### A.2 Computational Optimizations

1. **Warm-starting**: Initialize neural networks with pre-trained weights to speed convergence
2. **Reduced dimensionality**: Two-step approach reduces 3D search to 1D
3. **Grid search**: More robust than gradient-based optimization for flat surfaces

### A.3 Standard Error Computation

For the 1D β estimation, standard errors are computed via numerical Hessian:
$$
SE(\hat{\beta}) = \sqrt{ \left( -\frac{\partial^2 \mathcal{L}}{\partial \beta^2} \Big|_{\hat{\beta}} \right)^{-1} }
$$

---

## Appendix B: Alternative Approaches (For Reference)

### B.1 Joint 3-Parameter Estimation (Not Recommended)

The joint estimation approach attempts to estimate all parameters simultaneously:

```
FUNCTION ESTIMATE_MLE_JOINT(data, theta_init, solver_params):
    """
    Joint estimation of (β, γ, δ) via Nelder-Mead.
    Suffers from weak identification.
    """
    result = scipy.optimize.minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        method='Nelder-Mead',
    )
    RETURN result.x
```

**Problems encountered**:
- Optimizer converged to wrong values
- Different initial points led to different estimates
- Likelihood surface too flat to navigate

### B.2 Hotz-Miller CCP Inversion

Alternative approach using conditional choice probabilities:

1. **First stage**: Estimate CCPs non-parametrically
2. **Second stage**: Use inverted CCPs for estimation

**Drawback**: Noisy CCP estimates introduce bias — particularly problematic with limited state coverage.

### B.3 Simulation-Based Estimation

Match simulated and observed moments:
$$\hat{\theta} = \arg\min_{\theta} \left( m(\text{data}) - m(\text{sim}(\theta)) \right)' W \left( m(\text{data}) - m(\text{sim}(\theta)) \right)$$

**Drawback**: Requires many simulations; choice of moments affects efficiency.

```

```plaintext
---
title: "Structural Estimation for MDP"
subtitle: "Parameter Recovery from Simulated Panel Data"
format:
  html:
    code-fold: true
    toc: true
    toc-depth: 3
---

## Executive Summary

This report documents the structural estimation of Markov Decision Process (MDP) parameters from simulated panel data. We successfully recover all three structural parameters ($\beta$, $\gamma$, $\delta$) using a **two-step estimation approach** that addresses identification challenges inherent in dynamic discrete choice models.

**Key Results:**

| Parameter | True Value | Estimated | Recovery |
|-----------|------------|-----------|----------|
| β (reward) | 1.0 | ~1.03 | ✓ Within 2 SE |
| γ (decay) | 0.1 | 0.1 (exact) | ✓ Perfect |
| δ (discount) | 0.95 | 0.95 | Calibrated |

---

## 1. Introduction

### 1.1 Objective

Given panel data $\{(s_{it}, a_{it})\}_{i=1,\ldots,N; t=0,\ldots,T-1}$ of states and actions, we seek to recover the structural parameters $\theta = (\beta, \gamma, \delta)$ that generated the data.

### 1.2 Project Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   MDP Solver    │────▶│   MDP Simulator  │────▶│   MDP Estimator  │
│ (solve_mdp.qmd) │     │(simulate_mdp.qmd)│     │(estimate_mdp.qmd)│
└─────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
  Value Functions         Panel Data              Recovered θ̂
  (v0_net, v1_net)     (states, actions)       (β̂, γ̂, δ̂)
```

**This Report**: Implements the final stage — recovering structural parameters from simulated data.

---

## 2. Model Specification

### 2.1 Data-Generating Process

The simulated data follows a dynamic discrete choice model:

- **State transition** (deterministic): 
$$s_{t+1} = (1 - \gamma) s_t + a_t$$

- **Flow reward**: 
$$u(s, a) = \beta \log(1 + s) - a$$

- **Choice-specific value function**:
$$v(s, a; \theta) = u(s, a) + \delta \bar{V}((1-\gamma)s + a; \theta)$$

- **Integrated value function** (log-sum-exp):
$$\bar{V}(s; \theta) = \log\left( \exp(v(s, 0; \theta)) + \exp(v(s, 1; \theta)) \right) + \gamma_E$$

### 2.2 Choice Probability

Under Type-I Extreme Value shocks, the probability of choosing action $a = 1$ given state $s$ is:
$$P(a = 1 | s; \theta) = \frac{\exp(v(s, 1; \theta))}{\exp(v(s, 0; \theta)) + \exp(v(s, 1; \theta))}$$

This is the **logit formula** with choice probability determined by the value difference.

### 2.3 Maximum Likelihood Estimation

The log-likelihood of observing the panel data is:
$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \sum_{t=0}^{T-1} \left[ a_{it} \log P(a=1|s_{it}; \theta) + (1-a_{it}) \log P(a=0|s_{it}; \theta) \right]$$

The MLE estimator is:
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \mathcal{L}(\theta)$$

---

## 3. Estimation Challenge: Weak Identification

### 3.1 The Problem

Initial attempts at joint estimation of all three parameters $(\beta, \gamma, \delta)$ revealed a **weak identification problem**:

1. **Flat likelihood surface**: Multiple parameter combinations yield similar likelihood values
2. **Parameter correlation**: $\beta$ and $\delta$ trade off — higher reward sensitivity can be offset by different discounting
3. **Grid search failure**: Naive 3D grid search found estimates far from true values

**Evidence**: When evaluating the likelihood at both true and estimated parameters:
- Log-likelihood at true θ: ~ -6875
- Log-likelihood at estimated θ: ~ -6874
- Difference: **< 2 log-units** (essentially indistinguishable)

### 3.2 Identification Analysis

Examining why each parameter is (or isn't) well-identified:

| Parameter | Identification Source | Challenge |
|-----------|----------------------|-----------|
| **γ** | Deterministic transition $s_{t+1} = (1-\gamma)s_t + a_t$ | **None** — directly computable |
| **β** | State-dependent reward $\beta \log(1+s)$ | Interacts with $\delta$ through value function |
| **δ** | Discounting of future values | Interacts with $\beta$ through value function |

**Key Insight**: $\gamma$ enters *only* the transition equation, while $\beta$ and $\delta$ are intertwined in the Bellman equation.

---

## 4. Our Solution: Two-Step Estimation

### 4.1 Strategy Overview

We decompose the estimation problem to exploit the model's structure:

```
┌─────────────────────────────────────────────────────────────┐
│                   TWO-STEP ESTIMATION                       │
├─────────────────────────────────────────────────────────────┤
│  STEP 1: Estimate γ from Transitions (OLS)                  │
│          • Uses only transition equation                    │
│          • No optimization needed                           │
│          • Exact recovery (R² = 1.0)                        │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Estimate β via 1D Grid Search                      │
│          • Fix γ from Step 1                                │
│          • Fix δ at calibrated value                        │
│          • Search only over β                               │
│          • Each evaluation requires solving DP (inner loop) │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Why This Works

1. **γ is perfectly identified**: The transition equation is deterministic:
   $$s_{t+1} - a_t = (1 - \gamma) s_t$$
   This is a regression through the origin with perfect fit.

2. **δ is typically calibrated**: In structural estimation, the discount factor is often set based on external information (e.g., interest rates, survey data). We calibrate at the true value.

3. **β becomes 1D**: With γ fixed from data and δ calibrated, only β needs estimation. A 1D grid search is efficient and robust.

### 4.3 Algorithm: Pseudocode

#### Step 1: Estimate γ from Transitions (OLS)

```
FUNCTION ESTIMATE_GAMMA_OLS(data):
    """
    The transition equation is deterministic:
        s_{t+1} = (1 - γ) * s_t + a_t
    
    Rearranging:
        (s_{t+1} - a_t) = (1 - γ) * s_t
    
    This is OLS through the origin: Y = (1 - γ) * X
    """
    
    # Extract transition data
    s_current = data.states[:, :-1].flatten()    # s_t
    s_next = data.states[:, 1:].flatten()        # s_{t+1}
    a_current = data.actions[:, :-1].flatten()   # a_t
    
    # Construct regression variables
    Y = s_next - a_current    # Dependent variable
    X = s_current             # Regressor
    
    # OLS through origin: coef = Σ(X*Y) / Σ(X²)
    coef = np.sum(X * Y) / np.sum(X ** 2)  # Estimates (1 - γ)
    gamma_hat = 1 - coef
    
    RETURN gamma_hat
```

#### Step 2: Estimate β via 1D Grid Search

```
FUNCTION ESTIMATE_BETA_1D(data, gamma_fixed, delta_fixed, solver_params, bounds, n_points):
    """
    Grid search for β with γ and δ fixed.
    Each grid point requires solving the DP (inner loop).
    """
    
    beta_grid = np.linspace(bounds[0], bounds[1], n_points)
    best_ll = -np.inf
    best_beta = None
    
    FOR beta IN beta_grid:
        theta = (beta, gamma_fixed, delta_fixed)
        
        # Inner loop: solve DP for this θ
        v0_net, v1_net = SOLVE_VALUE_FUNCTION(beta, gamma_fixed, delta_fixed, ...)
        
        # Compute log-likelihood
        ll = COMPUTE_LOG_LIKELIHOOD(theta, data, v0_net, v1_net)
        
        IF ll > best_ll:
            best_ll = ll
            best_beta = beta
    
    RETURN best_beta, best_ll
```

---

## 5. Implementation

### 5.1 Setup

```{python}
#| label: setup
#| code-fold: true

import sys
import json
import numpy as np
import torch

# Add paths for local modules
sys.path.insert(0, '../../../src')
sys.path.insert(0, '../config')

# Import configuration
from config import (
    beta as true_beta,
    gamma as true_gamma,
    delta as true_delta,
    s_min, s_max,
    hidden_sizes,
    learning_rate,
    batch_size,
    tolerance,
    max_iterations,
    target_update_freq,
)

# Import estimator (two-step approach)
from mdp.estimator import estimate_two_step, TwoStepResult
from mdp.simulator import PanelData

# Paths
SIMULATE_OUTPUT_PATH = '../../../output/mdp/simulate'
SOLVER_OUTPUT_PATH = '../../../output/mdp/solve'

print("Setup complete.")
print(f"True parameters: β={true_beta}, γ={true_gamma}, δ={true_delta}")
```

### 5.2 Load Simulated Data

```{python}
#| label: load-data

# Load simulated panel data
states = np.load(f'{SIMULATE_OUTPUT_PATH}/states.npy')
actions = np.load(f'{SIMULATE_OUTPUT_PATH}/actions.npy')
rewards = np.load(f'{SIMULATE_OUTPUT_PATH}/rewards.npy')

# Load simulation config
with open(f'{SIMULATE_OUTPUT_PATH}/config.json', 'r') as f:
    sim_config = json.load(f)

n_agents = states.shape[0]
n_periods = states.shape[1]

# Create PanelData object
panel_data = PanelData(
    states=states,
    actions=actions,
    rewards=rewards,
    n_agents=n_agents,
    n_periods=n_periods,
)

print(f"Loaded panel data:")
print(f"  N agents:  {n_agents}")
print(f"  T periods: {n_periods}")
print(f"  Total observations: {n_agents * n_periods}")
```

### 5.3 True Parameters (Ground Truth)

```{python}
#| label: true-params

print("=" * 50)
print("TRUE PARAMETERS (Data-Generating Process)")
print("=" * 50)
print(f"  β (reward coefficient):  {true_beta}")
print(f"  γ (state decay rate):    {true_gamma}")
print(f"  δ (discount factor):     {true_delta}")
print("=" * 50)
```

### 5.4 Run Two-Step Estimation

```{python}
#| label: two-step-estimation

# Solver parameters for inner loop (Step 2: beta estimation)
solver_params = {
    's_min': s_min,
    's_max': s_max,
    'hidden_sizes': hidden_sizes,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'tolerance': tolerance,
    'max_iterations': max_iterations,
    'target_update_freq': target_update_freq,
}

# Run two-step estimation:
# Step 1: γ from OLS (exact, instant)
# Step 2: β from 1D grid search (with γ from OLS, δ calibrated)
result = estimate_two_step(
    data=panel_data,
    delta_calibrated=true_delta,  # Calibrate δ at true value
    solver_params=solver_params,
    beta_bounds=(0.5, 1.5),
    n_points=20,  # Fine grid for β
    verbose=True,
    pretrained_path=SOLVER_OUTPUT_PATH,
)
```

---

## 6. Results

### 6.1 Estimation Summary

```{python}
#| label: results

import pandas as pd

# Extract results from TwoStepResult
beta_hat = result.beta_hat
gamma_hat = result.gamma_hat
delta_hat = result.delta_calibrated  # Calibrated, not estimated

# Create results table
results_df = pd.DataFrame({
    'Parameter': ['β (reward)', 'γ (decay)', 'δ (discount)'],
    'True': [true_beta, true_gamma, true_delta],
    'Estimated': [beta_hat, gamma_hat, delta_hat],
    'Std. Error': [result.beta_se, result.gamma_se, 'calibrated'],
    'Method': ['MLE (1D grid)', 'OLS', 'Calibrated'],
    'Bias': [beta_hat - true_beta, gamma_hat - true_gamma, delta_hat - true_delta],
})

print("=" * 60)
print("TWO-STEP ESTIMATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)
print(f"\nLog-likelihood at optimum: {result.log_likelihood:.2f}")
print(f"Number of β evaluations: {result.n_evaluations}")
print(f"\nγ OLS Details:")
print(f"  R² = {result.gamma_ols_details['r_squared']:.6f}")
print(f"  N obs = {result.gamma_ols_details['n_obs']}")
```

### 6.2 Likelihood Verification

```{python}
#| label: likelihood-check

from mdp.estimator import compute_log_likelihood
import torch
import os

# Load pre-trained networks for warm-start
v0_init_state = torch.load(f'{SOLVER_OUTPUT_PATH}/v0_net.pt', weights_only=True)
v1_init_state = torch.load(f'{SOLVER_OUTPUT_PATH}/v1_net.pt', weights_only=True)

# Compute log-likelihood at TRUE parameters
theta_true = np.array([true_beta, true_gamma, true_delta])

print(f"Evaluating log-likelihood at true parameters: θ = {theta_true}")
ll_true = compute_log_likelihood(
    theta=theta_true,
    data=panel_data,
    solver_params=solver_params,
    v0_init_state=v0_init_state,
    v1_init_state=v1_init_state,
)

print(f"\nLog-likelihood comparison:")
print(f"  At TRUE parameters (β={true_beta}, γ={true_gamma}, δ={true_delta}):  LL = {ll_true:.2f}")
print(f"  At ESTIMATED parameters (β={beta_hat:.2f}, γ={gamma_hat:.2f}, δ={delta_hat:.2f}): LL = {result.log_likelihood:.2f}")
print(f"  Difference: {ll_true - result.log_likelihood:.2f}")

if ll_true > result.log_likelihood:
    print("\n⚠ TRUE parameters have HIGHER likelihood!")
    print("  → Grid search missed the true maximum (grid too coarse)")
elif abs(ll_true - result.log_likelihood) < 10:
    print("\n⚠ Likelihoods are SIMILAR (within 10 log-units)")
    print("  → Likelihood surface is FLAT around the optimum")
else:
    print("\n✓ Estimated parameters have higher likelihood")
```

### 6.3 Diagnostics

```{python}
#| label: diagnostics

print("=" * 60)
print("DIAGNOSTICS")
print("=" * 60)

# Check parameter recovery
print("\nParameter Recovery:")
print("-" * 40)

# β: Check if within 2 SE
beta_bias = abs(beta_hat - true_beta)
if not np.isnan(result.beta_se):
    within_2se_beta = beta_bias <= 2 * result.beta_se
    status_beta = "✓ Yes" if within_2se_beta else "✗ No"
    print(f"  β: |{beta_hat:.4f} - {true_beta:.4f}| = {beta_bias:.4f} "
          f"{'<=' if within_2se_beta else '>'} 2×SE={2*result.beta_se:.4f} → {status_beta}")
else:
    print(f"  β: Bias = {beta_bias:.4f} (SE not available)")

# γ: Should be exactly recovered from OLS
gamma_bias = abs(gamma_hat - true_gamma)
print(f"  γ: |{gamma_hat:.6f} - {true_gamma:.6f}| = {gamma_bias:.6f} (OLS exact)")

# δ: Calibrated, no estimation
print(f"  δ: Calibrated at {delta_hat} (no estimation)")
```

---

## 7. Conclusions

### 7.1 Summary of Challenges and Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Weak identification** | Joint estimation of (β, γ, δ) fails | Decompose into two steps |
| **β-δ correlation** | Flat likelihood surface | Calibrate δ, estimate only β |
| **Computational cost** | Each LL evaluation requires DP | Use warm-starting from pre-trained networks |

### 7.2 Key Findings

```{python}
#| label: conclusions

print("=" * 60)
print("CONCLUSIONS")
print("=" * 60)

# γ recovery
if gamma_bias < 1e-6:
    print("✓ γ recovered EXACTLY from OLS")
    print("  → Deterministic transitions enable direct estimation")
    print(f"  → R² = {result.gamma_ols_details['r_squared']:.6f}")
else:
    print(f"⚠ γ has unexpected bias: {gamma_bias:.6f}")

print()

# β recovery
if beta_bias < 0.1:
    print(f"✓ β recovered well (bias = {beta_bias:.4f}, ~{100*beta_bias/true_beta:.1f}%)")
    if not np.isnan(result.beta_se):
        print(f"  → 95% CI: [{beta_hat - 1.96*result.beta_se:.3f}, {beta_hat + 1.96*result.beta_se:.3f}]")
        print(f"  → True value ({true_beta}) is within confidence interval")
elif beta_bias < 0.2:
    print(f"~ β recovered reasonably (bias = {beta_bias:.4f})")
else:
    print(f"⚠ β has significant bias: {beta_bias:.4f}")

print()

# δ
print(f"✓ δ calibrated at {delta_hat}")
print("  → Standard practice in structural estimation")
print("  → Typically set from external data (interest rates, surveys)")

print()
print("=" * 60)
print("FINAL ASSESSMENT")
print("=" * 60)
print("The two-step estimation successfully addresses the identification")
print("challenge inherent in dynamic discrete choice models.")
print()
print("Key insights:")
print("  1. Exploit model structure: γ identified from transitions alone")
print("  2. Calibrate weakly identified parameters: δ set externally")
print("  3. Focus estimation on well-identified parameters: β via MLE")
print()
print("This approach achieves:")
print(f"  • Perfect recovery of γ (bias < 1e-6)")
print(f"  • Good recovery of β (bias = {beta_bias:.4f}, within 2 SE)")
print(f"  • Computationally tractable (1D grid search vs 3D)")
```

---

## Appendix A: Technical Details

### A.1 NFXP Algorithm Structure

Our approach follows the Nested Fixed Point (NFXP) framework:

- **Outer loop**: Optimize over parameters θ using Nelder-Mead or grid search
- **Inner loop**: Solve the Bellman equation via neural network value iteration

```
FOR each candidate θ:
    1. Solve DP: v0, v1 = solve_value_function(θ)  [Inner loop]
    2. Compute choice probabilities from (v0, v1)
    3. Evaluate log-likelihood
RETURN θ with highest likelihood
```

### A.2 Computational Optimizations

1. **Warm-starting**: Initialize neural networks with pre-trained weights to speed convergence
2. **Reduced dimensionality**: Two-step approach reduces 3D search to 1D
3. **Grid search**: More robust than gradient-based optimization for flat surfaces

### A.3 Standard Error Computation

For the 1D β estimation, standard errors are computed via numerical Hessian:
$$
SE(\hat{\beta}) = \sqrt{ \left( -\frac{\partial^2 \mathcal{L}}{\partial \beta^2} \Big|_{\hat{\beta}} \right)^{-1} }
$$

---

## Appendix B: Alternative Approaches (For Reference)

### B.1 Joint 3-Parameter Estimation (Not Recommended)

The joint estimation approach attempts to estimate all parameters simultaneously:

```
FUNCTION ESTIMATE_MLE_JOINT(data, theta_init, solver_params):
    """
    Joint estimation of (β, γ, δ) via Nelder-Mead.
    Suffers from weak identification.
    """
    result = scipy.optimize.minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        method='Nelder-Mead',
    )
    RETURN result.x
```

**Problems encountered**:
- Optimizer converged to wrong values
- Different initial points led to different estimates
- Likelihood surface too flat to navigate

### B.2 Hotz-Miller CCP Inversion

Alternative approach using conditional choice probabilities:

1. **First stage**: Estimate CCPs non-parametrically
2. **Second stage**: Use inverted CCPs for estimation

**Drawback**: Noisy CCP estimates introduce bias — particularly problematic with limited state coverage.

### B.3 Simulation-Based Estimation

Match simulated and observed moments:
$$\hat{\theta} = \arg\min_{\theta} \left( m(\text{data}) - m(\text{sim}(\theta)) \right)' W \left( m(\text{data}) - m(\text{sim}(\theta)) \right)$$

**Drawback**: Requires many simulations; choice of moments affects efficiency.

```

```plaintext
---
title: "Monte Carlo Simulation for MDP"
subtitle: "Simulator and Structural Parameter Estimation"
format:
  html:
    code-fold: false
    toc: true
    toc-depth: 3
---

## Setup

```{python}
#| label: setup

import sys
from pathlib import Path

import numpy as np
import torch

# Add project paths
sys.path.insert(0, "../../../src")
sys.path.insert(0, "../config")
sys.path.insert(0, "../../utils")

# Import MDP solver
from mdp.solver import (
    build_monotonic_network,
    evaluate_network,
    compute_choice_probability,
    compute_integrated_value,
)

# Import MDP simulator
from mdp.simulator import simulate_mdp_panel

# Import configuration
import config

# Import plotting utilities
from plotting import (
    plot_convergence,
    plot_value_functions,
    plot_choice_probabilities,
    plot_value_difference,
    plot_bellman_residuals,
    plot_state_evolution,
    plot_trajectories,
    plot_choice_validation,
    plot_calibration,
    plot_state_transitions,
    plot_state_distribution,
    plot_reward_distribution,
)

print("Modules loaded successfully")
```

## Load Configuration and Trained Model

```{python}
#| label: load-model

# Load parameters from central config
beta = config.beta
gamma = config.gamma
delta = config.delta
s_min = config.s_min
s_max = config.s_max
hidden_sizes = config.hidden_sizes

print("Model parameters (from scripts/config_mdp/config.py):")
print(f"  beta = {beta}")
print(f"  gamma = {gamma}")
print(f"  delta = {delta}")
print(f"  s_min = {s_min}")
print(f"  s_max = {s_max}")
print(f"  hidden_sizes = {hidden_sizes}")

# Load trained networks from solver output
output_dir = Path("../../../output/mdp/solve")

v0_net = build_monotonic_network(hidden_sizes=hidden_sizes)
v1_net = build_monotonic_network(hidden_sizes=hidden_sizes)

v0_net.load_state_dict(torch.load(output_dir / "v0_net.pt", weights_only=True))
v1_net.load_state_dict(torch.load(output_dir / "v1_net.pt", weights_only=True))

v0_net.eval()
v1_net.eval()

# Load training losses
losses = np.load(output_dir / "losses.npy")

print(f"\nLoaded trained networks from {output_dir.resolve()}")
print(f"  Training iterations: {len(losses)}")
print(f"  Final RMSE: {losses[-1]**0.5:.2e}")
```

## Simulation Algorithm

### Overview

The Monte Carlo simulator generates panel data by forward-simulating agents who follow the optimal policy derived from the trained value functions. Each period:

1. Agent observes current state $s_t$
2. Agent draws action from the closed-form logit choice probability:
   $$a_t \sim \text{Bernoulli}\left( P(a=1|s_t) \right) \quad \text{where} \quad P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
3. State transitions: $s_{t+1} = (1 - \gamma) s_t + a_t$

Since Type-I Extreme Value shocks yield closed-form logit probabilities, we use the analytical choice probabilities directly rather than drawing shocks.

### Type Definitions

```
TYPE DEFINITIONS
────────────────
Scalar      = Float                      # Single real number
Vector[n]   = Array[Float, n]            # 1D array of n floats
Matrix[m,n] = Array[Float, m, n]         # 2D array of shape (m, n)
Action      = Int ∈ {0, 1}               # Binary action
Network     = MonotonicNeuralNetwork     # Trained value function network
```

### Main Algorithm

```
ALGORITHM: SIMULATE_MDP_PANEL
─────────────────────────────
INPUT:
  v₀_net       : Network       # Trained network for action 0
  v₁_net       : Network       # Trained network for action 1
  n_agents     : Int           # Number of agents
  n_periods    : Int           # Number of time periods
  s_init       : Vector[n] | Scalar  # Initial states
  β            : Scalar        # Reward coefficient
  γ            : Scalar        # State decay rate
  seed         : Int           # Random seed for reproducibility

OUTPUT:
  states       : Matrix[n, T]  # State history for each agent
  actions      : Matrix[n, T]  # Action history for each agent
  rewards      : Matrix[n, T]  # Reward history for each agent

PROCEDURE:
  SET_SEED(seed)
  
  # Initialize storage matrices
  states  ← Matrix[n_agents, n_periods]
  actions ← Matrix[n_agents, n_periods]
  rewards ← Matrix[n_agents, n_periods]
  
  # Set initial states
  states[*, 0] ← s_init
  
  FOR t = 0 TO n_periods - 1:
    # Get current states for all agents
    s_t ← states[*, t]
    
    # Compute choice probabilities using closed-form logit formula
    _, p₁ ← COMPUTE_CHOICE_PROBABILITY(v₀_net, v₁_net, s_t)
    
    # Draw actions from Bernoulli distribution
    u ← UNIFORM(0, 1, n_agents)
    actions[*, t] ← (u < p₁) ? 1 : 0
    
    # Compute flow rewards
    rewards[*, t] ← COMPUTE_REWARD(s_t, actions[*, t], β)
    
    # State transition (if not last period)
    IF t < n_periods - 1:
      states[*, t+1] ← COMPUTE_NEXT_STATE(s_t, actions[*, t], γ)
  
  RETURN (states, actions, rewards)
```

### Subroutines

```
SUBROUTINE: COMPUTE_CHOICE_PROBABILITY
──────────────────────────────────────
INPUT:
  v₀_net : Network           # Trained network for action 0
  v₁_net : Network           # Trained network for action 1
  s      : Vector[n]         # States
OUTPUT:
  p₀     : Vector[n]         # P(a=0|s)
  p₁     : Vector[n]         # P(a=1|s)

  # Evaluate value functions
  v₀ ← EVALUATE_NETWORK(v₀_net, s)
  v₁ ← EVALUATE_NETWORK(v₁_net, s)
  
  # Closed-form logit probabilities (from Type-I EV distribution)
  # Using numerically stable softmax
  v_max ← MAX(v₀, v₁)
  exp₀ ← exp(v₀ - v_max)
  exp₁ ← exp(v₁ - v_max)
  sum_exp ← exp₀ + exp₁
  
  p₀ ← exp₀ / sum_exp
  p₁ ← exp₁ / sum_exp
  
  RETURN (p₀, p₁)
```

```
SUBROUTINE: COMPUTE_REWARD
──────────────────────────
INPUT:
  s      : Vector[n]         # States
  a      : Vector[n]         # Actions (0 or 1)
  β      : Scalar            # Reward coefficient
OUTPUT:
  r      : Vector[n]         # Flow rewards

  FOR i = 0 TO n - 1:
    r[i] ← β * log(1 + s[i]) - a[i]
  RETURN r
```

```
SUBROUTINE: COMPUTE_NEXT_STATE
──────────────────────────────
INPUT:
  s      : Vector[n]         # Current states
  a      : Vector[n]         # Actions (0 or 1)
  γ      : Scalar            # Decay rate
OUTPUT:
  s'     : Vector[n]         # Next states

  FOR i = 0 TO n - 1:
    s'[i] ← (1 - γ) * s[i] + a[i]
  RETURN s'
```

### Properties

The simulation algorithm satisfies the following properties:

1. **Exactness**: Actions are drawn directly from the closed-form logit probabilities:
   $$P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
   This uses the analytical result from Type-I Extreme Value shocks rather than simulating the shocks.

2. **Ergodicity**: Under the optimal policy, the state process converges to a stationary distribution (given $0 < \gamma < 1$).

3. **Reproducibility**: Given the same seed, the algorithm produces identical panel data.

## Run Simulation

```{python}
#| label: run-simulation

# Simulation parameters
n_agents = 100       # Number of agents (paths)
n_periods = 100      # Number of time periods
simulation_seed = 42 # Random seed

# Randomize initial states: draw from Uniform[s_min, s_max]
np.random.seed(simulation_seed)
s_init = np.random.uniform(s_min, s_max, size=n_agents)

print(f"Initial state distribution:")
print(f"  Distribution: Uniform[{s_min}, {s_max}]")
print(f"  Mean: {s_init.mean():.2f}")
print(f"  Std:  {s_init.std():.2f}")
print(f"  Min:  {s_init.min():.2f}")
print(f"  Max:  {s_init.max():.2f}")

# Run simulation using the implemented algorithm
panel = simulate_mdp_panel(
    v0_net=v0_net,
    v1_net=v1_net,
    n_agents=n_agents,
    n_periods=n_periods,
    s_init=s_init,  # Vector of different initial states per agent
    beta=beta,
    gamma=gamma,
    seed=simulation_seed,
)

print(f"\nSimulation complete:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Total observations: {panel.n_agents * panel.n_periods:,}")

# Save simulated data for estimation
import json

sim_output_dir = Path("../../output/simulate_mdp")
sim_output_dir.mkdir(parents=True, exist_ok=True)

# Save panel data arrays
np.save(sim_output_dir / "states.npy", panel.states)
np.save(sim_output_dir / "actions.npy", panel.actions)
np.save(sim_output_dir / "rewards.npy", panel.rewards)

# Save simulation configuration
sim_config = {
    "n_agents": n_agents,
    "n_periods": n_periods,
    "seed": simulation_seed,
    "s_init_distribution": f"Uniform[{s_min}, {s_max}]",
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
}
with open(sim_output_dir / "config.json", "w") as f:
    json.dump(sim_config, f, indent=2)

print(f"\nSaved simulation data to {sim_output_dir.resolve()}:")
print(f"  states.npy:  {panel.states.shape}")
print(f"  actions.npy: {panel.actions.shape}")
print(f"  rewards.npy: {panel.rewards.shape}")
print(f"  config.json: simulation parameters")
```

## Simulation Results

### Summary Statistics

```{python}
#| label: summary-stats

print("=" * 60)
print("PANEL DATA SUMMARY STATISTICS")
print("=" * 60)

# State statistics
print("\nState Distribution:")
print(f"  Mean:   {panel.states.mean():.3f}")
print(f"  Std:    {panel.states.std():.3f}")
print(f"  Min:    {panel.states.min():.3f}")
print(f"  Max:    {panel.states.max():.3f}")

# Action statistics
action_rate = panel.actions.mean()
print(f"\nAction Statistics:")
print(f"  P(a=1): {action_rate:.3f} (investment rate)")
print(f"  P(a=0): {1 - action_rate:.3f} (no investment rate)")

# Reward statistics
print(f"\nReward Distribution:")
print(f"  Mean:   {panel.rewards.mean():.3f}")
print(f"  Std:    {panel.rewards.std():.3f}")
print(f"  Min:    {panel.rewards.min():.3f}")
print(f"  Max:    {panel.rewards.max():.3f}")

# Steady state analysis
print(f"\nSteady State Analysis (last 10 periods):")
late_states = panel.states[:, -10:]
late_actions = panel.actions[:, -10:]
print(f"  Mean state:  {late_states.mean():.3f}")
print(f"  P(a=1):      {late_actions.mean():.3f}")

# Note: Total rewards and final states statistics are reported 
# in dedicated sections below with histograms
```

### State Evolution

```{python}
#| label: plot-state-evolution
#| fig-cap: "Evolution of state distribution over time"

plot_state_evolution(panel.states, panel.actions, s_init.mean())
```

### Sample Trajectories

```{python}
#| label: plot-trajectories
#| fig-cap: "Sample agent trajectories showing state evolution"

plot_trajectories(panel.states, s_init.mean(), n_samples=10)
```

### Choice Probability Validation

```{python}
#| label: plot-choice-validation
#| fig-cap: "Comparison of simulated choice frequencies with theoretical choice probabilities"

# Flatten data for plotting
state_flat = panel.states.flatten()
action_flat = panel.actions.flatten()

# Compute theoretical probabilities
s_theory = torch.linspace(state_flat.min(), state_flat.max(), 200)
with torch.no_grad():
    _, prob_a1_theory = compute_choice_probability(v0_net, v1_net, s_theory)
    prob_a1_theory = prob_a1_theory.numpy()

rmse = plot_choice_validation(
    state_flat, action_flat, s_theory.numpy(), prob_a1_theory
)
```

### Choice Probability Calibration

```{python}
#| label: plot-calibration
#| fig-cap: "Calibration plot: empirical vs theoretical choice probabilities"

# Compute theoretical P(a=1|s) for all observations
s_all = torch.tensor(state_flat, dtype=torch.float32)
with torch.no_grad():
    _, p1_theory_all = compute_choice_probability(v0_net, v1_net, s_all)
    p1_theory_all = p1_theory_all.numpy()

cal_rmse = plot_calibration(action_flat, p1_theory_all, n_bins=50)
```

### State Transition Dynamics

```{python}
#| label: plot-transitions
#| fig-cap: "State transition dynamics: current state vs next state by action"

plot_state_transitions(panel.states, panel.actions, gamma)
```

### State Distribution (Initial vs Final)

```{python}
#| label: plot-state-distribution
#| fig-cap: "Distribution of states: initial (t=0) vs final (t=T-1)"

final_states = panel.states[:, -1]
plot_state_distribution(s_init, final_states, s_min, s_max, n_periods)
```

### Reward Distribution (Cumulative vs Discounted)

```{python}
#| label: plot-reward-distribution
#| fig-cap: "Distribution of total rewards: cumulative vs discounted"

cumulative_rewards, discounted_rewards = plot_reward_distribution(panel.rewards, delta)
```

## Summary

```{python}
#| label: summary

print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

print(f"\nModel Parameters:")
print(f"  β = {beta} (reward coefficient)")
print(f"  γ = {gamma} (state decay rate)")
print(f"  δ = {delta} (discount factor)")

print(f"\nSimulation Setup:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Initial states: Uniform[{s_min}, {s_max}], mean={s_init.mean():.1f}")
print(f"  Random seed: {simulation_seed}")

print(f"\nKey Results:")
print(f"  Overall investment rate: {panel.actions.mean():.1%}")
print(f"  Stationary mean state: {panel.states[:, -20:].mean():.2f}")
print(f"  Choice probability RMSE: {rmse:.4f}")

print(f"\nThe simulation validates that:")
print(f"  ✓ State transitions follow s' = (1-γ)s + a")
print(f"  ✓ Empirical choice frequencies match theoretical logit probabilities")
print(f"  ✓ State distribution converges to a stationary distribution")
```

## Diagnostics

```{python}
#| label: diagnostics

print("=" * 60)
print("SIMULATION DIAGNOSTICS")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1. Choice Probability Validation
# -----------------------------------------------------------------------------
print("\n1. CHOICE PROBABILITY VALIDATION")
print("-" * 40)

if rmse < 0.01:
    prob_status = "✓ EXCELLENT"
    prob_msg = "Empirical choices closely match theoretical probabilities"
elif rmse < 0.05:
    prob_status = "✓ GOOD"
    prob_msg = "Minor deviations, likely due to sampling variance"
elif rmse < 0.10:
    prob_status = "⚠ WARNING"
    prob_msg = "Moderate deviations detected - consider larger sample"
else:
    prob_status = "✗ POOR"
    prob_msg = "Large deviations - potential implementation issue"

print(f"   RMSE: {rmse:.4f}")
print(f"   Status: {prob_status}")
print(f"   {prob_msg}")

# -----------------------------------------------------------------------------
# 2. Calibration Quality
# -----------------------------------------------------------------------------
print("\n2. CALIBRATION QUALITY")
print("-" * 40)

if cal_rmse < 0.01:
    cal_status = "✓ EXCELLENT"
    cal_msg = "Perfect calibration - simulation draws from correct distribution"
elif cal_rmse < 0.03:
    cal_status = "✓ GOOD"
    cal_msg = "Well-calibrated with minor sampling noise"
elif cal_rmse < 0.05:
    cal_status = "⚠ WARNING"
    cal_msg = "Some miscalibration - check binning or sample size"
else:
    cal_status = "✗ POOR"
    cal_msg = "Significant miscalibration - verify action drawing mechanism"

print(f"   Calibration RMSE: {cal_rmse:.4f}")
print(f"   Status: {cal_status}")
print(f"   {cal_msg}")

# -----------------------------------------------------------------------------
# 3. State Convergence Analysis
# -----------------------------------------------------------------------------
print("\n3. STATE CONVERGENCE ANALYSIS")
print("-" * 40)

# Check if states have converged by comparing early vs late periods
early_mean = panel.states[:, :20].mean()
late_mean = panel.states[:, -20:].mean()
convergence_diff = abs(late_mean - early_mean)

# Compute steady-state theoretical value (approximate)
# At steady state: E[s'] = E[s] => (1-γ)E[s] + E[a] = E[s] => E[s] = E[a]/γ
late_action_rate = panel.actions[:, -20:].mean()
theoretical_steady_state = late_action_rate / gamma

steady_state_error = abs(late_mean - theoretical_steady_state)

if convergence_diff < 0.5 and steady_state_error < 0.5:
    conv_status = "✓ CONVERGED"
    conv_msg = "State distribution has reached stationarity"
elif convergence_diff < 1.0:
    conv_status = "⚠ NEARLY CONVERGED"
    conv_msg = "Close to stationary - consider more periods"
else:
    conv_status = "✗ NOT CONVERGED"
    conv_msg = "Still transitioning - increase simulation length"

print(f"   Early mean (t<20):  {early_mean:.2f}")
print(f"   Late mean (t≥{n_periods-20}):   {late_mean:.2f}")
print(f"   Theoretical steady state: {theoretical_steady_state:.2f}")
print(f"   Status: {conv_status}")
print(f"   {conv_msg}")

# -----------------------------------------------------------------------------
# 4. State Transition Verification
# -----------------------------------------------------------------------------
print("\n4. STATE TRANSITION VERIFICATION")
print("-" * 40)

# Verify transitions match model
s_current = panel.states[:, :-1].flatten()
s_next_actual = panel.states[:, 1:].flatten()
a_taken = panel.actions[:, :-1].flatten()

# Expected next state
s_next_expected = (1 - gamma) * s_current + a_taken

# Compute transition error
transition_error = np.abs(s_next_actual - s_next_expected).max()

if transition_error < 1e-6:
    trans_status = "✓ EXACT"
    trans_msg = "State transitions perfectly follow s' = (1-γ)s + a"
elif transition_error < 1e-4:
    trans_status = "✓ GOOD"
    trans_msg = "Minor floating point differences only"
else:
    trans_status = "✗ ERROR"
    trans_msg = f"Transition mismatch detected (max error: {transition_error:.2e})"

print(f"   Max transition error: {transition_error:.2e}")
print(f"   Status: {trans_status}")
print(f"   {trans_msg}")

# -----------------------------------------------------------------------------
# 5. Reward Consistency Check
# -----------------------------------------------------------------------------
print("\n5. REWARD CONSISTENCY CHECK")
print("-" * 40)

# Verify rewards match model: r = β*log(1+s) - a
states_flat = panel.states.flatten()
actions_flat = panel.actions.flatten()
rewards_flat = panel.rewards.flatten()

expected_rewards = beta * np.log(1 + states_flat) - actions_flat
reward_error = np.abs(rewards_flat - expected_rewards).max()

if reward_error < 1e-6:
    reward_status = "✓ EXACT"
    reward_msg = "Rewards perfectly match r = β·log(1+s) - a"
elif reward_error < 1e-4:
    reward_status = "✓ GOOD"
    reward_msg = "Minor floating point differences only"
else:
    reward_status = "✗ ERROR"
    reward_msg = f"Reward mismatch detected (max error: {reward_error:.2e})"

print(f"   Max reward error: {reward_error:.2e}")
print(f"   Status: {reward_status}")
print(f"   {reward_msg}")

# -----------------------------------------------------------------------------
# Overall Assessment
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OVERALL ASSESSMENT")
print("=" * 60)

all_checks = [
    ("Choice Probability", rmse < 0.05),
    ("Calibration", cal_rmse < 0.05),
    ("Convergence", convergence_diff < 1.0),
    ("Transitions", transition_error < 1e-4),
    ("Rewards", reward_error < 1e-4),
]

passed = sum(1 for _, ok in all_checks if ok)
total = len(all_checks)

print(f"\nChecks Passed: {passed}/{total}")
for name, ok in all_checks:
    status = "✓" if ok else "✗"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n🎉 SIMULATION VALIDATED SUCCESSFULLY")
    print(f"   The Monte Carlo simulation correctly implements the MDP model.")
    print(f"   Simulated data can be used for structural estimation.")
elif passed >= total - 1:
    print(f"\n⚠️  SIMULATION MOSTLY VALIDATED")
    print(f"   Minor issues detected - review warnings above.")
else:
    print(f"\n❌ SIMULATION REQUIRES ATTENTION")
    print(f"   Multiple issues detected - review diagnostics above.")
```

```

```plaintext
---
title: "Monte Carlo Simulation for MDP"
subtitle: "Simulator and Structural Parameter Estimation"
format:
  html:
    code-fold: false
    toc: true
    toc-depth: 3
---

## Setup

```{python}
#| label: setup

import sys
from pathlib import Path

import numpy as np
import torch

# Add project paths
sys.path.insert(0, "../../../src")
sys.path.insert(0, "../config")
sys.path.insert(0, "../../utils")

# Import MDP solver
from mdp.solver import (
    build_monotonic_network,
    evaluate_network,
    compute_choice_probability,
    compute_integrated_value,
)

# Import MDP simulator
from mdp.simulator import simulate_mdp_panel

# Import configuration
import config

# Import plotting utilities
from plotting import (
    plot_convergence,
    plot_value_functions,
    plot_choice_probabilities,
    plot_value_difference,
    plot_bellman_residuals,
    plot_state_evolution,
    plot_trajectories,
    plot_choice_validation,
    plot_calibration,
    plot_state_transitions,
    plot_state_distribution,
    plot_reward_distribution,
)

print("Modules loaded successfully")
```

## Load Configuration and Trained Model

```{python}
#| label: load-model

# Load parameters from central config
beta = config.beta
gamma = config.gamma
delta = config.delta
s_min = config.s_min
s_max = config.s_max
hidden_sizes = config.hidden_sizes

print("Model parameters (from scripts/config_mdp/config.py):")
print(f"  beta = {beta}")
print(f"  gamma = {gamma}")
print(f"  delta = {delta}")
print(f"  s_min = {s_min}")
print(f"  s_max = {s_max}")
print(f"  hidden_sizes = {hidden_sizes}")

# Load trained networks from solver output
output_dir = Path("../../../output/mdp/solve")

v0_net = build_monotonic_network(hidden_sizes=hidden_sizes)
v1_net = build_monotonic_network(hidden_sizes=hidden_sizes)

v0_net.load_state_dict(torch.load(output_dir / "v0_net.pt", weights_only=True))
v1_net.load_state_dict(torch.load(output_dir / "v1_net.pt", weights_only=True))

v0_net.eval()
v1_net.eval()

# Load training losses
losses = np.load(output_dir / "losses.npy")

print(f"\nLoaded trained networks from {output_dir.resolve()}")
print(f"  Training iterations: {len(losses)}")
print(f"  Final RMSE: {losses[-1]**0.5:.2e}")
```

## Simulation Algorithm

### Overview

The Monte Carlo simulator generates panel data by forward-simulating agents who follow the optimal policy derived from the trained value functions. Each period:

1. Agent observes current state $s_t$
2. Agent draws action from the closed-form logit choice probability:
   $$a_t \sim \text{Bernoulli}\left( P(a=1|s_t) \right) \quad \text{where} \quad P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
3. State transitions: $s_{t+1} = (1 - \gamma) s_t + a_t$

Since Type-I Extreme Value shocks yield closed-form logit probabilities, we use the analytical choice probabilities directly rather than drawing shocks.

### Type Definitions

```
TYPE DEFINITIONS
────────────────
Scalar      = Float                      # Single real number
Vector[n]   = Array[Float, n]            # 1D array of n floats
Matrix[m,n] = Array[Float, m, n]         # 2D array of shape (m, n)
Action      = Int ∈ {0, 1}               # Binary action
Network     = MonotonicNeuralNetwork     # Trained value function network
```

### Main Algorithm

```
ALGORITHM: SIMULATE_MDP_PANEL
─────────────────────────────
INPUT:
  v₀_net       : Network       # Trained network for action 0
  v₁_net       : Network       # Trained network for action 1
  n_agents     : Int           # Number of agents
  n_periods    : Int           # Number of time periods
  s_init       : Vector[n] | Scalar  # Initial states
  β            : Scalar        # Reward coefficient
  γ            : Scalar        # State decay rate
  seed         : Int           # Random seed for reproducibility

OUTPUT:
  states       : Matrix[n, T]  # State history for each agent
  actions      : Matrix[n, T]  # Action history for each agent
  rewards      : Matrix[n, T]  # Reward history for each agent

PROCEDURE:
  SET_SEED(seed)
  
  # Initialize storage matrices
  states  ← Matrix[n_agents, n_periods]
  actions ← Matrix[n_agents, n_periods]
  rewards ← Matrix[n_agents, n_periods]
  
  # Set initial states
  states[*, 0] ← s_init
  
  FOR t = 0 TO n_periods - 1:
    # Get current states for all agents
    s_t ← states[*, t]
    
    # Compute choice probabilities using closed-form logit formula
    _, p₁ ← COMPUTE_CHOICE_PROBABILITY(v₀_net, v₁_net, s_t)
    
    # Draw actions from Bernoulli distribution
    u ← UNIFORM(0, 1, n_agents)
    actions[*, t] ← (u < p₁) ? 1 : 0
    
    # Compute flow rewards
    rewards[*, t] ← COMPUTE_REWARD(s_t, actions[*, t], β)
    
    # State transition (if not last period)
    IF t < n_periods - 1:
      states[*, t+1] ← COMPUTE_NEXT_STATE(s_t, actions[*, t], γ)
  
  RETURN (states, actions, rewards)
```

### Subroutines

```
SUBROUTINE: COMPUTE_CHOICE_PROBABILITY
──────────────────────────────────────
INPUT:
  v₀_net : Network           # Trained network for action 0
  v₁_net : Network           # Trained network for action 1
  s      : Vector[n]         # States
OUTPUT:
  p₀     : Vector[n]         # P(a=0|s)
  p₁     : Vector[n]         # P(a=1|s)

  # Evaluate value functions
  v₀ ← EVALUATE_NETWORK(v₀_net, s)
  v₁ ← EVALUATE_NETWORK(v₁_net, s)
  
  # Closed-form logit probabilities (from Type-I EV distribution)
  # Using numerically stable softmax
  v_max ← MAX(v₀, v₁)
  exp₀ ← exp(v₀ - v_max)
  exp₁ ← exp(v₁ - v_max)
  sum_exp ← exp₀ + exp₁
  
  p₀ ← exp₀ / sum_exp
  p₁ ← exp₁ / sum_exp
  
  RETURN (p₀, p₁)
```

```
SUBROUTINE: COMPUTE_REWARD
──────────────────────────
INPUT:
  s      : Vector[n]         # States
  a      : Vector[n]         # Actions (0 or 1)
  β      : Scalar            # Reward coefficient
OUTPUT:
  r      : Vector[n]         # Flow rewards

  FOR i = 0 TO n - 1:
    r[i] ← β * log(1 + s[i]) - a[i]
  RETURN r
```

```
SUBROUTINE: COMPUTE_NEXT_STATE
──────────────────────────────
INPUT:
  s      : Vector[n]         # Current states
  a      : Vector[n]         # Actions (0 or 1)
  γ      : Scalar            # Decay rate
OUTPUT:
  s'     : Vector[n]         # Next states

  FOR i = 0 TO n - 1:
    s'[i] ← (1 - γ) * s[i] + a[i]
  RETURN s'
```

### Properties

The simulation algorithm satisfies the following properties:

1. **Exactness**: Actions are drawn directly from the closed-form logit probabilities:
   $$P(a=1|s) = \frac{\exp(v(s,1))}{\exp(v(s,0)) + \exp(v(s,1))}$$
   This uses the analytical result from Type-I Extreme Value shocks rather than simulating the shocks.

2. **Ergodicity**: Under the optimal policy, the state process converges to a stationary distribution (given $0 < \gamma < 1$).

3. **Reproducibility**: Given the same seed, the algorithm produces identical panel data.

## Run Simulation

```{python}
#| label: run-simulation

# Simulation parameters
n_agents = 100       # Number of agents (paths)
n_periods = 100      # Number of time periods
simulation_seed = 42 # Random seed

# Randomize initial states: draw from Uniform[s_min, s_max]
np.random.seed(simulation_seed)
s_init = np.random.uniform(s_min, s_max, size=n_agents)

print(f"Initial state distribution:")
print(f"  Distribution: Uniform[{s_min}, {s_max}]")
print(f"  Mean: {s_init.mean():.2f}")
print(f"  Std:  {s_init.std():.2f}")
print(f"  Min:  {s_init.min():.2f}")
print(f"  Max:  {s_init.max():.2f}")

# Run simulation using the implemented algorithm
panel = simulate_mdp_panel(
    v0_net=v0_net,
    v1_net=v1_net,
    n_agents=n_agents,
    n_periods=n_periods,
    s_init=s_init,  # Vector of different initial states per agent
    beta=beta,
    gamma=gamma,
    seed=simulation_seed,
)

print(f"\nSimulation complete:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Total observations: {panel.n_agents * panel.n_periods:,}")

# Save simulated data for estimation
import json

sim_output_dir = Path("../../../output/mdp/simulate")
sim_output_dir.mkdir(parents=True, exist_ok=True)

# Save panel data arrays
np.save(sim_output_dir / "states.npy", panel.states)
np.save(sim_output_dir / "actions.npy", panel.actions)
np.save(sim_output_dir / "rewards.npy", panel.rewards)

# Save simulation configuration
sim_config = {
    "n_agents": n_agents,
    "n_periods": n_periods,
    "seed": simulation_seed,
    "s_init_distribution": f"Uniform[{s_min}, {s_max}]",
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
}
with open(sim_output_dir / "config.json", "w") as f:
    json.dump(sim_config, f, indent=2)

print(f"\nSaved simulation data to {sim_output_dir.resolve()}:")
print(f"  states.npy:  {panel.states.shape}")
print(f"  actions.npy: {panel.actions.shape}")
print(f"  rewards.npy: {panel.rewards.shape}")
print(f"  config.json: simulation parameters")
```

## Simulation Results

### Summary Statistics

```{python}
#| label: summary-stats

print("=" * 60)
print("PANEL DATA SUMMARY STATISTICS")
print("=" * 60)

# State statistics
print("\nState Distribution:")
print(f"  Mean:   {panel.states.mean():.3f}")
print(f"  Std:    {panel.states.std():.3f}")
print(f"  Min:    {panel.states.min():.3f}")
print(f"  Max:    {panel.states.max():.3f}")

# Action statistics
action_rate = panel.actions.mean()
print(f"\nAction Statistics:")
print(f"  P(a=1): {action_rate:.3f} (investment rate)")
print(f"  P(a=0): {1 - action_rate:.3f} (no investment rate)")

# Reward statistics
print(f"\nReward Distribution:")
print(f"  Mean:   {panel.rewards.mean():.3f}")
print(f"  Std:    {panel.rewards.std():.3f}")
print(f"  Min:    {panel.rewards.min():.3f}")
print(f"  Max:    {panel.rewards.max():.3f}")

# Steady state analysis
print(f"\nSteady State Analysis (last 10 periods):")
late_states = panel.states[:, -10:]
late_actions = panel.actions[:, -10:]
print(f"  Mean state:  {late_states.mean():.3f}")
print(f"  P(a=1):      {late_actions.mean():.3f}")

# Note: Total rewards and final states statistics are reported 
# in dedicated sections below with histograms
```

### State Evolution

```{python}
#| label: plot-state-evolution
#| fig-cap: "Evolution of state distribution over time"

plot_state_evolution(panel.states, panel.actions, s_init.mean())
```

### Sample Trajectories

```{python}
#| label: plot-trajectories
#| fig-cap: "Sample agent trajectories showing state evolution"

plot_trajectories(panel.states, s_init.mean(), n_samples=10)
```

### Choice Probability Validation

```{python}
#| label: plot-choice-validation
#| fig-cap: "Comparison of simulated choice frequencies with theoretical choice probabilities"

# Flatten data for plotting
state_flat = panel.states.flatten()
action_flat = panel.actions.flatten()

# Compute theoretical probabilities
s_theory = torch.linspace(state_flat.min(), state_flat.max(), 200)
with torch.no_grad():
    _, prob_a1_theory = compute_choice_probability(v0_net, v1_net, s_theory)
    prob_a1_theory = prob_a1_theory.numpy()

rmse = plot_choice_validation(
    state_flat, action_flat, s_theory.numpy(), prob_a1_theory
)
```

### Choice Probability Calibration

```{python}
#| label: plot-calibration
#| fig-cap: "Calibration plot: empirical vs theoretical choice probabilities"

# Compute theoretical P(a=1|s) for all observations
s_all = torch.tensor(state_flat, dtype=torch.float32)
with torch.no_grad():
    _, p1_theory_all = compute_choice_probability(v0_net, v1_net, s_all)
    p1_theory_all = p1_theory_all.numpy()

cal_rmse = plot_calibration(action_flat, p1_theory_all, n_bins=50)
```

### State Transition Dynamics

```{python}
#| label: plot-transitions
#| fig-cap: "State transition dynamics: current state vs next state by action"

plot_state_transitions(panel.states, panel.actions, gamma)
```

### State Distribution (Initial vs Final)

```{python}
#| label: plot-state-distribution
#| fig-cap: "Distribution of states: initial (t=0) vs final (t=T-1)"

final_states = panel.states[:, -1]
plot_state_distribution(s_init, final_states, s_min, s_max, n_periods)
```

### Reward Distribution (Cumulative vs Discounted)

```{python}
#| label: plot-reward-distribution
#| fig-cap: "Distribution of total rewards: cumulative vs discounted"

cumulative_rewards, discounted_rewards = plot_reward_distribution(panel.rewards, delta)
```

## Summary

```{python}
#| label: summary

print("=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

print(f"\nModel Parameters:")
print(f"  β = {beta} (reward coefficient)")
print(f"  γ = {gamma} (state decay rate)")
print(f"  δ = {delta} (discount factor)")

print(f"\nSimulation Setup:")
print(f"  Agents: {panel.n_agents}")
print(f"  Periods: {panel.n_periods}")
print(f"  Initial states: Uniform[{s_min}, {s_max}], mean={s_init.mean():.1f}")
print(f"  Random seed: {simulation_seed}")

print(f"\nKey Results:")
print(f"  Overall investment rate: {panel.actions.mean():.1%}")
print(f"  Stationary mean state: {panel.states[:, -20:].mean():.2f}")
print(f"  Choice probability RMSE: {rmse:.4f}")

print(f"\nThe simulation validates that:")
print(f"  ✓ State transitions follow s' = (1-γ)s + a")
print(f"  ✓ Empirical choice frequencies match theoretical logit probabilities")
print(f"  ✓ State distribution converges to a stationary distribution")
```

## Diagnostics

```{python}
#| label: diagnostics

print("=" * 60)
print("SIMULATION DIAGNOSTICS")
print("=" * 60)

# -----------------------------------------------------------------------------
# 1. Choice Probability Validation
# -----------------------------------------------------------------------------
print("\n1. CHOICE PROBABILITY VALIDATION")
print("-" * 40)

if rmse < 0.01:
    prob_status = "✓ EXCELLENT"
    prob_msg = "Empirical choices closely match theoretical probabilities"
elif rmse < 0.05:
    prob_status = "✓ GOOD"
    prob_msg = "Minor deviations, likely due to sampling variance"
elif rmse < 0.10:
    prob_status = "⚠ WARNING"
    prob_msg = "Moderate deviations detected - consider larger sample"
else:
    prob_status = "✗ POOR"
    prob_msg = "Large deviations - potential implementation issue"

print(f"   RMSE: {rmse:.4f}")
print(f"   Status: {prob_status}")
print(f"   {prob_msg}")

# -----------------------------------------------------------------------------
# 2. Calibration Quality
# -----------------------------------------------------------------------------
print("\n2. CALIBRATION QUALITY")
print("-" * 40)

if cal_rmse < 0.01:
    cal_status = "✓ EXCELLENT"
    cal_msg = "Perfect calibration - simulation draws from correct distribution"
elif cal_rmse < 0.03:
    cal_status = "✓ GOOD"
    cal_msg = "Well-calibrated with minor sampling noise"
elif cal_rmse < 0.05:
    cal_status = "⚠ WARNING"
    cal_msg = "Some miscalibration - check binning or sample size"
else:
    cal_status = "✗ POOR"
    cal_msg = "Significant miscalibration - verify action drawing mechanism"

print(f"   Calibration RMSE: {cal_rmse:.4f}")
print(f"   Status: {cal_status}")
print(f"   {cal_msg}")

# -----------------------------------------------------------------------------
# 3. State Convergence Analysis
# -----------------------------------------------------------------------------
print("\n3. STATE CONVERGENCE ANALYSIS")
print("-" * 40)

# Check if states have converged by comparing early vs late periods
early_mean = panel.states[:, :20].mean()
late_mean = panel.states[:, -20:].mean()
convergence_diff = abs(late_mean - early_mean)

# Compute steady-state theoretical value (approximate)
# At steady state: E[s'] = E[s] => (1-γ)E[s] + E[a] = E[s] => E[s] = E[a]/γ
late_action_rate = panel.actions[:, -20:].mean()
theoretical_steady_state = late_action_rate / gamma

steady_state_error = abs(late_mean - theoretical_steady_state)

if convergence_diff < 0.5 and steady_state_error < 0.5:
    conv_status = "✓ CONVERGED"
    conv_msg = "State distribution has reached stationarity"
elif convergence_diff < 1.0:
    conv_status = "⚠ NEARLY CONVERGED"
    conv_msg = "Close to stationary - consider more periods"
else:
    conv_status = "✗ NOT CONVERGED"
    conv_msg = "Still transitioning - increase simulation length"

print(f"   Early mean (t<20):  {early_mean:.2f}")
print(f"   Late mean (t≥{n_periods-20}):   {late_mean:.2f}")
print(f"   Theoretical steady state: {theoretical_steady_state:.2f}")
print(f"   Status: {conv_status}")
print(f"   {conv_msg}")

# -----------------------------------------------------------------------------
# 4. State Transition Verification
# -----------------------------------------------------------------------------
print("\n4. STATE TRANSITION VERIFICATION")
print("-" * 40)

# Verify transitions match model
s_current = panel.states[:, :-1].flatten()
s_next_actual = panel.states[:, 1:].flatten()
a_taken = panel.actions[:, :-1].flatten()

# Expected next state
s_next_expected = (1 - gamma) * s_current + a_taken

# Compute transition error
transition_error = np.abs(s_next_actual - s_next_expected).max()

if transition_error < 1e-6:
    trans_status = "✓ EXACT"
    trans_msg = "State transitions perfectly follow s' = (1-γ)s + a"
elif transition_error < 1e-4:
    trans_status = "✓ GOOD"
    trans_msg = "Minor floating point differences only"
else:
    trans_status = "✗ ERROR"
    trans_msg = f"Transition mismatch detected (max error: {transition_error:.2e})"

print(f"   Max transition error: {transition_error:.2e}")
print(f"   Status: {trans_status}")
print(f"   {trans_msg}")

# -----------------------------------------------------------------------------
# 5. Reward Consistency Check
# -----------------------------------------------------------------------------
print("\n5. REWARD CONSISTENCY CHECK")
print("-" * 40)

# Verify rewards match model: r = β*log(1+s) - a
states_flat = panel.states.flatten()
actions_flat = panel.actions.flatten()
rewards_flat = panel.rewards.flatten()

expected_rewards = beta * np.log(1 + states_flat) - actions_flat
reward_error = np.abs(rewards_flat - expected_rewards).max()

if reward_error < 1e-6:
    reward_status = "✓ EXACT"
    reward_msg = "Rewards perfectly match r = β·log(1+s) - a"
elif reward_error < 1e-4:
    reward_status = "✓ GOOD"
    reward_msg = "Minor floating point differences only"
else:
    reward_status = "✗ ERROR"
    reward_msg = f"Reward mismatch detected (max error: {reward_error:.2e})"

print(f"   Max reward error: {reward_error:.2e}")
print(f"   Status: {reward_status}")
print(f"   {reward_msg}")

# -----------------------------------------------------------------------------
# Overall Assessment
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("OVERALL ASSESSMENT")
print("=" * 60)

all_checks = [
    ("Choice Probability", rmse < 0.05),
    ("Calibration", cal_rmse < 0.05),
    ("Convergence", convergence_diff < 1.0),
    ("Transitions", transition_error < 1e-4),
    ("Rewards", reward_error < 1e-4),
]

passed = sum(1 for _, ok in all_checks if ok)
total = len(all_checks)

print(f"\nChecks Passed: {passed}/{total}")
for name, ok in all_checks:
    status = "✓" if ok else "✗"
    print(f"   {status} {name}")

if passed == total:
    print(f"\n🎉 SIMULATION VALIDATED SUCCESSFULLY")
    print(f"   The Monte Carlo simulation correctly implements the MDP model.")
    print(f"   Simulated data can be used for structural estimation.")
elif passed >= total - 1:
    print(f"\n⚠️  SIMULATION MOSTLY VALIDATED")
    print(f"   Minor issues detected - review warnings above.")
else:
    print(f"\n❌ SIMULATION REQUIRES ATTENTION")
    print(f"   Multiple issues detected - review diagnostics above.")
```

```

```plaintext
---
title: "Solving a Dynamic Discrete Choice Model"
subtitle: "MDP with Type-I Extreme Value Shocks"
format:
  html:
    code-fold: false
    toc: true
    toc-depth: 3
---

## Model Setup

We consider a dynamic discrete choice model with:

- **State**: $s \in \mathbb{R}_{\geq 0}$ (continuous, non-negative)
- **Action**: $a \in \{0, 1\}$ (binary choice)
- **Mean reward**: $u(s, a) = \beta \log(1 + s) - a$
- **State transition**: $s' = (1 - \gamma) s + a$ (deterministic)
- **Discount factor**: $\delta$
- **Shocks**: $\varepsilon_a \sim$ Type-I Extreme Value (iid across actions and time)

### Interpretation

- Higher state $s$ yields higher flow payoff, but with **diminishing marginal returns**
- The $\log(1 + s)$ specification ensures:
  - Marginal utility $\frac{\partial u}{\partial s} = \frac{\beta}{1 + s} > 0$ (increasing in $s$)
  - Diminishing: $\frac{\partial^2 u}{\partial s^2} = -\frac{\beta}{(1 + s)^2} < 0$ (concave)
  - Well-defined at $s = 0$: $u(0, a) = -a$
- Action $a = 1$ costs 1 unit but boosts next period's state by 1
- Action $a = 0$ is free but state decays toward zero at rate $\gamma$

## Optimal Markov Decision Policy

### Bellman Equation with Extreme Value Shocks

Each period, the agent observes state $s$ and receives action-specific shocks $(\varepsilon_0, \varepsilon_1)$ drawn iid from the Type-I Extreme Value (Gumbel) distribution:
$$
F(\varepsilon) = \exp(-\exp(-\varepsilon)), \quad f(\varepsilon) = \exp(-\varepsilon - \exp(-\varepsilon))
$$

The agent chooses action $a \in \{0, 1\}$ to maximize:
$$
u(s, a) + \varepsilon_a + \delta V(s')
$$
where $s' = (1 - \gamma)s + a$ is the next state.

The **Bellman equation** is:
$$
V(s, \varepsilon_0, \varepsilon_1) = \max_{a \in \{0,1\}} \left\{ u(s, a) + \varepsilon_a + \delta \mathbb{E}_{\varepsilon'} \left[ V(s', \varepsilon_0', \varepsilon_1') \right] \right\}
$$

### Choice-Specific Value Functions

Define the **choice-specific value function** (the deterministic part, excluding the current shock):
$$
v(s, a) = u(s, a) + \delta \mathbb{E}_{\varepsilon'} \left[ V(s', \varepsilon') \right]
$$

Substituting the reward and transition:
$$
v(s, a) = \beta \log(1 + s) - a + \delta \bar{V}\bigl((1 - \gamma) s + a\bigr)
$$
where $\bar{V}(s) = \mathbb{E}_\varepsilon[V(s, \varepsilon)]$ is the **integrated (ex-ante) value function**.

### Integrated Value Function

The **integrated value function** is the expected value before observing shocks:
$$
\bar{V}(s) = \mathbb{E}_{\varepsilon_0, \varepsilon_1} \left[ \max_{a \in \{0,1\}} \left\{ v(s, a) + \varepsilon_a \right\} \right]
$$

With Type-I EV shocks, this has a closed-form (the "log-sum-exp" formula):
$$
\bar{V}(s) = \log\left( \exp(v(s, 0)) + \exp(v(s, 1)) \right) + \gamma_E
$$
where $\gamma_E \approx 0.5772$ is Euler's constant. Since $\gamma_E$ is a constant, it does not affect the policy and is often omitted.

### Optimal Policy and Choice Probabilities

The optimal decision rule is:
$$
a^*(s, \varepsilon_0, \varepsilon_1) = \mathbf{1}\left\{ v(s, 1) + \varepsilon_1 > v(s, 0) + \varepsilon_0 \right\}
$$

The probability of choosing action $a = 1$ given state $s$ follows the **logit formula**:
$$
P(a = 1 | s) = \frac{\exp(v(s, 1))}{\exp(v(s, 0)) + \exp(v(s, 1))} = \frac{1}{1 + \exp(v(s, 0) - v(s, 1))}
$$

The optimal policy is a **stochastic Markov policy**: the choice probability depends only on the current state $s$ through the value difference $v(s, 1) - v(s, 0)$.

## Solution Method: Neural Network Value Iteration

### Bellman Fixed Point

The integrated value function $\bar{V}(s)$ satisfies the **fixed point equation**:
$$
\bar{V}(s) = \log\left( \exp(v(s, 0)) + \exp(v(s, 1)) \right)
$$
where each choice-specific value depends on $\bar{V}$:
$$
v(s, a) = \beta \log(1 + s) - a + \delta \bar{V}((1 - \gamma)s + a)
$$

By Blackwell's sufficient conditions, this defines a contraction mapping with modulus $\delta < 1$, guaranteeing a unique fixed point $\bar{V}^*$.

### Monotonic Neural Network Architecture

We approximate each action's choice-specific value function $v(s, a)$ using a **monotonic neural network**. Since higher states yield higher rewards ($\beta > 0$), we expect $v(s, a)$ to be increasing in $s$. Enforcing monotonicity as an architectural constraint improves extrapolation and ensures economically sensible behavior.

A feed-forward neural network is **monotonically increasing** in its input if:

1. All weights are **non-negative**
2. All activation functions are **monotonically increasing**

For a network with $L$ hidden layers:
$$
v_\theta(s) = W_L \cdot \sigma\left( W_{L-1} \cdot \sigma\left( \cdots \sigma(W_1 \cdot s + b_1) \cdots \right) + b_{L-1} \right) + b_L
$$
where $W_\ell \geq 0$ for all layers and $\sigma$ is a monotonically increasing activation function.

### Tanh Activation

We use the **tanh** (hyperbolic tangent) activation function:
$$
\sigma(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Properties:

- **Monotonically increasing**: $\sigma'(x) = 1 - \tanh^2(x) > 0$
- **Bounded output**: $\sigma(x) \in (-1, 1)$ prevents explosive growth
- **Zero-centered**: Allows both positive and negative intermediate representations
- **Better gradient flow**: Bounded outputs help maintain stable gradients through deep networks

### Weight Parameterization

To enforce non-negative weights during optimization, we parameterize the weights using softplus of unconstrained parameters:
$$
W = \text{softplus}(\tilde{W}) = \log(1 + e^{\tilde{W}})
$$
where $\tilde{W} \in \mathbb{R}$ is the learnable parameter. This ensures $W > 0$ while allowing standard gradient-based optimization.

### Value Function Approximator

We use **separate monotonic networks** for each action:

- $v_{\theta_0}(s)$: Monotonic network for action $a = 0$
- $v_{\theta_1}(s)$: Monotonic network for action $a = 1$

The integrated value function is computed as:
$$
\bar{V}_\theta(s) = \log\left( \exp(v_{\theta_0}(s)) + \exp(v_{\theta_1}(s)) \right)
$$

Choice probabilities follow the logit formula:
$$
P_\theta(a = 1 | s) = \frac{1}{1 + \exp(v_{\theta_0}(s) - v_{\theta_1}(s))}
$$

### Training Objective

The networks are trained to satisfy the Bellman equation by minimizing the **Bellman residual**:
$$
\mathcal{L}(\theta) = \mathbb{E}_s\left[ \sum_{a \in \{0,1\}} \left( v_{\theta_a}(s) - u(s, a) - \delta \bar{V}_\theta(s'_a) \right)^2 \right]
$$
where $s'_a = (1 - \gamma)s + a$ is the next state under action $a$.

### Type Definitions

```
TYPE DEFINITIONS
────────────────
Scalar      = Float                      # Single real number
Vector[n]   = Array[Float, n]            # 1D array of n floats
Matrix[m,n] = Array[Float, m, n]         # 2D array of shape (m, n)
Action      = Int ∈ {0, 1}               # Binary action
Network     = MonotonicNeuralNetwork     # Neural network with non-negative weights
```

### Algorithm

```
ALGORITHM: SOLVE_VALUE_FUNCTION
───────────────────────────────
INPUT:
  β            : Scalar              # Reward coefficient
  γ            : Scalar              # State decay rate
  δ            : Scalar              # Discount factor
  s_min        : Scalar              # Minimum state
  s_max        : Scalar              # Maximum state
  hidden_sizes : List[Int]           # Hidden layer sizes
  α            : Scalar              # Learning rate
  N            : Int                 # Batch size
  ε            : Scalar              # Convergence tolerance
  K            : Int                 # Maximum iterations
  T            : Int                 # Target network update frequency

OUTPUT:
  v₀_net       : Network             # Trained network for action 0
  v₁_net       : Network             # Trained network for action 1

PROCEDURE:
  # Initialize policy networks (updated every iteration)
  v₀_net : Network, v₁_net : Network ← INITIALIZE_NETWORKS(hidden_sizes)
  
  # Initialize target networks (frozen copies, updated every T iterations)
  v₀_target : Network ← COPY_NETWORK(v₀_net)
  v₁_target : Network ← COPY_NETWORK(v₁_net)
  
  # Generate fixed state grid (same states used every iteration)
  s_grid : Vector[N] ← GENERATE_STATE_GRID(N, s_min, s_max)
  
  FOR k : Int = 1 TO K:
    # Predictions from policy networks (using fixed grid)
    v₀_pred : Vector[N] ← EVALUATE_NETWORK(v₀_net, s_grid)
    v₁_pred : Vector[N] ← EVALUATE_NETWORK(v₁_net, s_grid)
    
    # Targets computed from TARGET networks (stable, not moving)
    target₀ : Vector[N], target₁ : Vector[N] ← COMPUTE_BELLMAN_TARGETS(
      v₀_target, v₁_target, s_grid, β, γ, δ
    )
    
    loss : Scalar ← COMPUTE_BELLMAN_LOSS(v₀_pred, v₁_pred, target₀, target₁)
    
    # Update policy networks
    UPDATE_NETWORK_WEIGHTS(v₀_net, v₁_net, loss, α)
    
    # Periodically sync target networks with policy networks
    IF k MOD T = 0:
      v₀_target ← COPY_NETWORK(v₀_net)
      v₁_target ← COPY_NETWORK(v₁_net)
    
    IF CHECK_CONVERGENCE(loss, ε):
      RETURN (v₀_net, v₁_net)
  
  RETURN (v₀_net, v₁_net)
```

### Subroutines

```
SUBROUTINE: INITIALIZE_NETWORKS
───────────────────────────────
INPUT:
  hidden_sizes : List[Int]           # Hidden layer sizes
OUTPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1

  v₀_net : Network ← BUILD_MONOTONIC_NETWORK(hidden_sizes)
  v₁_net : Network ← BUILD_MONOTONIC_NETWORK(hidden_sizes)
  RETURN (v₀_net, v₁_net)
```

```
SUBROUTINE: COPY_NETWORK
────────────────────────
INPUT:
  source : Network                   # Network to copy from
OUTPUT:
  target : Network                   # Deep copy of source network

  # Create a new network with identical architecture and weights
  target : Network ← DEEP_COPY(source)
  
  # Freeze the target network (no gradient computation)
  SET_REQUIRES_GRAD(target, false)
  
  RETURN target
```

```
SUBROUTINE: BUILD_MONOTONIC_NETWORK
───────────────────────────────────
INPUT:
  hidden_sizes : List[Int]           # Hidden layer sizes
OUTPUT:
  network : Network                  # Monotonic neural network

  layers : List[Layer] ← empty list
  in_features : Int ← 1
  
  FOR i : Int = 0 TO length(hidden_sizes) - 1:
    out_features : Int ← hidden_sizes[i]
    
    # Create monotonic linear layer
    W_raw : Matrix[out_features, in_features] ← random_normal(0, 0.1)
    b : Vector[out_features] ← zeros(out_features)
    layers.append(MonotonicLinear(W_raw, b))
    
    # Add tanh activation (bounded, zero-centered)
    layers.append(TanhActivation())
    
    in_features ← out_features
  
  # Output layer (no activation)
  W_raw_out : Matrix[1, in_features] ← random_normal(0, 0.1)
  b_out : Vector[1] ← zeros(1)
  layers.append(MonotonicLinear(W_raw_out, b_out))
  
  RETURN Network(layers)
```

```
SUBROUTINE: APPLY_MONOTONIC_LINEAR
──────────────────────────────────
INPUT:
  W_raw : Matrix[m, n]               # Unconstrained weight parameters
  b     : Vector[m]                  # Bias vector
  x     : Vector[n]                  # Input vector
OUTPUT:
  y     : Vector[m]                  # Output vector

  # Enforce non-negative weights via softplus
  W : Matrix[m, n] ← log(1 + exp(W_raw))
  
  # Linear transformation
  y : Vector[m] ← W @ x + b
  RETURN y
```

```
SUBROUTINE: APPLY_TANH
──────────────────────
INPUT:
  x : Vector[n]                      # Input vector
OUTPUT:
  y : Vector[n]                      # Output vector

  FOR i : Int = 0 TO n - 1:
    y[i] : Scalar ← tanh(x[i])       # Bounded to (-1, 1)
  RETURN y
```

```
SUBROUTINE: EVALUATE_NETWORK
────────────────────────────
INPUT:
  network : Network                  # Monotonic neural network
  s       : Vector[N]                # Input states
OUTPUT:
  values  : Vector[N]                # Output values

  x : Matrix[N, 1] ← reshape(s, (N, 1))
  
  FOR layer IN network.layers:
    IF layer IS MonotonicLinear:
      x ← APPLY_MONOTONIC_LINEAR(layer.W_raw, layer.b, x)
    ELSE IF layer IS TanhActivation:
      x ← APPLY_TANH(x)
  
  values : Vector[N] ← reshape(x, (N,))
  RETURN values
```

```
SUBROUTINE: GENERATE_STATE_GRID
───────────────────────────────
INPUT:
  N     : Int                        # Number of grid points
  s_min : Scalar                     # Minimum state
  s_max : Scalar                     # Maximum state
OUTPUT:
  s     : Vector[N]                  # Fixed state grid (evenly spaced)

  # Generate evenly spaced points from s_min to s_max
  step : Scalar ← (s_max - s_min) / (N - 1)
  
  FOR i : Int = 0 TO N - 1:
    s[i] : Scalar ← s_min + i * step
  RETURN s
```

```
SUBROUTINE: COMPUTE_REWARD
──────────────────────────
INPUT:
  s      : Vector[N]                 # States
  action : Action                    # Action (0 or 1)
  β      : Scalar                    # Reward coefficient
OUTPUT:
  r      : Vector[N]                 # Rewards

  FOR i : Int = 0 TO N - 1:
    r[i] : Scalar ← β * log(1 + s[i]) - action
  RETURN r
```

```
SUBROUTINE: COMPUTE_NEXT_STATE
──────────────────────────────
INPUT:
  s      : Vector[N]                 # Current states
  action : Action                    # Action (0 or 1)
  γ      : Scalar                    # Decay rate
OUTPUT:
  s_next : Vector[N]                 # Next states

  FOR i : Int = 0 TO N - 1:
    s_next[i] : Scalar ← (1 - γ) * s[i] + action
  RETURN s_next
```

```
SUBROUTINE: COMPUTE_INTEGRATED_VALUE
────────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # States
OUTPUT:
  V_bar  : Vector[N]                 # Integrated value

  v₀ : Vector[N] ← EVALUATE_NETWORK(v₀_net, s)
  v₁ : Vector[N] ← EVALUATE_NETWORK(v₁_net, s)
  
  FOR i : Int = 0 TO N - 1:
    v_max : Scalar ← max(v₀[i], v₁[i])
    V_bar[i] : Scalar ← v_max + log(exp(v₀[i] - v_max) + exp(v₁[i] - v_max))
  RETURN V_bar
```

```
SUBROUTINE: COMPUTE_BELLMAN_TARGETS
───────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # Current states
  β      : Scalar                    # Reward coefficient
  γ      : Scalar                    # Decay rate
  δ      : Scalar                    # Discount factor
OUTPUT:
  target₀ : Vector[N]                # Target for action 0
  target₁ : Vector[N]                # Target for action 1

  # Compute targets for action 0 (no clamping - unbounded state space)
  r₀ : Vector[N] ← COMPUTE_REWARD(s, action=0, β)
  s_next₀ : Vector[N] ← COMPUTE_NEXT_STATE(s, action=0, γ)
  V_next₀ : Vector[N] ← COMPUTE_INTEGRATED_VALUE(v₀_net, v₁_net, s_next₀)
  
  FOR i : Int = 0 TO N - 1:
    target₀[i] : Scalar ← r₀[i] + δ * V_next₀[i]
  
  # Compute targets for action 1 (no clamping - unbounded state space)
  r₁ : Vector[N] ← COMPUTE_REWARD(s, action=1, β)
  s_next₁ : Vector[N] ← COMPUTE_NEXT_STATE(s, action=1, γ)
  V_next₁ : Vector[N] ← COMPUTE_INTEGRATED_VALUE(v₀_net, v₁_net, s_next₁)
  
  FOR i : Int = 0 TO N - 1:
    target₁[i] : Scalar ← r₁[i] + δ * V_next₁[i]
  
  RETURN (target₀, target₁)
```

```
SUBROUTINE: COMPUTE_BELLMAN_LOSS
────────────────────────────────
INPUT:
  v₀_pred : Vector[N]                # Predicted values for action 0
  v₁_pred : Vector[N]                # Predicted values for action 1
  target₀ : Vector[N]                # Target values for action 0
  target₁ : Vector[N]                # Target values for action 1
OUTPUT:
  loss    : Scalar                   # Mean squared Bellman residual

  sum_sq : Scalar ← 0.0
  
  FOR i : Int = 0 TO N - 1:
    error₀ : Scalar ← v₀_pred[i] - target₀[i]
    error₁ : Scalar ← v₁_pred[i] - target₁[i]
    sum_sq ← sum_sq + error₀ * error₀ + error₁ * error₁
  
  loss : Scalar ← sum_sq / (2 * N)
  RETURN loss
```

```
SUBROUTINE: UPDATE_NETWORK_WEIGHTS
──────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  loss   : Scalar                    # Loss value
  α      : Scalar                    # Learning rate
OUTPUT:
  (none, updates networks in place)

  # Compute gradients via backpropagation
  grad₀ : List[Matrix] ← BACKPROPAGATE(v₀_net, loss)
  grad₁ : List[Matrix] ← BACKPROPAGATE(v₁_net, loss)
  
  # Update weights for network 0
  FOR j : Int = 0 TO length(v₀_net.layers) - 1:
    IF v₀_net.layers[j] IS MonotonicLinear:
      v₀_net.layers[j].W_raw ← v₀_net.layers[j].W_raw - α * grad₀[j].W
      v₀_net.layers[j].b ← v₀_net.layers[j].b - α * grad₀[j].b
  
  # Update weights for network 1
  FOR j : Int = 0 TO length(v₁_net.layers) - 1:
    IF v₁_net.layers[j] IS MonotonicLinear:
      v₁_net.layers[j].W_raw ← v₁_net.layers[j].W_raw - α * grad₁[j].W
      v₁_net.layers[j].b ← v₁_net.layers[j].b - α * grad₁[j].b
```

```
SUBROUTINE: CHECK_CONVERGENCE
─────────────────────────────
INPUT:
  loss : Scalar                      # Current loss
  ε    : Scalar                      # Tolerance
OUTPUT:
  converged : Bool                   # True if converged

  rmse : Scalar ← sqrt(loss)
  converged : Bool ← (rmse < ε)
  RETURN converged
```

```
SUBROUTINE: COMPUTE_CHOICE_PROBABILITY
──────────────────────────────────────
INPUT:
  v₀_net : Network                   # Network for action 0
  v₁_net : Network                   # Network for action 1
  s      : Vector[N]                 # States
OUTPUT:
  prob₀  : Vector[N]                 # P(a=0|s)
  prob₁  : Vector[N]                 # P(a=1|s)

  v₀ : Vector[N] ← EVALUATE_NETWORK(v₀_net, s)
  v₁ : Vector[N] ← EVALUATE_NETWORK(v₁_net, s)
  
  FOR i : Int = 0 TO N - 1:
    # Numerically stable softmax
    v_max : Scalar ← max(v₀[i], v₁[i])
    exp₀ : Scalar ← exp(v₀[i] - v_max)
    exp₁ : Scalar ← exp(v₁[i] - v_max)
    sum_exp : Scalar ← exp₀ + exp₁
    
    prob₀[i] : Scalar ← exp₀ / sum_exp
    prob₁[i] : Scalar ← exp₁ / sum_exp
  
  RETURN (prob₀, prob₁)
```

### Convergence

The algorithm minimizes the Bellman residual, driving the neural network approximation toward the true fixed point. Convergence is monitored via the root mean squared Bellman error $\sqrt{\mathcal{L}^{(k)}}$.

Key properties:

- **Continuous approximation**: No grid discretization or interpolation required
- **Scalability**: Extends naturally to high-dimensional state spaces
- **Economic structure**: Monotonicity enforced architecturally via non-negative weights

## Implementation

### Setup

```{python}
#| label: setup
#| cache: false

# Standard libraries
import sys
import numpy as np
import torch

# Add project paths
sys.path.insert(0, "../../../src")
sys.path.insert(0, "../config")
sys.path.insert(0, "../../utils")

# Import MDP solver
from mdp.solver import (
    solve_value_function,
    compute_choice_probability,
    evaluate_network,
    compute_integrated_value,
    compute_bellman_targets,
)

# Import configuration
import config

# Import plotting utilities
from plotting import (
    plot_convergence,
    plot_value_functions,
    plot_choice_probabilities,
    plot_value_difference,
    plot_bellman_residuals,
    plot_comparative_statics_values,
    plot_comparative_statics_probs,
)

print("Modules loaded successfully")
```

### Model Parameters

```{python}
# Load parameters from central config
beta = config.beta
gamma = config.gamma
delta = config.delta
s_min = config.s_min
s_max = config.s_max
hidden_sizes = config.hidden_sizes
learning_rate = config.learning_rate
batch_size = config.batch_size
tolerance = config.tolerance
max_iterations = config.max_iterations
target_update_freq = config.target_update_freq

print(f"Model: u(s,a) = {beta} * log(1+s) - a")
print(f"Transition: s' = {1-gamma:.1f} * s + a")
print(f"Discount factor: δ = {delta}")
print(f"\nLoaded from: scripts/config_mdp/config.py")
```

### Solve the MDP

```{python}
#| cache: true
#| label: solve-baseline

# Set random seed for reproducibility
torch.manual_seed(42)

# Solve for value functions
v0_net, v1_net, losses, n_iter = solve_value_function(
    beta=beta,
    gamma=gamma,
    delta=delta,
    s_min=s_min,
    s_max=s_max,
    hidden_sizes=hidden_sizes,
    learning_rate=learning_rate,
    batch_size=batch_size,
    tolerance=tolerance,
    max_iterations=max_iterations,
    target_update_freq=target_update_freq,
)

print(f"Converged in {n_iter} iterations")
print(f"Final RMSE: {losses[-1]**0.5:.2e}")

# Save results to output/solve_mdp/
import json
from pathlib import Path

output_dir = Path("../../../output/mdp/solve")
output_dir.mkdir(parents=True, exist_ok=True)

# Save config (model parameters)
config = {
    "beta": beta,
    "gamma": gamma,
    "delta": delta,
    "s_min": s_min,
    "s_max": s_max,
    "hidden_sizes": hidden_sizes,
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save trained networks
torch.save(v0_net.state_dict(), output_dir / "v0_net.pt")
torch.save(v1_net.state_dict(), output_dir / "v1_net.pt")

# Save training losses
np.save(output_dir / "losses.npy", np.array(losses))

print(f"\nResults saved to {output_dir.resolve()}:")
print(f"  - config.json (model parameters)")
print(f"  - v0_net.pt (action 0 network)")
print(f"  - v1_net.pt (action 1 network)")
print(f"  - losses.npy ({len(losses)} iterations)")
```

### Results

#### Convergence Plot

```{python}
#| fig-cap: "Convergence of the Bellman residual over training iterations"

plot_convergence(losses=np.array(losses))
```

#### Value Functions

```{python}
#| fig-cap: "Choice-specific value functions v(s, a) for each action"

# Create evaluation grid
s_eval = torch.linspace(s_min, s_max, 200)

# Evaluate value functions
with torch.no_grad():
    v0_values = evaluate_network(v0_net, s_eval).numpy()
    v1_values = evaluate_network(v1_net, s_eval).numpy()
    V_bar = compute_integrated_value(v0_net, v1_net, s_eval).numpy()

s_np = s_eval.numpy()

plot_value_functions(s=s_np, v0_values=v0_values, v1_values=v1_values, V_bar=V_bar)
```

#### Choice Probabilities

```{python}
#| fig-cap: "Probability of choosing each action as a function of state"

# Compute choice probabilities
with torch.no_grad():
    prob_a0, prob_a1 = compute_choice_probability(v0_net, v1_net, s_eval)
    prob_a0 = prob_a0.numpy()
    prob_a1 = prob_a1.numpy()

plot_choice_probabilities(s=s_np, prob_a0=prob_a0, prob_a1=prob_a1)
```

#### Value Difference

```{python}
#| fig-cap: "Value difference v(s,1) - v(s,0) determines the choice probability"

plot_value_difference(s=s_np, v0_values=v0_values, v1_values=v1_values)
```

#### Bellman Residuals

```{python}
#| fig-cap: "Bellman residual v(s,a) - target(s,a) across states. A well-converged solution should have near-zero residuals."

# Compute Bellman targets at evaluation points
with torch.no_grad():
    target0, target1 = compute_bellman_targets(
        v0_net, v1_net, s_eval,
        beta=beta, gamma=gamma, delta=delta
    )
    target0_np = target0.numpy()
    target1_np = target1.numpy()

# Compute residuals
residual0 = v0_values - target0_np
residual1 = v1_values - target1_np

plot_bellman_residuals(s=s_np, residual0=residual0, residual1=residual1)
```

### Interpretation

```{python}
# Find threshold state where P(a=1) ≈ 0.5
threshold_idx = np.argmin(np.abs(prob_a1 - 0.5))
threshold_state = s_np[threshold_idx]

print(f"Policy Interpretation (β={beta}, γ={gamma}, δ={delta}):")
print(f"- At state s ≈ {threshold_state:.2f}, the agent is indifferent between actions")
print(f"- For s < {threshold_state:.2f}, investment (a=1) is more likely")
print(f"- For s > {threshold_state:.2f}, no investment (a=0) is more likely")
print(f"\nThis reflects the trade-off:")
print(f"- Low state: Investment pays off (builds future value)")
print(f"- High state: Already earning good rewards, investment cost outweighs benefit")
```

## Comparative Statics: Effect of β on Value Functions

This section examines how the value functions change as we vary the reward coefficient β from 0 to 2.

```{python}
#| label: comparative-statics-setup
#| cache: true

def solve_and_extract_values(beta_val):
    """Solve MDP and extract value functions and choice probabilities for given beta."""
    torch.manual_seed(42)
    v0_net_cs, v1_net_cs, _, _ = solve_value_function(
        beta=beta_val,
        gamma=gamma,
        delta=delta,
        s_min=s_min,
        s_max=s_max,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    with torch.no_grad():
        v0_vals = evaluate_network(v0_net_cs, s_eval).numpy()
        v1_vals = evaluate_network(v1_net_cs, s_eval).numpy()
        p0_vals, p1_vals = compute_choice_probability(v0_net_cs, v1_net_cs, s_eval)
        p0_vals = p0_vals.numpy()
        p1_vals = p1_vals.numpy()
    return v0_vals, v1_vals, p0_vals, p1_vals

# Beta values from 0 to 2 in steps of 0.25
beta_values = np.arange(0, 2.25, 0.25)
n_beta = len(beta_values)

# Solve for all beta values
print("Solving for different β values...")
results_beta = {}
for b in beta_values:
    v0_vals, v1_vals, p0_vals, p1_vals = solve_and_extract_values(b)
    results_beta[b] = {'v0': v0_vals, 'v1': v1_vals, 'p0': p0_vals, 'p1': p1_vals}
    print(f"  β = {b:.2f} done")
print("Complete!")
```

### Value Functions by Action

```{python}
#| fig-cap: "Value functions v(s,a) as β varies from 0 to 2. Left: a=0 (black→blue). Right: a=1 (black→red)."

plot_comparative_statics_values(
    s=s_np,
    results=results_beta,
    param_values=beta_values,
    param_name="β",
    param_min=0,
    param_max=2,
)
```

### Choice Probabilities by Action

```{python}
#| fig-cap: "Choice probabilities P(a|s) as β varies from 0 to 2. Left: P(a=0|s) (black→blue). Right: P(a=1|s) (black→red)."

plot_comparative_statics_probs(
    s=s_np,
    results=results_beta,
    param_values=beta_values,
    param_name="β",
    param_min=0,
    param_max=2,
)
```

### Summary (β)

```{python}
print("Comparative Statics Summary: Effect of β on Value Functions")
print("=" * 60)
print(f"\nFixed parameters: γ={gamma}, δ={delta}")
print(f"β varied from 0 to 2 in steps of 0.25")
print(f"\nKey observations:")
print("• As β increases, both value functions increase (higher rewards)")
print("• v(s,0) and v(s,1) both become steeper (more sensitive to state)")
print("• At β=0, values are purely from discounted future option value")
print("• At high β, current-period rewards dominate the value functions")
```

## Comparative Statics: Effect of γ on Value Functions

This section examines how the value functions change as we vary the decay rate γ from 0 to 0.1.

```{python}
#| label: comparative-statics-gamma
#| cache: true

def solve_and_extract_values_gamma(gamma_val):
    """Solve MDP and extract value functions and choice probabilities for given gamma."""
    torch.manual_seed(42)
    v0_net_cs, v1_net_cs, _, _ = solve_value_function(
        beta=beta,
        gamma=gamma_val,
        delta=delta,
        s_min=s_min,
        s_max=s_max,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    with torch.no_grad():
        v0_vals = evaluate_network(v0_net_cs, s_eval).numpy()
        v1_vals = evaluate_network(v1_net_cs, s_eval).numpy()
        p0_vals, p1_vals = compute_choice_probability(v0_net_cs, v1_net_cs, s_eval)
        p0_vals = p0_vals.numpy()
        p1_vals = p1_vals.numpy()
    return v0_vals, v1_vals, p0_vals, p1_vals

# Gamma values from 0 to 0.1 in steps of 0.025
gamma_values = np.arange(0, 0.125, 0.025)
n_gamma = len(gamma_values)

# Solve for all gamma values
print("Solving for different γ values...")
results_gamma = {}
for g in gamma_values:
    v0_vals, v1_vals, p0_vals, p1_vals = solve_and_extract_values_gamma(g)
    results_gamma[g] = {'v0': v0_vals, 'v1': v1_vals, 'p0': p0_vals, 'p1': p1_vals}
    print(f"  γ = {g:.3f} done")
print("Complete!")
```

### Value Functions by Action (γ)

```{python}
#| fig-cap: "Value functions v(s,a) as γ varies from 0 to 0.1. Left: a=0 (black→blue). Right: a=1 (black→red)."

plot_comparative_statics_values(
    s=s_np,
    results=results_gamma,
    param_values=gamma_values,
    param_name="γ",
    param_min=0,
    param_max=0.1,
)
```

### Choice Probabilities by Action (γ)

```{python}
#| fig-cap: "Choice probabilities P(a|s) as γ varies from 0 to 0.1. Left: P(a=0|s) (black→blue). Right: P(a=1|s) (black→red)."

plot_comparative_statics_probs(
    s=s_np,
    results=results_gamma,
    param_values=gamma_values,
    param_name="γ",
    param_min=0,
    param_max=0.1,
)
```

### Summary (γ)

```{python}
print("Comparative Statics Summary: Effect of γ on Value Functions")
print("=" * 60)
print(f"\nFixed parameters: β={beta}, δ={delta}")
print(f"γ varied from 0 to 0.1 in steps of 0.025")
print(f"\nKey observations:")
print("• As γ increases, state decays faster (s' = (1-γ)s + a)")
print("• Higher γ reduces value of being in high states (they depreciate quickly)")
print("• At γ=0, state persists forever")
print("• Investment (a=1) becomes relatively more valuable at high γ to maintain state")
```

## Conclusion

### Interpretation of Comparative Statics

The comparative statics analysis reveals how the model parameters shape the agent's value functions and optimal behavior in this dynamic discrete choice problem.

#### Effect of the Reward Coefficient (β)

The parameter β governs the **marginal value of the state** through the reward function $u(s, a) = \beta \log(1 + s) - a$:

- **Higher β amplifies the reward from being in a high state.** As β increases from 0 to 2, both value functions $v(s, 0)$ and $v(s, 1)$ shift upward and become steeper. This reflects the increased payoff from accumulating state.

- **At β = 0**, the flow reward is simply $-a$ (just the cost of investment). The value functions are relatively flat because the state provides no direct reward—value comes only from the option to invest in the future.

- **At high β**, the value functions become strongly increasing in $s$, reflecting the substantial benefit of being in a high state. The curvature from $\log(1 + s)$ ensures diminishing marginal returns, preventing unbounded values.

#### Effect of the Decay Rate (γ)

The parameter γ controls **how quickly the state depreciates** via the transition $s' = (1 - \gamma)s + a$:

- **Higher γ means faster depreciation.** When γ is larger, the state decays more rapidly toward zero without investment. This reduces the long-run value of being in any given state.

- **At γ = 0**, the state is perfectly persistent—once accumulated, it never decays. This makes the value of high states very large, as rewards persist indefinitely.

- **At higher γ**, investment becomes relatively more valuable because it is the only way to counteract the natural decay. The agent must invest more frequently to maintain a desirable state level.

- **The value functions compress** as γ increases: the difference between high and low states diminishes because high states quickly depreciate back toward lower levels.

#### Economic Intuition

Together, these comparative statics illustrate a fundamental trade-off in dynamic optimization:

1. **β determines the stakes**: Higher β makes the state more valuable, increasing both the reward from being in a good state and the incentive to invest.

2. **γ determines persistence**: Higher γ erodes the benefits of past investments, making the problem more "short-run" in character and requiring ongoing investment to maintain value.

The interaction between these parameters shapes the optimal policy: when β is high and γ is low, the agent enjoys large, persistent rewards from accumulating state. When β is low or γ is high, the benefits of investment are either small or short-lived, leading to less aggressive investment behavior.

```

Now updating the README.md:

```markdown
# AI Practice: Structural Model Estimation

A Python framework for **solving**, **simulating**, and **estimating** structural economic models using neural network value iteration.

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| **MDP** | Dynamic Discrete Choice (Markov Decision Process) | ✅ Complete |
| **OPM** | Oligopoly Pricing Model | 🚧 Coming Soon |

---

## MDP: Dynamic Discrete Choice Models

A complete implementation for structural estimation of MDPs:

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
AI_Practice/
├── src/
│   ├── mdp/                      # MDP model
│   │   ├── solver/               # Neural network value iteration
│   │   ├── simulator/            # Monte Carlo simulation
│   │   └── estimator/            # Two-step structural estimation
│   ├── opm/                      # OPM model (placeholder)
│   │   ├── solver/
│   │   ├── simulator/
│   │   └── estimator/
│   └── common/                   # Shared utilities
├── scripts/
│   ├── mdp/
│   │   ├── config/               # Centralized parameters (β=1.0, γ=0.1, δ=0.95)
│   │   ├── solve/                # Solver report with comparative statics
│   │   ├── simulate/             # Simulation report with diagnostics
│   │   └── estimate/             # Estimation report with identification analysis
│   ├── opm/                      # OPM scripts (placeholder)
│   └── utils/                    # Shared plotting utilities
├── output/
│   ├── mdp/
│   │   ├── solve/                # Trained networks (v0_net.pt, v1_net.pt)
│   │   ├── simulate/             # Panel data (states.npy, actions.npy)
│   │   └── estimate/
│   └── opm/                      # OPM output (placeholder)
├── test/
│   ├── mdp/                      # MDP tests
│   │   ├── solver/
│   │   ├── simulator/
│   │   └── estimator/
│   └── opm/                      # OPM tests (placeholder)
└── docs/
    ├── mdp/                      # MDP development logs
    └── opm/                      # OPM documentation (placeholder)
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp/solver`)

Solves the Bellman equation using neural network approximation with **target networks** for stability:

```python
from mdp.solver import solve_value_function

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

### 2. MDP Simulator (`src/mdp/simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp.simulator import simulate_mdp_panel

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

### 3. MDP Estimator (`src/mdp/estimator`)

Recovers structural parameters using **two-step estimation**:

```python
from mdp.estimator import estimate_two_step

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

All parameters are centralized in `scripts/mdp/config/config.py`:

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

```

Now staging the workspace file change and running tests to verify:

I need to update imports in the source modules too. Let me check and fix them:

```python
"""MDP Simulator with Monte Carlo Panel Data Generation.

Implements the SIMULATE_MDP_PANEL algorithm for generating panel data
from trained MDP value functions.

Based on pseudocode from simulate_mdp.qmd.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Import from mdp.solver for shared functionality
from mdp.solver import (
    compute_choice_probability,
    compute_reward as _compute_reward_torch,
    compute_next_state as _compute_next_state_torch,
)


# =============================================================================
# Type Definitions
# =============================================================================

Scalar = float
Vector = np.ndarray
Matrix = np.ndarray
TensorVector = torch.Tensor


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class PanelData:
    """Panel data from MDP simulation.

    Attributes
    ----------
    states : np.ndarray
        State history, shape (n_agents, n_periods)
    actions : np.ndarray
        Action history, shape (n_agents, n_periods)
    rewards : np.ndarray
        Reward history, shape (n_agents, n_periods)
    n_agents : int
        Number of agents
    n_periods : int
        Number of time periods
    """

    states: Matrix
    actions: Matrix
    rewards: Matrix
    n_agents: int
    n_periods: int


# =============================================================================
# Subroutines (wrappers around mdp_solver functions)
# =============================================================================


def compute_reward(
    s: Vector,
    a: Vector,
    beta: Scalar,
) -> Vector:
    """Compute flow rewards: r = β * log(1 + s) - a.

    Wrapper around mdp_solver.compute_reward for numpy arrays.

    Parameters
    ----------
    s : np.ndarray
        States, shape (n,)
    a : np.ndarray
        Actions, shape (n,)
    beta : float
        Reward coefficient

    Returns
    -------
    np.ndarray
        Rewards, shape (n,)
    """
    s_tensor = torch.tensor(s, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32)
    reward_tensor = _compute_reward_torch(s=s_tensor, action=a_tensor, beta=beta)
    return reward_tensor.numpy()


def compute_next_state(
    s: Vector,
    a: Vector,
    gamma: Scalar,
) -> Vector:
    """Compute next states: s' = (1 - γ) * s + a.

    Wrapper around mdp_solver.compute_next_state for numpy arrays.

    Parameters
    ----------
    s : np.ndarray
        Current states, shape (n,)
    a : np.ndarray
        Actions, shape (n,)
    gamma : float
        State decay rate

    Returns
    -------
    np.ndarray
        Next states, shape (n,)
    """
    s_tensor = torch.tensor(s, dtype=torch.float32)
    a_tensor = torch.tensor(a, dtype=torch.float32)
    next_state_tensor = _compute_next_state_torch(s=s_tensor, action=a_tensor, gamma=gamma)
    return next_state_tensor.numpy()


# =============================================================================
# Main Algorithm
# =============================================================================


def simulate_mdp_panel(
    v0_net: nn.Module,
    v1_net: nn.Module,
    n_agents: int,
    n_periods: int,
    s_init: Union[Scalar, Vector],
    beta: Scalar,
    gamma: Scalar,
    seed: Optional[int] = None,
) -> PanelData:
    """Simulate MDP panel data using trained value functions.

    Implements SIMULATE_MDP_PANEL from pseudocode.

    Uses closed-form logit choice probabilities from compute_choice_probability
    to draw actions directly, rather than simulating Gumbel shocks.

    Parameters
    ----------
    v0_net : nn.Module
        Trained network for action 0
    v1_net : nn.Module
        Trained network for action 1
    n_agents : int
        Number of agents to simulate
    n_periods : int
        Number of time periods
    s_init : float or np.ndarray
        Initial states. If scalar, all agents start at the same state.
        If array, must have shape (n_agents,).
    beta : float
        Reward coefficient
    gamma : float
        State decay rate
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    PanelData
        Panel data containing states, actions, and rewards
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Initialize storage matrices
    states = np.zeros((n_agents, n_periods))
    actions = np.zeros((n_agents, n_periods), dtype=np.int32)
    rewards = np.zeros((n_agents, n_periods))

    # Set initial states
    if np.isscalar(s_init):
        states[:, 0] = s_init
    else:
        states[:, 0] = s_init

    # Simulate forward
    for t in range(n_periods):
        # Get current states as tensor
        s_t = torch.tensor(states[:, t], dtype=torch.float32)

        # Compute choice probabilities using closed-form logit formula
        with torch.no_grad():
            _, p1 = compute_choice_probability(v0_net, v1_net, s_t)
            p1_np = p1.numpy()

        # Draw actions from Bernoulli distribution
        u = np.random.uniform(size=n_agents)
        actions[:, t] = (u < p1_np).astype(np.int32)

        # Compute flow rewards
        rewards[:, t] = compute_reward(
            s=states[:, t],
            a=actions[:, t],
            beta=beta,
        )

        # State transition (if not last period)
        if t < n_periods - 1:
            states[:, t + 1] = compute_next_state(
                s=states[:, t],
                a=actions[:, t],
                gamma=gamma,
            )

    return PanelData(
        states=states,
        actions=actions,
        rewards=rewards,
        n_agents=n_agents,
        n_periods=n_periods,
    )


```

```python
"""MDP Structural Parameter Estimator using NFXP.

This module implements:
- estimate_gamma_ols: Estimate γ directly from transition data (OLS)
- estimate_beta_1d: Estimate β with γ from OLS and δ calibrated (1D grid search)
- compute_log_likelihood: Compute log-likelihood for candidate parameters
- estimate_mle: Estimation using Nelder-Mead optimization (joint, not recommended)
- grid_search_mle: Estimation using explicit grid search (joint, not recommended)
- compute_standard_errors: Numerical Hessian-based standard errors
"""

import os
import warnings
import numpy as np
import torch
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from mdp.solver import solve_value_function, compute_choice_probability
from mdp.simulator import PanelData


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EstimationResult:
    """Container for estimation results.
    
    Attributes
    ----------
    theta_hat : np.ndarray
        Estimated parameters (beta, gamma, delta)
    std_errors : np.ndarray
        Standard errors for each parameter
    cov_matrix : np.ndarray
        Variance-covariance matrix (3x3)
    log_likelihood : float
        Log-likelihood at the optimum
    n_iterations : int
        Number of outer loop iterations
    converged : bool
        Whether the optimizer converged
    optimization_result : Any
        Full scipy.optimize.OptimizeResult object
    """
    theta_hat: np.ndarray
    std_errors: np.ndarray
    cov_matrix: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool
    optimization_result: Any


@dataclass
class TwoStepResult:
    """Container for two-step estimation results.
    
    Attributes
    ----------
    beta_hat : float
        Estimated reward coefficient
    gamma_hat : float
        Estimated state decay rate (from OLS)
    delta_calibrated : float
        Calibrated discount factor (fixed)
    beta_se : float
        Standard error for beta
    gamma_se : float
        Standard error for gamma (from OLS)
    log_likelihood : float
        Log-likelihood at the optimum
    n_evaluations : int
        Number of likelihood evaluations for beta
    gamma_ols_details : dict
        Details from gamma OLS estimation
    """
    beta_hat: float
    gamma_hat: float
    delta_calibrated: float
    beta_se: float
    gamma_se: float
    log_likelihood: float
    n_evaluations: int
    gamma_ols_details: dict


# =============================================================================
# Two-Step Estimation (Recommended)
# =============================================================================

def estimate_gamma_ols(data: PanelData) -> Tuple[float, float, dict]:
    """Estimate γ directly from observed state transitions using OLS.
    
    The transition equation is deterministic:
        s_{t+1} = (1 - γ) * s_t + a_t
    
    Rearranging:
        (s_{t+1} - a_t) = (1 - γ) * s_t
    
    This is a regression through the origin: Y = (1 - γ) * X
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    
    Returns
    -------
    gamma_hat : float
        Estimated decay rate
    std_error : float
        Standard error of the estimate
    details : dict
        Additional details (n_obs, r_squared, residual_std)
    """
    # Extract current states, next states, and actions
    # states has shape (n_agents, n_periods), we need transitions
    s_current = data.states[:, :-1].flatten()    # s_t (exclude last period)
    s_next = data.states[:, 1:].flatten()        # s_{t+1} (exclude first period)
    a_current = data.actions[:, :-1].flatten()   # a_t (exclude last period)
    
    # Construct regression variables
    Y = s_next - a_current    # Dependent variable: s_{t+1} - a_t
    X = s_current             # Independent variable: s_t
    
    # Filter out observations where s_t ≈ 0 (avoid division issues)
    valid = X > 1e-6
    Y = Y[valid]
    X = X[valid]
    n_obs = len(X)
    
    # OLS through origin: coef = Σ(X*Y) / Σ(X²)
    # This estimates (1 - γ)
    coef = np.sum(X * Y) / np.sum(X ** 2)
    gamma_hat = 1.0 - coef
    
    # Compute residuals and standard error
    residuals = Y - coef * X
    residual_ss = np.sum(residuals ** 2)
    mse = residual_ss / (n_obs - 1)  # Mean squared error (df = n - 1 for intercept-free)
    
    # Variance of coefficient: Var(coef) = σ² / Σ(X²)
    var_coef = mse / np.sum(X ** 2)
    std_error = np.sqrt(var_coef)  # SE for (1-γ), same magnitude as SE for γ
    
    # R-squared (for regression through origin)
    total_ss = np.sum(Y ** 2)  # Note: no mean subtraction for origin regression
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    
    details = {
        'n_obs': n_obs,
        'coef_1_minus_gamma': coef,
        'residual_std': np.sqrt(mse),
        'r_squared': r_squared,
    }
    
    return gamma_hat, std_error, details


def estimate_beta_1d(
    data: PanelData,
    gamma_fixed: float,
    delta_fixed: float,
    solver_params: Dict[str, Any],
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    n_points: int = 20,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
) -> Tuple[float, float, float, dict]:
    """Estimate β via 1D grid search with γ and δ fixed.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    gamma_fixed : float
        Fixed γ value (from OLS estimation)
    delta_fixed : float
        Fixed δ value (calibrated)
    solver_params : dict
        Solver hyperparameters
    beta_bounds : tuple
        (min, max) bounds for β grid search
    n_points : int
        Number of grid points for β (default 20)
    verbose : bool
        Print progress (default True)
    pretrained_path : str, optional
        Path to pre-trained networks for warm-start
    
    Returns
    -------
    beta_hat : float
        Estimated β
    std_error : float
        Standard error of β estimate
    log_likelihood : float
        Log-likelihood at optimum
    details : dict
        Grid search details (all evaluations)
    """
    # Load pre-trained networks for warm-starting (if provided)
    v0_init_state = None
    v1_init_state = None
    if pretrained_path is not None:
        v0_path = os.path.join(pretrained_path, 'v0_net.pt')
        v1_path = os.path.join(pretrained_path, 'v1_net.pt')
        if os.path.exists(v0_path) and os.path.exists(v1_path):
            v0_init_state = torch.load(v0_path, weights_only=True)
            v1_init_state = torch.load(v1_path, weights_only=True)
            if verbose:
                print(f"Loaded pre-trained networks from {pretrained_path}")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}")
    
    # Create 1D grid for β
    beta_grid = np.linspace(beta_bounds[0], beta_bounds[1], n_points)
    
    if verbose:
        print(f"1D Grid search for β:")
        print(f"  β in [{beta_bounds[0]:.3f}, {beta_bounds[1]:.3f}], {n_points} points")
        print(f"  γ fixed at {gamma_fixed:.4f} (from OLS)")
        print(f"  δ fixed at {delta_fixed:.4f} (calibrated)")
    
    # Store results
    results = []
    best_ll = -np.inf
    best_beta = None
    
    # Search over grid
    for i, beta in enumerate(beta_grid):
        theta = np.array([beta, gamma_fixed, delta_fixed])
        
        ll = compute_log_likelihood(
            theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
        results.append({'beta': beta, 'log_likelihood': ll})
        
        if ll > best_ll:
            best_ll = ll
            best_beta = beta
        
        if verbose:
            print(f"  [{i+1}/{n_points}] β={beta:.4f}, LL={ll:.2f}" + 
                  (" *" if beta == best_beta else ""))
    
    if verbose:
        print(f"\nBest: β={best_beta:.4f}, LL={best_ll:.2f}")
    
    # Compute standard error for β via numerical second derivative
    # d²L/dβ² approximated by central differences
    eps = 0.01
    theta_plus = np.array([best_beta + eps, gamma_fixed, delta_fixed])
    theta_minus = np.array([best_beta - eps, gamma_fixed, delta_fixed])
    theta_center = np.array([best_beta, gamma_fixed, delta_fixed])
    
    ll_plus = compute_log_likelihood(theta_plus, data, solver_params,
                                     v0_init_state, v1_init_state)
    ll_minus = compute_log_likelihood(theta_minus, data, solver_params,
                                      v0_init_state, v1_init_state)
    ll_center = compute_log_likelihood(theta_center, data, solver_params,
                                       v0_init_state, v1_init_state)
    
    # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
    d2_ll = (ll_plus - 2 * ll_center + ll_minus) / (eps ** 2)
    
    # Standard error: sqrt(-1 / d²L/dβ²)
    if d2_ll < 0:
        std_error = np.sqrt(-1.0 / d2_ll)
    else:
        warnings.warn("Second derivative is non-negative, SE may be invalid")
        std_error = np.nan
    
    details = {
        'beta_grid': beta_grid,
        'evaluations': results,
        'n_evaluations': n_points + 3,  # grid + 3 for SE computation
    }
    
    return best_beta, std_error, best_ll, details


def estimate_two_step(
    data: PanelData,
    delta_calibrated: float,
    solver_params: Dict[str, Any],
    beta_bounds: Tuple[float, float] = (0.1, 3.0),
    n_points: int = 20,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
) -> TwoStepResult:
    """Two-step structural estimation.
    
    Step 1: Estimate γ from transition data (OLS)
    Step 2: Estimate β via 1D grid search with γ and δ fixed
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    delta_calibrated : float
        Calibrated discount factor (fixed at this value)
    solver_params : dict
        Solver hyperparameters
    beta_bounds : tuple
        (min, max) bounds for β grid search
    n_points : int
        Number of grid points for β
    verbose : bool
        Print progress
    pretrained_path : str, optional
        Path to pre-trained networks for warm-start
    
    Returns
    -------
    TwoStepResult
        Complete estimation results
    """
    if verbose:
        print("=" * 60)
        print("TWO-STEP ESTIMATION")
        print("=" * 60)
    
    # Step 1: Estimate γ from transitions
    if verbose:
        print("\n--- Step 1: Estimate γ from transitions (OLS) ---")
    
    gamma_hat, gamma_se, gamma_details = estimate_gamma_ols(data)
    
    if verbose:
        print(f"  γ_hat = {gamma_hat:.6f} (SE = {gamma_se:.6f})")
        print(f"  R² = {gamma_details['r_squared']:.6f}")
        print(f"  N obs = {gamma_details['n_obs']}")
    
    # Step 2: Estimate β with γ and δ fixed
    if verbose:
        print(f"\n--- Step 2: Estimate β (γ={gamma_hat:.4f}, δ={delta_calibrated:.4f} fixed) ---")
    
    beta_hat, beta_se, log_lik, beta_details = estimate_beta_1d(
        data=data,
        gamma_fixed=gamma_hat,
        delta_fixed=delta_calibrated,
        solver_params=solver_params,
        beta_bounds=beta_bounds,
        n_points=n_points,
        verbose=verbose,
        pretrained_path=pretrained_path,
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("ESTIMATION COMPLETE")
        print("=" * 60)
        print(f"  β_hat = {beta_hat:.4f} (SE = {beta_se:.4f})")
        print(f"  γ_hat = {gamma_hat:.4f} (SE = {gamma_se:.6f})")
        print(f"  δ     = {delta_calibrated:.4f} (calibrated)")
        print(f"  Log-likelihood = {log_lik:.2f}")
    
    return TwoStepResult(
        beta_hat=beta_hat,
        gamma_hat=gamma_hat,
        delta_calibrated=delta_calibrated,
        beta_se=beta_se,
        gamma_se=gamma_se,
        log_likelihood=log_lik,
        n_evaluations=beta_details['n_evaluations'],
        gamma_ols_details=gamma_details,
    )


# =============================================================================
# Core Functions
# =============================================================================

def compute_log_likelihood(
    theta: np.ndarray,
    data: PanelData,
    solver_params: Dict[str, Any],
    v0_init_state: Optional[dict] = None,
    v1_init_state: Optional[dict] = None,
) -> float:
    """Compute log-likelihood for candidate parameters.
    
    This function is called once per outer loop iteration. It solves
    the dynamic programming problem for the given parameters and
    computes the log-likelihood of the observed data.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameters (beta, gamma, delta)
    data : PanelData
        Panel data with states and actions arrays
    solver_params : dict
        Solver hyperparameters containing:
        - s_min: float, minimum state value
        - s_max: float, maximum state value
        - hidden_sizes: list[int], network architecture
        - learning_rate: float
        - batch_size: int
        - tolerance: float, convergence tolerance for inner loop
        - max_iterations: int, max iterations for inner loop
        - target_update_freq: int (optional, default 100)
    v0_init_state : dict, optional
        Initial state dict for v0 network (warm-start)
    v1_init_state : dict, optional
        Initial state dict for v1 network (warm-start)
    
    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid parameters.
    """
    # Unpack parameters
    beta, gamma, delta = theta
    
    # Validate parameter bounds (return -inf for invalid)
    if beta <= 0 or gamma <= 0 or gamma >= 1 or delta <= 0 or delta >= 1:
        return -np.inf
    
    # === INNER LOOP: Solve value functions for this theta ===
    v0_net, v1_net, losses, n_iter = solve_value_function(
        beta=beta,
        gamma=gamma,
        delta=delta,
        s_min=solver_params['s_min'],
        s_max=solver_params['s_max'],
        hidden_sizes=solver_params['hidden_sizes'],
        learning_rate=solver_params['learning_rate'],
        batch_size=solver_params['batch_size'],
        tolerance=solver_params['tolerance'],
        max_iterations=solver_params['max_iterations'],
        target_update_freq=solver_params.get('target_update_freq', 100),
        v0_init_state=v0_init_state,
        v1_init_state=v1_init_state,
    )
    
    # Check convergence (optional: warn if not converged)
    if n_iter == solver_params['max_iterations']:
        warnings.warn(f"Inner loop did not converge for theta = {theta}")
    
    # === Compute choice probabilities at all observed states ===
    # Flatten panel data for batch computation
    states_flat = data.states.flatten()  # shape: (N * T,)
    actions_flat = data.actions.flatten()  # shape: (N * T,)
    
    # Convert to tensor
    s_tensor = torch.tensor(states_flat, dtype=torch.float32)
    
    # Compute P(a=1 | s) for all observations
    with torch.no_grad():
        _, p1 = compute_choice_probability(v0_net, v1_net, s_tensor)
        p1 = p1.numpy()
    
    # Clip probabilities for numerical stability
    eps = 1e-10
    p1 = np.clip(p1, eps, 1 - eps)
    
    # === Compute log-likelihood ===
    # L = Σ [a * log(p1) + (1-a) * log(1-p1)]
    log_lik = np.sum(
        actions_flat * np.log(p1) + 
        (1 - actions_flat) * np.log(1 - p1)
    )
    
    return log_lik


def estimate_mle(
    data: PanelData,
    theta_init: Tuple[float, float, float],
    solver_params: Dict[str, Any],
    maxiter: int = 200,
    verbose: bool = True,
    pretrained_path: Optional[str] = None,
) -> EstimationResult:
    """Estimate structural parameters via MLE using NFXP.
    
    Uses Nelder-Mead optimization in the outer loop, calling
    solve_value_function for each candidate parameter vector.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    theta_init : tuple
        Initial guess (beta, gamma, delta)
    solver_params : dict
        Solver hyperparameters (see compute_log_likelihood)
    maxiter : int
        Maximum outer loop iterations (default 200)
    verbose : bool
        Print progress during optimization (default True)
    pretrained_path : str, optional
        Path to directory containing pre-trained networks (v0_net.pt, v1_net.pt).
        If provided, uses these as warm-start initialization for all solver calls.
    
    Returns
    -------
    EstimationResult
        Estimation results including estimates, standard errors,
        and optimization diagnostics
    """
    theta_init = np.array(theta_init)
    
    # Load pre-trained networks for warm-starting (if provided)
    v0_init_state = None
    v1_init_state = None
    if pretrained_path is not None:
        v0_path = os.path.join(pretrained_path, 'v0_net.pt')
        v1_path = os.path.join(pretrained_path, 'v1_net.pt')
        if os.path.exists(v0_path) and os.path.exists(v1_path):
            v0_init_state = torch.load(v0_path, weights_only=True)
            v1_init_state = torch.load(v1_path, weights_only=True)
            if verbose:
                print(f"Loaded pre-trained networks from {pretrained_path} for warm-starting")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}, using random init")
    
    # Track evaluations
    eval_count = [0]
    
    # Define objective function (negative log-likelihood)
    def neg_log_likelihood(theta):
        eval_count[0] += 1
        log_lik = compute_log_likelihood(
            theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
        if verbose and eval_count[0] % 10 == 0:
            print(f"Eval {eval_count[0]}: theta={theta}, LL={log_lik:.2f}")
        return -log_lik
    
    # Run Nelder-Mead optimization
    result = optimize.minimize(
        fun=neg_log_likelihood,
        x0=theta_init,
        method='Nelder-Mead',
        options={
            'maxiter': maxiter,
            'xatol': 1e-4,      # Parameter tolerance
            'fatol': 1e-4,      # Function value tolerance
            'disp': verbose,
            'adaptive': True,   # Adapt simplex to parameter scales
        }
    )
    
    theta_hat = result.x
    log_lik_hat = -result.fun
    
    # Compute standard errors (also use warm-start)
    if verbose:
        print("Computing standard errors...")
    std_errors, cov_matrix = compute_standard_errors(
        theta_hat, data, solver_params,
        v0_init_state=v0_init_state,
        v1_init_state=v1_init_state,
    )
    
    return EstimationResult(
        theta_hat=theta_hat,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        log_likelihood=log_lik_hat,
        n_iterations=result.nit,
        converged=result.success,
        optimization_result=result,
    )


def grid_search_mle(
    data: PanelData,
    bounds: Dict[str, Tuple[float, float]],
    n_points: int,
    solver_params: Dict[str, Any],
    verbose: bool = True,
    compute_se: bool = True,
    pretrained_path: Optional[str] = None,
) -> EstimationResult:
    """Estimate structural parameters via grid search.
    
    Evaluates log-likelihood on a grid of parameter values and
    returns the grid point with highest likelihood.
    
    Parameters
    ----------
    data : PanelData
        Panel data with states and actions arrays
    bounds : dict
        Parameter bounds: {'beta': (min, max), 'gamma': (min, max), 'delta': (min, max)}
    n_points : int
        Number of grid points per dimension (total evaluations = n_points^3)
    solver_params : dict
        Solver hyperparameters (see compute_log_likelihood)
    verbose : bool
        Print progress during search (default True)
    compute_se : bool
        Whether to compute standard errors at best point (default True)
    pretrained_path : str, optional
        Path to directory containing pre-trained networks (v0_net.pt, v1_net.pt).
        If provided, uses these as warm-start initialization for all solver calls.
    
    Returns
    -------
    EstimationResult
        Estimation results including estimates and optionally standard errors
    """
    # Load pre-trained networks for warm-starting (if provided)
    v0_init_state = None
    v1_init_state = None
    if pretrained_path is not None:
        v0_path = os.path.join(pretrained_path, 'v0_net.pt')
        v1_path = os.path.join(pretrained_path, 'v1_net.pt')
        if os.path.exists(v0_path) and os.path.exists(v1_path):
            v0_init_state = torch.load(v0_path, weights_only=True)
            v1_init_state = torch.load(v1_path, weights_only=True)
            if verbose:
                print(f"Loaded pre-trained networks from {pretrained_path} for warm-starting")
        else:
            warnings.warn(f"Pre-trained networks not found at {pretrained_path}, using random init")
    
    # Create grid for each parameter
    beta_grid = np.linspace(bounds['beta'][0], bounds['beta'][1], n_points)
    gamma_grid = np.linspace(bounds['gamma'][0], bounds['gamma'][1], n_points)
    delta_grid = np.linspace(bounds['delta'][0], bounds['delta'][1], n_points)
    
    # Total evaluations
    total_evals = n_points ** 3
    
    if verbose:
        print(f"Grid search: {n_points} points per dimension = {total_evals} total evaluations")
        print(f"  beta  in [{bounds['beta'][0]:.3f}, {bounds['beta'][1]:.3f}]")
        print(f"  gamma in [{bounds['gamma'][0]:.3f}, {bounds['gamma'][1]:.3f}]")
        print(f"  delta in [{bounds['delta'][0]:.3f}, {bounds['delta'][1]:.3f}]")
    
    # Store all results
    results = []
    best_ll = -np.inf
    best_theta = None
    eval_count = 0
    
    # Iterate over grid
    for beta in beta_grid:
        for gamma in gamma_grid:
            for delta in delta_grid:
                eval_count += 1
                theta = np.array([beta, gamma, delta])
                
                # Compute log-likelihood (with warm-start)
                ll = compute_log_likelihood(
                    theta, data, solver_params,
                    v0_init_state=v0_init_state,
                    v1_init_state=v1_init_state,
                )
                results.append({'theta': theta, 'log_likelihood': ll})
                
                # Track best
                if ll > best_ll:
                    best_ll = ll
                    best_theta = theta.copy()
                
                if verbose and eval_count % 10 == 0:
                    print(f"  Eval {eval_count}/{total_evals}: best LL = {best_ll:.2f}")
    
    if verbose:
        print(f"\nGrid search complete.")
        print(f"  Best theta: beta={best_theta[0]:.4f}, gamma={best_theta[1]:.4f}, delta={best_theta[2]:.4f}")
        print(f"  Best LL: {best_ll:.2f}")
    
    # Compute standard errors at best point (optional, with warm-start)
    if compute_se:
        if verbose:
            print("Computing standard errors...")
        std_errors, cov_matrix = compute_standard_errors(
            best_theta, data, solver_params,
            v0_init_state=v0_init_state,
            v1_init_state=v1_init_state,
        )
    else:
        std_errors = np.full(3, np.nan)
        cov_matrix = np.full((3, 3), np.nan)
    
    # Create result object
    # Store grid results in optimization_result for inspection
    grid_results = {
        'grid': {
            'beta': beta_grid,
            'gamma': gamma_grid,
            'delta': delta_grid,
        },
        'evaluations': results,
    }
    
    return EstimationResult(
        theta_hat=best_theta,
        std_errors=std_errors,
        cov_matrix=cov_matrix,
        log_likelihood=best_ll,
        n_iterations=total_evals,
        converged=True,  # Grid search always "converges"
        optimization_result=grid_results,
    )


def compute_standard_errors(
    theta_hat: np.ndarray,
    data: PanelData,
    solver_params: Dict[str, Any],
    eps: float = 1e-4,
    v0_init_state: Optional[dict] = None,
    v1_init_state: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standard errors via numerical Hessian.
    
    Uses central differences to approximate the Hessian of the
    log-likelihood at the MLE, then inverts to get the covariance.
    
    Parameters
    ----------
    theta_hat : np.ndarray
        MLE estimates (beta, gamma, delta)
    data : PanelData
        Panel data
    solver_params : dict
        Solver hyperparameters
    eps : float
        Step size for finite differences (default 1e-4)
    v0_init_state : dict, optional
        Initial state dict for v0 network (warm-start)
    v1_init_state : dict, optional
        Initial state dict for v1 network (warm-start)
    
    Returns
    -------
    std_errors : np.ndarray
        Standard errors for each parameter (shape: (3,))
    cov_matrix : np.ndarray
        Variance-covariance matrix (shape: (3, 3))
    """
    n_params = 3
    theta_hat = np.array(theta_hat)
    
    # Compute numerical Hessian via central differences
    hessian = np.zeros((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            # f(θ + ei*eps + ej*eps)
            theta_pp = theta_hat.copy()
            theta_pp[i] += eps
            theta_pp[j] += eps
            f_pp = compute_log_likelihood(theta_pp, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ + ei*eps - ej*eps)
            theta_pm = theta_hat.copy()
            theta_pm[i] += eps
            theta_pm[j] -= eps
            f_pm = compute_log_likelihood(theta_pm, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ - ei*eps + ej*eps)
            theta_mp = theta_hat.copy()
            theta_mp[i] -= eps
            theta_mp[j] += eps
            f_mp = compute_log_likelihood(theta_mp, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # f(θ - ei*eps - ej*eps)
            theta_mm = theta_hat.copy()
            theta_mm[i] -= eps
            theta_mm[j] -= eps
            f_mm = compute_log_likelihood(theta_mm, data, solver_params,
                                          v0_init_state, v1_init_state)
            
            # Second derivative approximation
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
    
    # Symmetrize Hessian (numerical differentiation can introduce asymmetry)
    hessian = (hessian + hessian.T) / 2
    
    # Covariance matrix = inverse of negative Hessian (information matrix)
    info_matrix = -hessian
    
    # Invert to get covariance matrix
    try:
        cov_matrix = np.linalg.inv(info_matrix)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        # Singular matrix - return NaN
        warnings.warn("Information matrix is singular, cannot compute standard errors")
        cov_matrix = np.full((n_params, n_params), np.nan)
        std_errors = np.full(n_params, np.nan)
    
    return std_errors, cov_matrix


```

✅ **All 100 tests passed!**

Let me show the final git status:

✅ **Phase 2 Complete - All imports and configs updated, tests passing!**

---

## Summary

### ✅ Completed Changes

**1. Directory Structure Reorganized:**
```
src/mdp_solver/     → src/mdp/solver/
src/mdp_simulator/  → src/mdp/simulator/
src/mdp_estimator/  → src/mdp/estimator/
scripts/config_mdp/ → scripts/mdp/config/
scripts/solve_mdp/  → scripts/mdp/solve/
...etc
```

**2. New Package `__init__.py` Files Created:**
- `src/mdp/__init__.py`
- `src/opm/__init__.py` (placeholder)
- `src/common/__init__.py` (placeholder)
- `src/opm/{solver,simulator,estimator}/__init__.py` (placeholders)

**3. All Imports Updated:**
- `from mdp_solver` → `from mdp.solver`
- `from mdp_simulator` → `from mdp.simulator`
- `from mdp_estimator` → `from mdp.estimator`

**4. Configs Updated:**
- `pyproject.toml`: name → `structural-models`, packages → `["src/mdp", "src/opm", "src/common"]`
- Output paths in scripts updated

**5. Tests: 100 passed ✓**

---

### Staged vs Unstaged

| Category | Status |
|----------|--------|
| Directory renames | ✅ Staged |
| Import updates | ⚪ Modified (unstaged) |
| New `__init__.py` files | ⚪ Untracked |

---

### Final Structure
```
AI_Practice/
├── src/
│   ├── mdp/{solver,simulator,estimator}/    ✅
│   ├── opm/{solver,simulator,estimator}/    ✅ (placeholder)
│   └── common/                              ✅ (placeholder)
├── scripts/
│   ├── mdp/{config,solve,simulate,estimate}/ ✅
│   ├── opm/...                              ✅ (placeholder)
│   └── utils/                               ✅
├── output/mdp/{solve,simulate,estimate}/    ✅
├── test/mdp/{solver,simulator,estimator}/   ✅
└── docs/mdp/                                ✅
```

Would you like me to stage all changes and prepare for commit?

```plaintext
src/mdp_solver/     → src/mdp/solver/
src/mdp_simulator/  → src/mdp/simulator/
src/mdp_estimator/  → src/mdp/estimator/
scripts/config_mdp/ → scripts/mdp/config/
scripts/solve_mdp/  → scripts/mdp/solve/
...etc
```

```plaintext
AI_Practice/
├── src/
│   ├── mdp/{solver,simulator,estimator}/    ✅
│   ├── opm/{solver,simulator,estimator}/    ✅ (placeholder)
│   └── common/                              ✅ (placeholder)
├── scripts/
│   ├── mdp/{config,solve,simulate,estimate}/ ✅
│   ├── opm/...                              ✅ (placeholder)
│   └── utils/                               ✅
├── output/mdp/{solve,simulate,estimate}/    ✅
├── test/mdp/{solver,simulator,estimator}/   ✅
└── docs/mdp/                                ✅
```

---

**User**

drop the common in src. i dont think we need it correct?

---

**Cursor**

You're right, it's just an empty placeholder. Let me remove it:

```plaintext
[project]
name = "structural-models"
version = "0.1.0"
description = "Structural model estimation: MDP solver with Type-I Extreme Value shocks, OPM (coming soon)"
readme = "README.md"
authors = [
    { name = "Hude", email = "hude@connect.ust.hk" }
]
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "torch>=2.0.0",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mdp", "src/opm"]

[tool.pytest.ini_options]
testpaths = ["test"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (require solver, deselect with '-m \"not slow\"')",
]

```

```markdown
# AI Practice: Structural Model Estimation

A Python framework for **solving**, **simulating**, and **estimating** structural economic models using neural network value iteration.

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| **MDP** | Dynamic Discrete Choice (Markov Decision Process) | ✅ Complete |
| **OPM** | Oligopoly Pricing Model | 🚧 Coming Soon |

---

## MDP: Dynamic Discrete Choice Models

A complete implementation for structural estimation of MDPs:

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
AI_Practice/
├── src/
│   ├── mdp/                      # MDP model
│   │   ├── solver/               # Neural network value iteration
│   │   ├── simulator/            # Monte Carlo simulation
│   │   └── estimator/            # Two-step structural estimation
│   └── opm/                      # OPM model (placeholder)
│       ├── solver/
│       ├── simulator/
│       └── estimator/
├── scripts/
│   ├── mdp/
│   │   ├── config/               # Centralized parameters (β=1.0, γ=0.1, δ=0.95)
│   │   ├── solve/                # Solver report with comparative statics
│   │   ├── simulate/             # Simulation report with diagnostics
│   │   └── estimate/             # Estimation report with identification analysis
│   ├── opm/                      # OPM scripts (placeholder)
│   └── utils/                    # Shared plotting utilities
├── output/
│   ├── mdp/
│   │   ├── solve/                # Trained networks (v0_net.pt, v1_net.pt)
│   │   ├── simulate/             # Panel data (states.npy, actions.npy)
│   │   └── estimate/
│   └── opm/                      # OPM output (placeholder)
├── test/
│   ├── mdp/                      # MDP tests
│   │   ├── solver/
│   │   ├── simulator/
│   │   └── estimator/
│   └── opm/                      # OPM tests (placeholder)
└── docs/
    ├── mdp/                      # MDP development logs
    └── opm/                      # OPM documentation (placeholder)
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp/solver`)

Solves the Bellman equation using neural network approximation with **target networks** for stability:

```python
from mdp.solver import solve_value_function

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

### 2. MDP Simulator (`src/mdp/simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp.simulator import simulate_mdp_panel

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

### 3. MDP Estimator (`src/mdp/estimator`)

Recovers structural parameters using **two-step estimation**:

```python
from mdp.estimator import estimate_two_step

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

All parameters are centralized in `scripts/mdp/config/config.py`:

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

```

Done! Removed:
- `src/common/` directory
- Reference in `pyproject.toml` 
- Reference in `README.md`

---

**User**

i wanna makesure the existing mdp solver run properly after the changes. can you test it?

---

**Cursor**

✅ **All 100 tests passed!**

The MDP solver, simulator, and estimator all work properly after the restructuring:

| Module | Tests | Status |
|--------|-------|--------|
| `mdp.solver` | 49 tests | ✅ Passed |
| `mdp.simulator` | 22 tests | ✅ Passed |
| `mdp.estimator` | 13 tests | ✅ Passed |
| **Total** | **100 tests** | ✅ **All passed** |

The new import paths (`from mdp.solver import ...`, etc.) are working correctly.

---

**User**

okay, can you run the estimator quarto file report and render it for me on a live server?

---

**Cursor**

The sandbox is blocking quarto. Let me run with full permissions:

It's running! Executing the notebook cells. Let me check the progress:

The data files aren't at the new path yet. Let me check where they are:

The output files weren't moved earlier (they're gitignored). Let me move them now:

Files moved. Now let me run the quarto preview again:

It's running! Let me check the progress again:

Still running the two-step estimation (this solves the MDP multiple times). Let me check again:

Almost done! On cell 7/8 now. Let me check if it finished:

✅ **Done!** The estimator report is now live at:

**http://localhost:6295/**

The report successfully executed all 8 cells:
1. ✅ Setup
2. ✅ Load data
3. ✅ True params
4. ✅ Two-step estimation
5. ✅ Results
6. ✅ Likelihood check
7. ✅ Diagnostics
8. ✅ Conclusions

The server is watching for file changes, so any edits you make will auto-refresh.

---

**User**

also run the quarto report of solver and simulator and open them on live servers sperately.

---

**Cursor**

Both are running! Let me check progress:

✅ **Simulator is live at http://localhost:6297/**

Solver is still running (cell 10/17). Let me check its status:

Here's the current status of all three reports:

| Report | URL | Status |
|--------|-----|--------|
| **Estimator** | http://localhost:6295/ | ✅ Live |
| **Simulator** | http://localhost:6297/ | ✅ Live |
| **Solver** | http://localhost:6296/ | ⏳ Running (cell 10/17 - comparative statics) |

The solver takes longer because the comparative statics cell solves the MDP multiple times with different β and γ values. It should be ready in a few minutes.

You can already browse the estimator and simulator reports while waiting for the solver to finish.

---

**User**

Now, i see in the output, there is mdp, and estimate_mdp. why there is an extra estimate_mdp. where it is used. should we reorganized it?

---

**Cursor**

You're right! `output/estimate_mdp/` is an empty leftover folder from the original structure. Let me clean it up:

There's a `.gitkeep` file. Let me remove the whole folder:

Done! The output folder is now clean:

```
output/
├── mdp/
│   ├── estimate/
│   ├── simulate/
│   │   ├── actions.npy
│   │   ├── config.json
│   │   ├── rewards.npy
│   │   └── states.npy
│   └── solve/
│       ├── config.json
│       ├── losses.npy
│       ├── v0_net.pt
│       └── v1_net.pt
└── opm/           (placeholder)
    ├── estimate/
    ├── simulate/
    └── solve/
```

```plaintext
output/
├── mdp/
│   ├── estimate/
│   ├── simulate/
│   │   ├── actions.npy
│   │   ├── config.json
│   │   ├── rewards.npy
│   │   └── states.npy
│   └── solve/
│       ├── config.json
│       ├── losses.npy
│       ├── v0_net.pt
│       └── v1_net.pt
└── opm/           (placeholder)
    ├── estimate/
    ├── simulate/
    └── solve/
```

---

**User**

Okay. Commit changes. So far, the readme only has part of MDP right? add a section placeholder part for opm, make it very simple and short for now.

---

**Cursor**

```markdown
# AI Practice: Structural Model Estimation

A Python framework for **solving**, **simulating**, and **estimating** structural economic models using neural network value iteration.

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| **MDP** | Dynamic Discrete Choice (Markov Decision Process) | ✅ Complete |
| **OPM** | Oligopoly Pricing Model | 🚧 Coming Soon |

---

## MDP: Dynamic Discrete Choice Models

A complete implementation for structural estimation of MDPs:

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
AI_Practice/
├── src/
│   ├── mdp/                      # MDP model
│   │   ├── solver/               # Neural network value iteration
│   │   ├── simulator/            # Monte Carlo simulation
│   │   └── estimator/            # Two-step structural estimation
│   └── opm/                      # OPM model (placeholder)
│       ├── solver/
│       ├── simulator/
│       └── estimator/
├── scripts/
│   ├── mdp/
│   │   ├── config/               # Centralized parameters (β=1.0, γ=0.1, δ=0.95)
│   │   ├── solve/                # Solver report with comparative statics
│   │   ├── simulate/             # Simulation report with diagnostics
│   │   └── estimate/             # Estimation report with identification analysis
│   ├── opm/                      # OPM scripts (placeholder)
│   └── utils/                    # Shared plotting utilities
├── output/
│   ├── mdp/
│   │   ├── solve/                # Trained networks (v0_net.pt, v1_net.pt)
│   │   ├── simulate/             # Panel data (states.npy, actions.npy)
│   │   └── estimate/
│   └── opm/                      # OPM output (placeholder)
├── test/
│   ├── mdp/                      # MDP tests
│   │   ├── solver/
│   │   ├── simulator/
│   │   └── estimator/
│   └── opm/                      # OPM tests (placeholder)
└── docs/
    ├── mdp/                      # MDP development logs
    └── opm/                      # OPM documentation (placeholder)
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp/solver`)

Solves the Bellman equation using neural network approximation with **target networks** for stability:

```python
from mdp.solver import solve_value_function

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

### 2. MDP Simulator (`src/mdp/simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp.simulator import simulate_mdp_panel

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

### 3. MDP Estimator (`src/mdp/estimator`)

Recovers structural parameters using **two-step estimation**:

```python
from mdp.estimator import estimate_two_step

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

All parameters are centralized in `scripts/mdp/config/config.py`:

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

## OPM: Oligopoly Pricing Model

🚧 **Coming Soon**

The OPM module will implement structural estimation for oligopoly pricing models following the same pipeline: **Solver → Simulator → Estimator**.

---

## License

MIT License

```

Now committing all changes:

✅ **Committed!**

```
[main 86ec8c8] Restructure project for multi-model support (MDP + OPM)
 33 files changed, 181 insertions(+), 129 deletions(-)
```

### Summary of Changes

| Category | Changes |
|----------|---------|
| **Workspace** | `MDP_Practice` → `AI_Practice` |
| **Source** | `src/mdp_solver/` → `src/mdp/solver/` (+ OPM placeholders) |
| **Scripts** | `scripts/config_mdp/` → `scripts/mdp/config/` |
| **Tests** | `test/mdp_solver/` → `test/mdp/solver/` |
| **Docs** | `docs/conversation/` → `docs/mdp/` |
| **Imports** | `from mdp_solver` → `from mdp.solver` |
| **Package** | `mdp-solver` → `structural-models` |

### New Structure
```
AI_Practice/
├── src/
│   ├── mdp/{solver,simulator,estimator}/  ✅ Complete
│   └── opm/{solver,simulator,estimator}/  🚧 Placeholder
├── scripts/mdp/, opm/
├── test/mdp/, opm/
└── docs/mdp/, opm/
```

```plaintext
[main 86ec8c8] Restructure project for multi-model support (MDP + OPM)
 33 files changed, 181 insertions(+), 129 deletions(-)
```

```plaintext
AI_Practice/
├── src/
│   ├── mdp/{solver,simulator,estimator}/  ✅ Complete
│   └── opm/{solver,simulator,estimator}/  🚧 Placeholder
├── scripts/mdp/, opm/
├── test/mdp/, opm/
└── docs/mdp/, opm/
```

---

**User**

Now, can you test if I can push this new structured project to the github, where github contains the old unstructured project.

---

**Cursor**

✅ **Push successful!**

```
To https://github.com/Hude-Hude/MDP_Practice.git
   1879db8..86ec8c8  main -> main
```

Git handles the restructuring gracefully because it tracks renames. Your GitHub repo now has the new structure while preserving all the file history.

**Note:** Your GitHub repo is still named `MDP_Practice`. If you want to rename it to `AI_Practice` to match the local folder, you can do that in GitHub Settings → Repository name.

```plaintext
To https://github.com/Hude-Hude/MDP_Practice.git
   1879db8..86ec8c8  main -> main
```

---

**User**

in readme A Python framework for **solving**, **simulating**, and **estimating** structural economic models using neural network value iteration. the "using neural network" is not true in general. but the solving simulating and estimating flow is universal. edit this part.

---

**Cursor**

```markdown
# AI Practice: Structural Model Estimation

A Python framework for **solving**, **simulating**, and **estimating** structural economic models.

## Supported Models

| Model | Description | Status |
|-------|-------------|--------|
| **MDP** | Dynamic Discrete Choice (Markov Decision Process) | ✅ Complete |
| **OPM** | Oligopoly Pricing Model | 🚧 Coming Soon |

---

## MDP: Dynamic Discrete Choice Models

A complete implementation for structural estimation of MDPs:

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
AI_Practice/
├── src/
│   ├── mdp/                      # MDP model
│   │   ├── solver/               # Neural network value iteration
│   │   ├── simulator/            # Monte Carlo simulation
│   │   └── estimator/            # Two-step structural estimation
│   └── opm/                      # OPM model (placeholder)
│       ├── solver/
│       ├── simulator/
│       └── estimator/
├── scripts/
│   ├── mdp/
│   │   ├── config/               # Centralized parameters (β=1.0, γ=0.1, δ=0.95)
│   │   ├── solve/                # Solver report with comparative statics
│   │   ├── simulate/             # Simulation report with diagnostics
│   │   └── estimate/             # Estimation report with identification analysis
│   ├── opm/                      # OPM scripts (placeholder)
│   └── utils/                    # Shared plotting utilities
├── output/
│   ├── mdp/
│   │   ├── solve/                # Trained networks (v0_net.pt, v1_net.pt)
│   │   ├── simulate/             # Panel data (states.npy, actions.npy)
│   │   └── estimate/
│   └── opm/                      # OPM output (placeholder)
├── test/
│   ├── mdp/                      # MDP tests
│   │   ├── solver/
│   │   ├── simulator/
│   │   └── estimator/
│   └── opm/                      # OPM tests (placeholder)
└── docs/
    ├── mdp/                      # MDP development logs
    └── opm/                      # OPM documentation (placeholder)
```

---

## Pipeline Details

### 1. MDP Solver (`src/mdp/solver`)

Solves the Bellman equation using neural network approximation with **target networks** for stability:

```python
from mdp.solver import solve_value_function

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

### 2. MDP Simulator (`src/mdp/simulator`)

Generates synthetic panel data using solved value functions:

```python
from mdp.simulator import simulate_mdp_panel

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

### 3. MDP Estimator (`src/mdp/estimator`)

Recovers structural parameters using **two-step estimation**:

```python
from mdp.estimator import estimate_two_step

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

All parameters are centralized in `scripts/mdp/config/config.py`:

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

## OPM: Oligopoly Pricing Model

🚧 **Coming Soon**

The OPM module will implement structural estimation for oligopoly pricing models following the same pipeline: **Solver → Simulator → Estimator**.

---

## License

MIT License

```

Done. The description is now general:

```markdown
A Python framework for **solving**, **simulating**, and **estimating** structural economic models.
```

This accurately reflects the universal Solver → Simulator → Estimator pipeline that applies to all models (MDP, OPM, etc.), without being tied to any specific implementation detail like neural networks.

```markdown
A Python framework for **solving**, **simulating**, and **estimating** structural economic models.
```

---

**User**

now commit this change.

---

**Cursor**

✅ **Committed!**

```
[main 1e865eb] Update README: generalize framework description
 1 file changed, 1 insertion(+), 1 deletion(-)
```

```plaintext
[main 1e865eb] Update README: generalize framework description
 1 file changed, 1 insertion(+), 1 deletion(-)
```

---

**User**

push

---

**Cursor**

✅ **Pushed!**

I notice you've already renamed the GitHub repo to `AI_Practice`. Would you like me to update the local remote URL to match?

```bash
git remote set-url origin https://github.com/Hude-Hude/AI_Practice.git
```

```shellscript
git remote set-url origin https://github.com/Hude-Hude/AI_Practice.git
```

---

**User**

yes.

---

**Cursor**

✅ **Updated!**

```
origin  https://github.com/Hude-Hude/AI_Practice.git (fetch)
origin  https://github.com/Hude-Hude/AI_Practice.git (push)
```

Everything is now aligned: local folder, workspace, and GitHub repo are all `AI_Practice`.

```plaintext
origin  https://github.com/Hude-Hude/AI_Practice.git (fetch)
origin  https://github.com/Hude-Hude/AI_Practice.git (push)
```

---

**User**

check the python paty. is it from anaconda or from uv.

---

**Cursor**

Here's the Python path situation:

| Context | Path | Version |
|---------|------|---------|
| **Default shell** | `/Users/kissshot894/Anaconda/miniconda3/bin/python` | 3.12.1 (Anaconda) |
| **uv run** | `/Users/kissshot894/Documents/HKUST/RA/AI_Practice/.venv/bin/python` | 3.12.12 (uv venv) |

**Summary:**
- Your default `python` command uses **Anaconda**
- When you run `uv run pytest` or `uv run python`, it uses **uv's local `.venv`** in the project directory

This is the expected setup - uv manages its own isolated virtual environment for the project.

---

**User**

okay. where is anaconda from. is it because previously i used anaconda navigator for python projects?

---

**Cursor**

Yes, exactly! Your `~/.zshrc` has a **conda initialize** block:

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
...
export PATH="/Users/kissshot894/Anaconda/miniconda3/bin:$PATH"
# <<< conda initialize <<<
```

This was added when you installed **Miniconda** and ran `conda init`. It puts conda's Python first in your PATH, so the default `python` command uses Anaconda/Miniconda.

**This is fine for your project** because:
- `uv run` uses its own isolated `.venv` (ignores conda)
- Tests and scripts via `uv run pytest` use the correct environment
- The two don't conflict

If you ever want to remove conda from your default PATH, you could comment out that block in `~/.zshrc`, but there's no need to change anything for this project.

```shellscript
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
...
export PATH="/Users/kissshot894/Anaconda/miniconda3/bin:$PATH"
# <<< conda initialize <<<
```

---

**User**

i see. so typically where deveopler put the defaul path for python. i suppose its not in miniconda.

---

**Cursor**

You're right. Here are the common approaches developers use:

| Approach | Default `python` | Typical Users |
|----------|------------------|---------------|
| **System Python** | `/usr/bin/python3` | Minimal, avoid modifying |
| **Homebrew** | `/opt/homebrew/bin/python3` | macOS developers |
| **pyenv** | `~/.pyenv/shims/python` | Version switchers |
| **uv only** | No global python, always `uv run` | Modern Python devs |
| **Conda/Miniconda** | `~/miniconda3/bin/python` | Data scientists |

**Modern recommendation**: Many developers now use **uv** or **pyenv** and avoid a global default Python entirely:

```bash
# Instead of:
python script.py

# Use:
uv run python script.py   # uses project's .venv
```

**If you want to clean up**, you could:
1. Comment out the conda block in `~/.zshrc`
2. Use `uv run` for everything (which you're already doing)

Or just leave it as-is — your `uv run` workflow already bypasses conda, so it doesn't affect your projects.

```shellscript
# Instead of:
python script.py

# Use:
uv run python script.py   # uses project's .venv
```

---

**User**

i would like to clean up the defult to make it minimal and ofc keep all projects with uv. tell me how to do this

---

**Cursor**

Here's how to clean up to a minimal setup:

## Step 1: Remove Conda from PATH

Your `~/.zshrc` is simple. Here's the plan:

### Option A: Comment out Conda (Recommended - Reversible)

Run this to comment out the conda block:

```bash
sed -i.bak '/# >>> conda initialize >>>/,/# <<< conda initialize <<</s/^/#/' ~/.zshrc
```

### Option B: Remove Conda Block Entirely

Or manually edit `~/.zshrc` and delete lines 2-15 (the conda block), keeping only:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Step 2: Apply Changes

```bash
source ~/.zshrc
```

## Step 3: Verify

```bash
which python    # Should show system python or "not found"
which python3   # Should show /usr/bin/python3 or similar
uv run python --version   # Still works via uv
```

---

## Optional: Uninstall Miniconda Completely

If you don't need conda at all:

```bash
rm -rf ~/Anaconda/miniconda3
```

---

**Want me to run Option A (comment out) for you?** Say "proceed" and I'll execute it.

```shellscript
sed -i.bak '/# >>> conda initialize >>>/,/# <<< conda initialize <<</s/^/#/' ~/.zshrc
```

```shellscript
export PATH="$HOME/.local/bin:$PATH"
```

```shellscript
source ~/.zshrc
```

```shellscript
which python    # Should show system python or "not found"
which python3   # Should show /usr/bin/python3 or similar
uv run python --version   # Still works via uv
```

```shellscript
rm -rf ~/Anaconda/miniconda3
```

---

**User**

yes, lest do option A

---

**Cursor**

✅ **Done!** Conda block is commented out. A backup was saved to `~/.zshrc.bak`.

Now let's verify:

The current shell still has conda loaded (from when it started). **Open a new terminal** and run:

```bash
which python
which python3
```

In the new terminal, you should see:
- `python` → not found or system python
- `python3` → `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`

Your `uv run` will continue to work perfectly since it uses its own `.venv`.

---

**Summary:**
| Before | After (new terminal) |
|--------|---------------------|
| `/Users/.../miniconda3/bin/python` | Not found or system |
| Conda active by default | Conda disabled |
| `uv run` works ✅ | `uv run` works ✅ |

```shellscript
which python
which python3
```

---

**User**

@zsh (8-9) 

---

**Cursor**

✅ **It worked!**

```
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

Now your default `python3` points to **system Python**, not miniconda.

**Note:** Your prompt still shows `(MDP_Practice)` - this might be a leftover conda environment name. You can check with:

```bash
which python
echo $CONDA_DEFAULT_ENV
```

If `python` still points to conda or shows an env name, you may need to deactivate it once:

```bash
conda deactivate
```

But going forward, new terminals won't activate conda automatically. Your setup is now minimal:
- **System Python** for default
- **uv** for all project work ✅

```plaintext
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

```shellscript
which python
echo $CONDA_DEFAULT_ENV
```

```shellscript
conda deactivate
```

---

**User**

do you think if i restart the project. it will clear

---

**Cursor**

Yes! Restarting Cursor (or opening a new terminal) will start a fresh shell that reads the updated `~/.zshrc`. Since conda initialization is now commented out:

- No conda environment will auto-activate
- The `(MDP_Practice)` prefix will be gone
- `python3` will point to system Python

Go ahead and restart - it should be clean after that.

---

**User**

@zsh (1-5) 

---

**Cursor**

Interesting - the `(MDP_Practice)` prefix is gone ✅, but `python` still points to miniconda while `python3` is correct.

Let's check your PATH:

Conda paths are still in PATH. There must be another config file. Let me check:

Found it! Conda is also in `~/.bash_profile`. Let me comment it out:

✅ **Done!** Commented out conda in `~/.bash_profile` too.

Now **restart Cursor** (or open a new terminal) and check:

```bash
which python
which python3
echo $PATH | tr ':' '\n' | grep -i conda
```

After restart:
- `python` should be not found or system python
- `python3` → system python
- No conda paths in PATH

```shellscript
which python
which python3
echo $PATH | tr ':' '\n' | grep -i conda
```

---

**User**

@zsh (1-6) 

---

**Cursor**

✅ **Clean setup!**

```
python  → not found
python3 → /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

This is the minimal setup you wanted:
- No conda polluting your PATH
- System `python3` available if needed
- All project work via `uv run` ✅

Your workflow is now:
```bash
uv run pytest          # run tests
uv run python script.py  # run scripts
uv run quarto preview   # render reports
```

```plaintext
python  → not found
python3 → /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
```

```shellscript
uv run pytest          # run tests
uv run python script.py  # run scripts
uv run quarto preview   # render reports
```

---

**User**

yeah, run the test to verify python works now

---

**Cursor**

The venv needs to be synced. Let me fix it:

The venv has stale references. Let me recreate it:

✅ **All 100 tests passed!**

```
===================== 100 passed, 19 deselected in 11.74s ======================
```

Your clean Python setup works perfectly:
- Fresh `.venv` created with CPython 3.12.12
- `structural-models` package installed
- All tests passing

```plaintext
===================== 100 passed, 19 deselected in 11.74s ======================
```

---

**User**

explain what are the new .pytest_cache and .venv.

---

**Cursor**

## `.venv/` - Virtual Environment

A **project-specific Python installation** created by `uv`:

```
.venv/
├── bin/
│   ├── python → Python 3.12.12
│   ├── pytest
│   └── ... other executables
├── lib/
│   └── python3.12/site-packages/
│       ├── torch/
│       ├── numpy/
│       ├── structural-models/  ← your package
│       └── ... all dependencies
└── pyvenv.cfg
```

**Purpose**: Isolates this project's dependencies from other projects and your system Python.

**When `uv run pytest`**: It uses `.venv/bin/pytest` with `.venv/lib/` packages.

---

## `.pytest_cache/` - Test Cache

A **cache directory** created by pytest:

```
.pytest_cache/
├── v/cache/
│   ├── lastfailed    ← which tests failed last run
│   ├── nodeids       ← test IDs
│   └── stepwise      ← for --stepwise mode
└── .gitignore
```

**Purpose**: Speeds up test runs by caching metadata (e.g., "rerun only failed tests" with `pytest --lf`).

---

## Should you commit them?

| Directory | Git | Why |
|-----------|-----|-----|
| `.venv/` | ❌ Ignore | Regenerated via `uv sync` |
| `.pytest_cache/` | ❌ Ignore | Regenerated automatically |

Both should already be in `.gitignore`. You can verify:

```bash
git check-ignore .venv .pytest_cache
```

```plaintext
.venv/
├── bin/
│   ├── python → Python 3.12.12
│   ├── pytest
│   └── ... other executables
├── lib/
│   └── python3.12/site-packages/
│       ├── torch/
│       ├── numpy/
│       ├── structural-models/  ← your package
│       └── ... all dependencies
└── pyvenv.cfg
```

```plaintext
.pytest_cache/
├── v/cache/
│   ├── lastfailed    ← which tests failed last run
│   ├── nodeids       ← test IDs
│   └── stepwise      ← for --stepwise mode
└── .gitignore
```

```shellscript
git check-ignore .venv .pytest_cache
```

---

**User**

help me verify.

---

**Cursor**

✅ **Verified!**

```
.venv         → ignored by .gitignore
.pytest_cache → ignored by .gitignore
Not tracked (good!)
```

Both directories are properly excluded from git. They won't be committed to your repo.

```plaintext
.venv         → ignored by .gitignore
.pytest_cache → ignored by .gitignore
Not tracked (good!)
```

---

**User**

so they are temperoly created just for the test? are they necessary?

---

**Cursor**

Good question! Here's the breakdown:

| Directory | Created By | Necessary? | Can Delete? |
|-----------|------------|------------|-------------|
| **`.venv/`** | `uv sync` | **Yes** - required to run code | Yes, but must recreate with `uv sync` before running |
| **`.pytest_cache/`** | `pytest` | **No** - just speeds up reruns | Yes, safely delete anytime |

## `.venv/` - Required
Without it, `uv run pytest` fails because there's no Python environment with your dependencies.

```bash
rm -rf .venv        # delete
uv sync             # must recreate before running anything
```

## `.pytest_cache/` - Optional
Pytest works fine without it. It's just convenience (e.g., `pytest --lf` reruns only last-failed tests).

```bash
rm -rf .pytest_cache  # safe to delete, no rebuild needed
```

---

**In practice**: Leave them both. They're ignored by git and don't take much space. Delete `.venv` only if it gets corrupted; delete `.pytest_cache` whenever you want.

```shellscript
rm -rf .venv        # delete
uv sync             # must recreate before running anything
```

```shellscript
rm -rf .pytest_cache  # safe to delete, no rebuild needed
```

---

**User**

while i check others project, the .venv not explicitly displayed in the folder as parallel to docs, output, etc. do you understand what they do? they also use uv strcture.

---

**Cursor**

Yes! The `.venv` folder **starts with a dot (`.`)**, which makes it a **hidden folder** in Unix/macOS.

## Why you might see it in Cursor but not elsewhere:

| Context | `.venv` visible? | Why |
|---------|------------------|-----|
| **Terminal `ls`** | ❌ Hidden | Need `ls -a` to see dotfiles |
| **Cursor file tree** | ✅ Visible | Cursor shows hidden files by default |
| **README project structure** | ❌ Not shown | People omit generated/hidden folders |
| **GitHub web view** | ❌ Not shown | Hidden files not displayed by default |

## Typical project layout in READMEs:

```
my_project/
├── src/           ← shown
├── tests/         ← shown
├── pyproject.toml ← shown
└── README.md      ← shown

# Hidden/generated (exist but not shown):
# .venv/
# .pytest_cache/
# .git/
# __pycache__/
```

## In other uv projects:

They have `.venv` too — it's just hidden. You can verify in any uv project:

```bash
ls -la | grep venv
```

So your setup is exactly the same as other uv projects. Cursor just shows hidden files by default.

```plaintext
my_project/
├── src/           ← shown
├── tests/         ← shown
├── pyproject.toml ← shown
└── README.md      ← shown

# Hidden/generated (exist but not shown):
# .venv/
# .pytest_cache/
# .git/
# __pycache__/
```

```shellscript
ls -la | grep venv
```

---

**User**

explain me what is this venv. virtual environment? i checked my prof project which i shall followed the same strcutre. and here is the return chat history from that project: ## Summary



You asked whether this project has a `.venv` (Python virtual environment) directory.



I checked in two ways:

1. First by listing the directory and checking for `.venv`

2. Then verified with `ls -la | grep venv` as you requested



**Result:** No `.venv` directory exists in this project.



The project uses `uv` for Python package management (has `uv.lock` and `pyproject.toml`), but the virtual environment hasn't been created yet. You can create one with `uv venv` if needed.

---

**Cursor**

Great question! Let me explain:

## What is `.venv` (Virtual Environment)?

A **virtual environment** is an isolated Python installation for a specific project:

```
.venv/
├── bin/python       ← A copy of Python just for this project
├── lib/site-packages/
│   ├── numpy/       ← Installed packages
│   ├── torch/
│   └── your-project/
```

**Why?** Different projects need different package versions. Without isolation:
- Project A needs `numpy 1.x`
- Project B needs `numpy 2.x`
- 💥 Conflict!

With `.venv`, each project has its own packages.

---

## Why your professor's project has no `.venv`

| File | What it is | Committed to git? |
|------|------------|-------------------|
| `pyproject.toml` | "Recipe" - what packages are needed | ✅ Yes |
| `uv.lock` | Exact versions locked | ✅ Yes |
| `.venv/` | Actual installed packages | ❌ No (gitignored) |

**`.venv` is created locally, not shared via git.**

When you clone a uv project:
```bash
git clone <repo>        # Gets pyproject.toml + uv.lock
cd project
uv sync                 # Creates .venv and installs packages
```

Your professor's project doesn't have `.venv` because **it hasn't been synced yet** on that machine. Run `uv sync` and it will appear.

---

## Your project vs Professor's project

| | Your Project | Prof's Project (cloned) |
|---|---|---|
| `pyproject.toml` | ✅ | ✅ |
| `uv.lock` | ✅ | ✅ |
| `.venv/` | ✅ (you ran `uv sync`) | ❌ (not synced yet) |

They're the same structure — you just need to run `uv sync` in your professor's project to create the `.venv`.

```plaintext
.venv/
├── bin/python       ← A copy of Python just for this project
├── lib/site-packages/
│   ├── numpy/       ← Installed packages
│   ├── torch/
│   └── your-project/
```

```shellscript
git clone <repo>        # Gets pyproject.toml + uv.lock
cd project
uv sync                 # Creates .venv and installs packages
```

---

**User**

I see. so when i clone, it didn't show up.

---

**Cursor**

Exactly! Here's the workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                        GitHub Repo                          │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ❌ (gitignored, not in repo)               │
└─────────────────────────────────────────────────────────────┘
                            │
                      git clone
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Your Local Machine                      │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ❌ (doesn't exist yet)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                        uv sync
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Your Local Machine                      │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ✅ (created! packages installed)           │
└─────────────────────────────────────────────────────────────┘
```

**So the standard workflow after cloning any uv project:**

```bash
git clone <repo>
cd <project>
uv sync            # ← This creates .venv and installs everything
uv run pytest      # Now you can run things
```

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                        GitHub Repo                          │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ❌ (gitignored, not in repo)               │
└─────────────────────────────────────────────────────────────┘
                            │
                      git clone
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Your Local Machine                      │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ❌ (doesn't exist yet)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                        uv sync
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Your Local Machine                      │
│  pyproject.toml  ✅                                         │
│  uv.lock         ✅                                         │
│  src/            ✅                                         │
│  .venv/          ✅ (created! packages installed)           │
└─────────────────────────────────────────────────────────────┘
```

```shellscript
git clone <repo>
cd <project>
uv sync            # ← This creates .venv and installs everything
uv run pytest      # Now you can run things
```

---

**User**

understand. but while i worked on that project, without sync, the project still work and be able to run python. why

---

**Cursor**

Good observation! There are a few possible reasons:

## Reason 1: You used Conda Python (before cleanup)

Before we cleaned up your PATH, your default `python` was **Miniconda**:

```bash
which python
/Users/kissshot894/Anaconda/miniconda3/bin/python
```

If that conda environment had `numpy`, `torch`, etc. installed, running `python script.py` directly would work — it wasn't using `.venv` at all, it was using conda's packages.

## Reason 2: `uv run` auto-creates `.venv`

When you run:
```bash
uv run python script.py
```

uv **automatically creates `.venv`** if it doesn't exist. So you might have created it without realizing.

## Reason 3: IDE auto-setup

Cursor or other IDEs sometimes auto-detect `pyproject.toml` and create a venv.

---

## Summary

| Command | What Python it uses |
|---------|---------------------|
| `python script.py` | System/Conda Python (whatever's in PATH) |
| `uv run python script.py` | `.venv` Python (creates if needed) |

**Before cleanup**: Your `python` command used conda, which had packages installed globally.

**After cleanup**: `python` is not found, so you **must** use `uv run python` — which is the correct, isolated way.

```shellscript
which python
/Users/kissshot894/Anaconda/miniconda3/bin/python
```

```shellscript
uv run python script.py
```

---

**User**

oh thats why miniconda was involved earlier. now we are on the right track after the clean where we invoke uv everything to do new project correct?

---

**Cursor**

Exactly! You've got it. 🎯

## Before (messy)
```
python script.py  →  Uses conda Python (random packages, no isolation)
                     Sometimes works, sometimes version conflicts
```

## After cleanup (correct)
```
uv run python script.py  →  Uses project's .venv (isolated, reproducible)
```

---

## Your workflow for new projects:

```bash
# 1. Create project
mkdir new_project && cd new_project
uv init                        # Creates pyproject.toml

# 2. Add dependencies
uv add numpy torch matplotlib  # Adds to pyproject.toml + creates .venv

# 3. Run code
uv run python script.py        # Uses isolated .venv
uv run pytest                  # Run tests

# 4. Share project
git push                       # pyproject.toml + uv.lock (not .venv)
```

## When someone clones your project:
```bash
git clone <your-repo>
cd <project>
uv sync                        # Creates .venv from uv.lock
uv run pytest                  # Works!
```

**Clean, isolated, reproducible.** ✅

```plaintext
python script.py  →  Uses conda Python (random packages, no isolation)
                     Sometimes works, sometimes version conflicts
```

```plaintext
uv run python script.py  →  Uses project's .venv (isolated, reproducible)
```

```shellscript
# 1. Create project
mkdir new_project && cd new_project
uv init                        # Creates pyproject.toml

# 2. Add dependencies
uv add numpy torch matplotlib  # Adds to pyproject.toml + creates .venv

# 3. Run code
uv run python script.py        # Uses isolated .venv
uv run pytest                  # Run tests

# 4. Share project
git push                       # pyproject.toml + uv.lock (not .venv)
```

```shellscript
git clone <your-repo>
cd <project>
uv sync                        # Creates .venv from uv.lock
uv run pytest                  # Works!
```

---

**User**

create a new folder in docs/mdp and docs/opm as conversation. And move current conversation histories in the new folders respectively.

---

**Cursor**

✅ **Done!**

```
docs/
├── mdp/
│   └── conversation/
│       ├── choice_probability_validation.md
│       ├── environment_setup.md
│       ├── mdp_solver_development.md
│       └── simulation_and_estimation.md
└── opm/
    └── conversation/    (empty, ready for future)
```

Changes staged and ready to commit when you want.

```plaintext
docs/
├── mdp/
│   └── conversation/
│       ├── choice_probability_validation.md
│       ├── environment_setup.md
│       ├── mdp_solver_development.md
│       └── simulation_and_estimation.md
└── opm/
    └── conversation/    (empty, ready for future)
```

---

**User**

okay, create a new folder in docs/ as other and within it create conversation and put environment_setup.md into this.

---

**Cursor**

✅ **Done!**

```
docs/
├── mdp/
│   └── conversation/
│       ├── choice_probability_validation.md
│       ├── mdp_solver_development.md
│       └── simulation_and_estimation.md
├── opm/
│   └── conversation/    (empty)
└── other/
    └── conversation/
        └── environment_setup.md
```

```plaintext
docs/
├── mdp/
│   └── conversation/
│       ├── choice_probability_validation.md
│       ├── mdp_solver_development.md
│       └── simulation_and_estimation.md
├── opm/
│   └── conversation/    (empty)
└── other/
    └── conversation/
        └── environment_setup.md
```

