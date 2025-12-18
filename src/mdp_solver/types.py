"""Type definitions for the MDP solver.

Based on pseudo code type definitions:
- Scalar = Float
- Vector[n] = Array[Float, n]
- Matrix[m,n] = Array[Float, m, n]
- Action = Int ∈ {0, 1}
- Network = MonotonicNeuralNetwork
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Scalar type
Scalar = float

# Array types (numpy)
Vector = np.ndarray  # 1D array
Matrix = np.ndarray  # 2D array

# Tensor types (torch)
TensorScalar = torch.Tensor
TensorVector = torch.Tensor
TensorMatrix = torch.Tensor

# Action type
Action = int  # 0 or 1

# Network type (will be defined in network.py)
Network = nn.Module

# Type aliases for function signatures
HiddenSizes = List[int]
NetworkPair = Tuple[Network, Network]
TargetPair = Tuple[TensorVector, TensorVector]

