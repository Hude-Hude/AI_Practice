"""Type definitions for the MDP solver."""

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Scalar types
Scalar: TypeAlias = float

# Array types
Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]

# Domain-specific types
StateGrid: TypeAlias = Vector
Action: TypeAlias = Literal[0, 1]

