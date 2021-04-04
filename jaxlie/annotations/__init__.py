from typing import NamedTuple, Union

import numpy as onp
from jax import numpy as jnp

Array = Union[onp.ndarray, jnp.ndarray]
"""Type alias for `Union[jnp.ndarray, onp.ndarray]`.
"""

Scalar = Union[float, Array]
"""Type alias for `Union[float, Array]`.
"""

Matrix = Array
"""Type alias for `Array`. Should not be instantiated.

Refers to a square matrix, typically with shape `(Group.matrix_dim, Group.matrix_dim)`.
For adjoints, shape should be `(Group.tangent_dim, Group.tangent_dim)`.
"""

Vector = Array
"""Type alias for `Array`. Should not be instantiated.

Refers to a general 1D array.
"""

TangentVector = Array
"""Type alias for `Array`. Should not be instantiated.

Refers to a 1D array with shape `(Group.tangent_dim,)`.
"""


class RollPitchYaw(NamedTuple):
    """Tuple containing roll, pitch, and yaw Euler angles."""

    roll: Scalar
    pitch: Scalar
    yaw: Scalar


__all__ = [
    "Array",
    "Scalar",
    "Matrix",
    "Vector",
    "TangentVector",
    "RollPitchYaw",
]
