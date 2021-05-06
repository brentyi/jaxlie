from typing import NamedTuple, Union

import numpy as onp
from jax import numpy as jnp

# Type aliases for JAX/Numpy arrays; primarily for function inputs

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

# Type aliases for JAX arrays; primarily for function outputs

ArrayJax = jnp.ndarray
"""Type alias for jnp.ndarray."""

ScalarJax = ArrayJax
"""Type alias for jnp.ndarray."""

MatrixJax = ArrayJax
"""Type alias for jnp.ndarray."""

VectorJax = ArrayJax
"""Type alias for jnp.ndarray."""

TangentVectorJax = ArrayJax
"""Type alias for jnp.ndarray."""


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
    "ArrayJax",
    "ScalarJax",
    "MatrixJax",
    "VectorJax",
    "TangentVectorJax",
    "RollPitchYaw",
]
