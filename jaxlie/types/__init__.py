from jax import numpy as jnp

Matrix = jnp.ndarray
"""Type alias for `jnp.ndarray`. Should not be instantiated.

Refers to a square matrix, typically with shape `(Group.matrix_dim, Group.matrix_dim)`.
For adjoints, shape should be `(Group.tangent_dim, Group.tangent_dim)`.
"""


Vector = jnp.ndarray
"""Type alias for `jnp.ndarray`. Should not be instantiated.

Refers to a general 1D array.
"""


TangentVector = jnp.ndarray
"""Type alias for `jnp.ndarray`. Should not be instantiated.

Refers to a 1D array with shape `(Group.tangent_dim,)`.
"""


__all__ = [
    "Matrix",
    "Vector",
    "TangentVector",
]
