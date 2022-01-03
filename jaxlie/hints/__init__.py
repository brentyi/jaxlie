from typing import NamedTuple, Union

import numpy as onp
from jax import numpy as jnp

# Type aliases for JAX/Numpy arrays; primarily for function inputs.

Array = Union[onp.ndarray, jnp.ndarray]
"""Type alias for `Union[jnp.ndarray, onp.ndarray]`."""

Scalar = Union[float, Array]
"""Type alias for `Union[float, Array]`."""


class RollPitchYaw(NamedTuple):
    """Tuple containing roll, pitch, and yaw Euler angles."""

    roll: Scalar
    pitch: Scalar
    yaw: Scalar


__all__ = [
    "Array",
    "Scalar",
    "RollPitchYaw",
]
