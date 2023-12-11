from typing import Any, NamedTuple, Union

import jax
import numpy as onp

# Type aliases for JAX/Numpy arrays; primarily for function inputs.

Array = Union[onp.ndarray, jax.Array]
"""Type alias for `Union[jax.Array, onp.ndarray]`."""

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
