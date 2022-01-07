from typing import Any, NamedTuple, Union

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


try:
    # This is only exposed in `jax>=0.2.21`.
    from jax.random import KeyArray
except ImportError:
    KeyArray = Any  # type: ignore
    """Backward-compatible alias for `jax.random.KeyArray`."""


__all__ = [
    "Array",
    "Scalar",
    "RollPitchYaw",
    "KeyArray",
]
