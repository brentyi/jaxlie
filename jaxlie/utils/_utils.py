from typing import TYPE_CHECKING, Callable, Type, TypeVar

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from .._base import MatrixLieGroup


T = TypeVar("T", bound="MatrixLieGroup")


def get_epsilon(dtype: jnp.dtype) -> float:
    """Helper for grabbing type-specific precision constants.

    Args:
        dtype: Datatype.

    Returns:
        Output float.
    """
    return {
        jnp.dtype("float32"): 1e-5,
        jnp.dtype("float64"): 1e-10,
    }[dtype]


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group dataclasses.

    Sets dimensionality class variables, and marks all methods for JIT compilation.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        # JIT all methods.
        for f in filter(
            lambda f: not f.startswith("_")
            and callable(getattr(cls, f))
            and f != "get_batch_axes",  # Avoid returning tracers.
            dir(cls),
        ):
            setattr(cls, f, jax.jit(getattr(cls, f)))

        return cls

    return _wrap
