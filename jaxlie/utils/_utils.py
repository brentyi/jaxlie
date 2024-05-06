from typing import TYPE_CHECKING, Callable, Tuple, Type, TypeVar, Union, cast

import jax_dataclasses as jdc
from jax import numpy as jnp

from jaxlie.hints import Array

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
            setattr(cls, f, jdc.jit(getattr(cls, f)))

        return cls

    return _wrap


TupleOfBroadcastable = TypeVar(
    "TupleOfBroadcastable",
    bound="Tuple[Union[MatrixLieGroup, Array], ...]",
)


def broadcast_leading_axes(inputs: TupleOfBroadcastable) -> TupleOfBroadcastable:
    """Broadcast leading axes of arrays. Takes tuples of either:
    - an array, which we assume has shape (*, D).
    - a Lie group object."""

    from .._base import MatrixLieGroup

    array_inputs = [
        (
            (x.parameters(), (x.parameters_dim,))
            if isinstance(x, MatrixLieGroup)
            else (x, x.shape[-1:])
        )
        for x in inputs
    ]
    for array, shape_suffix in array_inputs:
        assert array.shape[-len(shape_suffix) :] == shape_suffix
    batch_axes = jnp.broadcast_shapes(
        *[array.shape[: -len(suffix)] for array, suffix in array_inputs]
    )
    broadcasted_arrays = tuple(
        jnp.broadcast_to(array, batch_axes + shape_suffix)
        for (array, shape_suffix) in array_inputs
    )
    return cast(
        TupleOfBroadcastable,
        tuple(
            array if not isinstance(inp, MatrixLieGroup) else type(inp)(array)
            for array, inp in zip(broadcasted_arrays, inputs)
        ),
    )
