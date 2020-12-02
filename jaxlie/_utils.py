import dataclasses
from typing import TYPE_CHECKING, Callable, Tuple, TypeVar

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._base import MatrixLieGroup


def get_epsilon(dtype: jnp.dtype) -> float:
    if dtype is jnp.dtype("float32"):
        return 1e-5
    elif dtype is jnp.dtype("float64"):
        return 1e-10
    else:
        assert False, f"Unexpected array type: {dtype}"


T = TypeVar("T", bound="MatrixLieGroup")


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[T], T]:
    """Process a Lie group dataclass:
    - Sets static dimensionality attributes
    - Makes the group hashable
    - Marks all functions for JIT compilation
    - Adds flattening/unflattening ops for use as a PyTree node
    """

    def _wrap(cls: "MatrixLieGroup"):
        # Register dimensions as class attributes
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        # Hash based on object ID, rather than contents (arrays are not hashable)
        cls.__hash__ = object.__hash__

        # JIT for all functions
        for f in filter(
            lambda f: not f.startswith("_") and callable(getattr(cls, f)),
            dir(cls),
        ):
            setattr(cls, f, jax.jit(getattr(cls, f)))

        # Register as a PyTree node to make JIT compilation, etc easier in Jax
        def _flatten_group(
            v: "MatrixLieGroup",
        ) -> Tuple[Tuple[jnp.ndarray, ...], Tuple]:
            """Flatten a dataclass for use as a PyTree."""
            as_dict = dataclasses.asdict(v)
            return tuple(as_dict.values()), tuple(as_dict.keys())

        def _unflatten_group(
            treedef: Tuple, children: Tuple[jnp.ndarray, ...]
        ) -> "MatrixLieGroup":
            """Unflatten a dataclass for use as a PyTree."""
            # Treedef is names of fields
            return cls(**dict(zip(treedef, children)))

        jax.tree_util.register_pytree_node(cls, _flatten_group, _unflatten_group)

        return cls

    return _wrap
