import dataclasses
from typing import TYPE_CHECKING, Tuple

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._base import MatrixLieGroup


def get_epsilon(x: jnp.ndarray) -> float:
    if x.dtype is jnp.dtype("float32"):
        return 1e-5
    elif x.dtype is jnp.dtype("float64"):
        return 1e-10
    else:
        assert False, f"Unexpected array type: {x.dtype}"


def register_lie_group(*, matrix_dim: int, parameters_dim: int, tangent_dim: int):
    """Decorator for defining immutable dataclasses."""

    def _wrap(cls):
        # Hash based on object ID, rather than contents (arrays are not hashable)
        cls.__hash__ = object.__hash__

        # Register dimensions as class attributes
        # This needs to happen after dataclass processing
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim

        # Register as a PyTree node to make JIT compilation, etc easier in Jax
        def _flatten_group(
            v: "MatrixLieGroup",
        ) -> Tuple[Tuple[jnp.ndarray, ...], Tuple]:
            """Flatten a dataclass for use as a PyTree."""
            as_dict = dataclasses.asdict(v)

            array_data = {
                k: v for k, v in as_dict.items() if isinstance(v, jnp.ndarray)
            }
            aux_dict = {k: v for k, v in as_dict.items() if k not in array_data}

            return (
                tuple(array_data.values()),
                tuple(array_data.keys())
                + tuple(aux_dict.keys())
                + tuple(aux_dict.values()),
            )

        def _unflatten_group(
            treedef: Tuple, children: Tuple[jnp.ndarray, ...]
        ) -> "MatrixLieGroup":
            """Unflatten a dataclass for use as a PyTree."""
            array_keys = treedef[: len(children)]
            aux = treedef[len(children) :]
            aux_keys = aux[: len(aux) // 2]
            aux_values = aux[len(aux) // 2 :]

            return cls(
                **dict(zip(array_keys, children)),
                **dict(zip(aux_keys, aux_values)),
            )

        jax.tree_util.register_pytree_node(cls, _flatten_group, _unflatten_group)

        return cls

    return _wrap
