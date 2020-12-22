from typing import TYPE_CHECKING, Callable, Tuple, Type, TypeVar

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from ._base import MatrixLieGroup


def get_epsilon(dtype: jnp.dtype) -> float:
    return {
        jnp.dtype("float32"): 1e-5,
        jnp.dtype("float64"): 1e-10,
    }[dtype]


T = TypeVar("T", bound="MatrixLieGroup")


def register_lie_group(
    *,
    matrix_dim: int,
    parameters_dim: int,
    tangent_dim: int,
    space_dim: int,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group dataclasses.
    - Sets static dimensionality attributes
    - Makes the group hashable
    - Marks all functions for JIT compilation
    - Adds flattening/unflattening ops for use as a PyTree node

    Example:
    ```
    @register_lie_group(
        matrix_dim=2,
        parameters_dim=2,
        tangent_dim=1,
        space_dim=2,
    )
    @dataclasses.dataclass(frozen=True)
    class SO2(_base.MatrixLieGroup):
        ...
    ```
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes
        cls.matrix_dim = matrix_dim
        cls.parameters_dim = parameters_dim
        cls.tangent_dim = tangent_dim
        cls.space_dim = space_dim

        # Hash based on object ID, rather than contents (arrays are not hashable)
        setattr(cls, "__hash__", object.__hash__)

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
            as_dict = vars(v)
            return tuple(as_dict.values()), tuple(as_dict.keys())

        def _unflatten_group(
            treedef: Tuple, children: Tuple[jnp.ndarray, ...]
        ) -> "MatrixLieGroup":
            """Unflatten a dataclass for use as a PyTree."""
            # Treedef is names of fields
            return cls(**dict(zip(treedef, children)))  # type: ignore

        jax.tree_util.register_pytree_node(cls, _flatten_group, _unflatten_group)

        return cls

    return _wrap
