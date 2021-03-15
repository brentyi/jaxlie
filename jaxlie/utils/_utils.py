import dataclasses
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import flax
import jax
from jax import numpy as jnp

from .. import types

if TYPE_CHECKING:
    from .._base import MatrixLieGroup


T = TypeVar("T", bound="MatrixLieGroup")


def get_epsilon(dtype: jnp.dtype) -> float:
    """Helper for grabbing type-specific precision constants."""
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
    - Sets static dimensionality attributes
    - Makes the group hashable
    - Marks all functions for JIT compilation
    - Adds flattening/unflattening ops for use as a PyTree node
    - Adds serialization ops for `flax.serialization`

    Note that a significant amount of functionality here could be replaced by
    `flax.struct`, but `flax.struct` doesn't work very well with jedi or mypy.

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
            v: T,
        ) -> Tuple[Tuple[types.Vector, ...], Tuple[Hashable, ...]]:
            """Flatten a dataclass for use as a PyTree."""
            as_dict = vars(v)
            return tuple(as_dict.values()), tuple(as_dict.keys())

        def _unflatten_group(treedef: Any, children: Sequence[Any]) -> T:
            """Unflatten a dataclass for use as a PyTree."""
            # Treedef is names of fields
            return cls(**dict(zip(treedef, children)))  # type: ignore

        jax.tree_util.register_pytree_node(cls, _flatten_group, _unflatten_group)

        # Make object flax-serializable
        def _ty_to_state_dict(x: "MatrixLieGroup") -> Dict[str, types.Array]:
            return {
                key: flax.serialization.to_state_dict(value)
                for key, value in vars(x).items()
            }

        def _ty_from_state_dict(x: "MatrixLieGroup", state: Dict) -> "MatrixLieGroup":
            updates: Dict[str, Any] = {}
            for key, value in vars(x).items():
                updates[key] = flax.serialization.from_state_dict(
                    getattr(x, key), value
                )
            return dataclasses.replace(x, **updates)

        flax.serialization.register_serialization_state(
            ty=cls,
            ty_to_state_dict=_ty_to_state_dict,
            ty_from_state_dict=_ty_from_state_dict,
        )

        return cls

    return _wrap
