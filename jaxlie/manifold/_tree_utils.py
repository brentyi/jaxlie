from typing import Any, Callable, List, TypeVar

import jax
import numpy as onp
from jax._src.tree_util import _registry  # Dangerous!

from .._base import MatrixLieGroup

# Tangent structures are difficult to annotate, so we just mark everything via Any.
#
# An annotation that would work in most cases is:
#
#     def zero_tangents(structure: T) -> T
#
# But this is leaky; note that an input of List[SE3] should output List[jax.Array],
# Dict[str, SE3] should output Dict[str, SE3], etc.
#
# Another tempting option is to define a wrapper class:
#
#     @jdc.pytree_dataclass
#     class TangentPytree(Generic[PytreeType]):
#         wrapped: Any
#
# And have zero_tangents() return:
#
#     def zero_tangents(structure: T) -> TangentPytree[T]
#
# which we could also use to make `jaxlie.manifold.rplus()` type safe by adding
# overloads to make sure that the delta input is a TangentPytree, but it would be hard
# to accurately annotate the `grad()` and `value_and_grad()` functions with this wrapper
# type without sacrificing the ability to use them as drop-in replacements for
# `jax.grad()` and `jax.value_and_grad()`.
#
# Finally, NewType is also attractive:
#
#     TangentPytree: TypeAlias = NewType("TangentPytree", object)
#
# This seems reasonable, but doesn't play nice with how optax currently (a) annotates
# everything using chex.ArrayTree and (b) doesn't use any generics, leading to a mess of
# casts and `type: ignore` directives. We might consider using this if optax's gradient
# transform annotations change.
TangentPytree = Any


def _map_group_trees(
    f_lie_groups: Callable,
    f_other_arrays: Callable,
    *tree_args,
) -> Any:
    if isinstance(tree_args[0], MatrixLieGroup):
        return f_lie_groups(*tree_args)
    elif isinstance(tree_args[0], (jax.Array, onp.ndarray)):
        return f_other_arrays(*tree_args)
    else:
        # Handle PyTrees recursively.
        assert len(set(map(type, tree_args))) == 1
        registry_entry = _registry[type(tree_args[0])]  # type: ignore

        children: List[List[Any]] = []
        metadata: List[Any] = []
        for tree in tree_args:
            childs, meta = registry_entry.to_iter(tree)
            children.append(childs)
            metadata.append(meta)

        assert len(set(metadata)) == 1

        return registry_entry.from_iter(
            metadata[0],
            [
                _map_group_trees(
                    f_lie_groups,
                    f_other_arrays,
                    *list(children[i][j] for i in range(len(children))),
                )
                for j in range(len(children[0]))
            ],
        )


PytreeType = TypeVar("PytreeType")


def normalize_all(pytree: PytreeType) -> PytreeType:
    """Call `.normalize()` on each Lie group instance in a pytree.

    Results in a naive projection of each group instance to its respective manifold.
    """

    def _project(t: MatrixLieGroup) -> MatrixLieGroup:
        return t.normalize()

    return _map_group_trees(
        _project,
        lambda x: x,
        pytree,
    )
