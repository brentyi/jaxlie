from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Union, overload

import jax
from jax import numpy as jnp
from typing_extensions import ParamSpec

from .._base import MatrixLieGroup
from . import _deltas, _tree_utils


def zero_tangents(pytree: Any) -> _tree_utils.TangentPytree:
    """Replace all values in a Pytree with zero vectors on the corresponding tangent
    spaces."""

    def tangent_zero(t: MatrixLieGroup) -> jax.Array:
        return jnp.zeros(t.get_batch_axes() + (t.tangent_dim,))

    return _tree_utils._map_group_trees(
        tangent_zero,
        lambda array: jnp.zeros_like(array),
        pytree,
    )


AxisName = Any

P = ParamSpec("P")


@overload
def grad(
    fun: Callable[P, Any],
    argnums: int = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, _tree_utils.TangentPytree]:
    ...


@overload
def grad(
    fun: Callable[P, Any],
    argnums: Sequence[int],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, Tuple[_tree_utils.TangentPytree, ...]]:
    ...


def grad(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
):
    """Same as `jax.grad`, but computes gradients of Lie groups with respect to
    tangent spaces."""

    compute_value_and_grad = value_and_grad(
        fun=fun,
        argnums=argnums,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
    )
    return lambda *args, **kwargs: compute_value_and_grad(*args, **kwargs)[1]  # type: ignore


@overload
def value_and_grad(
    fun: Callable[P, Any],
    argnums: int = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, Tuple[Any, _tree_utils.TangentPytree]]:
    ...


@overload
def value_and_grad(
    fun: Callable[P, Any],
    argnums: Sequence[int],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[P, Tuple[Any, Tuple[_tree_utils.TangentPytree, ...]]]:
    ...


def value_and_grad(
    fun: Callable[P, Any],
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
):
    """Same as `jax.value_and_grad`, but computes gradients of Lie groups with respect to
    tangent spaces."""

    def wrapped_grad(*args, **kwargs):
        def tangent_fun(*tangent_args, **tangent_kwargs):
            return fun(  # type: ignore
                *_deltas.rplus(args, tangent_args),
                **_deltas.rplus(kwargs, tangent_kwargs),
            )

        # Put arguments onto tangent space.
        tangent_args = map(zero_tangents, args)
        tangent_kwargs = {k: zero_tangents(v) for k, v in kwargs.items()}

        value, grad = jax.value_and_grad(
            fun=tangent_fun,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )(*tangent_args, **tangent_kwargs)
        return value, grad

    return wrapped_grad  # type: ignore
