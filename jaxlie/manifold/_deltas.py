"""Helpers for recursively applying tangent-space deltas."""

from typing import Any, Callable, TypeVar, Union, cast, overload

import jax
import numpy as onp
from jax import numpy as jnp

from .. import hints
from .._base import MatrixLieGroup
from .._se2 import SE2
from .._se3 import SE3
from .._so2 import SO2
from .._so3 import SO3
from . import _tree_utils

PytreeType = TypeVar("PytreeType")
GroupType = TypeVar("GroupType", bound=MatrixLieGroup)
CallableType = TypeVar("CallableType", bound=Callable)


def _naive_auto_vmap(f: CallableType) -> CallableType:
    def inner(*args, **kwargs):
        batch_axes = None
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, MatrixLieGroup):
                if batch_axes is None:
                    batch_axes = arg.get_batch_axes()
                else:
                    assert arg.get_batch_axes() == batch_axes
        assert batch_axes is not None

        f_vmapped: Callable = f
        for i in range(len(batch_axes)):
            f_vmapped = jax.vmap(f_vmapped)
        return f_vmapped(*args, **kwargs)

    return inner  # type: ignore


@_naive_auto_vmap
def _rplus(transform: GroupType, delta: jax.Array) -> GroupType:
    assert isinstance(transform, MatrixLieGroup)
    assert isinstance(delta, (jax.Array, onp.ndarray))
    return transform @ type(transform).exp(delta)


@overload
def rplus(
    transform: GroupType,
    delta: hints.Array,
) -> GroupType:
    ...


@overload
def rplus(
    transform: PytreeType,
    delta: _tree_utils.TangentPytree,
) -> PytreeType:
    ...


# Using our typevars in the overloaded signature will cause errors.
def rplus(
    transform: Union[MatrixLieGroup, Any],
    delta: Union[hints.Array, Any],
) -> Union[MatrixLieGroup, Any]:
    """Manifold right plus. Computes `T' = T @ exp(delta)`.

    Supports pytrees containing Lie group instances recursively; simple Euclidean
    addition will be performed for all other arrays.
    """
    return _tree_utils._map_group_trees(_rplus, jnp.add, transform, delta)


@_naive_auto_vmap
def _rminus(a: GroupType, b: GroupType) -> jax.Array:
    assert isinstance(a, MatrixLieGroup) and isinstance(b, MatrixLieGroup)
    return (a.inverse() @ b).log()


@overload
def rminus(a: GroupType, b: GroupType) -> jax.Array:
    ...


@overload
def rminus(a: PytreeType, b: PytreeType) -> _tree_utils.TangentPytree:
    ...


# Using our typevars in the overloaded signature will cause errors.
def rminus(
    a: Union[MatrixLieGroup, Any], b: Union[MatrixLieGroup, Any]
) -> Union[jax.Array, _tree_utils.TangentPytree]:
    """Manifold right minus. Computes
    `delta = T_ab.log() = (T_wa.inverse() @ T_wb).log()`.

    Supports pytrees containing Lie group instances recursively; simple Euclidean
    subtraction will be performed for all other arrays.
    """
    return _tree_utils._map_group_trees(_rminus, jnp.subtract, a, b)


@jax.jit
def rplus_jacobian_parameters_wrt_delta(transform: MatrixLieGroup) -> jax.Array:
    """Analytical Jacobians for `jaxlie.manifold.rplus()`, linearized around a zero
    local delta.

    Mostly useful for reducing JIT compile times for tangent-space optimization.

    Equivalent to --
    ```
    def rplus_jacobian_parameters_wrt_delta(transform: MatrixLieGroup) -> jax.Array:
        # Since transform objects are pytree containers, note that `jacfwd` returns a
        # transformation object itself and that the Jacobian terms corresponding to the
        # parameters are grabbed explicitly.
        return jax.jacfwd(
            jaxlie.manifold.rplus,  # Args are (transform, delta)
            argnums=1,  # Jacobian wrt delta
        )(transform, onp.zeros(transform.tangent_dim)).parameters()
    ```

    Args:
        transform: Transform to linearize around.

    Returns:
        Jacobian. Shape should be `(Group.parameters_dim, Group.tangent_dim)`.
    """
    if type(transform) is SO2:
        # Jacobian row indices: cos, sin
        # Jacobian col indices: theta

        transform_so2 = cast(SO2, transform)
        J = jnp.zeros((2, 1))

        cos, sin = transform_so2.unit_complex
        J = J.at[0].set(-sin).at[1].set(cos)

    elif type(transform) is SE2:
        # Jacobian row indices: cos, sin, x, y
        # Jacobian col indices: vx, vy, omega

        transform_se2 = cast(SE2, transform)
        J = jnp.zeros((4, 3))

        # Translation terms.
        J = J.at[2:, :2].set(transform_se2.rotation().as_matrix())

        # Rotation terms.
        J = J.at[:2, 2:3].set(
            rplus_jacobian_parameters_wrt_delta(transform_se2.rotation())
        )

    elif type(transform) is SO3:
        # Jacobian row indices: qw, qx, qy, qz
        # Jacobian col indices: omega x, omega y, omega z

        transform_so3 = cast(SO3, transform)

        w, x, y, z = transform_so3.wxyz
        _unused_neg_w, neg_x, neg_y, neg_z = -transform_so3.wxyz

        J = (
            jnp.array(
                [
                    [neg_x, neg_y, neg_z],
                    [w, neg_z, y],
                    [z, w, neg_x],
                    [neg_y, x, w],
                ]
            )
            / 2.0
        )

    elif type(transform) is SE3:
        # Jacobian row indices: qw, qx, qy, qz, x, y, z
        # Jacobian col indices: vx, vy, vz, omega x, omega y, omega z

        transform_se3 = cast(SE3, transform)
        J = jnp.zeros((7, 6))

        # Translation terms.
        J = J.at[4:, :3].set(transform_se3.rotation().as_matrix())

        # Rotation terms.
        J = J.at[:4, 3:6].set(
            rplus_jacobian_parameters_wrt_delta(transform_se3.rotation())
        )

    else:
        assert False, f"Unsupported type: {type(transform)}"

    assert J.shape == (transform.parameters_dim, transform.tangent_dim)
    return J
