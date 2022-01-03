from typing import TypeVar, cast

import jax
from jax import numpy as jnp

from .. import hints
from .._base import MatrixLieGroup
from .._se2 import SE2
from .._se3 import SE3
from .._so2 import SO2
from .._so3 import SO3

T = TypeVar("T", bound=MatrixLieGroup)


@jax.jit
def rplus(transform: T, delta: hints.Array) -> T:
    """Manifold right plus.

    Computes `T_wb = T_wa @ exp(delta)`.

    Args:
        transform: `T_wa`
        delta: `T_ab.log()`

    Returns:
        T: `T_wb`
    """
    return transform @ type(transform).exp(delta)


@jax.jit
def rplus_jacobian_parameters_wrt_delta(transform: MatrixLieGroup) -> jnp.ndarray:
    """Analytical Jacobians for `jaxlie.manifold.rplus()`, linearized around a zero
    local delta.

    Sometimes useful for manifold optimization.

    Equivalent to --
    ```
    def rplus_jacobian_parameters_wrt_delta(
        transform: jaxlie.MatrixLieGroup,
    ) -> jnp.ndarray:
        return jax.jacfwd(
            lambda delta: jaxlie.manifold.rplus(transform, delta).parameters()
        )(onp.zeros(transform.tangent_dim))
    ```

    Args:
        transform

    Returns:
        Jacobian. Shape should be `(Group.parameters_dim, Group.tangent_dim)`.
    """
    if isinstance(transform, SO2):
        # Jacobian row indices: cos, sin
        # Jacobian col indices: theta

        transform_so2 = cast(SO2, transform)
        J = jnp.zeros((2, 1))

        cos, sin = transform_so2.unit_complex
        J = J.at[0].set(-sin).at[1].set(cos)

    elif isinstance(transform, SE2):
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

    elif isinstance(transform, SO3):
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

    elif isinstance(transform, SE3):
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


@jax.jit
def rminus(a: T, b: T) -> jnp.ndarray:
    """Manifold right minus.

    Computes `delta = (T_wa.inverse() @ T_wb).log()`.

    Args:
        a: `T_wa`
        b: `T_wb`

    Returns:
        `T_ab.log()`
    """
    return (a.inverse() @ b).log()
