from typing import TypeVar, cast

import jax
from jax import numpy as jnp

from .. import types
from .._base import MatrixLieGroup
from .._se2 import SE2
from .._se3 import SE3
from .._so2 import SO2
from .._so3 import SO3

T = TypeVar("T", bound=MatrixLieGroup)


@jax.jit
def rplus(transform: T, delta: types.TangentVector) -> T:
    """Manifold right plus.

    Computes `T_wb = T_wa @ exp(delta)`.

    Args:
        transform (T): `T_wa`
        delta (types.TangentVector): `T_ab.log()`

    Returns:
        T: `T_wb`
    """
    return transform @ type(transform).exp(delta)


@jax.jit
def rplus_jacobian_wrt_delta_at_zero(transform: T) -> T:
    """Analytical Jacobians for `jaxlie.manifold.rplus()`, linearized around a zero
    local delta.

    Useful for on-manifold optimization.

    Equivalent to --
    ```
        jax.jacfwd(
            jaxlie.manifold.rplus,  # Args are (transform, delta)
            argnums=1,  # Jacobian wrt delta
        )(transform, onp.zeros(transform.tangent_dim))
    ```

    Args:
        transform (T): transform

    Returns:
        T: Jacobian. Note that this will be a MatrixLieGroup object, and the matrix can
        be accessed with `.parameters`!
    """
    J = jnp.zeros((transform.parameters_dim, transform.tangent_dim))

    if type(transform) is SO2:
        # Jacobian row indices: cos, sin
        # Jacobian col indices: theta

        transform_so2 = cast(SO2, transform)

        cos, sin = transform_so2.unit_complex
        J = J.at[0].set(-sin).at[1].set(cos)

    elif type(transform) is SE2:
        # Jacobian row indices: x, y, cos, sin
        # Jacobian col indices: vx, vy, omega

        transform_se2 = cast(SE2, transform)

        # Translation terms
        J = J.at[:2, :2].set(transform_se2.rotation.as_matrix())

        # Rotation terms
        J = J.at[2:4, 2:3].set(
            rplus_jacobian_wrt_delta_at_zero(transform_se2.rotation).parameters
        )

    elif type(transform) is SO3:
        # Jacobian row indices: qw, qx, qy, qz
        # Jacobian col indices: omega x, omega y, omega z

        transform_so3 = cast(SO3, transform)

        # Jacobian
        w, x, y, z = transform_so3.wxyz
        neg_w, neg_x, neg_y, neg_z = -transform_so3.wxyz

        J = (
            jnp.array(
                [
                    [-x, -y, -z],
                    [w, -z, y],
                    [z, w, -x],
                    [-y, x, w],
                ]
            )
            / 2.0
        )

    elif type(transform) is SE3:
        # Jacobian row indices: x, y, z, qw, qx, qy, qz
        # Jacobian col indices: omega x, omega y, omega z

        transform_se3 = cast(SE3, transform)

        # Translation terms
        J = J.at[:3, :3].set(transform_se3.rotation.as_matrix())

        # Rotation terms
        J = J.at[3:7, 3:6].set(
            rplus_jacobian_wrt_delta_at_zero(transform_se3.rotation).parameters
        )

    else:
        assert False, f"Unsupported type: {type(transform)}"

    return type(transform)(J)


@jax.jit
def rminus(a: T, b: T) -> types.TangentVector:
    """Manifold right minus.

    Computes `delta = (T_wa.inverse() @ T_wb).log()`.

    Args:
        a (T): `T_wa`
        b (T): `T_wb`

    Returns:
        types.TangentVector: `T_ab.log()`
    """
    return (a.inverse() @ b).log()
