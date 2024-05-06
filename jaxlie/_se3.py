from __future__ import annotations

from typing import Tuple, cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import override

from . import _base, hints
from ._so3 import SO3
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


def _skew(omega: hints.Array) -> jax.Array:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = jnp.moveaxis(omega, -1, 0)
    zeros = jnp.zeros_like(wx)
    return jnp.stack(
        [zeros, -wz, wy, wz, zeros, -wx, -wy, wx, zeros],
        axis=-1,
    ).reshape((*omega.shape[:-1], 3, 3))


@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
@jdc.pytree_dataclass
class SE3(_base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D. Broadcasting
    rules are the same as for numpy.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    wxyz_xyz: jax.Array
    """Internal parameters. wxyz quaternion followed by xyz translation. Shape should be `(*, 7)`."""

    @override
    def __repr__(self) -> str:
        quat = jnp.round(self.wxyz_xyz[..., :4], 5)
        trans = jnp.round(self.wxyz_xyz[..., 4:], 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific.

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: hints.Array,
    ) -> SE3:
        assert translation.shape[-1:] == (3,)
        rotation, translation = broadcast_leading_axes((rotation, translation))
        return SE3(wxyz_xyz=jnp.concatenate([rotation.wxyz, translation], axis=-1))

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> jax.Array:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> SE3:
        return SE3(
            wxyz_xyz=jnp.broadcast_to(
                jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (*batch_axes, 7)
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SE3:
        assert matrix.shape[-2:] == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[..., :3, :3]),
            translation=matrix[..., :3, 3],
        )

    # Accessors.

    @override
    def as_matrix(self) -> jax.Array:
        return (
            jnp.zeros((*self.get_batch_axes(), 4, 4))
            .at[..., :3, :3]
            .set(self.rotation().as_matrix())
            .at[..., :3, 3]
            .set(self.translation())
            .at[..., 3, 3]
            .set(1.0)
        )

    @override
    def parameters(self) -> jax.Array:
        return self.wxyz_xyz

    # Operations.

    @classmethod
    @override
    def exp(cls, tangent: hints.Array) -> SE3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape[-1:] == (6,)

        rotation = SO3.exp(tangent[..., 3:])

        theta_squared = jnp.sum(jnp.square(tangent[..., 3:]), axis=-1)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = cast(
            jax.Array,
            jnp.where(
                use_taylor,
                jnp.ones_like(theta_squared),  # Any non-zero value should do here.
                theta_squared,
            ),
        )
        del theta_squared
        theta_safe = jnp.sqrt(theta_squared_safe)

        skew_omega = _skew(tangent[..., 3:])
        V = jnp.where(
            use_taylor[..., None, None],
            rotation.as_matrix(),
            (
                jnp.eye(3)
                + ((1.0 - jnp.cos(theta_safe)) / (theta_squared_safe))[..., None, None]
                * skew_omega
                + (
                    (theta_safe - jnp.sin(theta_safe))
                    / (theta_squared_safe * theta_safe)
                )[..., None, None]
                * jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
            ),
        )

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=jnp.einsum("...ij,...j->...i", V, tangent[..., :3]),
        )

    @override
    def log(self) -> jax.Array:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = jnp.sum(jnp.square(omega), axis=-1)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        skew_omega = _skew(omega)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = jnp.where(
            use_taylor,
            jnp.ones_like(theta_squared),  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = jnp.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        V_inv = jnp.where(
            use_taylor[..., None, None],
            jnp.eye(3)
            - 0.5 * skew_omega
            + jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega) / 12.0,
            (
                jnp.eye(3)
                - 0.5 * skew_omega
                + (
                    (
                        1.0
                        - theta_safe
                        * jnp.cos(half_theta_safe)
                        / (2.0 * jnp.sin(half_theta_safe))
                    )
                    / theta_squared_safe
                )[..., None, None]
                * jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
            ),
        )
        return jnp.concatenate(
            [jnp.einsum("...ij,...j->...i", V_inv, self.translation()), omega], axis=-1
        )

    @override
    def adjoint(self) -> jax.Array:
        R = self.rotation().as_matrix()
        return jnp.concatenate(
            [
                jnp.concatenate(
                    [R, jnp.einsum("...ij,...jk->...ik", _skew(self.translation()), R)],
                    axis=-1,
                ),
                jnp.concatenate(
                    [jnp.zeros((*self.get_batch_axes(), 3, 3)), R], axis=-1
                ),
            ],
            axis=-2,
        )

    @classmethod
    @override
    def sample_uniform(
        cls, key: jax.Array, batch_axes: jdc.Static[Tuple[int, ...]] = ()
    ) -> SE3:
        key0, key1 = jax.random.split(key)
        return SE3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(key0, batch_axes=batch_axes),
            translation=jax.random.uniform(
                key=key1, shape=(*batch_axes, 3), minval=-1.0, maxval=1.0
            ),
        )
