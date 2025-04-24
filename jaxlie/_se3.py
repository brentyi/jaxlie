from __future__ import annotations

from typing import Tuple, cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import override

from . import _base, hints
from ._so3 import _SO3_jac_left as _SO3_V, SO3, _skew, _SO3_jac_left_inv as _SO3_V_inv
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


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
        theta = tangent[..., 3:]
        rotation = SO3.exp(theta)
        V = _SO3_V(
            cast(jax.Array, theta), rotation.as_matrix()
        )  # Using _SO3_jac_left via import alias
        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=jnp.einsum("...ij,...j->...i", V, tangent[..., :3]),
        )

    @override
    def log(self) -> jax.Array:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        theta = self.rotation().log()
        V_inv = _SO3_V_inv(theta)  # Using _SO3_jac_left_inv via import alias
        return jnp.concatenate(
            [jnp.einsum("...ij,...j->...i", V_inv, self.translation()), theta], axis=-1
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

    @override
    def jlog(self) -> jax.Array:
        rotation = self.rotation()
        translation = self.translation()

        jlog_so3 = rotation.jlog()

        w = rotation.log()
        theta = jnp.linalg.norm(w, axis=-1)
        theta_squared = jnp.sum(jnp.square(w), axis=-1)

        use_taylor = theta_squared < get_epsilon(theta.dtype)
        theta_inv = cast(jax.Array, jnp.where(use_taylor, 1.0, 1.0 / theta))
        theta_squared_inv = theta_inv**2
        st, ct = jnp.sin(theta), jnp.cos(theta)
        inv_2_2ct = jnp.where(use_taylor, 0.5, 1 / (2 * (1 - ct)))

        # Use jnp.where for beta and beta_dot_over_theta.
        beta = theta_squared_inv - st * theta_inv * inv_2_2ct
        beta_dot_over_theta = (
            -2 * theta_squared_inv**2
            + (1 + st * theta_inv) * theta_squared_inv * inv_2_2ct
        )
        wTp = jnp.sum(w * translation, axis=-1, keepdims=True)
        v3_tmp = (beta_dot_over_theta[..., None] * wTp) * w - (
            theta_squared[..., None] * beta_dot_over_theta[..., None]
            + 2 * beta[..., None]
        ) * translation
        C = (
            jnp.einsum("...i,...j->...ij", v3_tmp, w)
            + beta[..., None, None] * jnp.einsum("...i,...j->...ij", w, translation)
            + wTp[..., None] * beta[..., None, None] * jnp.eye(3)
        )
        C = C + 0.5 * _skew(translation)

        B = jnp.einsum("...ij,...jk->...ik", C, jlog_so3)
        B_wh = jnp.where(use_taylor[..., None, None], 0.5 * _skew(translation), B)
        assert B_wh.shape == jlog_so3.shape

        jlog = (
            jnp.zeros((*theta.shape, 6, 6))
            .at[..., :3, :3]
            .set(jlog_so3)
            .at[..., 3:, 3:]
            .set(jlog_so3)
            .at[..., :3, 3:]
            .set(B_wh)
        )
        return jlog

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
