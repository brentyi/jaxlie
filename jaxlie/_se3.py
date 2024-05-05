from __future__ import annotations

from typing import cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import Annotated, override

from . import _base, hints
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group


def _skew(omega: hints.Array) -> jax.Array:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = omega
    return jnp.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
@jdc.pytree_dataclass
class SE3(jdc.EnforcedAnnotationsMixin, _base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    wxyz_xyz: Annotated[
        jax.Array,
        (..., 7),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. wxyz quaternion followed by xyz translation."""

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
        assert translation.shape == (3,)
        return SE3(wxyz_xyz=jnp.concatenate([rotation.wxyz, translation]))

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> jax.Array:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @classmethod
    @override
    def identity(cls) -> SE3:
        return SE3(wxyz_xyz=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SE3:
        assert matrix.shape == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    # Accessors.

    @override
    def as_matrix(self) -> jax.Array:
        return (
            jnp.eye(4)
            .at[:3, :3]
            .set(self.rotation().as_matrix())
            .at[:3, 3]
            .set(self.translation())
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
        assert tangent.shape == (6,)

        rotation = SO3.exp(tangent[3:])

        theta_squared = tangent[3:] @ tangent[3:]
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = cast(
            jax.Array,
            jnp.where(
                use_taylor,
                1.0,  # Any non-zero value should do here.
                theta_squared,
            ),
        )
        del theta_squared
        theta_safe = jnp.sqrt(theta_squared_safe)

        skew_omega = _skew(tangent[3:])
        V = jnp.where(
            use_taylor,
            rotation.as_matrix(),
            (
                jnp.eye(3)
                + (1.0 - jnp.cos(theta_safe)) / (theta_squared_safe) * skew_omega
                + (theta_safe - jnp.sin(theta_safe))
                / (theta_squared_safe * theta_safe)
                * (skew_omega @ skew_omega)
            ),
        )

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=V @ tangent[:3],
        )

    @override
    def log(self) -> jax.Array:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = omega @ omega
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        skew_omega = _skew(omega)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = jnp.where(
            use_taylor,
            1.0,  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = jnp.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        V_inv = jnp.where(
            use_taylor,
            jnp.eye(3) - 0.5 * skew_omega + (skew_omega @ skew_omega) / 12.0,
            (
                jnp.eye(3)
                - 0.5 * skew_omega
                + (
                    1.0
                    - theta_safe
                    * jnp.cos(half_theta_safe)
                    / (2.0 * jnp.sin(half_theta_safe))
                )
                / theta_squared_safe
                * (skew_omega @ skew_omega)
            ),
        )
        return jnp.concatenate([V_inv @ self.translation(), omega])

    @override
    def adjoint(self) -> jax.Array:
        R = self.rotation().as_matrix()
        return jnp.block(
            [
                [R, _skew(self.translation()) @ R],
                [jnp.zeros((3, 3)), R],
            ]
        )

    @classmethod
    @override
    def sample_uniform(cls, key: jax.Array) -> SE3:
        key0, key1 = jax.random.split(key)
        return SE3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(key0),
            translation=jax.random.uniform(
                key=key1, shape=(3,), minval=-1.0, maxval=1.0
            ),
        )
