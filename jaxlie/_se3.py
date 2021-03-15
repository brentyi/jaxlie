import dataclasses

import jax
from jax import numpy as jnp
from overrides import final, overrides

from . import _base, types
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group


def _skew(omega: types.Vector) -> types.Matrix:
    """Returns the skew-symmetric form of a length-3 vector. """

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
@dataclasses.dataclass(frozen=True)
class SE3(_base.SEBase):
    """Special Euclidean group for proper rigid transforms in 3D."""

    # SE3-specific

    wxyz_xyz: types.Vector
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @final
    @overrides
    def __repr__(self) -> str:
        quat = jnp.round(self.wxyz_xyz[..., :4], 5)
        trans = jnp.round(self.wxyz_xyz[..., 4:], 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific

    @staticmethod
    @final
    @overrides
    def from_rotation_and_translation(
        rotation: SO3,
        translation: types.Vector,
    ) -> "SE3":
        assert translation.shape == (3,)
        return SE3(wxyz_xyz=jnp.concatenate([rotation.wxyz, translation]))

    @final
    @overrides
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @final
    @overrides
    def translation(self) -> types.Vector:
        return self.wxyz_xyz[..., 4:]

    # Factory

    @staticmethod
    @final
    @overrides
    def identity() -> "SE3":
        return SE3(wxyz_xyz=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @final
    @overrides
    def from_matrix(matrix: types.Matrix) -> "SE3":
        assert matrix.shape == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    # Accessors

    @final
    @overrides
    def as_matrix(self) -> types.Matrix:
        return (
            jnp.eye(4)
            .at[:3, :3]
            .set(self.rotation().as_matrix())
            .at[:3, 3]
            .set(self.translation())
        )

    @final
    @overrides
    def parameters(self) -> types.Vector:
        return self.wxyz_xyz

    # Operations

    @staticmethod
    @final
    @overrides
    def exp(tangent: types.TangentVector) -> "SE3":
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape == (6,)

        rotation = SO3.exp(tangent[3:])

        theta_squared = tangent[3:] @ tangent[3:]
        theta = jnp.sqrt(theta_squared)
        skew_omega = _skew(tangent[3:])
        use_small_theta = theta < get_epsilon(theta_squared.dtype)
        V = jnp.where(
            use_small_theta,
            rotation.as_matrix(),
            (
                jnp.eye(3)
                + (1.0 - jnp.cos(theta)) / (theta_squared) * skew_omega
                + (theta - jnp.sin(theta))
                / (theta_squared * theta)
                * (skew_omega @ skew_omega)
            ),
        )

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=V @ tangent[:3],
        )

    @final
    @overrides
    def log(self: "SE3") -> types.TangentVector:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = omega @ omega
        skew_omega = _skew(omega)
        theta = jnp.sqrt(theta_squared)
        half_theta = theta / 2.0
        use_small_theta = theta < get_epsilon(theta_squared.dtype)
        V_inv = jnp.where(
            use_small_theta,
            jnp.eye(3) - 0.5 * skew_omega + (skew_omega @ skew_omega) / 12.0,
            (
                jnp.eye(3)
                - 0.5 * skew_omega
                + (1.0 - theta * jnp.cos(half_theta) / (2.0 * jnp.sin(half_theta)))
                / theta_squared
                * (skew_omega @ skew_omega)
            ),
        )
        return jnp.concatenate([V_inv @ self.translation(), omega])

    @final
    @overrides
    def adjoint(self: "SE3") -> types.Matrix:
        R = self.rotation().as_matrix()
        return jnp.block(
            [
                [R, _skew(self.translation()) @ R],
                [jnp.zeros((3, 3)), R],
            ]
        )

    @staticmethod
    @final
    @overrides
    def sample_uniform(key: jnp.ndarray) -> "SE3":
        key0, key1 = jax.random.split(key)
        return SE3.from_rotation_and_translation(
            rotation=SO3.sample_uniform(key0),
            translation=jax.random.uniform(
                key=key1, shape=(3,), minval=-1.0, maxval=1.0
            ),
        )
