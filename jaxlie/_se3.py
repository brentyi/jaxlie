import dataclasses

import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from . import _base, types
from ._so3 import SO3
from ._utils import get_epsilon, register_lie_group


def _skew(omega: jnp.ndarray) -> jnp.ndarray:
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
class SE3(_base.MatrixLieGroup):
    """Special Euclidean group for proper rigid transforms in 3D."""

    # SE3-specific

    xyz_wxyz: types.Vector
    """Internal parameters. Length-3 translation followed by wxyz quaternion."""

    @overrides
    def __repr__(self):
        trans = jnp.round(self.xyz_wxyz[..., :3], 5)
        quat = jnp.round(self.xyz_wxyz[..., 3:], 5)
        return f"{self.__class__.__name__}(xyz={trans}, wxyz={quat})"

    @staticmethod
    def from_rotation_and_translation(
        rotation: SO3,
        translation: types.Vector,
    ) -> "SE3":
        assert translation.shape == (3,)
        return SE3(xyz_wxyz=jnp.concatenate([translation, rotation.wxyz]))

    @property
    def rotation(self) -> SO3:
        return SO3(wxyz=self.xyz_wxyz[3:])

    @property
    def translation(self) -> types.Vector:
        return self.xyz_wxyz[:3]

    # Factory

    @staticmethod
    @overrides
    def identity() -> "SE3":
        return SE3(xyz_wxyz=onp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: types.Matrix) -> "SE3":
        assert matrix.shape == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1]
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[:3, :3]),
            translation=matrix[:3, 3],
        )

    # Accessors

    @overrides
    def as_matrix(self) -> types.Matrix:
        return (
            jnp.eye(4)
            .at[:3, :3]
            .set(self.rotation.as_matrix())
            .at[:3, 3]
            .set(self.translation)
        )

    @property  # type: ignore
    @overrides
    def parameters(self) -> types.Vector:
        return self.xyz_wxyz

    # Operations

    @overrides
    def apply(self: "SE3", target: types.Vector) -> types.Vector:
        assert target.shape == (3,)
        return self.rotation @ target + self.translation

    @overrides
    def multiply(self: "SE3", other: "SE3") -> "SE3":
        return SE3.from_rotation_and_translation(
            rotation=self.rotation @ other.rotation,
            translation=(self.rotation @ other.translation) + self.translation,
        )

    @staticmethod
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

    @overrides
    def log(self: "SE3") -> types.TangentVector:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation.log()
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
        return jnp.concatenate([V_inv @ self.translation, omega])

    @overrides
    def adjoint(self: "SE3") -> types.Matrix:
        R = self.rotation.as_matrix()
        return jnp.block(
            [
                [R, _skew(self.translation) @ R],
                [jnp.zeros((3, 3)), R],
            ]
        )

    @overrides
    def inverse(self: "SE3") -> "SE3":
        R_inv = self.rotation.inverse()
        return SE3.from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation),
        )

    @overrides
    def normalize(self: "SE3") -> "SE3":
        return SE3.from_rotation_and_translation(
            rotation=self.rotation.normalize(),
            translation=self.translation,
        )
