import dataclasses

import jax
from jax import numpy as jnp
from overrides import overrides

from . import _base, types
from ._so2 import SO2
from ._utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)
@dataclasses.dataclass(frozen=True)
class SE2(_base.MatrixLieGroup):
    """Special Euclidean group for proper rigid transforms in 2D."""

    # SE2-specific

    xy_unit_complex: types.Vector
    """Internal parameters. `(x, y, cos, sin)`."""

    @overrides
    def __repr__(self):
        xy = jnp.round(self.xy_unit_complex[..., :2], 5)
        unit_complex = jnp.round(self.xy_unit_complex[..., 2:], 5)
        return f"{self.__class__.__name__}(xy={xy}, unit_complex={unit_complex})"

    @staticmethod
    def from_xy_theta(x: float, y: float, theta: float) -> "SE2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SE2(xy_unit_complex=jnp.array([x, y, cos, sin]))

    @staticmethod
    def from_rotation_and_translation(
        rotation: SO2,
        translation: types.Vector,
    ) -> "SE2":
        assert translation.shape == (2,)
        return SE2(
            xy_unit_complex=jnp.concatenate([translation, rotation.unit_complex])
        )

    @property
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.xy_unit_complex[2:])

    @property
    def translation(self) -> types.Vector:
        return self.xy_unit_complex[:2]

    # Factory

    @staticmethod
    @overrides
    def identity() -> "SE2":
        return SE2(xy_unit_complex=jnp.array([0.0, 0.0, 1.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: types.Matrix) -> "SE2":
        assert matrix.shape == (3, 3)
        # Currently assumes bottom row is [0, 0, 1]
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[:2, :2]),
            translation=matrix[:2, 2],
        )

    # Accessors

    @property  # type: ignore
    @overrides
    def parameters(self) -> types.Vector:
        return self.xy_unit_complex

    @overrides
    def as_matrix(self) -> types.Matrix:
        x, y, cos, sin = self.xy_unit_complex
        return jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ]
        )

    # Operations

    @overrides
    def apply(self: "SE2", target: types.Vector) -> types.Vector:
        return self.rotation @ target + self.translation

    @overrides
    def multiply(self: "SE2", other: "SE2") -> "SE2":
        # Apply rotation to both the rotation and translation terms of `other`
        xy_unit_complex = jax.vmap(self.rotation.apply)(
            other.xy_unit_complex.reshape((2, 2))
        ).flatten()

        # Apply translation
        xy_unit_complex = xy_unit_complex.at[:2].add(self.translation)

        return SE2(xy_unit_complex=xy_unit_complex)

    @staticmethod
    @overrides
    def exp(tangent: types.TangentVector) -> "SE2":
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L558
        # Also see:
        # > http://ethaneade.com/lie.pdf

        assert tangent.shape == (3,)

        theta = tangent[2]
        use_taylor = jnp.abs(theta) < get_epsilon(tangent.dtype)
        theta_sq = theta ** 2
        sin_over_theta = jnp.where(
            use_taylor,
            1.0 - theta_sq / 6.0,
            jnp.sin(theta) / theta,
        )
        one_minus_cos_over_theta = jnp.where(
            use_taylor,
            0.5 * theta - theta * theta_sq / 24.0,
            (1.0 - jnp.cos(theta)) / theta,
        )

        V = jnp.array(
            [
                [sin_over_theta, -one_minus_cos_over_theta],
                [one_minus_cos_over_theta, sin_over_theta],
            ]
        )
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=V @ tangent[:2],
        )

    @overrides
    def log(self: "SE2") -> types.TangentVector:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L160
        # Also see:
        # > http://ethaneade.com/lie.pdf

        theta = self.rotation.log()[0]

        cos = jnp.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        use_taylor = jnp.abs(cos_minus_one) < get_epsilon(theta.dtype)
        half_theta_over_tan_half_theta = jnp.where(
            use_taylor,
            # First-order Taylor approximation
            1.0 - (theta ** 2) / 12.0,
            # Default
            -(half_theta * jnp.sin(theta)) / cos_minus_one,
        )

        V_inv = jnp.array(
            [
                [half_theta_over_tan_half_theta, half_theta],
                [-half_theta, half_theta_over_tan_half_theta],
            ]
        )

        tangent = jnp.concatenate([V_inv @ self.translation, theta[None]])
        return tangent

    @overrides
    def adjoint(self: "SE2") -> types.Matrix:
        x, y, cos, sin = self.xy_unit_complex
        return jnp.array(
            [
                [cos, -sin, y],
                [sin, cos, -x],
                [0.0, 0.0, 1.0],
            ]
        )

    @overrides
    def inverse(self: "SE2") -> "SE2":
        R_inv = self.rotation.inverse()
        return SE2.from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation),
        )

    @overrides
    def normalize(self: "SE2") -> "SE2":
        return SE2.from_rotation_and_translation(
            rotation=self.rotation.normalize(),
            translation=self.translation,
        )
