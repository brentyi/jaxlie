import dataclasses

import jax
from jax import numpy as jnp
from overrides import final, overrides

from . import _base, types
from ._so2 import SO2
from .utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)
@dataclasses.dataclass(frozen=True)
class SE2(_base.SEBase):
    """Special Euclidean group for proper rigid transforms in 2D."""

    # SE2-specific

    unit_complex_xy: types.Vector
    """Internal parameters. `(cos, sin, x, y)`."""

    @final
    @overrides
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex_xy[..., :2], 5)
        xy = jnp.round(self.unit_complex_xy[..., 2:], 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: types.Scalar, y: types.Scalar, theta: types.Scalar) -> "SE2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SE2(unit_complex_xy=jnp.array([cos, sin, x, y]))

    # SE-specific

    @staticmethod
    @final
    @overrides
    def from_rotation_and_translation(
        rotation: SO2,
        translation: types.Vector,
    ) -> "SE2":
        assert translation.shape == (2,)
        return SE2(
            unit_complex_xy=jnp.concatenate([rotation.unit_complex, translation])
        )

    @final
    @overrides
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @final
    @overrides
    def translation(self) -> types.Vector:
        return self.unit_complex_xy[..., 2:]

    # Factory

    @staticmethod
    @final
    @overrides
    def identity() -> "SE2":
        return SE2(unit_complex_xy=jnp.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @final
    @overrides
    def from_matrix(matrix: types.Matrix) -> "SE2":
        assert matrix.shape == (3, 3)
        # Currently assumes bottom row is [0, 0, 1]
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[:2, :2]),
            translation=matrix[:2, 2],
        )

    # Accessors

    @final
    @overrides
    def parameters(self) -> types.Vector:
        return self.unit_complex_xy

    @final
    @overrides
    def as_matrix(self) -> types.Matrix:
        cos, sin, x, y = self.unit_complex_xy
        return jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ]
        )

    # Operations

    @staticmethod
    @final
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

    @final
    @overrides
    def log(self: "SE2") -> types.TangentVector:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L160
        # Also see:
        # > http://ethaneade.com/lie.pdf

        theta = self.rotation().log()[0]

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

        tangent = jnp.concatenate([V_inv @ self.translation(), theta[None]])
        return tangent

    @final
    @overrides
    def adjoint(self: "SE2") -> types.Matrix:
        cos, sin, x, y = self.unit_complex_xy
        return jnp.array(
            [
                [cos, -sin, y],
                [sin, cos, -x],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    @final
    @overrides
    def sample_uniform(key: jnp.ndarray) -> "SE2":
        key0, key1 = jax.random.split(key)
        return SE2.from_rotation_and_translation(
            rotation=SO2.sample_uniform(key0),
            translation=jax.random.uniform(
                key=key1, shape=(2,), minval=-1.0, maxval=1.0
            ),
        )
