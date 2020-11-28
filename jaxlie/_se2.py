import dataclasses
from typing import Tuple, Type

import jax
from jax import numpy as jnp
from overrides import overrides

from ._base import MatrixLieGroup
from ._so2 import SO2
from ._types import Matrix, TangentVector, Vector
from ._utils import get_epsilon, register_lie_group


@register_lie_group
@dataclasses.dataclass(frozen=True)
class SE2(MatrixLieGroup):

    # SE2-specific
    xy_unit_complex: Vector

    @staticmethod
    def from_xy_theta(x: float, y: float, theta: float) -> "SE2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SE2(xy_unit_complex=jnp.array([x, y, cos, sin]))

    @staticmethod
    def from_rotation_and_translation(rotation: SO2, translation: jnp.array) -> "SE2":
        xy_unit_complex = jnp.zeros(4)
        xy_unit_complex = xy_unit_complex.at[:2].set(translation)
        xy_unit_complex = xy_unit_complex.at[2:].set(rotation.unit_complex)
        return SE2(xy_unit_complex=xy_unit_complex)

    def rotation(self) -> SO2:
        return SO2(unit_complex=self.xy_unit_complex[2:])

    def translation(self) -> Vector:
        return self.xy_unit_complex[:2]

    # Factory

    @staticmethod
    @overrides
    def identity() -> "SE2":
        return SE2(xy_unit_complex=jnp.array([0.0, 0.0, 1.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: Matrix) -> "SE2":
        assert matrix.shape == (3, 3)
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[:2, :2]), translation=matrix[:2, 2]
        )

    # Accessors

    @staticmethod
    @overrides
    def matrix_dim() -> int:
        return 3

    @staticmethod
    @overrides
    def compact_dim() -> int:
        return 4

    @staticmethod
    @overrides
    def tangent_dim() -> int:
        return 3

    @overrides
    def matrix(self) -> Matrix:
        x, y, cos, sin = self.xy_unit_complex
        return jnp.array(
            [
                [cos, -sin, x],
                [sin, cos, y],
                [0.0, 0.0, 1.0],
            ]
        )

    @overrides
    def compact(self) -> Vector:
        return self.xy_unit_complex

    # Operations

    @overrides
    def apply(self: "SE2", target: Vector) -> Vector:
        return self.rotation() @ target + self.translation()

    @overrides
    def product(self: "SE2", other: "SE2") -> "SE2":
        xy_unit_complex = jnp.zeros(4)

        # Compute translation terms
        xy_unit_complex = xy_unit_complex.at[:2].set(
            self.rotation() @ other.translation() + self.translation()
        )

        # Compute rotation terms
        xy_unit_complex = xy_unit_complex.at[2:].set(
            (self.rotation() @ other.rotation()).unit_complex
        )

        return SE2(xy_unit_complex=xy_unit_complex)

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> "SE2":
        # See Gallier and Xu:
        # > https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf

        assert tangent.shape == (3,)

        def compute_taylor(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            theta_sq = theta ** 2

            sin_over_theta = 1.0 - theta_sq / 6.0
            one_minus_cos_over_theta = 0.5 * theta - theta * theta_sq / 24.0
            return sin_over_theta, one_minus_cos_over_theta

        def compute_exact(theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            sin_over_theta = jnp.sin(theta) / theta
            one_minus_cos_over_theta = (1.0 - jnp.cos(theta)) / theta
            return sin_over_theta, one_minus_cos_over_theta

        theta = tangent[2]
        sin_over_theta, one_minus_cos_over_theta = jax.lax.cond(
            jnp.abs(theta) < get_epsilon(tangent),
            compute_taylor,
            compute_exact,
            operand=theta,
        )

        V = jnp.array(
            [
                [sin_over_theta, -one_minus_cos_over_theta],
                [one_minus_cos_over_theta, sin_over_theta],
            ]
        )
        return SE2.from_rotation_and_translation(SO2.from_theta(theta), V @ tangent[:2])

    @overrides
    def log(self: "SE2") -> TangentVector:
        # See Gallier and Xu:
        # > https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf

        theta = self.rotation().log()[0]

        cos = jnp.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        half_theta_over_tan_half_theta = jax.lax.cond(
            jnp.abs(cos_minus_one) < get_epsilon(theta),
            # First-order Taylor approximation
            lambda args: 1.0 - (args[0] ** 2) / 12.0,
            # Default
            lambda args: -(args[1] * jnp.sin(args[0])) / args[2],
            operand=(theta, half_theta, cos_minus_one),
        )

        V_inv = jnp.array(
            [
                [half_theta_over_tan_half_theta, half_theta],
                [-half_theta, half_theta_over_tan_half_theta],
            ]
        )

        tangent = jnp.zeros(3)
        tangent = tangent.at[:2].set(V_inv @ self.translation())
        tangent = tangent.at[2].set(theta)
        return tangent

    @overrides
    def inverse(self: "SE2") -> "SE2":
        R_inv = self.rotation().inverse()
        return SE2.from_rotation_and_translation(R_inv, -(R_inv @ self.translation()))

    @overrides
    def normalize(self: "SE2") -> "SE2":
        return SE2.from_rotation_and_translation(
            self.rotation().normalize(), self.translation()
        )
