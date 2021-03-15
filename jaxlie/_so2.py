import dataclasses

import jax
from jax import numpy as jnp
from overrides import final, overrides

from . import _base, types
from .utils import register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@dataclasses.dataclass(frozen=True)
class SO2(_base.SOBase):
    """Special orthogonal group for 2D rotations."""

    # SO2-specific

    unit_complex: types.Vector
    """Internal parameters. `(cos, sin)`."""

    @final
    @overrides
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: types.Scalar) -> "SO2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    def as_radians(self) -> jnp.ndarray:
        radians = self.log()[..., 0]
        return radians

    # Factory

    @staticmethod
    @final
    @overrides
    def identity() -> "SO2":
        return SO2(unit_complex=jnp.array([1.0, 0.0]))

    @staticmethod
    @final
    @overrides
    def from_matrix(matrix: types.Matrix) -> "SO2":
        assert matrix.shape == (2, 2)
        return SO2(unit_complex=matrix[:, 0])

    # Accessors

    @final
    @overrides
    def as_matrix(self) -> types.Matrix:
        cos, sin = self.unit_complex
        return jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

    @final
    @overrides
    def parameters(self) -> types.Vector:
        return self.unit_complex

    # Operations

    @final
    @overrides
    def apply(self: "SO2", target: types.Vector) -> types.Vector:
        assert target.shape == (2,)
        return self.as_matrix() @ target

    @final
    @overrides
    def multiply(self: "SO2", other: "SO2") -> "SO2":
        return SO2(unit_complex=self.as_matrix() @ other.unit_complex)

    @staticmethod
    @final
    @overrides
    def exp(tangent: types.TangentVector) -> "SO2":
        (theta,) = tangent
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    @final
    @overrides
    def log(self: "SO2") -> types.TangentVector:
        return jnp.arctan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @final
    @overrides
    def adjoint(self: "SO2") -> types.Matrix:
        return jnp.eye(1)

    @final
    @overrides
    def inverse(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex * jnp.array([1, -1]))

    @final
    @overrides
    def normalize(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex))

    @staticmethod
    @final
    @overrides
    def sample_uniform(key: jnp.ndarray) -> "SO2":
        return SO2.from_radians(
            jax.random.uniform(key=key, minval=0.0, maxval=2.0 * jnp.pi)
        )
