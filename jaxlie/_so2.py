import dataclasses

from jax import numpy as jnp
from overrides import overrides

from ._base import MatrixLieGroup
from ._types import Matrix, TangentVector, Vector
from ._utils import register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@dataclasses.dataclass(frozen=True)
class SO2(MatrixLieGroup):

    # SO2-specific

    unit_complex: Vector
    """Internal parameterization: `(cos, sin)`."""

    @overrides
    def __repr__(self):
        unit_complex = jnp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: float) -> "SO2":
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    # Factory

    @staticmethod
    @overrides
    def identity() -> "SO2":
        return SO2(unit_complex=jnp.array([1.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: Matrix) -> "SO2":
        assert matrix.shape == (2, 2)
        return SO2(unit_complex=matrix[:, 0])

    # Accessors

    @overrides
    def as_matrix(self) -> Matrix:
        cos, sin = self.unit_complex
        return jnp.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

    @property  # type: ignore
    @overrides
    def parameters(self) -> Vector:
        return self.unit_complex

    # Operations

    @overrides
    def apply(self: "SO2", target: Vector) -> Vector:
        assert target.shape == (2,)
        return self.as_matrix() @ target

    @overrides
    def product(self: "SO2", other: "SO2") -> "SO2":
        return SO2(unit_complex=self.as_matrix() @ other.unit_complex)

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> "SO2":
        assert tangent.shape == (1,)
        cos = jnp.cos(tangent[0])
        sin = jnp.sin(tangent[0])
        return SO2(unit_complex=jnp.array([cos, sin]))

    @overrides
    def log(self: "SO2") -> TangentVector:
        return jnp.arctan2(self.unit_complex[1, None], self.unit_complex[0, None])

    @overrides
    def inverse(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex.at[1].set(-self.unit_complex[1]))

    @overrides
    def normalize(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex))
