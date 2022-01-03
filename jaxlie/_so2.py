import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides
from typing_extensions import Annotated

from . import _base, hints
from .utils import register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@jdc.pytree_dataclass
class SO2(jdc.EnforcedAnnotationsMixin, _base.SOBase):
    """Special orthogonal group for 2D rotations.

    Internal parameterization is `(cos, sin)`. Tangent parameterization is `(omega,)`.
    """

    # SO2-specific.

    unit_complex: Annotated[
        jnp.ndarray,
        (2,),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. `(cos, sin)`."""

    @overrides
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> "SO2":
        """Construct a rotation object from a scalar angle."""
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    def as_radians(self) -> jnp.ndarray:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    # Factory.

    @staticmethod
    @overrides
    def identity() -> "SO2":
        return SO2(unit_complex=jnp.array([1.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: hints.Array) -> "SO2":
        assert matrix.shape == (2, 2)
        return SO2(unit_complex=jnp.asarray(matrix[:, 0]))

    # Accessors.

    @overrides
    def as_matrix(self) -> jnp.ndarray:
        cos_sin = self.unit_complex
        out = jnp.array(
            [
                # [cos, -sin],
                cos_sin.at[1].multiply(-1),
                # [sin, cos],
                cos_sin[::-1],
            ]
        )
        assert out.shape == (2, 2)
        return out

    @overrides
    def parameters(self) -> jnp.ndarray:
        return self.unit_complex

    # Operations.

    @overrides
    def apply(self: "SO2", target: hints.Array) -> jnp.ndarray:
        assert target.shape == (2,)
        return self.as_matrix() @ target  # type: ignore

    @overrides
    def multiply(self: "SO2", other: "SO2") -> "SO2":
        return SO2(unit_complex=self.as_matrix() @ other.unit_complex)

    @staticmethod
    @overrides
    def exp(tangent: hints.Array) -> "SO2":
        (theta,) = tangent
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    @overrides
    def log(self: "SO2") -> jnp.ndarray:
        return jnp.arctan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @overrides
    def adjoint(self: "SO2") -> jnp.ndarray:
        return jnp.eye(1)

    @overrides
    def inverse(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex * jnp.array([1, -1]))

    @overrides
    def normalize(self: "SO2") -> "SO2":
        return SO2(unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex))

    @staticmethod
    @overrides
    def sample_uniform(key: jax.random.KeyArray) -> "SO2":
        return SO2.from_radians(
            jax.random.uniform(key=key, minval=0.0, maxval=2.0 * jnp.pi)
        )
