from __future__ import annotations

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import Annotated, override

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
        jax.Array,
        (..., 2),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. `(cos, sin)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> SO2:
        """Construct a rotation object from a scalar angle."""
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    def as_radians(self) -> jax.Array:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    # Factory.

    @staticmethod
    @override
    def identity() -> SO2:
        return SO2(unit_complex=jnp.array([1.0, 0.0]))

    @staticmethod
    @override
    def from_matrix(matrix: hints.Array) -> SO2:
        assert matrix.shape == (2, 2)
        return SO2(unit_complex=jnp.asarray(matrix[:, 0]))

    # Accessors.

    @override
    def as_matrix(self) -> jax.Array:
        cos_sin = self.unit_complex
        out = jnp.array(
            [
                # [cos, -sin],
                cos_sin * jnp.array([1, -1]),
                # [sin, cos],
                cos_sin[::-1],
            ]
        )
        assert out.shape == (2, 2)
        return out

    @override
    def parameters(self) -> jax.Array:
        return self.unit_complex

    # Operations.

    @override
    def apply(self, target: hints.Array) -> jax.Array:
        assert target.shape == (2,)
        return self.as_matrix() @ target  # type: ignore

    @override
    def multiply(self, other: SO2) -> SO2:
        return SO2(unit_complex=self.as_matrix() @ other.unit_complex)

    @staticmethod
    @override
    def exp(tangent: hints.Array) -> SO2:
        (theta,) = tangent
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.array([cos, sin]))

    @override
    def log(self) -> jax.Array:
        return jnp.arctan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @override
    def adjoint(self) -> jax.Array:
        return jnp.eye(1)

    @override
    def inverse(self) -> SO2:
        return SO2(unit_complex=self.unit_complex * jnp.array([1, -1]))

    @override
    def normalize(self) -> SO2:
        return SO2(unit_complex=self.unit_complex / jnp.linalg.norm(self.unit_complex))

    @staticmethod
    @override
    def sample_uniform(key: jax.Array) -> SO2:
        return SO2.from_radians(
            jax.random.uniform(key=key, minval=0.0, maxval=2.0 * jnp.pi)
        )
