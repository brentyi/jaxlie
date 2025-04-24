from __future__ import annotations

from typing import Tuple

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import override

from . import _base, hints
from .utils import broadcast_leading_axes, register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@jdc.pytree_dataclass
class SO2(_base.SOBase):
    """Special orthogonal group for 2D rotations. Broadcasting rules are the
    same as for `numpy`.

    Internal parameterization is `(cos, sin)`. Tangent parameterization is `(omega,)`.
    """

    # SO2-specific.

    unit_complex: jax.Array
    """Internal parameters. `(cos, sin)`. Shape should be `(*, 2)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> SO2:
        """Construct a rotation object from a scalar angle."""
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SO2(unit_complex=jnp.stack([cos, sin], axis=-1))

    def as_radians(self) -> jax.Array:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> SO2:
        return SO2(
            unit_complex=jnp.stack(
                [jnp.ones(batch_axes), jnp.zeros(batch_axes)], axis=-1
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SO2:
        assert matrix.shape[-2:] == (2, 2)
        return SO2(unit_complex=jnp.asarray(matrix[..., :, 0]))

    # Accessors.

    @override
    def as_matrix(self) -> jax.Array:
        cos_sin = self.unit_complex
        out = jnp.stack(
            [
                # [cos, -sin],
                cos_sin * jnp.array([1, -1]),
                # [sin, cos],
                cos_sin[..., ::-1],
            ],
            axis=-2,
        )
        assert out.shape == (*self.get_batch_axes(), 2, 2)
        return out

    @override
    def parameters(self) -> jax.Array:
        return self.unit_complex

    # Operations.

    @override
    def apply(self, target: hints.Array) -> jax.Array:
        assert target.shape[-1:] == (2,)
        self, target = broadcast_leading_axes((self, target))
        return jnp.einsum("...ij,...j->...i", self.as_matrix(), target)

    @override
    def multiply(self, other: SO2) -> SO2:
        return SO2(
            unit_complex=jnp.einsum(
                "...ij,...j->...i", self.as_matrix(), other.unit_complex
            )
        )

    @classmethod
    @override
    def exp(cls, tangent: hints.Array) -> SO2:
        assert tangent.shape[-1] == 1
        cos = jnp.cos(tangent)
        sin = jnp.sin(tangent)
        return SO2(unit_complex=jnp.concatenate([cos, sin], axis=-1))

    @override
    def log(self) -> jax.Array:
        return jnp.arctan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @override
    def adjoint(self) -> jax.Array:
        return jnp.ones((*self.get_batch_axes(), 1, 1))

    @override
    def inverse(self) -> SO2:
        return SO2(unit_complex=self.unit_complex * jnp.array([1, -1]))

    @override
    def normalize(self) -> SO2:
        return SO2(
            unit_complex=self.unit_complex
            / jnp.linalg.norm(self.unit_complex, axis=-1, keepdims=True)
        )

    @override
    def jlog(self) -> jax.Array:
        batch_axes = self.get_batch_axes()
        ones = jnp.ones(batch_axes)
        return ones[..., None, None]

    @classmethod
    @override
    def sample_uniform(
        cls, key: jax.Array, batch_axes: jdc.Static[Tuple[int, ...]] = ()
    ) -> SO2:
        out = SO2.from_radians(
            jax.random.uniform(
                key=key, shape=batch_axes, minval=0.0, maxval=2.0 * jnp.pi
            )
        )
        assert out.get_batch_axes() == batch_axes
        return out
