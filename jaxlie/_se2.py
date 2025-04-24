from typing import Tuple, cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import override

from . import _base, hints
from ._so2 import SO2
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


def _SE2_jac_left(tangent: jax.Array) -> jax.Array:
    """Compute the left jacobian for the given SO(2) tangent vector. This only
    includes the translation terms (2x2), the orientation ones are less
    useful."""
    theta = tangent.squeeze(axis=-1)
    del tangent

    use_taylor = jnp.abs(theta) < get_epsilon(theta.dtype)

    # Shim to avoid NaNs in jnp.where branches, which cause failures for
    # reverse-mode AD.
    safe_theta = cast(
        jax.Array,
        jnp.where(use_taylor, 1.0, theta),
    )

    theta_sq = theta**2
    sin_over_theta = cast(
        jax.Array,
        jnp.where(
            use_taylor,
            1.0 - theta_sq / 6.0,
            jnp.sin(safe_theta) / safe_theta,
        ),
    )
    one_minus_cos_over_theta = cast(
        jax.Array,
        jnp.where(
            use_taylor,
            0.5 * theta - theta * theta_sq / 24.0,
            (1.0 - jnp.cos(safe_theta)) / safe_theta,
        ),
    )
    jac_left = (
        jnp.zeros((*theta.shape, 2, 2))
        .at[..., 0, 0]
        .set(sin_over_theta)
        .at[..., 0, 1]
        .set(-one_minus_cos_over_theta)
        .at[..., 1, 0]
        .set(one_minus_cos_over_theta)
        .at[..., 1, 1]
        .set(sin_over_theta)
    )
    return jac_left


def _SE2_jac_left_inv(tangent: jax.Array) -> jax.Array:
    """Compute the inverse of the left jacobian for the given SO(2) tangent
    vector. This only includes the translation terms (2x2), the orientation
    ones are less useful."""
    theta = tangent.squeeze(axis=-1)
    del tangent

    cos = jnp.cos(theta)
    cos_minus_one = cos - 1.0
    half_theta = theta / 2.0
    use_taylor = jnp.abs(cos_minus_one) < get_epsilon(theta.dtype)

    # Shim to avoid NaNs in jnp.where branches, which cause failures for
    # reverse-mode AD.
    safe_cos_minus_one = jnp.where(use_taylor, 1.0, cos_minus_one)
    half_theta_over_tan_half_theta = jnp.where(
        use_taylor,
        # Taylor approximation.
        1.0 - theta**2 / 12.0,
        # Default.
        -(half_theta * jnp.sin(theta)) / safe_cos_minus_one,
    )
    jac_left_inv = (
        jnp.zeros((*theta.shape, 2, 2))
        .at[..., 0, 0]
        .set(half_theta_over_tan_half_theta)
        .at[..., 0, 1]
        .set(half_theta)
        .at[..., 1, 0]
        .set(-half_theta)
        .at[..., 1, 1]
        .set(half_theta_over_tan_half_theta)
    )
    return jac_left_inv


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)
@jdc.pytree_dataclass
class SE2(_base.SEBase[SO2]):
    """Special Euclidean group for proper rigid transforms in 2D. Broadcasting
    rules are the same as for numpy.

    Internal parameterization is `(cos, sin, x, y)`. Tangent parameterization is `(vx,
    vy, omega)`.
    """

    # SE2-specific.

    unit_complex_xy: jax.Array
    """Internal parameters. `(cos, sin, x, y)`. Shape should be `(*, 4)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = jnp.round(self.unit_complex_xy[..., :2], 5)
        xy = jnp.round(self.unit_complex_xy[..., 2:], 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> "SE2":
        """Construct a transformation from standard 2D pose parameters.

        Note that this is not the same as integrating over a length-3 twist.
        """
        cos = jnp.cos(theta)
        sin = jnp.sin(theta)
        return SE2(unit_complex_xy=jnp.stack([cos, sin, x, y], axis=-1))

    # SE-specific.

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO2,
        translation: hints.Array,
    ) -> "SE2":
        assert translation.shape[-1:] == (2,)
        rotation, translation = broadcast_leading_axes((rotation, translation))
        return SE2(
            unit_complex_xy=jnp.concatenate(
                [rotation.unit_complex, translation], axis=-1
            )
        )

    @override
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @override
    def translation(self) -> jax.Array:
        return self.unit_complex_xy[..., 2:]

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> "SE2":
        return SE2(
            unit_complex_xy=jnp.broadcast_to(
                jnp.array([1.0, 0.0, 0.0, 0.0]), (*batch_axes, 4)
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> "SE2":
        assert matrix.shape[-2:] == (3, 3)
        # Currently assumes bottom row is [0, 0, 1].
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[..., :2, :2]),
            translation=matrix[..., :2, 2],
        )

    # Accessors.

    @override
    def parameters(self) -> jax.Array:
        return self.unit_complex_xy

    @override
    def as_matrix(self) -> jax.Array:
        cos, sin, x, y = jnp.moveaxis(self.unit_complex_xy, -1, 0)
        out = jnp.stack(
            [
                cos,
                -sin,
                x,
                sin,
                cos,
                y,
                jnp.zeros_like(x),
                jnp.zeros_like(x),
                jnp.ones_like(x),
            ],
            axis=-1,
        ).reshape((*self.get_batch_axes(), 3, 3))
        return out

    # Operations.

    @classmethod
    @override
    def exp(cls, tangent: hints.Array) -> "SE2":
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L558
        # Also see:
        # > http://ethaneade.com/lie.pdf

        assert tangent.shape[-1:] == (3,)
        so2_tangent = cast(jax.Array, tangent[..., 2:3])
        V = _SE2_jac_left(so2_tangent)
        return SE2.from_rotation_and_translation(
            rotation=SO2.exp(so2_tangent),
            translation=jnp.einsum("...ij,...j->...i", V, tangent[..., :2]),
        )

    @override
    def log(self) -> jax.Array:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L160
        # Also see:
        # > http://ethaneade.com/lie.pdf
        so2_tangent = self.rotation().log()
        V_inv = _SE2_jac_left_inv(so2_tangent)
        tangent = jnp.concatenate(
            [
                jnp.einsum("...ij,...j->...i", V_inv, self.translation()),
                so2_tangent,
            ],
            axis=-1,
        )
        return tangent

    @override
    def adjoint(self: "SE2") -> jax.Array:
        cos, sin, x, y = jnp.moveaxis(self.unit_complex_xy, -1, 0)
        return jnp.stack(
            [
                cos,
                -sin,
                y,
                sin,
                cos,
                -x,
                jnp.zeros_like(x),
                jnp.zeros_like(x),
                jnp.ones_like(x),
            ],
            axis=-1,
        ).reshape((*self.get_batch_axes(), 3, 3))

    @override
    def jlog(self) -> jax.Array:
        # Reference:
        # This is inverse of matrix (163) from Micro-Lie theory:
        # > https://arxiv.org/pdf/1812.01537

        tangent = self.log()
        theta = tangent[..., 2]

        # Handle the case where theta is small to avoid division by zero.
        use_taylor = jnp.abs(theta) < get_epsilon(theta.dtype)

        V_inv_theta = _SE2_jac_left_inv(theta[..., None])
        V_inv_theta_T = jnp.swapaxes(
            V_inv_theta, -2, -1
        )  # Transpose the last two dimensions

        # Calculate r, handling the small theta case separately.
        batch_shape = self.get_batch_axes()
        eye_2 = jnp.eye(2).reshape((1,) * len(batch_shape) + (2, 2))

        # Shim to avoid NaNs in jnp.where branches, which cause failures for reverse-mode AD.
        safe_theta = jnp.where(use_taylor, jnp.ones_like(theta), theta)
        A = jnp.where(
            use_taylor[..., None, None],
            jnp.stack(
                [
                    jnp.stack([theta / 12.0, jnp.full_like(theta, 0.5)], axis=-1),
                    jnp.stack([jnp.full_like(theta, -0.5), theta / 12.0], axis=-1),
                ],
                axis=-2,
            ),
            (eye_2 - V_inv_theta_T) / safe_theta[..., None, None],
        )
        r = jnp.einsum("...ij,...j->...i", A, tangent[..., :2])

        # Create the jlog matrix.
        jlog = (
            jnp.zeros((*batch_shape, 3, 3))
            .at[..., :2, :2]
            .set(V_inv_theta_T)
            .at[..., :2, 2]
            .set(r)
            .at[..., 2, 2]
            .set(1)
        )
        return jlog

    @classmethod
    @override
    def sample_uniform(
        cls, key: jax.Array, batch_axes: jdc.Static[Tuple[int, ...]] = ()
    ) -> "SE2":
        key0, key1 = jax.random.split(key)
        return SE2.from_rotation_and_translation(
            rotation=SO2.sample_uniform(key0, batch_axes=batch_axes),
            translation=jax.random.uniform(
                key=key1,
                shape=(
                    *batch_axes,
                    2,
                ),
                minval=-1.0,
                maxval=1.0,
            ),
        )
