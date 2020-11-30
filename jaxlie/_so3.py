import dataclasses

import jax
from jax import numpy as jnp
from overrides import overrides

from ._base import MatrixLieGroup
from ._types import Matrix, TangentVector, Vector
from ._utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
)
@dataclasses.dataclass(frozen=True)
class SO3(MatrixLieGroup):

    # SO3-specific

    wxyz: Vector
    """Internal parameters; wxyz quaternion."""

    # Factory

    @staticmethod
    @overrides
    def identity() -> "SO3":
        return SO3(wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: Matrix) -> "SO3":
        assert matrix.shape == (3, 3)

        # Reference:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        #
        # Note that we use the wxyz convention here, rather than xyzw.

        def case0(m):
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = jnp.array([m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]])
            return t, q

        def case1(m):
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = jnp.array([m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[2, 1] + m[2, 1]])
            return t, q

        def case2(m):
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = jnp.array([m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t])
            return t, q

        def case3(m):
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = jnp.array([t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]])
            return t, q

        t, q = jax.lax.cond(
            matrix[2, 2] < 0,
            true_fun=lambda matrix: jax.lax.cond(
                matrix[0, 0] > matrix[1, 1],
                true_fun=case0,
                false_fun=case1,
                operand=matrix,
            ),
            false_fun=lambda matrix: jax.lax.cond(
                matrix[0, 0] < -matrix[1, 1],
                true_fun=case2,
                false_fun=case3,
                operand=matrix,
            ),
            operand=matrix,
        )

        return SO3(wxyz=q * 0.5 / jnp.sqrt(t))

    # Accessors

    @overrides
    def as_matrix(self) -> Matrix:
        norm = self.wxyz @ self.wxyz
        q = self.wxyz * jnp.sqrt(2.0 / norm)
        q = jnp.outer(q, q)
        return jnp.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )

    @property  # type: ignore
    @overrides
    def parameters(self) -> Vector:
        return self.wxyz

    # Operations

    @overrides
    def apply(self: "SO3", target: Vector) -> Vector:
        assert target.shape == (3,)

        # Compute using quaternion products
        padded_target = jnp.zeros(4).at[1:].set(target)
        return (self.inverse() @ SO3(wxyz=padded_target) @ self).wxyz[1:]

    @overrides
    def product(self: "SO3", other: "SO3") -> "SO3":
        w0, x0, y0, z0 = self.wxyz
        w1, x1, y1, z1 = other.wxyz
        return SO3(
            wxyz=jnp.array(
                [
                    -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                ]
            )
        )

    @staticmethod
    @overrides
    def exp(tangent: TangentVector) -> "SO3":
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583

        assert tangent.shape == (3,)

        def compute_taylor(theta_sq):
            # First-order approximation of axis-angle => quaternion factor formula
            theta_pow_4 = theta_squared * theta_squared
            real_factor = 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0
            imaginary_factor = 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0
            return real_factor, imaginary_factor

        def compute_exact(theta_squared):
            # Standard axis-angle => quaternion factor formula
            theta = jnp.sqrt(theta_squared)
            half_theta = 0.5 * theta
            real_factor = jnp.cos(half_theta)
            imaginary_factor = jnp.sin(half_theta) / theta
            return real_factor, imaginary_factor

        theta_squared = tangent @ tangent
        real_factor, imaginary_factor = jax.lax.cond(
            theta_squared < get_epsilon(tangent.dtype) ** 2,
            true_fun=compute_taylor,
            false_fun=compute_exact,
            operand=theta_squared,
        )

        return SO3(
            wxyz=jnp.concatenate(
                [
                    real_factor[None],
                    imaginary_factor * tangent,
                ]
            )
        )

    @overrides
    def log(self: "SO3") -> TangentVector:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w = self.wxyz[0]
        norm_sq = self.wxyz[1:] @ self.wxyz[1:]

        def compute_taylor(operand):
            w, norm_sq = operand
            return 2.0 / w - 2.0 / 3.0 * norm_sq / (w ** 3)

        def compute_exact(operand):
            f, norm_sq = operand
            norm = jnp.sqrt(norm_sq)
            return jax.lax.cond(
                jnp.abs(w) < get_epsilon(w.dtype),
                true_fun=lambda w_norm: jax.lax.cond(
                    w_norm[0] > 0,
                    lambda norm: jnp.pi / norm,
                    lambda norm: -jnp.pi / norm,
                    operand=jnp.sqrt(w_norm[1]),
                ),
                false_fun=lambda w_norm: 2.0
                * jnp.arctan(w_norm[1] / w_norm[0])
                / w_norm[1],
                operand=(w, norm),
            )

        atan_factor = jax.lax.cond(
            norm_sq < get_epsilon(norm_sq.dtype) ** 2,
            true_fun=compute_taylor,
            false_fun=compute_exact,
            operand=(w, norm_sq),
        )

        return atan_factor * self.wxyz[1:]

    @overrides
    def inverse(self: "SO3") -> "SO3":
        # Negate complex terms
        return SO3(wxyz=self.wxyz.at[1:].set(-self.wxyz[1:]))

    @overrides
    def normalize(self: "SO3") -> "SO3":
        return SO3(wxyz=self.wxyz / jnp.linalg.norm(self.wxyz))
