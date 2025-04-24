from __future__ import annotations

from typing import Tuple, cast

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from typing_extensions import override

from . import _base, hints
from .utils import broadcast_leading_axes, get_epsilon, register_lie_group


def _skew(omega: hints.Array) -> jax.Array:
    """Returns the skew-symmetric form of a length-3 vector."""

    wx, wy, wz = jnp.moveaxis(omega, -1, 0)
    zeros = jnp.zeros_like(wx)
    return jnp.stack(
        [zeros, -wz, wy, wz, zeros, -wx, -wy, wx, zeros],
        axis=-1,
    ).reshape((*omega.shape[:-1], 3, 3))


def _SO3_jac_left(theta: jax.Array, rotation_matrix: jax.Array) -> jax.Array:
    """Compute the left jacobian for the given theta and rotation matrix.

    This function calculates the left jacobian, which is used in various geometric transformations.
    It handles both small and large theta values using different computation methods.

    Args:
        theta (jax.Array): The input angle(s) in axis-angle representation.
        rotation_matrix (jax.Array): The corresponding rotation matrix.

    Returns:
        jax.Array: A 3x3 matrix (or batch of 3x3 matrices) representing the left jacobian.
    """
    theta_squared = jnp.sum(jnp.square(theta), axis=-1)
    use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

    # Shim to avoid NaNs in jnp.where branches, which cause failures for
    # reverse-mode AD.
    theta_squared_safe = cast(
        jax.Array,
        jnp.where(
            use_taylor,
            # Any non-zero value should do here.
            jnp.ones_like(theta_squared),
            theta_squared,
        ),
    )
    del theta_squared
    theta_safe = jnp.sqrt(theta_squared_safe)

    skew_omega = _skew(theta)
    jac_left = jnp.where(
        use_taylor[..., None, None],
        rotation_matrix,
        (
            jnp.eye(3)
            + ((1.0 - jnp.cos(theta_safe)) / (theta_squared_safe))[..., None, None]
            * skew_omega
            + ((theta_safe - jnp.sin(theta_safe)) / (theta_squared_safe * theta_safe))[
                ..., None, None
            ]
            * jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
        ),
    )
    return jac_left


def _SO3_jac_left_inv(theta: jax.Array) -> jax.Array:
    """
    Compute the inverse of the left jacobian for the given theta.

    This function calculates the inverse of the left jacobian, which is used in various
    geometric transformations. It handles both small and large theta values
    using different computation methods.

    Args:
        theta (jax.Array): The input angle(s) in axis-angle representation.

    Returns:
        jax.Array: A 3x3 matrix (or batch of 3x3 matrices) representing the inverse left jacobian.
    """
    theta_squared = jnp.sum(jnp.square(theta), axis=-1)
    use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

    # Shim to avoid NaNs in jnp.where branches, which cause failures for
    # reverse-mode AD.
    theta_squared_safe = jnp.where(
        use_taylor,
        jnp.ones_like(theta_squared),  # Any non-zero value should do here.
        theta_squared,
    )
    del theta_squared
    theta_safe = jnp.sqrt(theta_squared_safe)
    half_theta_safe = theta_safe / 2.0

    skew_omega = _skew(theta)
    jac_left_inv = jnp.where(
        use_taylor[..., None, None],
        jnp.eye(3)
        - 0.5 * skew_omega
        + jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega) / 12.0,
        (
            jnp.eye(3)
            - 0.5 * skew_omega
            + (
                (
                    1.0
                    - theta_safe
                    * jnp.cos(half_theta_safe)
                    / (2.0 * jnp.sin(half_theta_safe))
                )
                / theta_squared_safe
            )[..., None, None]
            * jnp.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
        ),
    )
    return jac_left_inv


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
)
@jdc.pytree_dataclass
class SO3(_base.SOBase):
    """Special orthogonal group for 3D rotations. Broadcasting rules are the same as
    for numpy.

    Internal parameterization is `(qw, qx, qy, qz)`. Tangent parameterization is
    `(omega_x, omega_y, omega_z)`.
    """

    wxyz: jax.Array
    """Internal parameters. `(w, x, y, z)` quaternion. Shape should be `(*, 4)`."""

    @override
    def __repr__(self) -> str:
        wxyz = jnp.round(self.wxyz, 5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    @staticmethod
    def from_x_radians(theta: hints.Scalar) -> SO3:
        """Generates a x-axis rotation.

        Args:
            angle: X rotation, in radians.

        Returns:
            Output.
        """
        zeros = jnp.zeros_like(theta)
        return SO3.exp(jnp.stack([theta, zeros, zeros], axis=-1))

    @staticmethod
    def from_y_radians(theta: hints.Scalar) -> SO3:
        """Generates a y-axis rotation.

        Args:
            angle: Y rotation, in radians.

        Returns:
            Output.
        """
        zeros = jnp.zeros_like(theta)
        return SO3.exp(jnp.stack([zeros, theta, zeros], axis=-1))

    @staticmethod
    def from_z_radians(theta: hints.Scalar) -> SO3:
        """Generates a z-axis rotation.

        Args:
            angle: Z rotation, in radians.

        Returns:
            Output.
        """
        zeros = jnp.zeros_like(theta)
        return SO3.exp(jnp.stack([zeros, zeros, theta], axis=-1))

    @staticmethod
    def from_rpy_radians(
        roll: hints.Scalar,
        pitch: hints.Scalar,
        yaw: hints.Scalar,
    ) -> SO3:
        """Generates a transform from a set of Euler angles. Uses the ZYX mobile robot
        convention.

        Args:
            roll: X rotation, in radians. Applied first.
            pitch: Y rotation, in radians. Applied second.
            yaw: Z rotation, in radians. Applied last.

        Returns:
            Output.
        """
        return (
            SO3.from_z_radians(yaw)
            @ SO3.from_y_radians(pitch)
            @ SO3.from_x_radians(roll)
        )

    @staticmethod
    def from_quaternion_xyzw(xyzw: hints.Array) -> SO3:
        """Construct a rotation from an `xyzw` quaternion.

        Note that `wxyz` quaternions can be constructed using the default dataclass
        constructor.

        Args:
            xyzw: xyzw quaternion. Shape should be (*, 4).

        Returns:
            Output.
        """
        assert xyzw.shape[-1:] == (4,)
        return SO3(jnp.roll(xyzw, axis=-1, shift=1))

    def as_quaternion_xyzw(self) -> jax.Array:
        """Grab parameters as xyzw quaternion."""
        return jnp.roll(self.wxyz, axis=-1, shift=-1)

    def as_rpy_radians(self) -> hints.RollPitchYaw:
        """Computes roll, pitch, and yaw angles. Uses the ZYX mobile robot convention.

        Returns:
            Named tuple containing Euler angles in radians.
        """
        return hints.RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def compute_roll_radians(self) -> jax.Array:
        """Compute roll angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = jnp.moveaxis(self.wxyz, -1, 0)
        return jnp.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> jax.Array:
        """Compute pitch angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = jnp.moveaxis(self.wxyz, -1, 0)
        return jnp.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> jax.Array:
        """Compute yaw angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = jnp.moveaxis(self.wxyz, -1, 0)
        return jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    # Factory.

    @classmethod
    @override
    def identity(cls, batch_axes: jdc.Static[Tuple[int, ...]] = ()) -> SO3:
        return SO3(
            wxyz=jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0, 0.0]), (*batch_axes, 4))
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: hints.Array) -> SO3:
        assert matrix.shape[-2:] == (3, 3)

        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

        def case0(m):
            t = 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
            q = jnp.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    t,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                axis=-1,
            )
            return t, q

        def case1(m):
            t = 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
            q = jnp.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    t,
                    m[..., 2, 1] + m[..., 1, 2],
                ],
                axis=-1,
            )
            return t, q

        def case2(m):
            t = 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]
            q = jnp.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] + m[..., 1, 2],
                    t,
                ],
                axis=-1,
            )
            return t, q

        def case3(m):
            t = 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
            q = jnp.stack(
                [
                    t,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                axis=-1,
            )
            return t, q

        # Compute four cases, then pick the most precise one.
        # Probably worth revisiting this!
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[..., 2, 2] < 0
        cond1 = matrix[..., 0, 0] > matrix[..., 1, 1]
        cond2 = matrix[..., 0, 0] < -matrix[..., 1, 1]

        t = jnp.where(
            cond0,
            jnp.where(cond1, case0_t, case1_t),
            jnp.where(cond2, case2_t, case3_t),
        )
        q = jnp.where(
            cond0[..., None],
            jnp.where(cond1[..., None], case0_q, case1_q),
            jnp.where(cond2[..., None], case2_q, case3_q),
        )

        # We can also choose to branch, but this is slower.
        # t, q = jax.lax.cond(
        #     matrix[2, 2] < 0,
        #     true_fun=lambda matrix: jax.lax.cond(
        #         matrix[0, 0] > matrix[1, 1],
        #         true_fun=case0,
        #         false_fun=case1,
        #         operand=matrix,
        #     ),
        #     false_fun=lambda matrix: jax.lax.cond(
        #         matrix[0, 0] < -matrix[1, 1],
        #         true_fun=case2,
        #         false_fun=case3,
        #         operand=matrix,
        #     ),
        #     operand=matrix,
        # )

        return SO3(wxyz=q * 0.5 / jnp.sqrt(t[..., None]))

    # Accessors.

    @override
    def as_matrix(self) -> jax.Array:
        norm_sq = jnp.sum(jnp.square(self.wxyz), axis=-1, keepdims=True)
        q = self.wxyz * jnp.sqrt(2.0 / norm_sq)  # (*, 4)
        q_outer = jnp.einsum("...i,...j->...ij", q, q)  # (*, 4, 4)
        return jnp.stack(
            [
                1.0 - q_outer[..., 2, 2] - q_outer[..., 3, 3],
                q_outer[..., 1, 2] - q_outer[..., 3, 0],
                q_outer[..., 1, 3] + q_outer[..., 2, 0],
                q_outer[..., 1, 2] + q_outer[..., 3, 0],
                1.0 - q_outer[..., 1, 1] - q_outer[..., 3, 3],
                q_outer[..., 2, 3] - q_outer[..., 1, 0],
                q_outer[..., 1, 3] - q_outer[..., 2, 0],
                q_outer[..., 2, 3] + q_outer[..., 1, 0],
                1.0 - q_outer[..., 1, 1] - q_outer[..., 2, 2],
            ],
            axis=-1,
        ).reshape(*q.shape[:-1], 3, 3)

    @override
    def parameters(self) -> jax.Array:
        return self.wxyz

    # Operations.

    @override
    def apply(self, target: hints.Array) -> jax.Array:
        assert target.shape[-1:] == (3,)
        self, target = broadcast_leading_axes((self, target))

        # Compute using quaternion multiplys.
        padded_target = jnp.concatenate(
            [jnp.zeros((*self.get_batch_axes(), 1)), target], axis=-1
        )
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[..., 1:]

    @override
    def multiply(self, other: SO3) -> SO3:
        # Original implementation:
        #
        # w0, x0, y0, z0 = jnp.moveaxis(self.wxyz, -1, 0)
        # w1, x1, y1, z1 = jnp.moveaxis(other.wxyz, -1, 0)
        # return SO3(
        #     wxyz=jnp.stack(
        #         [
        #             -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
        #             x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
        #             -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
        #             x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
        #         ],
        #         axis=-1,
        #     )
        # )
        #
        # This is great/fine/standard, but there are a lot of operations. This
        # puts a lot of burden on the JIT compiler.
        #
        # Here's another implementation option. The JIT time is much faster, but the
        # runtime is ~10% slower:
        #
        # inds = jnp.array([0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0])
        # signs = jnp.array([1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1])
        # return SO3(
        #     wxyz=jnp.einsum(
        #         "...ij,...j->...i",
        #         (self.wxyz[..., inds] * signs).reshape((*self.wxyz.shape, 4)),
        #         other.wxyz,
        #     )
        # )
        #
        # For pose graph optimization on the sphere2500 dataset, the following
        # speeds up *overall* JIT times by over 35%, without any runtime
        # penalties.

        # Hamilton product constants.
        terms_i = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
        terms_j = jnp.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 0, 1, 3], [3, 0, 2, 1]])
        signs = jnp.array(
            [
                [1, -1, -1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, -1],
                [1, 1, 1, -1],
            ]
        )

        # Compute all components at once
        q_outer = jnp.einsum("...i,...j->...ij", self.wxyz, other.wxyz)
        return SO3(
            jnp.sum(
                signs * q_outer[..., terms_i, terms_j],
                axis=-1,
            )
        )

    @classmethod
    @override
    def exp(cls, tangent: hints.Array) -> SO3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583

        assert tangent.shape[-1:] == (3,)

        theta_squared = jnp.sum(jnp.square(tangent), axis=-1)
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        safe_theta = jnp.sqrt(
            jnp.where(
                use_taylor,
                # Any constant value should do here.
                jnp.ones_like(theta_squared),
                theta_squared,
            )
        )
        safe_half_theta = 0.5 * safe_theta

        real_factor = jnp.where(
            use_taylor,
            1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0,
            jnp.cos(safe_half_theta),
        )

        imaginary_factor = jnp.where(
            use_taylor,
            0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0,
            jnp.sin(safe_half_theta) / safe_theta,
        )

        return SO3(
            wxyz=jnp.concatenate(
                [
                    real_factor[..., None],
                    imaginary_factor[..., None] * tangent,
                ],
                axis=-1,
            )
        )

    @override
    def log(self) -> jax.Array:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w = self.wxyz[..., 0]
        norm_sq = jnp.sum(jnp.square(self.wxyz[..., 1:]), axis=-1)
        use_taylor = norm_sq < get_epsilon(norm_sq.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        norm_safe = jnp.sqrt(
            jnp.where(
                use_taylor,
                1.0,  # Any non-zero value should do here.
                norm_sq,
            )
        )
        w_safe = jnp.where(use_taylor, w, 1.0)
        atan_n_over_w = jnp.arctan2(
            jnp.where(w < 0, -norm_safe, norm_safe),
            jnp.abs(w),
        )
        atan_factor = jnp.where(
            use_taylor,
            2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3,
            jnp.where(
                jnp.abs(w) < get_epsilon(w.dtype),
                jnp.where(w > 0, 1.0, -1.0) * jnp.pi / norm_safe,
                2.0 * atan_n_over_w / norm_safe,
            ),
        )

        return atan_factor[..., None] * self.wxyz[..., 1:]

    @override
    def adjoint(self) -> jax.Array:
        return self.as_matrix()

    @override
    def inverse(self) -> SO3:
        # Negate complex terms.
        return SO3(wxyz=self.wxyz * jnp.array([1, -1, -1, -1]))

    @override
    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / jnp.linalg.norm(self.wxyz, axis=-1, keepdims=True))

    @override
    def jlog(self) -> jax.Array:
        # Reference:
        # Equations (144, 147, 174) from Micro-Lie theory:
        # > https://arxiv.org/pdf/1812.01537
        V_inv = _SO3_jac_left_inv(self.log())
        return jnp.swapaxes(V_inv, -1, -2)  # Transpose the last two dimensions

    @classmethod
    @override
    def sample_uniform(
        cls, key: jax.Array, batch_axes: jdc.Static[Tuple[int, ...]] = ()
    ) -> SO3:
        # Uniformly sample over S^3.
        # > Reference: http://planning.cs.uiuc.edu/node198.html
        u1, u2, u3 = jnp.moveaxis(
            jax.random.uniform(
                key=key,
                shape=(*batch_axes, 3),
                minval=jnp.zeros(3),
                maxval=jnp.array([1.0, 2.0 * jnp.pi, 2.0 * jnp.pi]),
            ),
            -1,
            0,
        )
        a = jnp.sqrt(1.0 - u1)
        b = jnp.sqrt(u1)

        return SO3(
            wxyz=jnp.stack(
                [
                    a * jnp.sin(u2),
                    a * jnp.cos(u2),
                    b * jnp.sin(u3),
                    b * jnp.cos(u3),
                ],
                axis=-1,
            )
        )
