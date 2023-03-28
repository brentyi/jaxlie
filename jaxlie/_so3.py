from __future__ import annotations

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from overrides import overrides
from typing_extensions import Annotated

from . import _base, hints
from .utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
)
@jdc.pytree_dataclass
class SO3(jdc.EnforcedAnnotationsMixin, _base.SOBase):
    """Special orthogonal group for 3D rotations.

    Internal parameterization is `(qw, qx, qy, qz)`. Tangent parameterization is
    `(omega_x, omega_y, omega_z)`.
    """

    # SO3-specific.

    wxyz: Annotated[
        jnp.ndarray,
        (..., 4),  # Shape.
        jnp.floating,  # Data-type.
    ]
    """Internal parameters. `(w, x, y, z)` quaternion."""

    @overrides
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
        return SO3.exp(jnp.array([theta, 0.0, 0.0]))

    @staticmethod
    def from_y_radians(theta: hints.Scalar) -> SO3:
        """Generates a y-axis rotation.

        Args:
            angle: Y rotation, in radians.

        Returns:
            Output.
        """
        return SO3.exp(jnp.array([0.0, theta, 0.0]))

    @staticmethod
    def from_z_radians(theta: hints.Scalar) -> SO3:
        """Generates a z-axis rotation.

        Args:
            angle: Z rotation, in radians.

        Returns:
            Output.
        """
        return SO3.exp(jnp.array([0.0, 0.0, theta]))

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
            xyzw: xyzw quaternion. Shape should be (4,).

        Returns:
            Output.
        """
        assert xyzw.shape == (4,)
        return SO3(jnp.roll(xyzw, shift=1))

    def as_quaternion_xyzw(self) -> jnp.ndarray:
        """Grab parameters as xyzw quaternion."""
        return jnp.roll(self.wxyz, shift=-1)

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

    def compute_roll_radians(self) -> jnp.ndarray:
        """Compute roll angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz
        return jnp.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> jnp.ndarray:
        """Compute pitch angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz
        return jnp.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> jnp.ndarray:
        """Compute yaw angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz
        return jnp.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    # Factory.

    @staticmethod
    @overrides
    def identity() -> SO3:
        return SO3(wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    @overrides
    def from_matrix(matrix: hints.Array) -> SO3:
        assert matrix.shape == (3, 3)

        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

        def case0(m):
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[2, 1] - m[1, 2],
                    t,
                    m[1, 0] + m[0, 1],
                    m[0, 2] + m[2, 0],
                ]
            )
            return t, q

        def case1(m):
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = jnp.array(
                [
                    m[0, 2] - m[2, 0],
                    m[1, 0] + m[0, 1],
                    t,
                    m[2, 1] + m[1, 2],
                ]
            )
            return t, q

        def case2(m):
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    m[1, 0] - m[0, 1],
                    m[0, 2] + m[2, 0],
                    m[2, 1] + m[1, 2],
                    t,
                ]
            )
            return t, q

        def case3(m):
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = jnp.array(
                [
                    t,
                    m[2, 1] - m[1, 2],
                    m[0, 2] - m[2, 0],
                    m[1, 0] - m[0, 1],
                ]
            )
            return t, q

        # Compute four cases, then pick the most precise one.
        # Probably worth revisiting this!
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[2, 2] < 0
        cond1 = matrix[0, 0] > matrix[1, 1]
        cond2 = matrix[0, 0] < -matrix[1, 1]

        t = jnp.where(
            cond0,
            jnp.where(cond1, case0_t, case1_t),
            jnp.where(cond2, case2_t, case3_t),
        )
        q = jnp.where(
            cond0,
            jnp.where(cond1, case0_q, case1_q),
            jnp.where(cond2, case2_q, case3_q),
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

        return SO3(wxyz=q * 0.5 / jnp.sqrt(t))

    # Accessors.

    @overrides
    def as_matrix(self) -> jnp.ndarray:
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

    @overrides
    def parameters(self) -> jnp.ndarray:
        return self.wxyz

    # Operations.

    @overrides
    def apply(self, target: hints.Array) -> jnp.ndarray:
        assert target.shape == (3,)

        # Compute using quaternion multiplys.
        padded_target = jnp.zeros(4).at[1:].set(target)
        return (self @ SO3(wxyz=padded_target) @ self.inverse()).wxyz[1:]

    @overrides
    def multiply(self, other: SO3) -> SO3:
        w0, x0, y0, z0 = self.wxyz
        w1, x1, y1, z1 = other.wxyz
        return SO3(
            wxyz=jnp.array(
                [
                    -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                    x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                    -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                    x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
                ]
            )
        )

    @staticmethod
    @overrides
    def exp(tangent: hints.Array) -> SO3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583

        assert tangent.shape == (3,)

        theta_squared = tangent @ tangent
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        safe_theta = jnp.sqrt(
            jnp.where(
                use_taylor,
                1.0,  # Any constant value should do here.
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
                    real_factor[None],
                    imaginary_factor * tangent,
                ]
            )
        )

    @overrides
    def log(self) -> jnp.ndarray:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w = self.wxyz[..., 0]
        norm_sq = self.wxyz[..., 1:] @ self.wxyz[..., 1:]
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

        return atan_factor * self.wxyz[1:]

    @overrides
    def adjoint(self) -> jnp.ndarray:
        return self.as_matrix()

    @overrides
    def inverse(self) -> SO3:
        # Negate complex terms.
        return SO3(wxyz=self.wxyz * jnp.array([1, -1, -1, -1]))

    @overrides
    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / jnp.linalg.norm(self.wxyz))

    @staticmethod
    @overrides
    def sample_uniform(key: hints.KeyArray) -> SO3:
        # Uniformly sample over S^3.
        # > Reference: http://planning.cs.uiuc.edu/node198.html
        u1, u2, u3 = jax.random.uniform(
            key=key,
            shape=(3,),
            minval=jnp.zeros(3),
            maxval=jnp.array([1.0, 2.0 * jnp.pi, 2.0 * jnp.pi]),
        )
        a = jnp.sqrt(1.0 - u1)
        b = jnp.sqrt(u1)

        return SO3(
            wxyz=jnp.array(
                [
                    a * jnp.sin(u2),
                    a * jnp.cos(u2),
                    b * jnp.sin(u3),
                    b * jnp.cos(u3),
                ]
            )
        )
