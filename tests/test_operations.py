"""Tests for general operation definitions."""

from typing import Tuple, Type

import numpy as onp
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import jaxlie


@general_group_test
def test_sample_uniform_valid(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that sample_uniform() returns valid group members."""
    T = sample_transform(Group, batch_axes)  # Calls sample_uniform under the hood.
    assert_transforms_close(T, T.normalize())


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so2_from_to_radians_bijective(_random_module):
    """Check that we can convert from and to radians."""
    radians = onp.random.uniform(low=-onp.pi, high=onp.pi)
    assert_arrays_close(jaxlie.SO2.from_radians(radians).as_radians(), radians)


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_xyzw_bijective(_random_module):
    """Check that we can convert between xyzw and wxyz quaternions."""
    T = sample_transform(jaxlie.SO3)
    assert_transforms_close(T, jaxlie.SO3.from_quaternion_xyzw(T.as_quaternion_xyzw()))


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_so3_rpy_bijective(_random_module):
    """Check that we can convert between quaternions and Euler angles."""
    T = sample_transform(jaxlie.SO3)
    assert_transforms_close(T, jaxlie.SO3.from_rpy_radians(*T.as_rpy_radians()))


@general_group_test
def test_log_exp_bijective(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check 1-to-1 mapping for log <=> exp operations."""
    transform = sample_transform(Group, batch_axes)

    tangent = transform.log()
    assert tangent.shape == (*batch_axes, Group.tangent_dim)

    exp_transform = Group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    assert_arrays_close(tangent, exp_transform.log())


@general_group_test
def test_inverse_bijective(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check inverse of inverse."""
    transform = sample_transform(Group, batch_axes)
    assert_transforms_close(transform, transform.inverse().inverse())


@general_group_test
def test_matrix_bijective(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check that we can convert to and from matrices."""
    transform = sample_transform(Group, batch_axes)
    assert_transforms_close(transform, Group.from_matrix(transform.as_matrix()))


@general_group_test
def test_adjoint(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check adjoint definition."""
    transform = sample_transform(Group, batch_axes)
    omega = onp.random.randn(*batch_axes, Group.tangent_dim)
    assert_transforms_close(
        transform @ Group.exp(omega),
        Group.exp(onp.einsum("...ij,...j->...i", transform.adjoint(), omega))
        @ transform,
    )


@general_group_test
def test_repr(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Smoke test for __repr__ implementations."""
    transform = sample_transform(Group, batch_axes)
    print(transform)


@general_group_test
def test_apply(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check group action interfaces."""
    T_w_b = sample_transform(Group, batch_axes)
    p_b = onp.random.randn(*batch_axes, Group.space_dim)

    if Group.matrix_dim == Group.space_dim:
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            onp.einsum("...ij,...j->...i", T_w_b.as_matrix(), p_b),
        )
    else:
        # Homogeneous coordinates.
        assert Group.matrix_dim == Group.space_dim + 1
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            onp.einsum(
                "...ij,...j->...i",
                T_w_b.as_matrix(),
                onp.concatenate([p_b, onp.ones_like(p_b[..., :1])], axis=-1),
            )[..., :-1],
        )


@general_group_test
def test_multiply(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check multiply interfaces."""
    T_w_b = sample_transform(Group, batch_axes)
    T_b_a = sample_transform(Group, batch_axes)
    assert_arrays_close(
        onp.einsum(
            "...ij,...jk->...ik", T_w_b.as_matrix(), T_w_b.inverse().as_matrix()
        ),
        onp.broadcast_to(
            onp.eye(Group.matrix_dim), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
    )
    assert_arrays_close(
        onp.einsum(
            "...ij,...jk->...ik", T_w_b.as_matrix(), jnp.linalg.inv(T_w_b.as_matrix())
        ),
        onp.broadcast_to(
            onp.eye(Group.matrix_dim), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
    )
    assert_transforms_close(T_w_b @ T_b_a, Group.multiply(T_w_b, T_b_a))
