from typing import List, Type

import jax
import numpy as onp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from jax.config import config

import jaxlie

# Run all tests with double-precision
config.update("jax_enable_x64", True)


##############
# Utilities
##############


def sample_transform(Group: Type[jaxlie.MatrixLieGroup]) -> jaxlie.MatrixLieGroup:
    """Sample a random transform from a group."""
    i = onp.random.randint(0, 3)
    if i == 0:
        tangent = onp.random.randn(Group.tangent_dim) * 1e-8
    else:
        tangent = onp.random.randn(Group.tangent_dim)
    return Group.exp(tangent)


def general_group_test(f):
    """Decorator for defining tests that run on all group types."""
    # Disable timing consistency tests (first, JITed run will be faster)
    f = settings(deadline=None)(f)

    # Add _random_module parameter
    f = given(_random_module=st.randoms())(f)

    # Parametrize tests wtih each group type
    f = pytest.mark.parametrize(
        "Group",
        [
            jaxlie.SO2,
            jaxlie.SE2,
            jaxlie.SO3,
            jaxlie.SE3,
        ],
    )(f)
    return f


def assert_transforms_close(a: jaxlie.MatrixLieGroup, b: jaxlie.MatrixLieGroup):
    """Make sure two matrices are equivalent."""
    assert_arrays_close(a.as_matrix(), b.as_matrix())
    assert_arrays_close(a.parameters, b.parameters)


def assert_arrays_close(*arrays: jnp.array):
    """Make sure two arrays are close. (and not NaN)"""
    for array1, array2 in zip(arrays[1:], arrays[:-1]):
        onp.testing.assert_allclose(array1, array2, rtol=1e-8, atol=1e-8)
        assert not onp.any(onp.isnan(array1))
        assert not onp.any(onp.isnan(array2))


##############
# Tests: group properties
##############


@general_group_test
def test_closure(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check closure property."""

    transform_a = sample_transform(Group)
    transform_b = sample_transform(Group)

    composed = transform_a @ transform_b
    assert_transforms_close(composed, composed.normalize())
    composed = transform_b @ transform_a
    assert_transforms_close(composed, composed.normalize())
    composed = Group.product(transform_a, transform_b)
    assert_transforms_close(composed, composed.normalize())
    composed = Group.product(transform_b, transform_a)
    assert_transforms_close(composed, composed.normalize())


@general_group_test
def test_identity(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check inverse property."""

    transform = sample_transform(Group)
    identity = Group.identity()
    assert_transforms_close(transform, identity @ transform)
    assert_transforms_close(transform, transform @ identity)
    assert_arrays_close(
        transform.as_matrix(), identity.as_matrix() @ transform.as_matrix()
    )
    assert_arrays_close(
        transform.as_matrix(), transform.as_matrix() @ identity.as_matrix()
    )


@general_group_test
def test_inverse(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check inverse property."""

    transform = sample_transform(Group)
    identity = Group.identity()
    assert_transforms_close(identity, transform @ transform.inverse())
    assert_transforms_close(identity, transform.inverse() @ transform)
    assert_transforms_close(identity, Group.product(transform, transform.inverse()))
    assert_transforms_close(identity, Group.product(transform.inverse(), transform))
    assert_arrays_close(
        onp.eye(Group.matrix_dim),
        transform.as_matrix() @ transform.inverse().as_matrix(),
    )
    assert_arrays_close(
        onp.eye(Group.matrix_dim),
        transform.inverse().as_matrix() @ transform.as_matrix(),
    )


@general_group_test
def test_associative(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check associativity property."""

    transform_a = sample_transform(Group)
    transform_b = sample_transform(Group)
    transform_c = sample_transform(Group)
    assert_transforms_close(
        (transform_a @ transform_b) @ transform_c,
        transform_a @ (transform_b @ transform_c),
    )


##############
# Tests: invertability
##############


@general_group_test
def test_log_exp_bijective(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check 1-to-1 mapping for log <=> exp operations."""
    transform = sample_transform(Group)

    tangent = transform.log()
    assert tangent.shape == (Group.tangent_dim,)

    exp_transform = Group.exp(tangent)
    assert_transforms_close(transform, exp_transform)
    assert_arrays_close(tangent, exp_transform.log())


@general_group_test
def test_inverse_bijective(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check inverse of inverse."""
    transform = sample_transform(Group)
    assert_transforms_close(transform, transform.inverse().inverse())


##############
# Tests: other general ops
##############


@general_group_test
def test_repr(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Smoke test for __repr__ implementations."""
    transform = sample_transform(Group)
    print(transform)


@general_group_test
def test_apply(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check product interfaces."""
    T_w_b = sample_transform(Group)
    p_b = onp.random.randn(Group.space_dim)

    if Group.matrix_dim == Group.space_dim:
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            T_w_b.as_matrix() @ p_b,
        )
    else:
        # Homogeneous coordinates
        assert Group.matrix_dim == Group.space_dim + 1
        assert_arrays_close(
            T_w_b @ p_b,
            T_w_b.apply(p_b),
            (T_w_b.as_matrix() @ onp.append(p_b, 1.0))[:-1],
        )


@general_group_test
def test_product(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check product interfaces."""
    T_w_b = sample_transform(Group)
    T_b_a = sample_transform(Group)
    assert_transforms_close(T_w_b @ T_b_a, Group.product(T_w_b, T_b_a))


##############
# Tests: simple direction checks
##############


@given(_random_module=st.randoms())
def test_se2_translation(_random_module):
    """Simple test for SE(2) translation terms."""
    translation = onp.random.randn(2)
    T = jaxlie.SE2.from_rotation_and_translation(
        rotation=jaxlie.SO2.identity(),
        translation=translation,
    )
    assert_arrays_close(T @ translation, translation * 2)


@given(_random_module=st.randoms())
def test_se3_translation(_random_module):
    """Simple test for SE(3) translation terms."""
    translation = onp.random.randn(3)
    T = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.identity(),
        translation=translation,
    )
    assert_arrays_close(T @ translation, translation * 2)


def test_se2_rotation():
    """Simple test for SE(2) rotation terms."""
    T_w_b = jaxlie.SE2.from_rotation_and_translation(
        rotation=jaxlie.SO2.from_radians(onp.pi / 2.0),
        translation=onp.zeros(2),
    )
    p_b = onp.array([1.0, 0.0])
    p_w = onp.array([0.0, 1.0])
    assert_arrays_close(T_w_b @ p_b, p_w)


def test_se3_rotation():
    """Simple test for SE(3) rotation terms."""
    T_w_b = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_rpy_radians(onp.pi / 2.0, 0.0, 0.0),
        translation=onp.zeros(3),
    )
    p_b = onp.array([0.0, 1.0, 0.0])
    p_w = onp.array([0.0, 0.0, -1.0])
    assert_arrays_close(T_w_b @ p_b, p_w)
