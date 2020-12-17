from typing import Type

import jaxlie
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
    composed = Group.multiply(transform_a, transform_b)
    assert_transforms_close(composed, composed.normalize())
    composed = Group.multiply(transform_b, transform_a)
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
    assert_transforms_close(identity, Group.multiply(transform, transform.inverse()))
    assert_transforms_close(identity, Group.multiply(transform.inverse(), transform))
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


@general_group_test
def test_matrix_recovery(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check that we can convert to and from matrices."""
    transform = sample_transform(Group)
    assert_transforms_close(transform, Group.from_matrix(transform.as_matrix()))


##############
# Tests: other general ops
##############


@general_group_test
def test_adjoint(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check adjoint definition."""
    transform = sample_transform(Group)
    omega = onp.random.randn(Group.tangent_dim)
    assert_transforms_close(
        transform @ Group.exp(omega),
        Group.exp(transform.adjoint() @ omega) @ transform,
    )


@general_group_test
def test_repr(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Smoke test for __repr__ implementations."""
    transform = sample_transform(Group)
    print(transform)


@general_group_test
def test_apply(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check group action interfaces."""
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
def test_multiply(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check multiply interfaces."""
    T_w_b = sample_transform(Group)
    T_b_a = sample_transform(Group)
    assert_arrays_close(
        T_w_b.as_matrix() @ T_w_b.inverse().as_matrix(), onp.eye(Group.matrix_dim)
    )
    assert_arrays_close(
        T_w_b.as_matrix() @ jnp.linalg.inv(T_w_b.as_matrix()), onp.eye(Group.matrix_dim)
    )
    assert_transforms_close(T_w_b @ T_b_a, Group.multiply(T_w_b, T_b_a))


##############
# Tests: simple transforms
##############


@given(_random_module=st.random_module())
def test_se2_translation(_random_module):
    """Simple test for SE(2) translation terms."""
    translation = onp.random.randn(2)
    T = jaxlie.SE2.from_rotation_and_translation(
        rotation=jaxlie.SO2.identity(),
        translation=translation,
    )
    assert_arrays_close(T @ translation, translation * 2)


@given(_random_module=st.random_module())
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
    p_w = onp.array([0.0, 0.0, 1.0])
    assert_arrays_close(T_w_b @ p_b, p_w)


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_se3_compose(_random_module):
    """Compare SE3 composition in matrix form vs compact form."""
    T1 = sample_transform(jaxlie.SE3)
    T2 = sample_transform(jaxlie.SE3)
    assert_arrays_close(T1.as_matrix() @ T2.as_matrix(), (T1 @ T2).as_matrix())
    assert_transforms_close(
        jaxlie.SE3.from_matrix(T1.as_matrix() @ T2.as_matrix()), T1 @ T2
    )
