"""Tests for general operation definitions.
"""

from typing import Type

import numpy as onp
from jax import numpy as jnp
from utils import (assert_arrays_close, assert_transforms_close,
                   general_group_test, sample_transform)

import jaxlie


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
