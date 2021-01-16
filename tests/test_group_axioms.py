"""Tests for group axioms.

https://proofwiki.org/wiki/Definition:Group_Axioms
"""

from typing import Type

import numpy as onp
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import jaxlie


@general_group_test
def test_closure(Group: Type[jaxlie.MatrixLieGroup]):
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
def test_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check identity property."""
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
def test_inverse(Group: Type[jaxlie.MatrixLieGroup]):
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
def test_associative(Group: Type[jaxlie.MatrixLieGroup]):
    """Check associative property."""
    transform_a = sample_transform(Group)
    transform_b = sample_transform(Group)
    transform_c = sample_transform(Group)
    assert_transforms_close(
        (transform_a @ transform_b) @ transform_c,
        transform_a @ (transform_b @ transform_c),
    )
