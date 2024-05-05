"""Tests for group axioms.

https://proofwiki.org/wiki/Definition:Group_Axioms
"""

from typing import Tuple, Type

import numpy as onp
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import jaxlie


@general_group_test
def test_closure(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check closure property."""
    transform_a = sample_transform(Group, batch_axes)
    transform_b = sample_transform(Group, batch_axes)

    composed = transform_a @ transform_b
    assert_transforms_close(composed, composed.normalize())
    composed = transform_b @ transform_a
    assert_transforms_close(composed, composed.normalize())
    composed = Group.multiply(transform_a, transform_b)
    assert_transforms_close(composed, composed.normalize())
    composed = Group.multiply(transform_b, transform_a)
    assert_transforms_close(composed, composed.normalize())


@general_group_test
def test_identity(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check identity property."""
    transform = sample_transform(Group, batch_axes)
    identity = Group.identity(batch_axes)
    assert_transforms_close(transform, identity @ transform)
    assert_transforms_close(transform, transform @ identity)
    assert_arrays_close(
        transform.as_matrix(),
        onp.einsum("...ij,...jk->...ik", identity.as_matrix(), transform.as_matrix()),
    )
    assert_arrays_close(
        transform.as_matrix(),
        onp.einsum("...ij,...jk->...ik", transform.as_matrix(), identity.as_matrix()),
    )


@general_group_test
def test_inverse(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check inverse property."""
    transform = sample_transform(Group, batch_axes)
    identity = Group.identity(batch_axes)
    assert_transforms_close(identity, transform @ transform.inverse())
    assert_transforms_close(identity, transform.inverse() @ transform)
    assert_transforms_close(identity, Group.multiply(transform, transform.inverse()))
    assert_transforms_close(identity, Group.multiply(transform.inverse(), transform))
    assert_arrays_close(
        onp.broadcast_to(
            onp.eye(Group.matrix_dim), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
        onp.einsum(
            "...ij,...jk->...ik",
            transform.as_matrix(),
            transform.inverse().as_matrix(),
        ),
    )
    assert_arrays_close(
        onp.broadcast_to(
            onp.eye(Group.matrix_dim), (*batch_axes, Group.matrix_dim, Group.matrix_dim)
        ),
        onp.einsum(
            "...ij,...jk->...ik",
            transform.inverse().as_matrix(),
            transform.as_matrix(),
        ),
    )


@general_group_test
def test_associative(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check associative property."""
    transform_a = sample_transform(Group, batch_axes)
    transform_b = sample_transform(Group, batch_axes)
    transform_c = sample_transform(Group, batch_axes)
    assert_transforms_close(
        (transform_a @ transform_b) @ transform_c,
        transform_a @ (transform_b @ transform_c),
    )
