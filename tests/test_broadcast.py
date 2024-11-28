"""Shape tests for broadcasting."""

from typing import Tuple, Type

import numpy as onp
from utils import (
    general_group_test,
    sample_transform,
)

import jaxlie


@general_group_test
def test_broadcast_multiply(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    if batch_axes == ():
        return

    T = sample_transform(Group, batch_axes) @ sample_transform(Group)
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes) @ sample_transform(Group, batch_axes=(1,))
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes) @ sample_transform(
        Group, batch_axes=(1,) * len(batch_axes)
    )
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group) @ sample_transform(Group, batch_axes)
    assert T.get_batch_axes() == batch_axes

    T = sample_transform(Group, batch_axes=(1,)) @ sample_transform(Group, batch_axes)
    assert T.get_batch_axes() == batch_axes


@general_group_test
def test_broadcast_apply(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    if batch_axes == ():
        return

    T = sample_transform(Group, batch_axes)
    points = onp.random.randn(Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes)
    points = onp.random.randn(1, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes)
    points = onp.random.randn(*((1,) * len(batch_axes)), Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group)
    points = onp.random.randn(*batch_axes, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)

    T = sample_transform(Group, batch_axes=(1,))
    points = onp.random.randn(*batch_axes, Group.space_dim)
    assert (T @ points).shape == (*batch_axes, Group.space_dim)
