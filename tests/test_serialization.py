"""Test transform serialization, for things like saving calibrated transforms to
disk."""

from typing import Tuple, Type

import flax.serialization
from utils import assert_transforms_close, general_group_test, sample_transform

import jaxlie


@general_group_test
def test_serialization_state_dict_bijective(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check bijectivity of state dict representation conversions."""
    T = sample_transform(Group, batch_axes)
    T_recovered = flax.serialization.from_state_dict(
        T, flax.serialization.to_state_dict(T)
    )
    assert_transforms_close(T, T_recovered)


@general_group_test
def test_serialization_bytes_bijective(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Check bijectivity of byte representation conversions."""
    T = sample_transform(Group, batch_axes)
    T_recovered = flax.serialization.from_bytes(T, flax.serialization.to_bytes(T))
    assert_transforms_close(T, T_recovered)
