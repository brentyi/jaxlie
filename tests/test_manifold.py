"""Test manifold helpers.
"""

from typing import Type

from utils import (assert_arrays_close, assert_transforms_close,
                   general_group_test, sample_transform)

import jaxlie


@general_group_test
def test_rplus_rminus(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check closure property."""
    T_wa = sample_transform(Group)
    T_wb = sample_transform(Group)
    T_ab = T_wa.inverse() @ T_wb

    assert_transforms_close(jaxlie.manifold.rplus(T_wa, T_ab.log()), T_wb)
    assert_arrays_close(jaxlie.manifold.rminus(T_wa, T_wb), T_ab.log())
