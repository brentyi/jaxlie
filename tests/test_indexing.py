"""Tests with explicit examples."""
import jaxlie
import numpy as onp
from hypothesis import given, settings
from hypothesis import strategies as st

from utils import assert_transforms_close, sample_transform


def index_via_parameters(T: jaxlie.SE3, *key):
    return jaxlie.SE3(T.wxyz_xyz[key])

@given(_random_module=st.random_module())
@settings(deadline=None)
def test_se3_indexing(_random_module):
    """Compare SE3 composition in matrix form vs compact form."""
    T1 = sample_transform(jaxlie.SE3, batch_axes=(10, 20))

    T2 = T1[3]
    T2b = jaxlie.SE3(T1.wxyz_xyz[3])
    assert_transforms_close(T2, T2b)

    T2 = T1[:3, -10]
    T2b = jaxlie.SE3(T1.wxyz_xyz[:3, -10])
    assert_transforms_close(T2, T2b)

    T2 = T1[:4, :10]
    T2b = jaxlie.SE3(T1.wxyz_xyz[:4, :10])
    assert_transforms_close(T2, T2b)

    T2 = T1[:5, ..., :]
    T2b = jaxlie.SE3(T1.wxyz_xyz[:5, ..., :])
    assert_transforms_close(T2, T2b)

    T2 = T1[:, 10:40, :]
    T2b = jaxlie.SE3(T1.wxyz_xyz[:, 10:40, :])
    assert_transforms_close(T2, T2b)

