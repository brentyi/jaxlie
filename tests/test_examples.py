"""Tests with explicit examples."""
import numpy as onp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from utils import assert_arrays_close, assert_transforms_close, sample_transform

import jaxlie


@settings(deadline=None)
@given(_random_module=st.random_module())
def test_se2_translation(_random_module):
    """Simple test for SE(2) translation terms."""
    translation = onp.random.randn(2)
    T = jaxlie.SE2.from_xy_theta(*translation, theta=0.0)
    assert_arrays_close(T @ translation, translation * 2)


@settings(deadline=None)
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
    T_w_b_alt = jaxlie.SE3.from_rotation(
        jaxlie.SO3.from_rpy_radians(onp.pi / 2.0, 0.0, 0.0),
    )
    p_b = onp.array([0.0, 1.0, 0.0])
    p_w = onp.array([0.0, 0.0, 1.0])
    assert_arrays_close(T_w_b @ p_b, T_w_b_alt @ p_b, p_w)


def test_so3_xyzw_basic():
    """Check that we can create an SO3 object from an xyzw quaternion."""
    assert_transforms_close(
        jaxlie.SO3.from_quaternion_xyzw(onp.array([0.0, 0.0, 0.0, 1.0])),
        jaxlie.SO3.identity(),
    )


def test_so3_xyzw_dtype_error():
    """Check that an incorrect data-type results in an AssertionError."""
    with pytest.raises(AssertionError):
        jaxlie.SO3(onp.array([1, 0, 0, 0])),


def test_so3_xyzw_shape_error():
    """Check that an incorrect shape results in an AssertionError."""
    with pytest.raises(AssertionError):
        jaxlie.SO3(onp.array([1.0, 0.0, 0.0, 0.0, 0.0]))


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
