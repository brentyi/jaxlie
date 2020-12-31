"""Test manifold helpers.
"""
from typing import Type

import jax
import numpy as onp
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    sample_transform,
)

import jaxlie


@general_group_test
def test_rplus_rminus(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check closure property."""
    T_wa = sample_transform(Group)
    T_wb = sample_transform(Group)
    T_ab = T_wa.inverse() @ T_wb

    assert_transforms_close(jaxlie.manifold.rplus(T_wa, T_ab.log()), T_wb)
    assert_arrays_close(jaxlie.manifold.rminus(T_wa, T_wb), T_ab.log())


_rplus_automatic_jacobian = jax.jit(
    jax.jacfwd(
        jaxlie.manifold.rplus,  # Args are (transform, delta)
        argnums=1,  # Jacobian wrt delta
    )
)


@general_group_test
def test_rplus_jacobian(Group: Type[jaxlie.MatrixLieGroup], _random_module):
    """Check closure property."""
    T_wa = sample_transform(Group)

    J_ours = jaxlie.manifold.rplus_jacobian_wrt_delta_at_zero(T_wa)

    J_jacfwd = _rplus_automatic_jacobian(T_wa, onp.zeros(Group.tangent_dim))

    assert_arrays_close(J_ours.parameters, J_jacfwd.parameters)
