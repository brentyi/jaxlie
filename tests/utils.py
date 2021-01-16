from typing import Callable, Type, TypeVar

import numpy as onp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from jax.config import config

import jaxlie

# Run all tests with double-precision
config.update("jax_enable_x64", True)

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


def sample_transform(Group: Type[T]) -> T:
    """Sample a random transform from a group."""
    tangent = onp.random.randn(Group.tangent_dim)
    return Group.exp(tangent)


def general_group_test(
    f: Callable[[Type[jaxlie.MatrixLieGroup]], None], max_examples: int = 100
) -> Callable[[Type[jaxlie.MatrixLieGroup]], None]:
    """Decorator for defining tests that run on all group types."""

    # Disregard unused argument
    def f_wrapped(Group: Type[jaxlie.MatrixLieGroup], _random_module):
        f(Group)

    # Disable timing check (first run requires JIT tracing and will be slower)
    f_wrapped = settings(deadline=None)(f_wrapped)

    # Add _random_module parameter
    f_wrapped = given(_random_module=st.random_module())(f_wrapped)

    # Parametrize tests with each group type
    f_wrapped = pytest.mark.parametrize(
        "Group",
        [
            jaxlie.SO2,
            jaxlie.SE2,
            jaxlie.SO3,
            jaxlie.SE3,
        ],
    )(f_wrapped)
    return f_wrapped


def assert_transforms_close(a: jaxlie.MatrixLieGroup, b: jaxlie.MatrixLieGroup):
    """Make sure two transforms are equivalent."""
    # Check matrix representation
    assert_arrays_close(a.as_matrix(), b.as_matrix())

    # Flip signs for quaternions
    # We use `jnp.asarray` here in case inputs are onp arrays and don't support `.at()`
    p1 = jnp.asarray(a.parameters)
    p2 = jnp.asarray(b.parameters)
    if isinstance(a, jaxlie.SO3):
        p1 = p1 * jnp.sign(jnp.sum(p1))
        p2 = p2 * jnp.sign(jnp.sum(p2))
    elif isinstance(a, jaxlie.SE3):
        p1 = p1.at[3:].mul(jnp.sign(jnp.sum(p1[3:])))
        p2 = p2.at[3:].mul(jnp.sign(jnp.sum(p2[3:])))

    # Make sure parameters are equal
    assert_arrays_close(p1, p2)


def assert_arrays_close(*arrays: jnp.array):
    """Make sure two arrays are close. (and not NaN)"""
    for array1, array2 in zip(arrays[:-1], arrays[1:]):
        onp.testing.assert_allclose(array1, array2, rtol=1e-8, atol=1e-8)
        assert not onp.any(onp.isnan(array1))
        assert not onp.any(onp.isnan(array2))
