import random
from typing import Any, Callable, List, Type, TypeVar, cast

import jax
import numpy as onp
import pytest
import scipy.optimize
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import numpy as jnp
from jax.config import config

import jaxlie

# Run all tests with double-precision.
config.update("jax_enable_x64", True)

T = TypeVar("T", bound=jaxlie.MatrixLieGroup)


def sample_transform(Group: Type[T]) -> T:
    """Sample a random transform from a group."""
    seed = random.getrandbits(32)
    strategy = random.randint(0, 2)

    if strategy == 0:
        # Uniform sampling.
        return cast(T, Group.sample_uniform(key=jax.random.PRNGKey(seed=seed)))
    elif strategy == 1:
        # Sample from normally-sampled tangent vector.
        return cast(T, Group.exp(onp.random.randn(Group.tangent_dim)))
    elif strategy == 2:
        # Sample near identity.
        return cast(T, Group.exp(onp.random.randn(Group.tangent_dim) * 1e-7))
    else:
        assert False


def general_group_test(
    f: Callable[[Type[jaxlie.MatrixLieGroup]], None], max_examples: int = 30
) -> Callable[[Type[jaxlie.MatrixLieGroup], Any], None]:
    """Decorator for defining tests that run on all group types."""

    # Disregard unused argument.
    def f_wrapped(Group: Type[jaxlie.MatrixLieGroup], _random_module) -> None:
        f(Group)

    # Disable timing check (first run requires JIT tracing and will be slower).
    f_wrapped = settings(deadline=None, max_examples=max_examples)(f_wrapped)

    # Add _random_module parameter.
    f_wrapped = given(_random_module=st.random_module())(f_wrapped)

    # Parametrize tests with each group type.
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
    # Check matrix representation.
    assert_arrays_close(a.as_matrix(), b.as_matrix())

    # Flip signs for quaternions.
    # We use `jnp.asarray` here in case inputs are onp arrays and don't support `.at()`.
    p1 = jnp.asarray(a.parameters())
    p2 = jnp.asarray(b.parameters())
    if isinstance(a, jaxlie.SO3):
        p1 = p1 * jnp.sign(jnp.sum(p1))
        p2 = p2 * jnp.sign(jnp.sum(p2))
    elif isinstance(a, jaxlie.SE3):
        p1 = p1.at[:4].mul(jnp.sign(jnp.sum(p1[:4])))
        p2 = p2.at[:4].mul(jnp.sign(jnp.sum(p2[:4])))

    # Make sure parameters are equal.
    assert_arrays_close(p1, p2)


def assert_arrays_close(
    *arrays: jaxlie.hints.Array,
    rtol: float = 1e-8,
    atol: float = 1e-8,
):
    """Make sure two arrays are close. (and not NaN)"""
    for array1, array2 in zip(arrays[:-1], arrays[1:]):
        onp.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)
        assert not onp.any(onp.isnan(array1))
        assert not onp.any(onp.isnan(array2))


def jacnumerical(
    f: Callable[[jaxlie.hints.Array], jnp.ndarray]
) -> Callable[[jaxlie.hints.Array], jnp.ndarray]:
    """Decorator for computing numerical Jacobians of vector->vector functions."""

    def wrapped(primal: jaxlie.hints.Array) -> jnp.ndarray:
        output_dim: int
        (output_dim,) = f(primal).shape

        jacobian_rows: List[onp.ndarray] = []
        for i in range(output_dim):
            jacobian_row: onp.ndarray = scipy.optimize.approx_fprime(
                primal, lambda p: f(p)[i], epsilon=1e-5
            )
            assert jacobian_row.shape == primal.shape
            jacobian_rows.append(jacobian_row)

        return jnp.stack(jacobian_rows, axis=0)

    return wrapped
