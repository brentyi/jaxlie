"""Compare forward- and reverse-mode Jacobians with a numerical Jacobian."""

from functools import lru_cache
from typing import Callable, Type, cast

import jax
import numpy as onp
from jax import numpy as jnp
from utils import assert_arrays_close, general_group_test, jacnumerical

import jaxlie

# We cache JITed Jacobians to improve runtime.
cached_jacfwd = lru_cache(maxsize=None)(
    lambda f: jax.jit(jax.jacfwd(f, argnums=1), static_argnums=0)
)
cached_jacrev = lru_cache(maxsize=None)(
    lambda f: jax.jit(jax.jacrev(f, argnums=1), static_argnums=0)
)
cached_jit = lru_cache(maxsize=None)(jax.jit)


def _assert_jacobians_close(
    Group: Type[jaxlie.MatrixLieGroup],
    f: Callable[[Type[jaxlie.MatrixLieGroup], jax.Array], jax.Array],
    primal: jaxlie.hints.Array,
) -> None:
    jacobian_fwd = cached_jacfwd(f)(Group, primal)
    jacobian_rev = cached_jacrev(f)(Group, primal)
    jacobian_numerical = jacnumerical(
        lambda primal: cached_jit(f, static_argnums=0)(Group, primal)
    )(primal)

    assert_arrays_close(jacobian_fwd, jacobian_rev)
    assert_arrays_close(jacobian_fwd, jacobian_numerical, rtol=5e-4, atol=5e-4)


# Exp tests.
def _exp(Group: Type[jaxlie.MatrixLieGroup], generator: jax.Array) -> jax.Array:
    return cast(jax.Array, Group.exp(generator).parameters())


def test_so3_nan():
    """Make sure we don't get NaNs from division when w == 0.

    https://github.com/brentyi/jaxlie/issues/9"""

    @jax.jit
    @jax.grad
    def func(x):
        return jaxlie.SO3.exp(x).log().sum()

    for omega in jnp.eye(3) * jnp.pi:
        a = jnp.array(omega, dtype=jnp.float32)
        assert all(onp.logical_not(onp.isnan(func(a))))


@general_group_test
def test_exp_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that exp Jacobians are consistent, with randomly sampled transforms."""
    generator = onp.random.randn(Group.tangent_dim)
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


@general_group_test
def test_exp_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that exp Jacobians are consistent, with transforms close to the
    identity."""
    generator = onp.random.randn(Group.tangent_dim) * 1e-6
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


# Log tests.
def _log(Group: Type[jaxlie.MatrixLieGroup], params: jax.Array) -> jax.Array:
    return Group.log(Group(params))


@general_group_test
def test_log_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that log Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


@general_group_test
def test_log_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that log Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


# Adjoint tests.
def _adjoint(Group: Type[jaxlie.MatrixLieGroup], params: jax.Array) -> jax.Array:
    return cast(jax.Array, Group(params).adjoint().flatten())


@general_group_test
def test_adjoint_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that adjoint Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


@general_group_test
def test_adjoint_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that adjoint Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


# Apply tests.
def _apply(Group: Type[jaxlie.MatrixLieGroup], params: jax.Array) -> jax.Array:
    return Group(params) @ onp.ones(Group.space_dim)


@general_group_test
def test_apply_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that apply Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


@general_group_test
def test_apply_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that apply Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


# Multiply tests.
def _multiply(Group: Type[jaxlie.MatrixLieGroup], params: jax.Array) -> jax.Array:
    return cast(jax.Array, (Group(params) @ Group(params)).parameters())


@general_group_test
def test_multiply_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that multiply Jacobians are consistent, with randomly sampled
    transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


@general_group_test
def test_multiply_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that multiply Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


# Inverse tests.
def _inverse(Group: Type[jaxlie.MatrixLieGroup], params: jax.Array) -> jax.Array:
    return cast(jax.Array, Group(params).inverse().parameters())


@general_group_test
def test_inverse_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that inverse Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)


@general_group_test
def test_inverse_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that inverse Jacobians are consistent, with transforms close to the
    identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)
