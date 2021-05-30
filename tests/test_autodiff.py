"""Compare forward- and reverse-mode Jacobians with a numerical Jacobian.
"""

from typing import Callable, Dict, Tuple, Type, cast

import jax
import numpy as onp
from jax import numpy as jnp
from utils import assert_arrays_close, general_group_test, jacnumerical

import jaxlie

# Helper methods to test + shared Jacobian helpers
# We cache JITed Jacobian helpers to improve runtime
_jacfwd_jacrev_cache: Dict[Callable, Tuple[Callable, Callable]] = {}


def _assert_jacobians_close(
    Group: Type[jaxlie.MatrixLieGroup],
    f: Callable[
        [Type[jaxlie.MatrixLieGroup], jaxlie.hints.Array], jaxlie.hints.ArrayJax
    ],
    primal: jaxlie.hints.Array,
) -> None:

    if f not in _jacfwd_jacrev_cache:
        _jacfwd_jacrev_cache[f] = (
            jax.jit(jax.jacfwd(f, argnums=1), static_argnums=0),
            jax.jit(jax.jacrev(f, argnums=1), static_argnums=0),
        )

    jacfwd, jacrev = _jacfwd_jacrev_cache[f]
    jacobian_fwd = jacfwd(Group, primal)
    jacobian_rev = jacrev(Group, primal)
    jacobian_numerical = jacnumerical(
        lambda primal: jax.jit(f, static_argnums=0)(Group, primal)
    )(primal)

    assert_arrays_close(jacobian_fwd, jacobian_rev)
    assert_arrays_close(jacobian_fwd, jacobian_numerical, rtol=5e-4, atol=5e-4)


# Exp
def _exp(
    Group: Type[jaxlie.MatrixLieGroup], generator: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return cast(jnp.ndarray, Group.exp(generator).parameters())


@general_group_test
def test_exp_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that exp Jacobians are consistent, with randomly sampled transforms."""
    generator = onp.random.randn(Group.tangent_dim)
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


@general_group_test
def test_exp_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that exp Jacobians are consistent, with transforms close to the identity."""
    generator = onp.random.randn(Group.tangent_dim) * 1e-6
    _assert_jacobians_close(Group=Group, f=_exp, primal=generator)


# Log
def _log(
    Group: Type[jaxlie.MatrixLieGroup], params: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return Group.log(Group(params))


@general_group_test
def test_log_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that log Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


@general_group_test
def test_log_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that log Jacobians are consistent, with transforms close to the identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_log, primal=params)


# Adjoint
def _adjoint(
    Group: Type[jaxlie.MatrixLieGroup], params: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return cast(jnp.ndarray, Group(params).adjoint().flatten())


@general_group_test
def test_adjoint_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that adjoint Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


@general_group_test
def test_adjoint_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that adjoint Jacobians are consistent, with transforms close to the identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_adjoint, primal=params)


# Apply
def _apply(
    Group: Type[jaxlie.MatrixLieGroup], params: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return Group(params) @ onp.ones(Group.space_dim)


@general_group_test
def test_apply_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that apply Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


@general_group_test
def test_apply_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that apply Jacobians are consistent, with transforms close to the identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_apply, primal=params)


# Multiply
def _multiply(
    Group: Type[jaxlie.MatrixLieGroup], params: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return cast(jnp.ndarray, (Group(params) @ Group(params)).parameters())


@general_group_test
def test_multiply_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that multiply Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


@general_group_test
def test_multiply_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that multiply Jacobians are consistent, with transforms close to the identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_multiply, primal=params)


# Inverse
def _inverse(
    Group: Type[jaxlie.MatrixLieGroup], params: jaxlie.hints.Array
) -> jaxlie.hints.ArrayJax:
    return cast(jnp.ndarray, Group(params).inverse().parameters())


@general_group_test
def test_inverse_random(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that inverse Jacobians are consistent, with randomly sampled transforms."""
    params = Group.exp(onp.random.randn(Group.tangent_dim)).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)


@general_group_test
def test_inverse_identity(Group: Type[jaxlie.MatrixLieGroup]):
    """Check that inverse Jacobians are consistent, with transforms close to the identity."""
    params = Group.exp(onp.random.randn(Group.tangent_dim) * 1e-6).parameters()
    _assert_jacobians_close(Group=Group, f=_inverse, primal=params)
