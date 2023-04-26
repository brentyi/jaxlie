"""Test manifold helpers."""
from typing import Type

import jax
import numpy as onp
import pytest
from jax import numpy as jnp
from jax import tree_util
from utils import (
    assert_arrays_close,
    assert_transforms_close,
    general_group_test,
    general_group_test_faster,
    sample_transform,
)

import jaxlie


@general_group_test
def test_rplus_rminus(Group: Type[jaxlie.MatrixLieGroup]):
    """Check rplus and rminus on random inputs."""
    T_wa = sample_transform(Group)
    T_wb = sample_transform(Group)
    T_ab = T_wa.inverse() @ T_wb

    assert_transforms_close(jaxlie.manifold.rplus(T_wa, T_ab.log()), T_wb)
    assert_arrays_close(jaxlie.manifold.rminus(T_wa, T_wb), T_ab.log())


@general_group_test
def test_rplus_jacobian(Group: Type[jaxlie.MatrixLieGroup]):
    """Check analytical rplus Jacobian.."""
    T_wa = sample_transform(Group)

    J_ours = jaxlie.manifold.rplus_jacobian_parameters_wrt_delta(T_wa)
    J_jacfwd = _rplus_jacobian_parameters_wrt_delta(T_wa)

    assert_arrays_close(J_ours, J_jacfwd)


@jax.jit
def _rplus_jacobian_parameters_wrt_delta(
    transform: jaxlie.MatrixLieGroup,
) -> jax.Array:
    # Copied from docstring for `rplus_jacobian_parameters_wrt_delta()`.
    return jax.jacfwd(
        lambda delta: jaxlie.manifold.rplus(transform, delta).parameters()
    )(onp.zeros(transform.tangent_dim))


@general_group_test_faster
def test_sgd(Group: Type[jaxlie.MatrixLieGroup]):
    def loss(transform: jaxlie.MatrixLieGroup):
        return (transform.log() ** 2).sum()

    transform = Group.exp(sample_transform(Group).log())
    original_loss = loss(transform)

    @jax.jit
    def step(t):
        return jaxlie.manifold.rplus(t, -1e-3 * jaxlie.manifold.grad(loss)(t))

    for i in range(5):
        transform = step(transform)

    assert loss(transform) < original_loss


def test_rplus_euclidean():
    assert_arrays_close(
        jaxlie.manifold.rplus(jnp.ones(2), jnp.ones(2)), 2 * jnp.ones(2)
    )


def test_rminus_auto_vmap():
    deltas = jaxlie.manifold.rminus(
        tree_util.tree_map(
            lambda *args: jnp.stack(args),
            [jaxlie.SE3.sample_uniform(jax.random.PRNGKey(0)), jaxlie.SE3.identity()],
        ),
        tree_util.tree_map(
            lambda *args: jnp.stack(args),
            [jaxlie.SE3.identity(), jaxlie.SE3.sample_uniform(jax.random.PRNGKey(0))],
        ),
    )
    assert_arrays_close(deltas[0], -deltas[1])


def test_normalize():
    container = {"key": (jaxlie.SO3(jnp.array([2.0, 0.0, 0.0, 0.0])),)}
    container_valid = {"key": (jaxlie.SO3(jnp.array([1.0, 0.0, 0.0, 0.0])),)}
    with pytest.raises(AssertionError):
        assert_transforms_close(container["key"][0], container_valid["key"][0])
    assert_transforms_close(
        jaxlie.manifold.normalize_all(container)["key"][0], container_valid["key"][0]
    )
