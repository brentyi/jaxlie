"""Test left jacobian functions for SO2 and SO3 groups. These are submatrices
of the left Jacobian of the Lie group and its inverse respectively. We can
check our analytical implementations against autodiff."""

from typing import Callable, Dict, Tuple, Type

import jax
import jax.numpy as jnp
import jaxlie

from utils import assert_arrays_close, general_group_test, sample_transform

# Dictionary mapping group classes to their corresponding left jacobian functions.
_V_FUNCS: Dict[Type[jaxlie.MatrixLieGroup], Tuple[Callable, Callable]] = {
    jaxlie.SE2: (
        jax.jit(jaxlie._se2._SE2_jac_left),
        jax.jit(jaxlie._se2._SE2_jac_left_inv),
    ),
    jaxlie.SO3: (
        jax.jit(jaxlie._so3._SO3_jac_left),
        jax.jit(jaxlie._so3._SO3_jac_left_inv),
    ),
}


# Autodiff versions of left jacobian functions. We could very reasonably use these
# directly in jaxlie, but the analytical versions give us a bit more control for
# things like Taylor expansion. In the future we might be able to handle that
# automatically with jet types though:
# https://docs.jax.dev/en/latest/jax.experimental.jet.html
#
# For these autodiff implementations:
# > https://arxiv.org/pdf/1812.01537
@jax.jit
def compute_autodiff_jac_left(transform: jaxlie.MatrixLieGroup):
    def left_plus(tangent_at_identity):
        Group = type(transform)
        return (Group.exp(tangent_at_identity) @ transform).log()

    # Jacobian of tangent at `transform` wrt tangent at identity.
    pullback = jax.jacrev(left_plus)(jnp.zeros(transform.tangent_dim))

    # The pushforward is the left Jacobian. This transforms tangent vectors at
    # identity to tangent vectors at `transform`.
    pushforward = jnp.linalg.inv(pullback)
    return pushforward


compute_autodiff_jac_left_vmap = jax.jit(jax.vmap(compute_autodiff_jac_left))


@jax.jit
def compute_autodiff_jac_left_inv(transform: jaxlie.MatrixLieGroup):
    def left_plus(tangent_at_identity):
        Group = type(transform)
        return (Group.exp(tangent_at_identity) @ transform).log()

    pullback = jax.jacrev(left_plus)(jnp.zeros(transform.tangent_dim))
    return pullback


compute_autodiff_jac_left_inv_vmap = jax.jit(jax.vmap(compute_autodiff_jac_left_inv))


@general_group_test
def test_jac_left_autodiff(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Test left jacobian and its inverse against automatic differentiation."""
    # Skip groups that don't have left jacobian functions.
    if Group not in _V_FUNCS:
        return

    # Create identity transform with appropriate batch shape.
    transform = sample_transform(Group, batch_axes)
    theta = transform.log()

    # For SE2, the input should be just the rotation part.
    if Group is jaxlie.SE2:
        theta = theta[..., 2:3]

    # Compute _V_inv using the implementation.
    V, V_inv = _V_FUNCS[Group]
    analytical_V_inv = V_inv(theta)
    if Group is jaxlie.SO3:
        analytical_V = V(theta, transform.as_matrix())
    else:
        analytical_V = V(theta)
    assert_arrays_close(
        jnp.linalg.inv(analytical_V_inv), analytical_V, rtol=1e-5, atol=1e-5
    )

    # Compute _V_inv using autodiff.
    autodiff_jac_left_inv = (
        compute_autodiff_jac_left_inv(transform)
        if len(transform.get_batch_axes()) == 0
        else compute_autodiff_jac_left_inv_vmap(transform)
    )
    autodiff_jac_left = (
        compute_autodiff_jac_left(transform)
        if len(transform.get_batch_axes()) == 0
        else compute_autodiff_jac_left_vmap(transform)
    )

    # For SE2, the output should be just the translation part.
    if Group is jaxlie.SE2:
        autodiff_jac_left = autodiff_jac_left[..., :2, :2]
        autodiff_jac_left_inv = autodiff_jac_left_inv[..., :2, :2]

    # Compare the results.
    assert_arrays_close(analytical_V, autodiff_jac_left, rtol=1e-5, atol=1e-5)
    assert_arrays_close(analytical_V_inv, autodiff_jac_left_inv, rtol=1e-5, atol=1e-5)
