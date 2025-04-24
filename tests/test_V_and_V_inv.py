"""Test _V_inv and _V functions for SE2 and SO3 groups. These are equivalent to
the left Jacobian of the Lie group and its inverse respectively. We can compute
these using autodiff."""

from typing import Callable, Dict, Tuple, Type

import jax
import jax.numpy as jnp
import jaxlie

from utils import assert_arrays_close, general_group_test, sample_transform

# Run all tests with double-precision
jax.config.update("jax_enable_x64", True)

# Dictionary mapping group classes to their corresponding _V_inv functions
_V_FUNCS: Dict[Type[jaxlie.MatrixLieGroup], Tuple[Callable, Callable]] = {
    jaxlie.SE2: (jax.jit(jaxlie._se2._SE2_V), jax.jit(jaxlie._se2._SE2_V_inv)),
    jaxlie.SO3: (jax.jit(jaxlie._so3._SO3_V), jax.jit(jaxlie._so3._SO3_V_inv)),
}


# Define function to compute autodiff version of _V_inv at identity
@jax.jit
def compute_autodiff_V_inv(transform: jaxlie.MatrixLieGroup):
    def wrapped_function(tau):
        return type(transform).exp(tau).multiply(transform).log()

    jacobian = jax.jacobian(wrapped_function)(jnp.zeros(transform.tangent_dim))
    return jacobian


compute_autodiff_V_inv_vmap = jax.jit(jax.vmap(compute_autodiff_V_inv))


@general_group_test
def test_V_inv_autodiff(
    Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]
):
    """Test _V_inv against automatic differentiation at identity."""
    # Skip groups that don't have _V_inv function
    if Group not in _V_FUNCS:
        return

    # Create identity transform with appropriate batch shape
    transform = sample_transform(Group, batch_axes)
    theta = transform.log()

    # For SE2, the input should be just the rotation part
    if Group is jaxlie.SE2:
        theta = theta[..., 2:3]

    # Compute _V_inv using the implementation
    V, V_inv = _V_FUNCS[Group]
    analytical_V_inv = V_inv(theta)
    if Group is jaxlie.SO3:
        analytical_V = V(theta, transform.as_matrix())
    else:
        analytical_V = V(theta)
    assert_arrays_close(
        jnp.linalg.inv(analytical_V_inv), analytical_V, rtol=1e-5, atol=1e-5
    )

    # Compute _V_inv using autodiff
    autodiff_V_inv = (
        compute_autodiff_V_inv(transform)
        if len(transform.get_batch_axes()) == 0
        else compute_autodiff_V_inv_vmap(transform)
    )

    # For SE2, the output should be just the translation part
    if Group is jaxlie.SE2:
        autodiff_V_inv = autodiff_V_inv[..., :2, :2]

    # Compare the results
    assert_arrays_close(analytical_V_inv, autodiff_V_inv, rtol=1e-5, atol=1e-5)
