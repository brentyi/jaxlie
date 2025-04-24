import time
from functools import partial
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import jaxlie

from utils import assert_arrays_close, general_group_test, sample_transform


def autodiff_jlog(group_element) -> jnp.ndarray:
    """
    Compute the Jacobian of the logarithm map for a Lie group element using automatic differentiation.

    Args:
        group_element (Union[SO2, SO3, SE2, SE3]): A Lie group element.

    Returns:
        jnp.ndarray: The Jacobian matrix.
    """

    def wrapped_function(tau):
        Group = type(group_element)
        return (group_element @ Group.exp(tau)).log()

    return jax.jacobian(wrapped_function)(jnp.zeros(group_element.tangent_dim))


def analytical_jlog(group_element) -> jnp.ndarray:
    """
    Analytical computation of the Jacobian of the logarithm map for a Lie group element.

    Args:
        group_element (Union[SO2, SO3, SE2, SE3]): A Lie group element.

    Returns:
        jnp.ndarray: The Jacobian matrix.
    """
    return group_element.jlog()


@general_group_test
def test_jlog_accuracy(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check accuracy of analytical jlog against autodiff jlog."""
    transform = sample_transform(Group, batch_axes)

    # Create jitted versions of both functions.
    jitted_autodiff = jax.jit(autodiff_jlog)
    jitted_analytical = jax.jit(analytical_jlog)

    # Get results from both implementations.
    result_analytical = jitted_analytical(transform)
    result_autodiff = jitted_autodiff(transform)

    # Compare results with appropriate tolerance.
    assert_arrays_close(result_analytical, result_autodiff, rtol=1e-5, atol=1e-5)


@partial(general_group_test, max_examples=1)
def test_jlog_runtime(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Compare runtime of analytical jlog and autodiff jlog."""
    if Group is jaxlie.SO2:
        # Skip SO(2) since it has a trivial Jacobian.
        return

    transform = sample_transform(Group, batch_axes)

    # JIT compile both functions.
    jitted_autodiff = jax.jit(autodiff_jlog)
    jitted_analytical = jax.jit(analytical_jlog)

    # Warm-up run to ensure compilation happens before timing.
    jax.block_until_ready(jitted_autodiff(transform))
    jax.block_until_ready(jitted_analytical(transform))

    # Create a new transform for timing.
    num_runs = 30

    # Time autodiff implementation.
    times = []
    for _ in range(num_runs):
        transform = jax.block_until_ready(sample_transform(Group, batch_axes))
        start = time.perf_counter()
        result = jitted_autodiff(transform)
        result = jax.block_until_ready(result)  # Wait for all operations to complete.
        times.append(time.perf_counter() - start)
    autodiff_runtime = min(times) * 1000  # Convert to ms.

    # Time analytical implementation.
    times = []
    for _ in range(num_runs):
        transform = jax.block_until_ready(sample_transform(Group, batch_axes))
        start = time.perf_counter()
        result = jitted_analytical(transform)
        result = jax.block_until_ready(result)  # Wait for all operations to complete.
        times.append(time.perf_counter() - start)
    analytical_runtime = min(times) * 1000  # Convert to ms.

    assert (
        analytical_runtime <= autodiff_runtime
    ), f"Autodiff jlog is slower than analytical jlog: {analytical_runtime:.2f}ms vs {autodiff_runtime:.2f}ms"
