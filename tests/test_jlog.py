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

    def f(element):
        return element.log()

    def wrapped_function(tau):
        perturbed_element = group_element.multiply(group_element.__class__.exp(tau))
        result = f(perturbed_element)
        return result

    jacobian = jax.jacobian(wrapped_function)(jnp.zeros(group_element.tangent_dim))

    return jacobian


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

    # Create jitted versions of both functions
    jitted_autodiff = jax.jit(autodiff_jlog)
    jitted_analytical = jax.jit(analytical_jlog)

    # Get results from both implementations
    result_analytical = jitted_analytical(transform)
    result_autodiff = jitted_autodiff(transform)

    # Compare results with appropriate tolerance
    assert_arrays_close(result_analytical, result_autodiff, rtol=1e-5, atol=1e-5)


@partial(general_group_test, max_examples=10)
def test_jlog_runtime(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Compare runtime of analytical jlog and autodiff jlog."""
    if Group is jaxlie.SO2:
        # Skip SO(2) since it has a trivial Jacobian.
        return

    transform = sample_transform(Group, batch_axes)

    # JIT compile both functions
    jitted_autodiff = jax.jit(autodiff_jlog)
    jitted_analytical = jax.jit(analytical_jlog)

    # Warm-up run to ensure compilation happens before timing
    result_autodiff = jitted_autodiff(transform)
    result_analytical = jitted_analytical(transform)

    # Block until compilation and execution is complete
    _ = jax.block_until_ready(result_autodiff)
    _ = jax.block_until_ready(result_analytical)

    # Create a new transform for timing
    transform = sample_transform(Group, batch_axes)
    num_runs = 10

    # Time autodiff implementation
    start_time = time.perf_counter()
    result = None
    for _ in range(num_runs):
        result = jitted_autodiff(transform)
    assert result is not None
    result = jax.block_until_ready(result)  # Wait for all operations to complete
    autodiff_runtime = (
        (time.perf_counter() - start_time) / num_runs
    ) * 1000  # Convert to ms

    # Time analytical implementation
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = jitted_analytical(transform)
    result = jax.block_until_ready(result)  # Wait for all operations to complete
    analytical_runtime = (
        (time.perf_counter() - start_time) / num_runs
    ) * 1000  # Convert to ms

    assert (
        analytical_runtime <= autodiff_runtime * 2.0
    ), f"Autodiff jlog is slower than analytical jlog by more than 2x: {analytical_runtime:.2f}ms vs {autodiff_runtime:.2f}ms"
