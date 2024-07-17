import jax
import jax.numpy as jnp
from typing import Type, Tuple
import jaxlie
import time
from utils import sample_transform, assert_arrays_close, general_group_test

# @jax.jit
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
        perturbed_element = group_element.multiply(
            group_element.__class__.exp(tau)
        )
        result = f(perturbed_element)
        return result

    jacobian = jax.jacobian(wrapped_function)(
        jnp.zeros(group_element.tangent_dim)
    )

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
    jlog_jit = jax.jit(autodiff_jlog)
    jlog_analytical = analytical_jlog(transform)
    jlog_autodiff_result = jlog_jit(transform)
    assert_arrays_close(jlog_analytical, jlog_autodiff_result)

@general_group_test
def test_jlog_runtime(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Compare runtime of analytical jlog and autodiff jlog."""
    transform = sample_transform(Group, batch_axes)

    # JIT compile both functions
    jitted_autodiff = jax.jit(autodiff_jlog)
    jitted_analytical = jax.jit(analytical_jlog)

    # Warm-up run
    _ = jitted_autodiff(transform)
    _ = jitted_analytical(transform)

    transform = sample_transform(Group, batch_axes)
    # Measure runtime
    num_runs = 100

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = jax.block_until_ready(jitted_autodiff(transform))
    autodiff_runtime = ((time.perf_counter() - start_time) / num_runs) * 1000  # Convert to ms

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = jax.block_until_ready(jitted_analytical(transform))
    analytical_runtime = ((time.perf_counter() - start_time) / num_runs) * 1000  # Convert to ms

    assert analytical_runtime <= autodiff_runtime #* 1.1


# @general_group_test
# def test_jlog_compilation_and_first_call(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
#     """Compare compilation time and first call time of analytical jlog and autodiff jlog."""
#     transform = sample_transform(Group, batch_axes)

#     # Analytical jlog
#     start_time = time.perf_counter()
#     jitted_analytical = jax.jit(analytical_jlog)
#     jitted_analytical = jax.block_until_ready(jitted_analytical)
#     analytical_compilation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

#     start_time = time.perf_counter()
#     _ = jax.block_until_ready(jitted_analytical(transform))
#     analytical_first_call_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

#     # Autodiff jlog
#     start_time = time.perf_counter()
#     jitted_autodiff = jax.jit(autodiff_jlog)
#     jitted_autodiff = jax.block_until_ready(jitted_autodiff)
#     autodiff_compilation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

#     start_time = time.perf_counter()
#     _ = jax.block_until_ready(jitted_autodiff(transform))
#     autodiff_first_call_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

#     assert analytical_compilation_time < autodiff_compilation_time
#     assert analytical_first_call_time < autodiff_first_call_time