import jax
import jax.numpy as jnp
import numpy as onp

from typing import Type, Tuple, cast
import jaxlie
from jaxlie import SE3, SO3, SE2, SO2
import time

# Assuming the utility functions are defined in a file named 'utils.py'
from utils import (
    sample_transform,
    assert_transforms_close,
    assert_arrays_close,
    jacnumerical
)


@jax.jit
def jlog(group_element) -> jnp.ndarray:
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
            group_element.__class__.exp(tau))
        result = f(perturbed_element)
        return result

    jacobian = jax.jacobian(wrapped_function)(
        jnp.zeros(group_element.tangent_dim))

    return jacobian


def test_jlog_accuracy(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Check accuracy of analytical jlog against autodiff jlog."""
    transform = sample_transform(Group, batch_axes)

    jlog_analytical = transform.jlog()
    jlog_autodiff_result = jlog(transform)
    # print(onp.array(jlog_analytical))
    # print(onp.array(jlog_autodiff_result))
    try:
        assert_arrays_close(jlog_analytical, jlog_autodiff_result, rtol=1e-5, atol=1e-5)
        print(f"{Group.__name__} Accuracy Test: Passed")
    except AssertionError:
        print(f"{Group.__name__} Accuracy Test: Failed")


def test_jlog_compilation_and_first_call(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Compare compilation time and first call time of analytical jlog and autodiff jlog."""
    transform = sample_transform(Group, batch_axes)

    # Analytical jlog
    start_time = time.perf_counter()
    jitted_analytical = jax.jit(lambda x: x.jlog())
    analytical_compilation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    start_time = time.perf_counter()
    _ = jitted_analytical(transform)
    analytical_first_call_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    # Autodiff jlog
    start_time = time.perf_counter()
    jitted_autodiff = jax.jit(jlog)
    autodiff_compilation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    start_time = time.perf_counter()
    _ = jitted_autodiff(transform)
    autodiff_first_call_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

    print(f"\n{Group.__name__} Compilation Time:")
    print(f"Autodiff: {autodiff_compilation_time:.4f} ms")
    print(f"Analytical: {analytical_compilation_time:.4f} ms")
    print(f"Compilation Speedup: {autodiff_compilation_time / analytical_compilation_time:.2f}x")

    print(f"\n{Group.__name__} First Call Time:")
    print(f"Autodiff: {autodiff_first_call_time:.4f} ms")
    print(f"Analytical: {analytical_first_call_time:.4f} ms")
    print(f"First Call Speedup: {autodiff_first_call_time / analytical_first_call_time:.2f}x")

    if analytical_compilation_time < autodiff_compilation_time:
        print("Compilation Time Test: Passed")
    else:
        print("Compilation Time Test: Failed")

    if analytical_first_call_time < autodiff_first_call_time:
        print("First Call Time Test: Passed")
    else:
        print("First Call Time Test: Failed")


def test_jlog_runtime(Group: Type[jaxlie.MatrixLieGroup], batch_axes: Tuple[int, ...]):
    """Compare runtime of analytical jlog and autodiff jlog."""
    transform = sample_transform(Group, batch_axes)

    # JIT compile both functions
    jitted_autodiff = jax.jit(jlog)
    jitted_analytical = jax.jit(lambda x: x.jlog())

    # Warm-up run
    _ = jitted_autodiff(transform)
    _ = jitted_analytical(transform)

    # Measure runtime
    num_runs = 1000

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = jax.block_until_ready(jitted_autodiff(transform))
    autodiff_runtime = ((time.perf_counter() - start_time) / num_runs) * 1000  # Convert to ms

    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = jax.block_until_ready(jitted_analytical(transform))
    analytical_runtime = ((time.perf_counter() - start_time) / num_runs) * 1000  # Convert to ms

    print(f"\n{Group.__name__} Runtime (average over {num_runs} runs):")
    print(f"Autodiff: {autodiff_runtime:.6f} ms")
    print(f"Analytical: {analytical_runtime:.6f} ms")
    print(f"Speedup: {autodiff_runtime / analytical_runtime:.2f}x")

    if analytical_runtime <= autodiff_runtime * 1.1:
        print("Runtime Test: Passed")
    else:
        print("Runtime Test: Failed")


def run_tests():
    groups = [SO2, SE2, SO3]
    batch_axes_list = [(),
                       (1,),
                       (3, 1, 2, 1),
                       ]

    for Group in groups:
        for batch_axes in batch_axes_list:
            print(f"\n--- Testing {Group.__name__} with batch_axes {batch_axes} ---")
            test_jlog_accuracy(Group, batch_axes)
            test_jlog_compilation_and_first_call(Group, batch_axes)
            test_jlog_runtime(Group, batch_axes)


if __name__ == "__main__":
    run_tests()
