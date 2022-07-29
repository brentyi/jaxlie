"""Example that uses helpers in `jaxlie.manifold.*` to run an ADAM optimizer on SE(3)
variables. We compute gradients, statistics, and updates on a local tangent space, which
are then retracted back to the quaternion-based global parameterization at each step.

This has better stability, equivariance, and convergence characteristics than naive
approaches to optimization over poses (parameterizing via exponential coordinates, Euler
angles, projected gradient descent, etc)."""

import time
from typing import Tuple

import jax
import jax_dataclasses as jdc
import optax
from jax import numpy as jnp
from typing_extensions import reveal_type

import jaxlie


@jdc.pytree_dataclass
class Parameters:
    """The parameters we'll be optimizing over.

    Note that there's redundancy here: given T_ab and T_bc, T_ca can be computed as
    (T_ab @ T_bc).inverse(). Our optimization will be focused on making these redundant
    transforms consistent with each other."""

    T_ab: jaxlie.SE3
    T_bc: jaxlie.SE3
    T_ca: jaxlie.SE3


@jdc.pytree_dataclass
class State:
    params: Parameters
    optimizer: optax.GradientTransformation = jdc.static_field()
    optimizer_state: optax.OptState


def initialize() -> State:
    """Initialize the state of our system. Transforms won't initially be consistent;
    `T_ab @ T_bc != T_ca.inverse()`."""

    prngs = jax.random.split(jax.random.PRNGKey(0), num=1)
    params = Parameters(
        jaxlie.SE3.sample_uniform(prngs[0]),
        jaxlie.SE3.sample_uniform(prngs[1]),
        jaxlie.SE3.sample_uniform(prngs[2]),
    )

    # (1) Make optimizer. We keep momentum values on the tangent space.
    optimizer = optax.adam(learning_rate=1e-2)
    return State(
        params=params,
        optimizer=optimizer,
        optimizer_state=optimizer.init(
            params=jaxlie.manifold.zero_tangents(params),  # type: ignore
        ),
    )


@jax.jit
def step(state: State) -> Tuple[jnp.ndarray, State]:
    """Take one ADAM optimization step."""

    def consistency_loss(params: Parameters) -> jnp.ndarray:
        return jnp.sum((params.T_ab @ params.T_bc @ params.T_ca).log() ** 2)

    # (2) `jaxlie.manifold.value_and_grad()` is a drop-in replacement for
    # `jax.value_and_grad()`, but for Lie group instances computes gradients on the
    # tangent space.
    loss, grads = jaxlie.manifold.value_and_grad(consistency_loss)(state.params)

    updates, new_optimizer_state = state.optimizer.update(
        grads, state.optimizer_state, state.params  # type: ignore
    )

    # (3) We replace standard addition (optax.apply_updates) with a topology-aware
    # `rplus` operator.
    new_params = jaxlie.manifold.rplus(state.params, updates)
    delt = jaxlie.manifold.rminus(new_params, new_params)

    return loss, State(
        params=new_params,  # type: ignore
        optimizer=state.optimizer,
        optimizer_state=new_optimizer_state,
    )


if __name__ == "__main__":
    state = initialize()
    step(state)  # Don't include JIT compile in timing.

    start_time = time.time()
    for i in range(200):
        loss, state = step(state)
        if i % 20 == 0:
            print("Loss", loss, flush=True)
    print()
    print(f"Converged in {time.time() - start_time} seconds")
    print()
    print("After optimization, the following transforms should be the same:")
    print("\tT_ac #1:", state.params.T_ab @ state.params.T_bc)
    print("\tT_ac #2:", state.params.T_ca.inverse())
