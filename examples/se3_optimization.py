"""Example that uses helpers in `jaxlie.manifold.*` to compare algorithms for running an
ADAM optimizer on SE(3) variables.

We compare three approaches:

(1) Tangent-space ADAM: computing updates on a local tangent space, which are then
retracted back to the global parameterization at each step. This should generally be the
most stable.

(2) Projected ADAM: running standard ADAM directly on the global parameterization, then
projecting after each step.

(3) Standard ADAM with exponential coordinates: using a log-space underlying
parameterization lets us run ADAM without any modifications.

Note that the number of training steps and learning rate can be configured, see:

    python se3_optimization.py --help

"""

from __future__ import annotations

import time
from typing import List, Literal, Tuple, Union

import dcargs
import jax
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import optax
from jax import numpy as jnp
from typing_extensions import assert_never

import jaxlie


@jdc.pytree_dataclass
class Parameters:
    """Parameters to optimize over, in their global representation. Rotations are
    quaternions under the hood.

    Note that there's redundancy here: given T_ab and T_bc, T_ca can be computed as
    (T_ab @ T_bc).inverse(). Our optimization will be focused on making these redundant
    transforms consistent with each other.
    """

    T_ab: jaxlie.SE3
    T_bc: jaxlie.SE3
    T_ca: jaxlie.SE3


@jdc.pytree_dataclass
class ExponentialCoordinatesParameters:
    """Same as `Parameters`, but using exponential coordinates."""

    log_T_ab: jnp.ndarray
    log_T_bc: jnp.ndarray
    log_T_ca: jnp.ndarray

    @property
    def T_ab(self) -> jaxlie.SE3:
        return jaxlie.SE3.exp(self.log_T_ab)

    @property
    def T_bc(self) -> jaxlie.SE3:
        return jaxlie.SE3.exp(self.log_T_bc)

    @property
    def T_ca(self) -> jaxlie.SE3:
        return jaxlie.SE3.exp(self.log_T_ca)

    @staticmethod
    def from_global(params: Parameters) -> ExponentialCoordinatesParameters:
        return ExponentialCoordinatesParameters(
            params.T_ab.log(),
            params.T_bc.log(),
            params.T_ca.log(),
        )


def compute_loss(
    params: Union[Parameters, ExponentialCoordinatesParameters]
) -> jnp.ndarray:
    """As our loss, we enforce (a) priors on our transforms and (b) a consistency
    constraint."""
    T_ba_prior = jaxlie.SE3.sample_uniform(jax.random.PRNGKey(4))
    T_cb_prior = jaxlie.SE3.sample_uniform(jax.random.PRNGKey(5))

    return jnp.sum(
        # Consistency term.
        (params.T_ab @ params.T_bc @ params.T_ca).log() ** 2
        # Priors.
        + (params.T_ab @ T_ba_prior).log() ** 2
        + (params.T_bc @ T_cb_prior).log() ** 2
    )


Algorithm = Literal["tangent_space", "projected", "exponential_coordinates"]


@jdc.pytree_dataclass
class State:
    params: Union[Parameters, ExponentialCoordinatesParameters]
    optimizer: jdc.Static[optax.GradientTransformation]
    optimizer_state: optax.OptState
    algorithm: jdc.Static[Algorithm]

    @staticmethod
    def initialize(algorithm: Algorithm, learning_rate: float) -> State:
        """Initialize the state of our optimization problem. Note that the transforms
        parameters won't initially be consistent; `T_ab @ T_bc != T_ca.inverse()`.
        """
        prngs = jax.random.split(jax.random.PRNGKey(0), num=1)
        global_params = Parameters(
            jaxlie.SE3.sample_uniform(prngs[0]),
            jaxlie.SE3.sample_uniform(prngs[1]),
            jaxlie.SE3.sample_uniform(prngs[2]),
        )

        # Make optimizer.
        params: Union[Parameters, ExponentialCoordinatesParameters]
        optimizer = optax.adam(learning_rate=learning_rate)
        if algorithm == "tangent_space":
            # Initialize gradient statistics as on the tangent space.
            params = global_params
            optimizer_state = optimizer.init(jaxlie.manifold.zero_tangents(params))
        elif algorithm == "projected":
            # Initialize gradient statistics directly in quaternion space.
            params = global_params
            optimizer_state = optimizer.init(params)  # type: ignore
        elif algorithm == "exponential_coordinates":
            # Switch to a log-space parameterization.
            params = ExponentialCoordinatesParameters.from_global(global_params)
            optimizer_state = optimizer.init(params)  # type: ignore
        else:
            assert_never(algorithm)

        return State(
            params=params,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            algorithm=algorithm,
        )

    @jax.jit
    def step(self: State) -> Tuple[jnp.ndarray, State]:
        """Take one ADAM optimization step."""

        if self.algorithm == "tangent_space":
            # ADAM step on manifold.
            #
            # `jaxlie.manifold.value_and_grad()` is a drop-in replacement for
            # `jax.value_and_grad()`, but for Lie group instances computes gradients on
            # the tangent space.
            loss, grads = jaxlie.manifold.value_and_grad(compute_loss)(self.params)
            updates, new_optimizer_state = self.optimizer.update(
                grads, self.optimizer_state, self.params  # type: ignore
            )
            new_params = jaxlie.manifold.rplus(self.params, updates)

        elif self.algorithm == "projected":
            # Projection-based approach.
            loss, grads = jax.value_and_grad(compute_loss)(self.params)
            updates, new_optimizer_state = self.optimizer.update(
                grads, self.optimizer_state, self.params  # type: ignore
            )
            new_params = optax.apply_updates(self.params, updates)  # type: ignore

            # Project back to manifold.
            new_params = jaxlie.manifold.project_all(new_params)

        elif self.algorithm == "exponential_coordinates":
            # If we parameterize with exponential coordinates, we can
            loss, grads = jax.value_and_grad(compute_loss)(self.params)
            updates, new_optimizer_state = self.optimizer.update(
                grads, self.optimizer_state, self.params  # type: ignore
            )
            new_params = optax.apply_updates(self.params, updates)  # type: ignore

        else:
            assert assert_never(self.algorithm)

        # Return updated structure.
        with jdc.copy_and_mutate(self, validate=True) as new_state:
            new_state.params = new_params  # type: ignore
            new_state.optimizer_state = new_optimizer_state

        return loss, new_state


def run_experiment(
    algorithm: Algorithm, learning_rate: float, train_steps: int
) -> List[float]:
    """Run the optimization problem, either using a tangent-space approach or via
    projection."""

    print(algorithm)
    state = State.initialize(algorithm, learning_rate)
    state.step()  # Don't include JIT compile in timing.

    start_time = time.time()
    losses = []
    for i in range(train_steps):
        loss, state = state.step()
        if i % 20 == 0:
            print(f"\t(step {i:03d}) Loss", loss, flush=True)
        losses.append(float(loss))
    print()
    print(f"\tConverged in {time.time() - start_time} seconds")
    print()
    print("\tAfter optimization, the following transforms should be consistent:")
    print(f"\t\t{state.params.T_ab @ state.params.T_bc=}")
    print(f"\t\t{state.params.T_ca.inverse()=}")

    return losses


def main(train_steps: int = 250, learning_rate: float = 1e-1) -> None:
    """Run pose optimization experiments.

    Args:
        train_steps: Number of training steps to take.
        learning_rate: Learning rate for our ADAM optimizers.
    """
    xs = range(train_steps)

    algorithms: Tuple[Algorithm, ...] = (
        "tangent_space",
        "projected",
        "exponential_coordinates",
    )
    for algorithm in algorithms:
        plt.plot(
            xs,
            run_experiment(algorithm, learning_rate, train_steps),
            label=algorithm,
        )
        print()
    plt.yscale("log", base=2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dcargs.cli(main)
