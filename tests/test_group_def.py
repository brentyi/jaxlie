from typing import Type

import jax
import jaxlie
import numpy as onp
from jax import numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


def test_identity():
    Group: Type[jaxlie.MatrixLieGroup]
    for Group in (jaxlie.SO2, jaxlie.SE2, jaxlie.SO3):
        T_a = Group.identity()
        T_b = Group.identity()
        onp.testing.assert_allclose((T_a @ T_b).parameters, T_b.parameters)
        onp.testing.assert_allclose((T_a @ T_b).as_matrix(), T_b.as_matrix())

        x = onp.random.randn(Group.space_dim)
        onp.testing.assert_allclose(T_a @ x, x)

        if Group.matrix_dim == Group.space_dim:
            onp.testing.assert_allclose(T_a.as_matrix() @ x, x)
        else:
            # Homogeneous
            assert Group.matrix_dim == Group.space_dim + 1
            onp.testing.assert_allclose(
                (T_a.as_matrix() @ jnp.ones(Group.matrix_dim).at[:-1].set(x))[:-1], x
            )

        onp.testing.assert_allclose(T_a.inverse() @ x, x)
        onp.testing.assert_allclose(Group.exp(T_a.log()) @ x, x)
        onp.testing.assert_allclose(
            Group.from_matrix(T_a.as_matrix()).parameters, T_a.parameters
        )


def test_generator():
    for Group in (jaxlie.SO2, jaxlie.SE2, jaxlie.SO3):
        for i in range(5):
            tangent = onp.random.randn(Group.tangent_dim)
            if i == 0:
                tangent = tangent * 0
            elif i == 1:
                tangent = tangent * 1e-10
            elif i == 2:
                tangent = tangent * -1e-10

            if jnp.linalg.norm(tangent) > jnp.pi:
                # Somewhat sketchy logic for making sure our exp/log operations are
                # bijective for rotations
                tangent = tangent / jnp.linalg.norm(tangent) * jnp.pi

            T = jax.jit(Group.exp)(tangent)
            onp.testing.assert_allclose(T.log(), tangent, atol=1e-6)

            onp.testing.assert_allclose(
                (T @ T.inverse()).parameters, Group.identity().parameters, atol=1e-6
            )

            onp.testing.assert_allclose(
                T.inverse().as_matrix(), jnp.linalg.inv(T.as_matrix())
            )

            x = onp.random.randn(Group.space_dim)
            if Group.matrix_dim == Group.space_dim:
                onp.testing.assert_allclose(T @ x, T.as_matrix() @ x)
            else:
                # Homogeneous
                assert Group.matrix_dim == Group.space_dim + 1
                onp.testing.assert_allclose(
                    T @ x,
                    (T.as_matrix() @ jnp.ones(Group.matrix_dim).at[:-1].set(x))[:-1],
                )

            assert not jnp.any(jnp.isnan(T.parameters))
            assert not jnp.any(jnp.isnan(T.as_matrix()))
            assert not jnp.any(jnp.isnan(T @ x))
            assert not jnp.any(jnp.isnan(T.log()))
