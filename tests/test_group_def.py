import jax
import numpy as onp
from jax.config import config

import jaxlie

config.update("jax_enable_x64", True)


def test_identity():
    for Group in (jaxlie.SO2, jaxlie.SE2):
        a = Group.identity()
        b = Group.identity()
        onp.testing.assert_allclose((a @ b).compact(), b.compact())
        onp.testing.assert_allclose((a @ b).matrix(), b.matrix())

        x = onp.ones(2)
        onp.testing.assert_allclose(a @ x, x)
        onp.testing.assert_allclose(a.inverse() @ x, x)
        onp.testing.assert_allclose(Group.exp(a.log()) @ x, x)
        onp.testing.assert_allclose(
            Group.from_matrix(a.matrix()).compact(), a.compact()
        )


def test_generator():
    for Group in (jaxlie.SO2, jaxlie.SE2):
        for _ in range(5):
            tangent = onp.random.randn(Group.tangent_dim())
            T = Group.exp(tangent)
            onp.testing.assert_allclose(T.log(), tangent, atol=1e-6)

            onp.testing.assert_allclose(
                (T @ T.inverse()).compact(), Group.identity().compact(), atol=1e-6
            )
