import jaxlie
import numpy as onp


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
