import numpy as onp

import jaxlie


def test_identity():
    a = jaxlie.SO2.identity()
    b = jaxlie.SO2.identity()
    onp.testing.assert_allclose((a @ b).compact(), b.compact())
    onp.testing.assert_allclose((a @ b).matrix(), b.matrix())
    onp.testing.assert_allclose((a @ b).unit_complex, b.unit_complex)
