"""Independent SE(3) exponential and differential oracles near zero rotation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import jaxlie

jax.config.update("jax_enable_x64", True)


def _skew_numpy(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _matrix_exp(tangent):
    algebra = np.zeros((4, 4))
    algebra[:3, :3] = _skew_numpy(tangent[3:])
    algebra[:3, 3] = tangent[:3]
    return expm(algebra)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_se3_exp_small_rotation_matches_independent_matrix_exponential(dtype):
    step = 1e-3 if dtype == jnp.float32 else 1e-6
    tolerance = 3e-6 if dtype == jnp.float32 else 1e-12
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    for translation in ([1.0, 2.0, -3.0], [-2.0, 0.5, 1.5]):
        for angle in (0.0, step, -step):
            tangent = jnp.asarray(np.r_[translation, angle * axis], dtype=dtype)
            expected = _matrix_exp(np.asarray(tangent))
            actual = jaxlie.SE3.exp(tangent).as_matrix()
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_se3_exp_translation_rotation_jacobian_at_zero(dtype):
    translation = jnp.asarray([1.0, 2.0, -3.0], dtype=dtype)

    def translation_from_rotation(rotation):
        return jaxlie.SE3.exp(jnp.concatenate([translation, rotation])).translation()

    # V(omega) = I + [omega]/2 + [omega]^2/6 + ...
    expected = -0.5 * _skew_numpy(np.asarray(translation))
    zero = jnp.zeros(3, dtype=dtype)
    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    for differentiate in (jax.jacfwd, jax.jacrev):
        actual = differentiate(translation_from_rotation)(zero)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_se3_exp_translation_rotation_hessian_at_zero(dtype):
    translation = jnp.asarray([1.0, 2.0, -3.0], dtype=dtype)

    def translation_from_rotation(rotation):
        return jaxlie.SE3.exp(jnp.concatenate([translation, rotation])).translation()

    generators = [_skew_numpy(axis) for axis in np.eye(3)]
    expected = np.zeros((3, 3, 3))
    for i, left in enumerate(generators):
        for j, right in enumerate(generators):
            expected[:, i, j] = (
                (left @ right + right @ left) @ np.asarray(translation) / 6.0
            )
    actual = jax.jacfwd(jax.jacrev(translation_from_rotation))(
        jnp.zeros(3, dtype=dtype)
    )
    assert np.isfinite(actual).all()
    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_se3_exp_mixed_batch_and_log_roundtrip(dtype):
    step = 1e-3 if dtype == jnp.float32 else 1e-6
    tolerance = 3e-6 if dtype == jnp.float32 else 1e-12
    tangents = jnp.asarray(
        [
            [1.0, 2.0, -3.0, 0.0, 0.0, 0.0],
            [-2.0, 0.5, 1.5, step, 0.0, 0.0],
            [0.5, -1.0, 2.0, 0.1, -0.2, 0.3],
        ],
        dtype=dtype,
    )
    expected = np.stack([_matrix_exp(row) for row in np.asarray(tangents)])
    actual = jaxlie.SE3.exp(tangents)
    np.testing.assert_allclose(
        actual.as_matrix(), expected, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(actual.log(), tangents, atol=tolerance, rtol=tolerance)
    for i in range(len(tangents)):
        np.testing.assert_allclose(
            jaxlie.SE3.exp(tangents[i]).as_matrix(),
            expected[i],
            atol=tolerance,
            rtol=tolerance,
        )
