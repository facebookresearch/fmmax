"""Tests for `fmmax.vector`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, vector


class ChangeBasisTest(unittest.TestCase):
    def test_basis_cycle(self):
        x = jnp.array([0.2, 0.8])
        y = jnp.array([-0.1, 0.3])
        u = jnp.array([-0.1, -0.7])
        v = jnp.array([0.0, 1.2])
        tu = jax.random.uniform(jax.random.PRNGKey(0), (10, 12))
        tv = jax.random.uniform(jax.random.PRNGKey(1), (10, 12))
        tx, ty = vector.change_vector_field_basis(tu, tv, u, v, x, y)
        tu_recovered, tv_recovered = vector.change_vector_field_basis(
            tx, ty, x, y, u, v
        )
        onp.testing.assert_allclose(tu_recovered, tu, rtol=1e-4)
        onp.testing.assert_allclose(tv_recovered, tv, rtol=1e-4)


class NormalizeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                vector.normalize_normal,
                [[1 / jnp.sqrt(2), 1.0, 0.0, 1.0, 0.0]],
                [[1 / jnp.sqrt(2), 0.0, 1.0, 0.0, 0.0]],
            ),
            (
                vector.normalize_pol,
                [[1 / jnp.sqrt(2), 0.2 / jnp.sqrt(2), 0.0, 0.01 / jnp.sqrt(2), 0.0]],
                [[1 / jnp.sqrt(2), 0.0, 0.2 / jnp.sqrt(2), 0.0, 0.0]],
            ),
            (
                vector.normalize_jones,
                [[0.5 + 0.5j, 0.734, 0.680, 0.707, 0.707]],
                [[0.5 + 0.5j, 0.680j, 0.734j, 0.707j, 0.707j]],
            ),
        ]
    )
    def test_normalized_matches_expected(self, normalize_fn, expected_tx, expected_ty):
        tx = jnp.asarray([[1.0, 0.2, 0.0, 0.01, 0.0]], dtype=float)
        ty = jnp.asarray([[1.0, 0.0, 0.2, 0.0, 0.0]], dtype=float)
        tx, ty = normalize_fn(tx, ty)
        onp.testing.assert_allclose(tx, jnp.asarray(expected_tx), rtol=1e-3)
        onp.testing.assert_allclose(ty, jnp.asarray(expected_ty), rtol=1e-3)

    @parameterized.parameterized.expand(
        [(vector.normalize_normal,), (vector.normalize_pol,), (vector.normalize_jones,)]
    )
    def test_zeros_no_nan(self, normalize_fn):
        tx = jnp.zeros((20, 20))
        ty = jnp.zeros((20, 20))
        tx, ty = normalize_fn(tx, ty)
        self.assertFalse(onp.any(onp.isnan(tx)))
        self.assertFalse(onp.any(onp.isnan(ty)))

    @parameterized.parameterized.expand(
        [(vector.normalize_normal,), (vector.normalize_pol,), (vector.normalize_jones,)]
    )
    def test_gradient_no_nan(self, normalize_fn):
        def loss_fn(tx, ty):
            tx, ty = normalize_fn(tx, ty)
            return jnp.real(jnp.sum(tx) + jnp.sum(ty))

        gx, gy = jax.grad(loss_fn, argnums=(0, 1))(jnp.zeros((5, 5)), jnp.zeros((5, 5)))
        self.assertFalse(onp.any(onp.isnan(gx)))
        self.assertFalse(onp.any(onp.isnan(gy)))


# -----------------------------------------------------------------------------
# Tests related to the `tangent_field` function.
# -----------------------------------------------------------------------------


class TangentFieldTest(unittest.TestCase):
    def test_optimize(self):
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        tx, ty = vector.tangent_field(
            arr,
            use_jones=False,
            optimizer=vector.OPTIMIZER,
            alignment_weight=vector.ALIGNMENT_WEIGHT,
            smoothness_weight=vector.SMOOTHNESS_WEIGHT,
            steps_dim_multiple=vector.STEPS_DIM_MULTIPLE,
            smoothing_kernel=jnp.ones((1, 1)),
        )
        expected_tx = [
            [
                0.083,
                0.25,
                0.417,
                0.583,
                0.75,
                0.667,
                0.333,
                0.0,
                -0.333,
                -0.667,
                -0.75,
                -0.583,
                -0.417,
                -0.25,
                -0.083,
            ]
        ]
        onp.testing.assert_allclose(tx, expected_tx, rtol=0.02)
        onp.testing.assert_allclose(ty, 0.0, atol=1e-7)

    def test_optimize_jones(self):
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        tx, ty = vector.tangent_field(
            arr,
            use_jones=True,
            optimizer=vector.OPTIMIZER,
            alignment_weight=vector.ALIGNMENT_WEIGHT,
            smoothness_weight=vector.SMOOTHNESS_WEIGHT,
            steps_dim_multiple=vector.STEPS_DIM_MULTIPLE,
            smoothing_kernel=jnp.ones((1, 1)),
        )
        expected_tx_magnitude = jnp.ones_like(arr)
        onp.testing.assert_allclose(jnp.abs(tx), expected_tx_magnitude)
        onp.testing.assert_allclose(ty, 0.0, atol=1e-7)


class SchemesTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(vector.JONES_DIRECT,), (vector.JONES,), (vector.POL,), (vector.NORMAL,)]
    )
    def test_batch_matches_single_exact(self, scheme):
        key = jax.random.PRNGKey(0)
        arr = jax.random.uniform(key, shape=(5, 10, 10))
        tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
            arr=arr,
            primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
        )
        for i in range(5):
            expected_tx_i, expected_ty_i = vector.VECTOR_FIELD_SCHEMES[scheme](
                arr=arr[i, :, :],
                primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
            )
            onp.testing.assert_array_equal(tx[i, :, :], expected_tx_i)
            onp.testing.assert_array_equal(ty[i, :, :], expected_ty_i)

    @parameterized.parameterized.expand(
        [(scheme,) for scheme in vector.VECTOR_FIELD_SCHEMES]
    )
    def test_uniform_array_no_nan(self, scheme):
        # The tangent field calculation requires special logic to handle uniform arrays,
        #  otherwise `nan` will show up in the result.
        arr = jnp.ones((10, 10))
        tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
            arr=arr,
            primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
        )
        self.assertFalse(onp.any(onp.isnan(tx)))
        self.assertFalse(onp.any(onp.isnan(ty)))

    @parameterized.parameterized.expand(
        [(scheme,) for scheme in vector.VECTOR_FIELD_SCHEMES]
    )
    def test_gradient_no_nan(self, scheme):
        def _loss_fn(arr):
            tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
                arr=arr,
                primitive_lattice_vectors=basis.LatticeVectors(u=basis.X, v=basis.Y),
            )
            return jnp.sum(jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2)

        arr = jnp.asarray(
            [
                [1.0 + 1.0j, 1.0, 1.0, 1.0, 1.0 + 1.1j, 1.0 + 1.1j],
                [1.0 + 1.0j, 0.0, 0.0, 0.0, 1.0 + 1.1j, 1.0 + 1.1j],
                [1.0 + 1.0j, 0.0, 0.0, 0.0, 1.0 + 1.1j, 1.0 + 1.1j],
                [1.0 + 1.0j, 1.0, 1.0, 1.0, 1.0 + 1.1j, 1.0 + 1.1j],
            ]
        )
        grad = jax.grad(_loss_fn)(arr)
        self.assertFalse(onp.any(onp.isnan(grad)))


class LossTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (1.0, 0.0, 1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0, 0.0, -1.0),
            (0.0, 1.0, 1.0, 0.0, 0.0),
        ]
    )
    def test_self_alignment_loss(self, tx, ty, tx0, ty0, expected):
        tx = jnp.asarray(tx)[jnp.newaxis, jnp.newaxis]
        ty = jnp.asarray(ty)[jnp.newaxis, jnp.newaxis]
        tx0 = jnp.asarray(tx0)[jnp.newaxis, jnp.newaxis]
        ty0 = jnp.asarray(ty0)[jnp.newaxis, jnp.newaxis]
        loss = vector._self_alignment_loss(tx, ty, tx0, ty0)
        onp.testing.assert_allclose(loss, expected)

    @parameterized.parameterized.expand(
        [
            (
                jnp.ones((2, 2)),
                jnp.zeros((2, 2)),
                jnp.ones((2, 2)),
                jnp.zeros((2, 2)),
                -400,
            ),
            (
                jnp.asarray([[1, 1], [-1, -1], [-1, -1], [1, 1]]),
                jnp.zeros((4, 2)),
                jnp.ones((4, 2)),
                jnp.zeros((4, 2)),
                -768.0,
            ),
        ]
    )
    def test_field_loss(self, tx, ty, tx0, ty0, expected):
        loss = vector._field_loss(
            tx, ty, tx0, ty0, alignment_weight=100, smoothness_weight=2
        )
        onp.testing.assert_allclose(loss, expected)

    def test_field_loss_batch_matches_single(self):
        key = jax.random.PRNGKey(0)
        tx, ty, tx0, ty0 = jax.random.uniform(key, (4, 8, 5, 10))
        loss = vector._field_loss(
            tx, ty, tx0, ty0, alignment_weight=100, smoothness_weight=2
        )
        expected_loss = 0
        for tx_slice, ty_slice, tx0_slice, ty0_slice in zip(tx, ty, tx0, ty0):
            expected_loss += vector._field_loss(
                tx_slice,
                ty_slice,
                tx0_slice,
                ty0_slice,
                alignment_weight=100,
                smoothness_weight=2,
            )
        onp.testing.assert_allclose(loss, expected_loss, rtol=1e-6)


class UtilitiesTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ((1.0, 1.0), 1.0, (jnp.sqrt(0.5), jnp.sqrt(0.5))),
            ((0.1, 0.1), 1.0, (0.1, 0.1)),
            ((2.0, 0.0), 1.0, (1.0, 0.0)),
            ((2.0, 0.0), 1.5, (1.5, 0.0)),
            ((2.0j, 0.0), 1.5, (1.5j, 0.0)),
        ]
    )
    def test_clip_magnitude(self, tx_ty, max_magnitude, expected):
        result = vector._clip_magnitude(*tx_ty, max_magnitude)
        onp.testing.assert_allclose(result, expected)

    @parameterized.parameterized.expand(
        [
            ((1.0, 0.0), (1.0, 0.0)),
            ((1.0j, 0.0), (1.0, 0.0)),
            ((1.0j, 1.0), (jnp.sqrt(0.5) * (1.0 + 1.0j), jnp.sqrt(0.5) * (1.0 - 1.0j))),
        ]
    )
    def test_remove_average_phase(self, tx_ty, expected):
        tx, ty = tx_ty
        tx = jnp.asarray(tx)[jnp.newaxis, jnp.newaxis]
        ty = jnp.asarray(ty)[jnp.newaxis, jnp.newaxis]
        result_tx, result_ty = vector._remove_mean_phase(tx, ty)
        onp.testing.assert_allclose(
            (result_tx.squeeze(), result_ty.squeeze()), expected
        )
