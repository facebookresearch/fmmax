"""Tests for `fmmax.vector_fourier`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from parameterized import parameterized
from scipy import ndimage

from fmmax import basis, vector_fourier


class TangentVectorTest(unittest.TestCase):
    def _compute_field(
        self,
        approximate_num_terms,
        scale,
        shape,
        arr_scale,
        binarize_arr,
        use_jones_direct,
    ):
        primitive_lattice_vectors = basis.LatticeVectors(
            basis.X * scale, basis.Y * scale
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=approximate_num_terms,
            truncation=basis.Truncation.CIRCULAR,
        )
        x, y = jnp.meshgrid(
            jnp.linspace(0, scale, shape[0]),
            jnp.linspace(0, scale, shape[1]),
            indexing="ij",
        )
        distance = jnp.sqrt((x - scale / 2) ** 2 + (y - scale / 2) ** 2)
        arr = jnp.exp(-(distance**2) / scale**2 * 30)
        if binarize_arr:
            arr = (arr > 0.5).astype(float)
        arr *= arr_scale

        return vector_fourier.compute_tangent_field(
            arr,
            expansion,
            primitive_lattice_vectors,
            use_jones_direct,
            fourier_loss_weight=0.001,
        )

    @parameterized.expand(
        [
            # Cases with grayscale density and `use_jones_direct = False`.
            (1, (160, 120), 1, False, False),  # Resolution invariance.
            (1000, (80, 80), 1, False, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), False, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), False, False),
            (1, (80, 80), (0 + 1j), False, False),
            # Cases with binarized density and `use_jones_direct = False`.
            (1, (160, 120), 1, True, False),  # Resolution invariance.
            (1000, (80, 80), 1, True, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), True, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), True, False),
            (1, (80, 80), (0 + 1j), True, False),
            # Cases with `use_jones_direct = True`.
            (1, (160, 120), 1, False, True),  # Grayscale, resolution invariance.
            (1000, (80, 80), 1, False, True),  # Grayscale, scale invariance.
            (1, (160, 120), 1, True, True),  # Binarized, resolution invariance.
            (1000, (80, 80), 1, True, True),  # Binarized, scale invariance.
        ]
    )
    def test_invariances(self, scale, shape, arr_scale, binarize_arr, use_jones_direct):
        reference_tx, reference_ty = self._compute_field(
            approximate_num_terms=200,
            scale=1,
            shape=(80, 80),
            arr_scale=1,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
        )
        zoom = (shape[0] / 80, shape[1] / 80)
        expected_tx = ndimage.zoom(reference_tx, zoom, order=1) * arr_scale
        expected_ty = ndimage.zoom(reference_ty, zoom, order=1) * arr_scale

        tx, ty = self._compute_field(
            approximate_num_terms=200,
            scale=scale,
            shape=shape,
            arr_scale=arr_scale,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
        )
        onp.testing.assert_allclose(tx, expected_tx, atol=0.05)
        onp.testing.assert_allclose(ty, expected_ty, atol=0.05)

    @parameterized.expand([[True], [False]])
    def test_batch_calculation_matches_single(self, use_jones_direct):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=200,
            truncation=basis.Truncation.CIRCULAR,
        )
        x, y = jnp.meshgrid(
            jnp.linspace(0, 1, 80), jnp.linspace(0, 1, 80), indexing="ij"
        )
        distance = jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        arr = jnp.exp(
            -(distance**2) * jnp.arange(10, 40, 5)[:, jnp.newaxis, jnp.newaxis]
        )
        assert arr.shape == (6, 80, 80)

        tx_batch, ty_batch = vector_fourier.compute_tangent_field(
            arr,
            expansion,
            primitive_lattice_vectors,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.01,
        )

        for i in range(arr.shape[0]):
            tx, ty = vector_fourier.compute_tangent_field(
                arr[i, :, :],
                expansion,
                primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.01,
            )
            onp.testing.assert_allclose(tx, tx_batch[i, :, :], rtol=1e-5)
            onp.testing.assert_allclose(ty, ty_batch[i, :, :], rtol=1e-5)

    @parameterized.expand([[True], [False]])
    def test_gradient_no_nan(self, use_jones_direct):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=200,
            truncation=basis.Truncation.CIRCULAR,
        )

        def loss_fn(arr):
            tx, ty = vector_fourier.compute_tangent_field(
                arr=arr,
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.01,
            )
            return jnp.sum(jnp.abs(tx) ** 2) + jnp.sum(jnp.abs(ty) ** 2)

        x, y = jnp.meshgrid(
            jnp.linspace(0, 1, 80), jnp.linspace(0, 1, 80), indexing="ij"
        )
        distance = jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        arr = jnp.exp(
            -(distance**2) * jnp.arange(10, 40, 5)[:, jnp.newaxis, jnp.newaxis]
        )
        grad = jax.grad(loss_fn)(arr)
        self.assertFalse(jnp.any(jnp.isnan(grad)))

    @parameterized.expand([[True], [False]])
    def test_valid_result_with_uniform_array(self, use_jones_direct):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=200,
            truncation=basis.Truncation.CIRCULAR,
        )

        def loss_fn(arr):
            tx, ty = vector_fourier.compute_tangent_field(
                arr=arr,
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.01,
            )
            return jnp.sum(jnp.abs(tx) ** 2) + jnp.sum(jnp.abs(ty) ** 2), (tx, ty)

        arr = jnp.zeros((80, 80))
        (_, (tx, ty)), grad = jax.value_and_grad(loss_fn, has_aux=True)(arr)
        self.assertFalse(jnp.any(jnp.isnan(tx)))
        self.assertFalse(jnp.any(jnp.isnan(ty)))
        self.assertFalse(jnp.any(jnp.isnan(grad)))

    # def test_field_converges(self):
    #     pass


class NormalizeTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                vector_fourier.normalize_elementwise,
                [[1 / jnp.sqrt(2), 1.0, 0.0, 1.0, 0.0]],
                [[1 / jnp.sqrt(2), 0.0, 1.0, 0.0, 0.0]],
            ),
            (
                vector_fourier.normalize,
                [[1 / jnp.sqrt(2), 0.2 / jnp.sqrt(2), 0.0, 0.01 / jnp.sqrt(2), 0.0]],
                [[1 / jnp.sqrt(2), 0.0, 0.2 / jnp.sqrt(2), 0.0, 0.0]],
            ),
            (
                vector_fourier.normalize_jones,
                [[0.5 + 0.5j, 0.734, 0.680, 0.707, 0.707]],
                [[0.5 + 0.5j, 0.680j, 0.734j, 0.707j, 0.707j]],
            ),
        ]
    )
    def test_normalized_matches_expected(self, normalize_fn, expected_tx, expected_ty):
        tx = jnp.asarray([[1.0, 0.2, 0.0, 0.01, 0.0]], dtype=float)
        ty = jnp.asarray([[1.0, 0.0, 0.2, 0.0, 0.0]], dtype=float)
        txty = normalize_fn(jnp.stack([tx, ty], axis=-1))
        tx, ty = txty[..., 0], txty[..., 1]
        onp.testing.assert_allclose(tx, jnp.asarray(expected_tx), rtol=1e-3)
        onp.testing.assert_allclose(ty, jnp.asarray(expected_ty), rtol=1e-3)

    @parameterized.expand(
        [
            (vector_fourier.normalize_elementwise,),
            (vector_fourier.normalize,),
            (vector_fourier.normalize_jones,),
        ]
    )
    def test_zeros_no_nan(self, normalize_fn):
        tx = jnp.zeros((20, 20))
        ty = jnp.zeros((20, 20))
        txty = normalize_fn(jnp.stack([tx, ty], axis=-1))
        tx, ty = txty[..., 0], txty[..., 1]
        self.assertFalse(onp.any(onp.isnan(tx)))
        self.assertFalse(onp.any(onp.isnan(ty)))

    @parameterized.expand(
        [
            (vector_fourier.normalize_elementwise,),
            (vector_fourier.normalize,),
            (vector_fourier.normalize_jones,),
        ]
    )
    def test_gradient_no_nan(self, normalize_fn):
        def loss_fn(tx, ty):
            txty = normalize_fn(jnp.stack([tx, ty], axis=-1))
            tx, ty = txty[..., 0], txty[..., 1]
            return jnp.real(jnp.sum(tx) + jnp.sum(ty))

        gx, gy = jax.grad(loss_fn, argnums=(0, 1))(jnp.zeros((5, 5)), jnp.zeros((5, 5)))
        self.assertFalse(onp.any(onp.isnan(gx)))
        self.assertFalse(onp.any(onp.isnan(gy)))


class MagnitudeTest(unittest.TestCase):
    @parameterized.expand(
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, jnp.sqrt(2)),
        ]
    )
    def test_magnitude(self, tx, ty, expected):
        result = vector_fourier._field_magnitude(jnp.stack([tx, ty], axis=-1))
        onp.testing.assert_allclose(result, expected)

    def test_magnitude_gradient_no_nan(self):
        grad = jax.grad(lambda x: jnp.squeeze(vector_fourier._field_magnitude(x)))(
            jnp.zeros((2,))
        )
        self.assertFalse(onp.any(onp.isnan(grad)))


class AngleTest(unittest.TestCase):
    @parameterized.expand(
        [
            (0, 0, 0),
            (0, 1, jnp.pi / 2),
            (1, 0, 0),
            (1, 1, jnp.pi / 4),
        ]
    )
    def test_angle(self, tx, ty, expected):
        result = vector_fourier._angle(tx + 1j * ty)
        onp.testing.assert_allclose(result, expected)

    def test_angle_gradient_no_nan(self):
        grad = jax.grad(vector_fourier._angle)(0.0)
        self.assertFalse(onp.any(onp.isnan(grad)))


class GradientTest(unittest.TestCase):
    pass


class TestNewtonIteration(unittest.TestCase):
    pass


# class LossTest(unittest.TestCase):
#     @parameterized.parameterized.expand(
#         [
#             (1.0, 0.0, 1.0, 0.0, -1.0),
#             (-1.0, 0.0, 1.0, 0.0, -1.0),
#             (0.0, 1.0, 1.0, 0.0, 0.0),
#         ]
#     )
#     def test_self_alignment_loss(self, tx, ty, tx0, ty0, expected):
#         tx = jnp.asarray(tx)[jnp.newaxis, jnp.newaxis]
#         ty = jnp.asarray(ty)[jnp.newaxis, jnp.newaxis]
#         tx0 = jnp.asarray(tx0)[jnp.newaxis, jnp.newaxis]
#         ty0 = jnp.asarray(ty0)[jnp.newaxis, jnp.newaxis]
#         loss = vector_fourier._self_alignment_loss(tx, ty, tx0, ty0)
#         onp.testing.assert_allclose(loss, expected)

#     @parameterized.parameterized.expand(
#         [
#             (
#                 jnp.ones((2, 2)),
#                 jnp.zeros((2, 2)),
#                 jnp.ones((2, 2)),
#                 jnp.zeros((2, 2)),
#                 -400,
#             ),
#             (
#                 jnp.asarray([[1, 1], [-1, -1], [-1, -1], [1, 1]]),
#                 jnp.zeros((4, 2)),
#                 jnp.ones((4, 2)),
#                 jnp.zeros((4, 2)),
#                 -768.0,
#             ),
#         ]
#     )
#     def test_field_loss(self, tx, ty, tx0, ty0, expected):
#         loss = vector_fourier._field_loss(
#             tx, ty, tx0, ty0, alignment_weight=100, smoothness_weight=2
#         )
#         onp.testing.assert_allclose(loss, expected)

#     def test_field_loss_batch_matches_single(self):
#         key = jax.random.PRNGKey(0)
#         tx, ty, tx0, ty0 = jax.random.uniform(key, (4, 8, 5, 10))
#         loss = vector_fourier._field_loss(
#             tx, ty, tx0, ty0, alignment_weight=100, smoothness_weight=2
#         )
#         expected_loss = 0
#         for tx_slice, ty_slice, tx0_slice, ty0_slice in zip(tx, ty, tx0, ty0):
#             expected_loss += vector_fourier._field_loss(
#                 tx_slice,
#                 ty_slice,
#                 tx0_slice,
#                 ty0_slice,
#                 alignment_weight=100,
#                 smoothness_weight=2,
#             )
#         onp.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
