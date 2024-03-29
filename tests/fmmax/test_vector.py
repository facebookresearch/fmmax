"""Tests for `fmmax.vector`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized
from scipy import ndimage

from fmmax import basis, vector

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)


def _generate_array(shape, arr_scale, binarize_arr):
    x, y = jnp.meshgrid(
        jnp.linspace(0, 1, shape[0]),
        jnp.linspace(0, 1, shape[1]),
        indexing="ij",
    )
    distance = jnp.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    arr = jnp.exp(-(distance**2) * 30)
    if binarize_arr:
        arr = (arr > 0.5).astype(float)
    return arr * arr_scale


class TangentVectorTest(unittest.TestCase):
    def _compute_field(
        self,
        approximate_num_terms,
        scale,
        shape,
        arr_scale,
        binarize_arr,
        use_jones_direct,
        fourier_loss_weight,
        smoothness_loss_weight,
        steps=1,
    ):
        primitive_lattice_vectors = basis.LatticeVectors(
            basis.X * scale, basis.Y * scale
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=approximate_num_terms,
            truncation=basis.Truncation.CIRCULAR,
        )
        arr = _generate_array(shape, arr_scale, binarize_arr)
        return vector.compute_tangent_field(
            arr,
            expansion,
            primitive_lattice_vectors,
            use_jones_direct,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=steps,
        )

    @parameterized.expand(
        [
            # Cases with grayscale density and `use_jones_direct = False`.
            (1, (160, 120), 1, False, False),  # Resolution invariance.
            (1000, (80, 80), 1, False, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), False, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), False, False),
            (1, (80, 80), (0 + 1j), False, False),
            (1, (80, 80), 0.1, False, False),  # Amplitude invariance.
            (1, (80, 80), 10.0, False, False),
            # Cases with binarized density and `use_jones_direct = False`.
            (1, (160, 120), 1, True, False),  # Resolution invariance.
            (1000, (80, 80), 1, True, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), True, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), True, False),
            (1, (80, 80), (0 + 1j), True, False),
            (1, (80, 80), 0.1, True, False),
            (1, (80, 80), 10.0, True, False),
            # Cases with `use_jones_direct = True`.
            (1, (160, 120), 1, False, True),  # Grayscale, resolution invariance.
            (1000, (80, 80), 1, False, True),  # Grayscale, scale invariance.
            (1, (160, 120), 1, True, True),  # Binarized, resolution invariance.
            (1000, (80, 80), 1, True, True),  # Binarized, scale invariance.
        ]
    )
    def test_invariances_fourier_loss(
        self, scale, shape, arr_scale, binarize_arr, use_jones_direct
    ):
        reference_tx, reference_ty = self._compute_field(
            approximate_num_terms=200,
            scale=1,
            shape=(80, 80),
            arr_scale=1,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.1,
            smoothness_loss_weight=0.0,
        )
        zoom = (shape[0] / 80, shape[1] / 80)
        norm = arr_scale / jnp.abs(arr_scale)
        expected_tx = ndimage.zoom(reference_tx, zoom, order=1) * norm
        expected_ty = ndimage.zoom(reference_ty, zoom, order=1) * norm

        tx, ty = self._compute_field(
            approximate_num_terms=200,
            scale=scale,
            shape=shape,
            arr_scale=arr_scale,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.1,
            smoothness_loss_weight=0.0,
        )
        onp.testing.assert_allclose(tx, expected_tx, atol=0.05)
        onp.testing.assert_allclose(ty, expected_ty, atol=0.05)

    @parameterized.expand(
        [
            # Cases with grayscale density and `use_jones_direct = False`.
            (1, (160, 120), 1, False, False),  # Resolution invariance.
            (1000, (80, 80), 1, False, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), False, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), False, False),
            (1, (80, 80), (0 + 1j), False, False),
            (1, (80, 80), 0.1, False, False),  # Amplitude invariance.
            (1, (80, 80), 10.0, False, False),
            # Cases with binarized density and `use_jones_direct = False`.
            (1, (160, 120), 1, True, False),  # Resolution invariance.
            (1000, (80, 80), 1, True, False),  # Scale invariance.
            (1, (80, 80), (1 + 0j), True, False),  # Phase invariance.
            (1, (80, 80), (1 / jnp.sqrt(2) + 1j / jnp.sqrt(2)), True, False),
            (1, (80, 80), (0 + 1j), True, False),
            (1, (80, 80), 0.1, True, False),
            (1, (80, 80), 10.0, True, False),
            # Cases with `use_jones_direct = True`.
            (1, (160, 120), 1, False, True),  # Grayscale, resolution invariance.
            (1000, (80, 80), 1, False, True),  # Grayscale, scale invariance.
            (1, (160, 120), 1, True, True),  # Binarized, resolution invariance.
            (1000, (80, 80), 1, True, True),  # Binarized, scale invariance.
        ]
    )
    def test_invariances_smoothness_loss(
        self, scale, shape, arr_scale, binarize_arr, use_jones_direct
    ):
        reference_tx, reference_ty = self._compute_field(
            approximate_num_terms=200,
            scale=1,
            shape=(80, 80),
            arr_scale=1,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.0,
            smoothness_loss_weight=1.0,
        )
        zoom = (shape[0] / 80, shape[1] / 80)
        norm = arr_scale / jnp.abs(arr_scale)
        expected_tx = ndimage.zoom(reference_tx, zoom, order=1) * norm
        expected_ty = ndimage.zoom(reference_ty, zoom, order=1) * norm

        tx, ty = self._compute_field(
            approximate_num_terms=200,
            scale=scale,
            shape=shape,
            arr_scale=arr_scale,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.0,
            smoothness_loss_weight=1.0,
        )
        onp.testing.assert_allclose(tx, expected_tx, atol=0.05)
        onp.testing.assert_allclose(ty, expected_ty, atol=0.05)

    @parameterized.expand(
        [
            [True, True, 0.1, 0.0],
            [True, False, 0.1, 0.0],
            [False, True, 0.1, 0.0],
            [False, False, 0.1, 0.0],
            [True, True, 0.0, 1.0],
            [True, False, 0.0, 1.0],
            [False, True, 0.0, 1.0],
            [False, False, 0.0, 1.0],
        ]
    )
    def test_supercell(
        self,
        binarize_arr,
        use_jones_direct,
        fourier_loss_weight,
        smoothness_loss_weight,
    ):
        approximate_num_terms = 200

        # Compute the non-supercell example.
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=approximate_num_terms,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        arr = _generate_array((80, 80), arr_scale=1, binarize_arr=binarize_arr)
        tx, ty = vector.compute_tangent_field(
            arr,
            expansion,
            primitive_lattice_vectors,
            use_jones_direct,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=1,
        )

        # Compute the supercell example.
        supercell_primitive_lattice_vectors = basis.LatticeVectors(
            basis.X * 2, basis.Y * 2
        )
        supercell_expansion = basis.generate_expansion(
            primitive_lattice_vectors=supercell_primitive_lattice_vectors,
            approximate_num_terms=4 * expansion.num_terms,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        supercell_tx, supercell_ty = vector.compute_tangent_field(
            jnp.tile(arr, (2, 2)),
            supercell_expansion,
            supercell_primitive_lattice_vectors,
            use_jones_direct,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
            steps=1,
        )
        onp.testing.assert_allclose(supercell_tx, jnp.tile(tx, (2, 2)), atol=5e-2)
        onp.testing.assert_allclose(supercell_ty, jnp.tile(ty, (2, 2)), atol=5e-2)

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

        tx_batch, ty_batch = vector.compute_tangent_field(
            arr,
            expansion,
            primitive_lattice_vectors,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=0.1,
            smoothness_loss_weight=1.0,
        )

        for i in range(arr.shape[0]):
            tx, ty = vector.compute_tangent_field(
                arr[i, :, :],
                expansion,
                primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.1,
                smoothness_loss_weight=1.0,
            )
            onp.testing.assert_allclose(tx, tx_batch[i, :, :], atol=1e-6)
            onp.testing.assert_allclose(ty, ty_batch[i, :, :], atol=1e-6)

    @parameterized.expand([[True], [False]])
    def test_gradient_no_nan(self, use_jones_direct):
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=200,
            truncation=basis.Truncation.CIRCULAR,
        )

        def loss_fn(arr):
            tx, ty = vector.compute_tangent_field(
                arr=arr,
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.01,
                smoothness_loss_weight=0.0,
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
            tx, ty = vector.compute_tangent_field(
                arr=arr,
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
                use_jones_direct=use_jones_direct,
                fourier_loss_weight=0.1,
                smoothness_loss_weight=0.0,
            )
            return jnp.sum(jnp.abs(tx) ** 2) + jnp.sum(jnp.abs(ty) ** 2), (tx, ty)

        arr = jnp.zeros((80, 80))
        (_, (tx, ty)), grad = jax.value_and_grad(loss_fn, has_aux=True)(arr)
        self.assertFalse(jnp.any(jnp.isnan(tx)))
        self.assertFalse(jnp.any(jnp.isnan(ty)))
        self.assertFalse(jnp.any(jnp.isnan(grad)))

    @parameterized.expand(
        [
            [True, True, 0.1, 0.0],
            [False, True, 0.1, 0.0],
            [True, False, 0.1, 0.0],
            [False, False, 0.1, 0.0],
            [True, True, 0.0, 1.0],
            [False, True, 0.0, 1.0],
            [True, False, 0.0, 1.0],
            [False, False, 0.0, 1.0],
        ]
    )
    def test_field_converges(
        self,
        binarize_arr,
        use_jones_direct,
        fourier_loss_weight,
        smoothness_loss_weight,
    ):
        # Test that a single Newton iteration is sufficient.
        field_fn = functools.partial(
            self._compute_field,
            approximate_num_terms=200,
            scale=1,
            shape=(80, 80),
            arr_scale=1,
            binarize_arr=binarize_arr,
            use_jones_direct=use_jones_direct,
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )

        tx, ty = field_fn(steps=2)
        tx_multi_step, ty_multi_step = field_fn(steps=3)
        onp.testing.assert_allclose(tx, tx_multi_step, atol=1e-4)
        onp.testing.assert_allclose(ty, ty_multi_step, atol=1e-4)


class NormalizeTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                vector.normalize_elementwise,
                [[1 / jnp.sqrt(2), 1.0, 0.0, 1.0, 0.0]],
                [[1 / jnp.sqrt(2), 0.0, 1.0, 0.0, 0.0]],
            ),
            (
                vector.normalize,
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
        txty = normalize_fn(jnp.stack([tx, ty], axis=-1))
        tx, ty = txty[..., 0], txty[..., 1]
        onp.testing.assert_allclose(tx, jnp.asarray(expected_tx), rtol=1e-3)
        onp.testing.assert_allclose(ty, jnp.asarray(expected_ty), rtol=1e-3)

    @parameterized.expand(
        [
            (vector.normalize_elementwise,),
            (vector.normalize,),
            (vector.normalize_jones,),
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
            (vector.normalize_elementwise,),
            (vector.normalize,),
            (vector.normalize_jones,),
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

    @parameterized.expand(
        [
            [vector.normalize_elementwise],
            [vector.normalize],
            [vector.normalize_jones],
        ]
    )
    def test_batch_calculation_matches_single(self, normalize_fn):
        onp.random.seed(0)
        arr = onp.random.randn(8, 10, 10, 2)
        arr = ndimage.zoom(arr, (1, 5, 5, 1))
        normalized = normalize_fn(arr)
        for i, arr_single in enumerate(arr):
            normalized_single = normalize_fn(arr_single)
            onp.testing.assert_array_equal(normalized[i, ...], normalized_single)


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
        result = vector._field_magnitude(jnp.stack([tx, ty], axis=-1))
        onp.testing.assert_allclose(result, expected)

    def test_magnitude_gradient_no_nan(self):
        grad = jax.grad(lambda x: jnp.squeeze(vector._field_magnitude(x)))(
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
        result = vector._angle(tx + 1j * ty)
        onp.testing.assert_allclose(result, expected)

    def test_angle_gradient_no_nan(self):
        grad = jax.grad(vector._angle)(0.0)
        self.assertFalse(onp.any(onp.isnan(grad)))


class TangentFieldMatchesExpectedTest(unittest.TestCase):
    @parameterized.expand(
        [
            [0.1, 0.0],
            [0.0, 1.0],
        ]
    )
    def test_field_pol(self, fourier_loss_weight, smoothness_loss_weight):
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        # Create an array that has a relatively large y-gradient, and a relatively
        # small x-gradient. This avoids the codepath which manually gives fields
        # when the array only varies in one direction.
        arr = jnp.concatenate([arr, arr * 0.99, arr * 0.98], axis=0)
        tx, ty = vector.compute_field_pol(
            arr,
            basis.Expansion(
                basis_coefficients=jnp.asarray(
                    [
                        [0, 0],
                        [0, 1],
                        [0, -1],
                        [0, 2],
                        [0, -2],
                        [0, 3],
                        [0, -3],
                        [1, 0],
                        [-1, 0],
                    ]
                )
            ),
            basis.LatticeVectors(basis.X, basis.Y),
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        expected_tx = jnp.asarray(
            [
                [
                    0.085,
                    0.300,
                    0.570,
                    0.840,
                    1.000,
                    0.910,
                    0.550,
                    0.000,
                    -0.550,
                    -0.910,
                    -1.000,
                    -0.840,
                    -0.570,
                    -0.300,
                    -0.085,
                ]
            ]
        )
        expected_tx = jnp.concatenate([expected_tx, expected_tx, expected_tx], axis=0)
        onp.testing.assert_allclose(tx, expected_tx, atol=0.05)
        onp.testing.assert_allclose(ty, 0.0, atol=0.05)

    def test_field_pol_gradient_y(self):
        # Create an array that has only a y-gradient. Only tx is nonzero.
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        tx, ty = vector.compute_field_pol(
            arr,
            basis.Expansion(
                basis_coefficients=jnp.asarray(
                    [[0, 0], [0, 1], [0, -1], [0, 2], [0, -2], [0, 3], [0, -3]]
                )
            ),
            basis.LatticeVectors(basis.X, basis.Y),
            fourier_loss_weight=0.1,
            smoothness_loss_weight=0.0,
        )
        onp.testing.assert_allclose(jnp.abs(tx.real), jnp.ones_like(tx), atol=1e-7)
        onp.testing.assert_allclose(jnp.abs(tx.imag), 0.0, atol=1e-6)
        onp.testing.assert_allclose(ty, 0.0, atol=1e-6)

    def test_field_pol_gradient_x(self):
        # Create an array that has only a x-gradient. Only ty is nonzero.
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        ).T
        tx, ty = vector.compute_field_pol(
            arr,
            basis.Expansion(
                basis_coefficients=jnp.asarray(
                    [[0, 0], [1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [-3, 0]]
                )
            ),
            basis.LatticeVectors(basis.X, basis.Y),
            fourier_loss_weight=0.1,
            smoothness_loss_weight=0.0,
        )
        onp.testing.assert_allclose(tx, 0.0, atol=1e-6)
        onp.testing.assert_allclose(jnp.abs(ty.real), jnp.ones_like(ty), atol=1e-7)
        onp.testing.assert_allclose(jnp.abs(ty.imag), 0.0, atol=1e-6)

    @parameterized.expand(
        [
            [0.1, 0.0],
            [0.0, 1.0],
        ]
    )
    def test_optimize_jones(self, fourier_loss_weight, smoothness_loss_weight):
        arr = jnp.array(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=jnp.float32
        )
        tx, ty = vector.compute_field_jones_direct(
            arr,
            basis.Expansion(
                basis_coefficients=jnp.asarray(
                    [[0, 0], [0, 1], [0, -1], [0, 2], [0, -2], [0, 3], [0, -3]]
                )
            ),
            basis.LatticeVectors(basis.X, basis.Y),
            fourier_loss_weight=fourier_loss_weight,
            smoothness_loss_weight=smoothness_loss_weight,
        )
        expected_tx_magnitude = jnp.ones_like(arr)
        onp.testing.assert_allclose(jnp.abs(tx), expected_tx_magnitude)
        onp.testing.assert_allclose(ty, 0.0, atol=1e-7)


class SchemesTest(unittest.TestCase):
    @parameterized.expand([(scheme,) for scheme in vector.VECTOR_FIELD_SCHEMES])
    def test_batch_matches_single_exact(self, scheme):
        key = jax.random.PRNGKey(0)
        arr = jax.random.uniform(key, shape=(5, 10, 10))
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )
        tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
            arr=arr,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        for i in range(5):
            expected_tx_i, expected_ty_i = vector.VECTOR_FIELD_SCHEMES[scheme](
                arr=arr[i, :, :],
                expansion=expansion,
                primitive_lattice_vectors=primitive_lattice_vectors,
            )
            onp.testing.assert_allclose(tx[i, :, :], expected_tx_i)
            onp.testing.assert_allclose(ty[i, :, :], expected_ty_i)

    @parameterized.expand([(scheme,) for scheme in vector.VECTOR_FIELD_SCHEMES])
    def test_uniform_array_no_nan(self, scheme):
        # The tangent field calculation requires special logic to handle uniform arrays,
        #  otherwise `nan` will show up in the result.
        arr = jnp.ones((10, 10))
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )
        tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
            arr=arr,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        self.assertFalse(onp.any(onp.isnan(tx)))
        self.assertFalse(onp.any(onp.isnan(ty)))

    @parameterized.expand([(scheme,) for scheme in vector.VECTOR_FIELD_SCHEMES])
    def test_gradient_no_nan(self, scheme):
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )

        def _loss_fn(arr):
            tx, ty = vector.VECTOR_FIELD_SCHEMES[scheme](
                arr=arr,
                expansion=expansion,
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

    def test_permittivity_with_singleton(self):
        # Manually create expansion for a 1D permittivity.
        nmax = 150
        ix = onp.zeros((2 * nmax + 1,), dtype=int)
        ix[1::2] = -onp.arange(1, nmax + 1, dtype=int)
        ix[2::2] = onp.arange(1, nmax + 1, dtype=int)
        assert tuple(ix[:5].tolist()) == (0, -1, 1, -2, 2)
        expansion = basis.Expansion(
            basis_coefficients=onp.stack([ix, onp.zeros_like(ix)], axis=-1)
        )

        primitive_lattice_vectors = basis.LatticeVectors(
            u=basis.X * 10,
            v=basis.Y,
        )
        arr = jax.random.uniform(jax.random.PRNGKey(0), (30, 1)) > 0.5
        arr = jnp.kron(arr, jnp.ones((20, 1))).astype(float)

        field = vector._compute_tangent_field_no_batch(
            arr=arr,
            expansion=expansion,
            primitive_lattice_vectors=primitive_lattice_vectors,
            use_jones_direct=True,
            fourier_loss_weight=0.05,
            smoothness_loss_weight=0.0,
            steps=1,
        )

        self.assertFalse(jnp.any(jnp.isnan(field)))
