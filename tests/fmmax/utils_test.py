"""Tests for `fmmax.utils`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import utils

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
RTOL_FD = 1e-3


def _jacfwd_fd(fn, delta=1e-6):
    """Forward mode jacobian by finite differences."""

    def _jac_fn(x):
        f0 = fn(x)
        jac = jnp.zeros(f0.shape + x.shape, dtype=f0.dtype)
        for inds in itertools.product(*[range(dim) for dim in x.shape]):
            offset = jnp.zeros_like(x).at[inds].set(delta)
            grad = (fn(x + offset / 2) - fn(x - offset / 2)) / delta
            jac_inds = tuple([slice(0, d) for d in f0.shape]) + inds
            jac = jac.at[jac_inds].set(grad)
        return jac

    return _jac_fn


def _sort_eigs(eigvals, eigvecs):
    """Sorts eigenvalues/eigenvectors and enforces a phase convention."""
    assert eigvals.shape[:-1] == eigvecs.shape[:-2]
    assert eigvecs.shape[-2:] == (eigvals.shape[-1],) * 2
    order = jnp.argsort(jnp.abs(eigvals), axis=-1)
    sorted_eigvals = jnp.take_along_axis(eigvals, order, axis=-1)
    sorted_eigvecs = jnp.take_along_axis(eigvecs, order[..., jnp.newaxis, :], axis=-1)
    assert eigvals.shape == sorted_eigvals.shape
    assert eigvecs.shape == sorted_eigvecs.shape
    # Set the phase of the largest component to zero.
    max_ind = jnp.argmax(jnp.abs(sorted_eigvecs), axis=-2)
    max_component = jnp.take_along_axis(
        sorted_eigvecs, max_ind[..., jnp.newaxis, :], axis=-2
    )
    sorted_eigvecs = sorted_eigvecs / jnp.exp(1j * jnp.angle(max_component))
    assert eigvecs.shape == sorted_eigvecs.shape
    return sorted_eigvals, sorted_eigvecs


def _reference_padded_conv(x, kernel, padding_mode):
    # Conservative padding equal to twice the kernel max dimension.
    p = 2 * max(kernel.shape)
    x_padded = onp.pad(x, ((p, p), (p, p)), padding_mode)
    y = jax.scipy.signal.convolve2d(x_padded, kernel[::-1, ::-1], mode="same")
    i_lo = p if kernel.shape[0] % 2 == 1 else p + 1
    j_lo = p if kernel.shape[1] % 2 == 1 else p + 1
    return y[i_lo : (i_lo + x.shape[0]), j_lo : (j_lo + x.shape[1])]


class DiagTest(unittest.TestCase):
    def test_diag_matches_expected(self):
        shapes = ((5,), (2, 5), (9, 1, 8))
        for shape in shapes:
            with self.subTest(shape):
                v = jax.random.uniform(jax.random.PRNGKey(0), shape)
                d = utils.diag(v)
                expected = jnp.zeros(shape + (shape[-1],))
                for ind in itertools.product(*[range(dim) for dim in shape[:-1]]):
                    expected = expected.at[ind].set(jnp.diag(v[ind]))
                onp.testing.assert_allclose(d, expected)


class AngularFrequencyTest(unittest.TestCase):
    def test_value_matches_expected(self):
        self.assertEqual(
            utils.angular_frequency_for_wavelength(2.71), 2 * jnp.pi / 2.71
        )


class MatrixAdjointTest(unittest.TestCase):
    def test_adjoint_matches_expected(self):
        shapes = ((5, 5), (2, 5, 5), (9, 1, 8, 8))
        for shape in shapes:
            with self.subTest(shape):
                m = jax.random.uniform(jax.random.PRNGKey(0), shape)
                ma = utils.matrix_adjoint(m)
                expected = jnp.zeros(shape)
                for ind in itertools.product(*[range(dim) for dim in shape[:-2]]):
                    expected = expected.at[ind].set(jnp.conj(m[ind]).T)
                onp.testing.assert_allclose(ma, expected)


class BatchCompatibleTest(unittest.TestCase):
    def test_value_matches_expected(self):
        shapes_and_expected = [
            ([(1, 2), (2,)], True),
            ([(1, 2), (3,)], False),
            ([(1, 2), (4, 2, 2)], True),
            ([(1, 2), ()], True),
        ]
        for shapes, expected in shapes_and_expected:
            with self.subTest(shapes):
                self.assertEqual(utils.batch_compatible_shapes(*shapes), expected)


class AtLeastNDTest(unittest.TestCase):
    def test_shape_matches_expected(self):
        shapes_n_expected = [
            [(2, 1), 1, (2, 1)],
            [(2, 1), 2, (2, 1)],
            [(2, 1), 3, (1, 2, 1)],
            [(2, 1), 4, (1, 1, 2, 1)],
        ]
        for shape, n, expected_shape in shapes_n_expected:
            with self.subTest(n):
                x = onp.zeros(shape)
                self.assertSequenceEqual(utils.atleast_nd(x, n).shape, expected_shape)


class InterpolateTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (4.0, 2.0, 0.0, 2.0),
            (4.0, 2.0, 1.0, 4.0),
            (4.0, 2.0, 0.5, (jnp.sqrt(2) * 0.5 + jnp.sqrt(4) * 0.5) ** 2),
            (4.0 + 1.0j, 2.0, 0.0, 2.0),
            (4.0 + 1.0j, 2.0, 1.0, 4.0 + 1.0j),
            (4.0 + 1.0j, 2.0, 0.5, (jnp.sqrt(2) * 0.5 + jnp.sqrt(4 + 1.0j) * 0.5) ** 2),
        ]
    )
    def test_interpolated_matches_expected(self, p_solid, p_void, density, expected):
        result = utils.interpolate_permittivity(p_solid, p_void, density)
        onp.testing.assert_allclose(result, expected)


class EigTest(unittest.TestCase):
    def test_no_nan_gradient_with_degenerate_eigenvalues(self):
        matrix = jnp.asarray([[2.0, 0.0, 2.0], [0.0, -2.0, 0.0], [2.0, 0.0, -1.0]])
        eigval_grad = jax.grad(lambda m: jnp.sum(jnp.abs(utils.eig(m)[0])))(matrix)
        eigvec_grad = jax.grad(lambda m: jnp.sum(jnp.abs(utils.eig(m)[1])))(matrix)
        self.assertFalse(onp.any(onp.isnan(eigval_grad)))
        self.assertFalse(onp.any(onp.isnan(eigvec_grad)))

    def test_value_matches_eig_with_nondegenerate_eigenvalues(self):
        matrix = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4))
        matrix += 1j * jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4))
        expected_eigval, expected_eigvec = jnp.linalg.eig(
            jax.device_put(matrix, device=jax.devices("cpu")[0])
        )
        eigval, eigvec = utils.eig(matrix)
        onp.testing.assert_array_equal(eigval, expected_eigval)
        onp.testing.assert_array_equal(eigvec, expected_eigvec)

    def test_eigvalue_jacobian_matches_expected_real_matrix(self):
        matrix = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4)).astype(complex)
        expected_jac = jax.jacrev(jnp.linalg.eigvals, holomorphic=True)(
            jax.device_put(matrix, device=jax.devices("cpu")[0])
        )
        jac = jax.jacrev(lambda x: utils.eig(x)[0], holomorphic=True)(matrix)
        onp.testing.assert_allclose(jac, expected_jac, rtol=RTOL)

    def test_eigvalue_jacobian_matches_expected_complex_matrix(self):
        matrix = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4))
        matrix += 1j * jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4))
        expected_jac = jax.jacrev(jnp.linalg.eigvals, holomorphic=True)(
            jax.device_put(matrix, device=jax.devices("cpu")[0])
        )
        jac = jax.jacrev(lambda x: utils.eig(x)[0], holomorphic=True)(matrix)
        onp.testing.assert_allclose(jac, expected_jac, rtol=RTOL)

    def test_matches_eigh_hermetian_real_matrix(self):
        # Compares against `eigh`, which is valid only for Hermetian matrices. `eig`
        # and `eigh` return eigenvalues in different, random order. We must sort
        # them to facilitiate comparison.
        def _eigh(m):
            return _sort_eigs(*jnp.linalg.eigh(m, symmetrize_input=False))

        def _eig(m):
            return _sort_eigs(*utils.eig(m))

        matrix = jax.random.normal(jax.random.PRNGKey(0), (32,))
        matrix = matrix.reshape((2, 4, 4)).astype(complex)
        matrix = matrix + utils.matrix_adjoint(matrix)
        onp.testing.assert_array_equal(matrix, jnp.transpose(matrix, (0, 2, 1)))

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(_eig(matrix)[0], _eigh(matrix)[0])

        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(_eig(matrix)[1], _eigh(matrix)[1])

        with self.subTest("eigenvalue_jac"):
            expected_eigval_jac = jax.jacrev(lambda m: _eigh(m)[0])(matrix)
            eigval_jac = jax.jacrev(lambda m: _eig(m)[0], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigval_jac, expected_eigval_jac, rtol=RTOL)

        with self.subTest("eigenvectors_jac"):
            expected_eigvec_jac = jax.jacrev(
                lambda m: _eigh(m)[1],
                holomorphic=True,
            )(matrix)
            eigvec_jac = jax.jacrev(lambda m: _eig(m)[1], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigvec_jac, expected_eigvec_jac, rtol=RTOL)

    def test_matches_eigh_hermetian_complex_matrix(self):
        # Compares against `eigh`, which is valid only for Hermetian matrices. `eig`
        # and `eigh` return eigenvalues in different, random order. We must sort
        # them to facilitiate comparison.
        def _eigh(m):
            return _sort_eigs(*jnp.linalg.eigh(m, symmetrize_input=False))

        def _eig(m):
            return _sort_eigs(*utils.eig(m))

        matrix = jax.random.normal(jax.random.PRNGKey(0), (32,))
        matrix = matrix + 1j * jax.random.normal(jax.random.PRNGKey(1), (32,))
        matrix = matrix.reshape((2, 4, 4)).astype(complex)
        matrix = matrix + utils.matrix_adjoint(matrix)
        onp.testing.assert_array_equal(matrix, utils.matrix_adjoint(matrix))

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(_eig(matrix)[0], _eigh(matrix)[0])

        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(_eig(matrix)[1], _eigh(matrix)[1])

        with self.subTest("eigenvalues_jac"):
            expected_eigval_jac = jax.jacrev(lambda m: _eigh(m)[0])(matrix)
            eigval_jac = jax.jacrev(lambda m: _eig(m)[0], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigval_jac, expected_eigval_jac, rtol=RTOL)

        with self.subTest("eigenvectors_jac"):
            expected_eigvec_jac = jax.jacrev(
                lambda m: _eigh(m)[1],
                holomorphic=True,
            )(matrix)
            eigvec_jac = jax.jacrev(lambda m: _eig(m)[1], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigvec_jac, expected_eigvec_jac, rtol=RTOL)

    def test_eigvec_jac_matches_fd_hermetian_matrix(self):
        # Tests that a finite-difference jacobian matches that computed by the
        # custom vjp rule. Here, the input and output to the function are real,
        # but internally a complex, Hermetian matrix is passed to the
        # eigendecomposition.
        def fn(x):
            x = x + 1j * x
            x = x + utils.matrix_adjoint(x)
            _, eigvec = utils.eig(x)
            return jnp.abs(eigvec)

        matrix = jax.random.normal(jax.random.PRNGKey(0), (32,))
        matrix = matrix.reshape((2, 4, 4))

        jac = jax.jacrev(fn)(matrix)
        expected_jac = _jacfwd_fd(fn)(matrix)
        onp.testing.assert_allclose(jac, expected_jac, rtol=RTOL_FD)

    def test_eigvec_jac_matches_fd_general_complex_matrix(self):
        # Tests that a finite-difference jacobian matches that computed by the
        # custom vjp rule. Here, the input and output to the function are real,
        # but internally a complex, non-Hermetian matrix is passed to the
        # eigendecomposition.
        def fn(x):
            x = x + 1j * x
            _, eigvec = _sort_eigs(*utils.eig(x))
            return jnp.abs(eigvec)

        matrix = jax.random.normal(jax.random.PRNGKey(0), (32,))
        matrix = matrix.reshape((2, 4, 4))

        jac = jax.jacrev(fn)(matrix)
        expected_jac = _jacfwd_fd(fn)(matrix)
        onp.testing.assert_allclose(jac, expected_jac, rtol=RTOL_FD)


class MagnitudeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, jnp.sqrt(2)),
        ]
    )
    def test_magnitude(self, tx, ty, expected):
        result = utils.magnitude(tx, ty)
        onp.testing.assert_allclose(result, expected)

    def test_magnitude_gradient_no_nan(self):
        grad = jax.grad(utils.magnitude, argnums=(0, 1))(0.0, 0.0)
        self.assertFalse(onp.any(onp.isnan(grad)))


class AngleTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (0, 0, 0),
            (0, 1, jnp.pi / 2),
            (1, 0, 0),
            (1, 1, jnp.pi / 4),
        ]
    )
    def test_angle(self, tx, ty, expected):
        result = utils.angle(tx + 1j * ty)
        onp.testing.assert_allclose(result, expected)

    def test_angle_gradient_no_nan(self):
        grad = jax.grad(utils.angle)(0.0)
        self.assertFalse(onp.any(onp.isnan(grad)))


class PaddedConvTest(unittest.TestCase):
    def test_conv_matches_reference_single(self):
        onp.random.seed(0)
        x = onp.random.rand(20, 25)
        kernel = onp.random.rand(5, 4)
        result = utils.padded_conv(x, kernel, "edge")
        expected = _reference_padded_conv(x, kernel, "edge")
        onp.testing.assert_allclose(result, expected)

    def test_conv_matches_reference_batch(self):
        onp.random.seed(0)
        x = onp.random.rand(2, 3, 20, 25)
        kernel = onp.random.rand(5, 4)

        result = utils.padded_conv(x, kernel, "edge")

        expected = onp.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                expected[i, j, :, :] = _reference_padded_conv(
                    x[i, j, :, :], kernel, "edge"
                )

        onp.testing.assert_allclose(result, expected)


class KernelTest(unittest.TestCase):
    def test_gaussian_kernel_matches_expected(self):
        kernel = utils.gaussian_kernel((3, 3), 2)
        expected = jnp.asarray(
            [
                [0.25, 0.50, 0.25],
                [0.50, 1.00, 0.50],
                [0.25, 0.50, 0.25],
            ]
        )
        onp.testing.assert_allclose(kernel, expected, rtol=1e-6)

    def test_gaussian_kernel_is_scale_independent(self):
        kernel_coarse = utils.gaussian_kernel((5, 7), 3)
        kernel_fine = utils.gaussian_kernel((25, 35), 15)
        onp.testing.assert_allclose(kernel_coarse, kernel_fine[2::5, 2::5], rtol=1e-6)


class ResampleTest(unittest.TestCase):
    @parameterized.parameterized.expand([[(1, 3)], [(2, 3)], [(1, 6)]])
    def test_downsampled_matches_expected(self, target_shape):
        # Downsampling where the target shape evenly divides the original
        # shape is equivalent to box downsampling.
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = utils.box_downsample(x, target_shape)
        result = utils.resample(x, target_shape, method=jax.image.ResizeMethod.CUBIC)
        onp.testing.assert_allclose(result, expected)

    @parameterized.parameterized.expand([[(4, 12)], [(2, 12)], [(3, 9)]])
    def test_upsampled_matches_expected(self, target_shape):
        # Upsampling is equivalent to `jax.image.resize`.
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = jax.image.resize(
            x, target_shape, method=jax.image.ResizeMethod.CUBIC
        )
        result = utils.resample(x, target_shape, method=jax.image.ResizeMethod.CUBIC)
        onp.testing.assert_allclose(result, expected)


class BoxDownsampleTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            [(4, 4), (3, 3)],
            [(4, 3), (3, 1)],
            [(4, 4, 1), (1, 1)],
        ]
    )
    def test_downsample_factor_validation(self, arr_shape, target_shape):
        with self.assertRaisesRegex(
            ValueError, "Each axis of `shape` must evenly divide "
        ):
            utils.box_downsample(jnp.ones(arr_shape), shape=target_shape)

    def test_downsampled_matches_expected(self):
        x = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        expected = jnp.asarray([[14 / 4, 22 / 4, 30 / 4]])
        result = utils.box_downsample(x, (1, 3))
        onp.testing.assert_allclose(result, expected)

    def test_upsample_downsample(self):
        onp.random.seed(0)
        shape = (5, 2, 8, 3, 3)
        factor = 4
        original = jnp.asarray(onp.random.rand(*shape))

        kernel = jnp.ones((factor,) * original.ndim, dtype=original.dtype)
        upsampled = jnp.kron(original, kernel)

        expected_shape = tuple([factor * d for d in shape])
        self.assertSequenceEqual(upsampled.shape, expected_shape)
        downsampled_upsampled = utils.box_downsample(upsampled, original.shape)
        onp.testing.assert_allclose(downsampled_upsampled, original, rtol=1e-6)


class AbsoluteAxesTest(unittest.TestCase):
    @parameterized.parameterized.expand(([(0, 0), 3], [(1, -2), 3]))
    def test_absolute_axes_duplicates(self, axes, ndim):
        with self.assertRaisesRegex(ValueError, "Found duplicates in `axes`"):
            utils.absolute_axes(axes, ndim)

    @parameterized.parameterized.expand(([(3,), 3], [(-4,), 3]))
    def test_absolute_axes_out_of_range(self, axes, ndim):
        with self.assertRaisesRegex(
            ValueError, "All elements of `axes` must be in the range"
        ):
            utils.absolute_axes(axes, ndim)

    @parameterized.parameterized.expand(
        (
            [(0, 1, 2, 3), 4, (0, 1, 2, 3)],
            [(0, 1, 2, 3), 6, (0, 1, 2, 3)],
            [(-4, -3, -2, -1), 4, (0, 1, 2, 3)],
            [(-6, -5, -4, -3), 6, (0, 1, 2, 3)],
        )
    )
    def test_absolute_axes_match_expected(self, axes, ndim, expected_axes):
        absolute_axes = utils.absolute_axes(axes, ndim)
        self.assertSequenceEqual(absolute_axes, expected_axes)
