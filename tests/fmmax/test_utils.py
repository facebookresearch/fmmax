"""Tests for `fmmax.utils`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

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
    @parameterized.expand(
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
        onp.testing.assert_allclose(eigval, expected_eigval, rtol=1e-12)
        onp.testing.assert_allclose(eigvec, expected_eigvec, rtol=1e-12)

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

    @parameterized.expand(
        [
            (1.0, 0.0),
            (1.0, 1e3),
            (1.0, 1e6),
            (1e-6, 0.0),
            (1e-6, 1e3),
            (1e6, 0.0),
        ]
    )
    def test_matches_eigh_hermetian_real_matrix(self, matrix_scale, eigval_shift):
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
        matrix *= matrix_scale
        matrix += jnp.eye(matrix.shape[-1]) * eigval_shift
        onp.testing.assert_array_equal(matrix, jnp.transpose(matrix, (0, 2, 1)))

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(_eig(matrix)[0], _eigh(matrix)[0])

        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(_eig(matrix)[1], _eigh(matrix)[1], rtol=1e-5)

        with self.subTest("eigenvalue_jac"):
            expected_eigval_jac = jax.jacrev(lambda m: _eigh(m)[0])(matrix)
            eigval_jac = jax.jacrev(lambda m: _eig(m)[0], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigval_jac, expected_eigval_jac, rtol=1e-4)

        with self.subTest("eigenvectors_jac"):
            expected_eigvec_jac = jax.jacrev(
                lambda m: _eigh(m)[1],
                holomorphic=True,
            )(matrix)
            eigvec_jac = jax.jacrev(lambda m: _eig(m)[1], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigvec_jac, expected_eigvec_jac, rtol=1e-4)

    @parameterized.expand(
        [
            (1.0, 0.0),
            (1.0, 1e2),
            (1.0, 1e6),
            (1e-6, 0.0),
            (1e-6, 1e3),
            (1e6, 0.0),
        ]
    )
    def test_matches_eigh_hermetian_complex_matrix(self, matrix_scale, eigval_shift):
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
        matrix *= matrix_scale
        matrix += jnp.eye(matrix.shape[-1]) * eigval_shift
        onp.testing.assert_array_equal(matrix, utils.matrix_adjoint(matrix))

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(_eig(matrix)[0], _eigh(matrix)[0])

        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(_eig(matrix)[1], _eigh(matrix)[1], rtol=1e-5)

        with self.subTest("eigenvalues_jac"):
            expected_eigval_jac = jax.jacrev(lambda m: _eigh(m)[0])(matrix)
            eigval_jac = jax.jacrev(lambda m: _eig(m)[0], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigval_jac, expected_eigval_jac, rtol=1e-4)

        with self.subTest("eigenvectors_jac"):
            expected_eigvec_jac = jax.jacrev(
                lambda m: _eigh(m)[1],
                holomorphic=True,
            )(matrix)
            eigvec_jac = jax.jacrev(lambda m: _eig(m)[1], holomorphic=True)(matrix)
            onp.testing.assert_allclose(eigvec_jac, expected_eigvec_jac, rtol=1e-4)

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

    def test_can_vmap(self):
        matrix = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4))
        matrix += 1j * jax.random.normal(jax.random.PRNGKey(1), (2, 4, 4))

        batch_eigval, batch_eigvec = utils.eig(matrix)
        vmap_eigval, vmap_eigvec = jax.vmap(utils.eig)(matrix)

        onp.testing.assert_array_equal(vmap_eigval, batch_eigval)
        onp.testing.assert_array_equal(vmap_eigvec, vmap_eigvec)


class AbsoluteAxesTest(unittest.TestCase):
    @parameterized.expand(([(0, 0), 3], [(1, -2), 3]))
    def test_absolute_axes_duplicates(self, axes, ndim):
        with self.assertRaisesRegex(ValueError, "Found duplicates in `axes`"):
            utils.absolute_axes(axes, ndim)

    @parameterized.expand(([(3,), 3], [(-4,), 3]))
    def test_absolute_axes_out_of_range(self, axes, ndim):
        with self.assertRaisesRegex(
            ValueError, "All elements of `axes` must be in the range"
        ):
            utils.absolute_axes(axes, ndim)

    @parameterized.expand(
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
