"""Tests for `fmmax.scattering`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import unittest

import grcwa
import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import basis, fmm, layer, scattering, sources

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)


WAVELENGTH = jnp.asarray(0.628)
PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(
    u=onp.asarray((0.9, 0.1)), v=onp.asarray((0.2, 0.8))
)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=4,
    truncation=basis.Truncation.CIRCULAR,
)
IN_PLANE_WAVEVECTOR = basis.plane_wave_in_plane_wavevector(
    wavelength=WAVELENGTH,
    polar_angle=jnp.pi * 0.1,
    azimuthal_angle=jnp.pi * 0.2,
    permittivity=jnp.asarray(1.0),
)


def _random_normal_complex(key, shape):
    x = jax.random.normal(key, (2,) + shape)
    return x[0, ...] + 1j * x[1, ...]


def _dummy_solve_result(
    key,
    wavelength=WAVELENGTH,
    in_plane_wavevector=IN_PLANE_WAVEVECTOR,
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    expansion=EXPANSION,
):
    key_eigenvalues, key_eigenvectors, key_omega_k = jax.random.split(key, 3)
    dim = expansion.basis_coefficients.shape[0]
    return layer.LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=_random_normal_complex(key_eigenvalues, (2 * dim,)),
        eigenvectors=_random_normal_complex(key_eigenvectors, (2 * dim, 2 * dim)),
        eta_matrix=_random_normal_complex(key_eigenvectors, (dim, dim)),
        z_permittivity_matrix=_random_normal_complex(key_eigenvectors, (dim, dim)),
        omega_script_k_matrix=_random_normal_complex(key_omega_k, (2 * dim, 2 * dim)),
    )


def _stack_solve_result(
    key,
    wavelength=WAVELENGTH,
    in_plane_wavevector=IN_PLANE_WAVEVECTOR,
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    expansion=EXPANSION,
):
    permittivities = [
        jnp.array([[1.0 + 0.0j]]),
        jnp.array([[1.0 + 0.0j]]),
        jax.random.normal(key, (128, 128)) * 0.1 + (3.0 + 0.0j),
        jnp.array([[5.0 + 0.0j]]),
        jnp.array([[5.0 + 0.0j]]),
    ]
    return [
        layer.eigensolve_isotropic_media(
            wavelength,
            in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        for p in permittivities
    ]


class ScatteringMatrixTest(unittest.TestCase):
    def test_compare_to_grcwa(self):
        solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        thicknesses = [1.0, 1.5, 2.0, 2.5]
        result = scattering.stack_s_matrix(solve_results, thicknesses)
        expected_s11, expected_s12, expected_s21, expected_s22 = grcwa.rcwa.GetSMatrix(
            indi=0,
            indj=len(solve_results) - 1,
            q_list=[r.eigenvalues for r in solve_results],
            phi_list=[r.eigenvectors for r in solve_results],
            kp_list=[r.omega_script_k_matrix for r in solve_results],
            thickness_list=thicknesses,
        )
        with self.subTest("s11"):
            onp.testing.assert_allclose(result.s11, expected_s11)
        with self.subTest("s12"):
            onp.testing.assert_allclose(result.s12, expected_s12)
        with self.subTest("s21"):
            onp.testing.assert_allclose(result.s21, expected_s21)
        with self.subTest("s22"):
            onp.testing.assert_allclose(result.s22, expected_s22)

    def test_compare_to_grcwa_actual_solve(self):
        solve_results = _stack_solve_result(jax.random.PRNGKey(0))
        thicknesses = [1.0, 1.5, 2.0, 2.5, 1.0]
        result = scattering.stack_s_matrix(solve_results, thicknesses)
        expected_s11, expected_s12, expected_s21, expected_s22 = grcwa.rcwa.GetSMatrix(
            indi=0,
            indj=len(solve_results) - 1,
            q_list=[r.eigenvalues for r in solve_results],
            phi_list=[r.eigenvectors for r in solve_results],
            kp_list=[r.omega_script_k_matrix for r in solve_results],
            thickness_list=thicknesses,
        )
        with self.subTest("s11"):
            onp.testing.assert_allclose(result.s11, expected_s11)
        with self.subTest("s12"):
            onp.testing.assert_allclose(result.s12, expected_s12)
        with self.subTest("s21"):
            onp.testing.assert_allclose(result.s21, expected_s21)
        with self.subTest("s22"):
            onp.testing.assert_allclose(result.s22, expected_s22)

    def test_construct_s_matrix_with_prepend(self):
        solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        thicknesses = [1.0, 1.5, 2.0, 2.5]
        expected = scattering.stack_s_matrix(solve_results, thicknesses)

        s_matrix = scattering.stack_s_matrix(solve_results[-2:], thicknesses[-2:])
        s_matrix = scattering.prepend_layer(s_matrix, solve_results[1], thicknesses[1])
        s_matrix = scattering.prepend_layer(s_matrix, solve_results[0], thicknesses[0])

        with self.subTest("s11"):
            onp.testing.assert_allclose(s_matrix.s11, expected.s11)
        with self.subTest("s12"):
            onp.testing.assert_allclose(s_matrix.s12, expected.s12)
        with self.subTest("s21"):
            onp.testing.assert_allclose(s_matrix.s21, expected.s21)
        with self.subTest("s22"):
            onp.testing.assert_allclose(s_matrix.s22, expected.s22)

    def test_interior_s_matrices(self):
        solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        thicknesses = [1.0, 1.5, 2.0, 2.5]
        s_matrices_interior = scattering.stack_s_matrices_interior(
            solve_results, thicknesses
        )

        # Test that that the "above" s-matrix for the first layer
        # and the "below" s-matrix for the last layer are equal.
        a, b = (s_matrices_interior[0][1], s_matrices_interior[-1][0])
        with self.subTest("s11-above/below"):
            onp.testing.assert_allclose(a.s11, b.s11)
        with self.subTest("s12-above/below"):
            onp.testing.assert_allclose(a.s12, b.s12)
        with self.subTest("s21-above/below"):
            onp.testing.assert_allclose(a.s21, b.s21)
        with self.subTest("s22-above/below"):
            onp.testing.assert_allclose(a.s22, b.s22)

        # Test that that the "below" s-matrix for the first layer
        # and the "above" s-matrix for the last layer are equal.
        a, b = (s_matrices_interior[0][0], s_matrices_interior[-1][1])
        with self.subTest("s11-below/above"):
            onp.testing.assert_allclose(a.s11, b.s11)
        with self.subTest("s12-below/above"):
            onp.testing.assert_allclose(a.s12, b.s12)
        with self.subTest("s21-below/above"):
            onp.testing.assert_allclose(a.s21, b.s21)
        with self.subTest("s22-below/above"):
            onp.testing.assert_allclose(a.s22, b.s22)

    def test_interior_s_matrices_actual_solve(self):
        solve_results = _stack_solve_result(jax.random.PRNGKey(0))
        thicknesses = [1.0, 1.5, 2.0, 2.5, 1.0]
        s_matrices_interior = scattering.stack_s_matrices_interior(
            solve_results, thicknesses
        )

        # Test that that the "above" s-matrix for the first layer
        # and the "below" s-matrix for the last layer are equal.
        a, b = (s_matrices_interior[0][1], s_matrices_interior[-1][0])
        with self.subTest("s11-above/below"):
            onp.testing.assert_allclose(a.s11, b.s11)
        with self.subTest("s12-above/below"):
            onp.testing.assert_allclose(a.s12, b.s12)
        with self.subTest("s21-above/below"):
            onp.testing.assert_allclose(a.s21, b.s21)
        with self.subTest("s22-above/below"):
            onp.testing.assert_allclose(a.s22, b.s22)

        # Test that that the "below" s-matrix for the first layer
        # and the "above" s-matrix for the last layer are equal.
        a, b = (s_matrices_interior[0][0], s_matrices_interior[-1][1])
        with self.subTest("s11-below/above"):
            onp.testing.assert_allclose(a.s11, b.s11)
        with self.subTest("s12-below/above"):
            onp.testing.assert_allclose(a.s12, b.s12)
        with self.subTest("s21-below/above"):
            onp.testing.assert_allclose(a.s21, b.s21)
        with self.subTest("s22-below/above"):
            onp.testing.assert_allclose(a.s22, b.s22)

    def test_can_jit(self):
        solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        thicknesses = [1.0, 1.5, 2.0, 2.5]
        expected = scattering.stack_s_matrix(solve_results, thicknesses)
        result = jax.jit(scattering.stack_s_matrix)(solve_results, thicknesses)

        with self.subTest("s11"):
            onp.testing.assert_allclose(result.s11, expected.s11)
        with self.subTest("s12"):
            onp.testing.assert_allclose(result.s12, expected.s12)
        with self.subTest("s21"):
            onp.testing.assert_allclose(result.s21, expected.s21)
        with self.subTest("s22"):
            onp.testing.assert_allclose(result.s22, expected.s22)

    def test_insensitive_to_eigenvalue_ordering(self):
        # Sanity check which validates that the ordering of the eigenvalues and
        # eigenvectors in an internal layer does not impact the final scattering
        # matrix.
        solve_results = _stack_solve_result(jax.random.PRNGKey(0))
        thicknesses = [1.0, 1.5, 2.0, 2.5, 1.0]

        expected = scattering.stack_s_matrix(solve_results, thicknesses)

        q = solve_results[2].eigenvalues
        phi = solve_results[2].eigenvectors
        swapped_q = jnp.concatenate([q[::2], q[1::2]])
        swapped_phi = jnp.concatenate([phi[:, ::2], phi[:, 1::2]], axis=1)

        swapped = dataclasses.replace(
            solve_results[2], eigenvalues=swapped_q, eigenvectors=swapped_phi
        )
        swapped_solve_results = [
            solve_results[0],
            solve_results[1],
            swapped,
            solve_results[3],
            solve_results[4],
        ]
        result = scattering.stack_s_matrix(swapped_solve_results, thicknesses)

        with self.subTest("s11"):
            onp.testing.assert_allclose(result.s11, expected.s11)
        with self.subTest("s12"):
            onp.testing.assert_allclose(result.s12, expected.s12)
        with self.subTest("s21"):
            onp.testing.assert_allclose(result.s21, expected.s21)
        with self.subTest("s22"):
            onp.testing.assert_allclose(result.s22, expected.s22)


class ChangeLayerThicknessTest(unittest.TestCase):
    def test_change_start_layer_thickness(self):
        layer_solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        original = scattering.stack_s_matrix(
            layer_solve_results, layer_thicknesses=[0.5, 0.2, 0.3, 0.2]
        )
        adjusted = scattering.set_start_layer_thickness(original, 0.2)
        expected = scattering.stack_s_matrix(
            layer_solve_results, layer_thicknesses=[0.2, 0.2, 0.3, 0.2]
        )

        onp.testing.assert_allclose(adjusted.s11, expected.s11)
        onp.testing.assert_allclose(adjusted.s21, expected.s21)
        onp.testing.assert_allclose(adjusted.s22, expected.s22)
        onp.testing.assert_allclose(adjusted.s12, expected.s12)
        onp.testing.assert_allclose(
            adjusted.start_layer_thickness, expected.start_layer_thickness
        )
        onp.testing.assert_allclose(
            adjusted.end_layer_thickness, expected.end_layer_thickness
        )

    def test_change_end_layer_thickness(self):
        layer_solve_results = [
            _dummy_solve_result(jax.random.PRNGKey(0)),
            _dummy_solve_result(jax.random.PRNGKey(1)),
            _dummy_solve_result(jax.random.PRNGKey(2)),
            _dummy_solve_result(jax.random.PRNGKey(3)),
        ]
        original = scattering.stack_s_matrix(
            layer_solve_results, layer_thicknesses=[0.2, 0.2, 0.3, 0.5]
        )
        adjusted = scattering.set_end_layer_thickness(original, 0.25)
        expected = scattering.stack_s_matrix(
            layer_solve_results, layer_thicknesses=[0.2, 0.2, 0.3, 0.25]
        )

        onp.testing.assert_allclose(adjusted.s11, expected.s11)
        onp.testing.assert_allclose(adjusted.s21, expected.s21)
        onp.testing.assert_allclose(adjusted.s22, expected.s22)
        onp.testing.assert_allclose(adjusted.s12, expected.s12)
        onp.testing.assert_allclose(
            adjusted.start_layer_thickness, expected.start_layer_thickness
        )
        onp.testing.assert_allclose(
            adjusted.end_layer_thickness, expected.end_layer_thickness
        )
