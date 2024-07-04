"""Tests for `fmmax.fmm`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fmm, utils

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


class LayerEigensolveTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(formulation,) for formulation in fmm.Formulation]
    )
    def test_uniform_matches_patterned(self, formulation):
        permittivity = jnp.asarray([[3.14]])
        uniform_result = fmm._eigensolve_uniform_isotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
        )
        uniform_eigenvalues, uniform_eigenvectors = _sort_eigs(
            uniform_result.eigenvalues, uniform_result.eigenvectors
        )
        patterned_result = fmm._eigensolve_patterned_isotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=jnp.broadcast_to(permittivity, (64, 64)),
            expansion=EXPANSION,
            formulation=formulation,
        )
        patterned_eigenvalues, patterned_eigenvectors = _sort_eigs(
            patterned_result.eigenvalues, patterned_result.eigenvectors
        )
        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(
                uniform_eigenvalues**2, patterned_eigenvalues**2
            )
        with self.subTest("z_permittivity_matrix"):
            onp.testing.assert_allclose(
                uniform_result.z_permittivity_matrix,
                patterned_result.z_permittivity_matrix,
            )
        with self.subTest("omega_script_k_matrix"):
            onp.testing.assert_allclose(
                uniform_result.omega_script_k_matrix,
                patterned_result.omega_script_k_matrix,
            )

    def test_uniform_layer_batch_matches_single(self):
        wavelength = jnp.asarray([[[0.2]], [[0.3]], [[0.4]]])
        in_plane_wavevector = jnp.asarray([[[0.0, 0.0]], [[0.05, 0.08]]])
        permittivity = jnp.asarray([[[3.14]], [[6.28]]])
        result = fmm._eigensolve_uniform_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
        )
        for i, w in enumerate(wavelength[:, 0, 0]):
            for j, ipwv in enumerate(in_plane_wavevector[:, 0, :]):
                for k, p in enumerate(permittivity[:, :, :]):
                    single_result = fmm._eigensolve_uniform_isotropic_media(
                        wavelength=w,
                        in_plane_wavevector=ipwv,
                        primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                        permittivity=p,
                        expansion=EXPANSION,
                    )
                    onp.testing.assert_array_equal(
                        result.eigenvalues[i, j, k, :],
                        single_result.eigenvalues,
                    )
                    onp.testing.assert_array_equal(
                        result.eigenvectors[i, j, k, :, :],
                        single_result.eigenvectors,
                    )
                    onp.testing.assert_array_equal(
                        result.z_permittivity_matrix[i, j, k, :, :],
                        single_result.z_permittivity_matrix,
                    )
                    onp.testing.assert_array_equal(
                        result.omega_script_k_matrix[i, j, k, :, :],
                        single_result.omega_script_k_matrix,
                    )

    @parameterized.parameterized.expand(
        [(formulation,) for formulation in fmm.Formulation]
    )
    def test_patterned_layer_batch_matches_single(self, formulation):
        wavelength = jnp.asarray([[[0.2]], [[0.3]], [[0.4]]])
        in_plane_wavevector = jnp.asarray([[[0.0, 0.0]], [[0.05, 0.08]]])
        permittivity = jnp.asarray(
            [
                1.0 + jax.random.uniform(jax.random.PRNGKey(0), shape=(64, 60)),
                1.0 + jax.random.uniform(jax.random.PRNGKey(1), shape=(64, 60)),
            ]
        )
        result = fmm._eigensolve_patterned_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=formulation,
        )
        for i, w in enumerate(wavelength[:, 0, 0]):
            for j, ipwv in enumerate(in_plane_wavevector[:, 0, :]):
                for k, p in enumerate(permittivity[:, :, :]):
                    single_result = fmm._eigensolve_patterned_isotropic_media(
                        wavelength=w,
                        in_plane_wavevector=ipwv,
                        primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                        permittivity=p,
                        expansion=EXPANSION,
                        formulation=formulation,
                    )
                    eigenvalues, eigenvectors = _sort_eigs(
                        result.eigenvalues[i, j, k, :],
                        result.eigenvectors[i, j, k, :, :],
                    )
                    expected_eigenvalues, expected_eigenvectors = _sort_eigs(
                        single_result.eigenvalues, single_result.eigenvectors
                    )
                    onp.testing.assert_allclose(eigenvalues**2, expected_eigenvalues**2)
                    onp.testing.assert_allclose(
                        result.z_permittivity_matrix[i, j, k, :, :],
                        single_result.z_permittivity_matrix,
                    )
                    onp.testing.assert_allclose(
                        result.omega_script_k_matrix[i, j, k, :, :],
                        single_result.omega_script_k_matrix,
                    )

    def test_shape_insufficient_validation(self):
        permittivity = jnp.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            fmm._eigensolve_patterned_isotropic_media(
                wavelength=WAVELENGTH,
                in_plane_wavevector=IN_PLANE_WAVEVECTOR,
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                permittivity=permittivity,
                expansion=EXPANSION,
                formulation=fmm.Formulation.FFT,
            )


class EigensolveJitTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.FFT,),
            (fmm.Formulation.POL,),
        ]
    )
    def test_can_jit(self, formulation):
        wavelength = jnp.asarray([[[0.2]], [[0.3]], [[0.4]]])
        in_plane_wavevector = jnp.asarray([[[0.0, 0.0]], [[0.05, 0.08]]])
        permittivity = jnp.asarray(
            [
                1.0 + jax.random.uniform(jax.random.PRNGKey(0), shape=(64, 60)),
                1.0 + jax.random.uniform(jax.random.PRNGKey(1), shape=(64, 60)),
            ]
        )
        result = fmm._eigensolve_patterned_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=formulation,
        )
        jit_fn = jax.jit(
            functools.partial(
                fmm._eigensolve_patterned_isotropic_media,
                expansion=EXPANSION,
            )
        )
        jit_result = jit_fn(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            formulation=formulation,
        )
        onp.testing.assert_allclose(result.eigenvalues**2, jit_result.eigenvalues**2)


class AnistropicLayerEigensolveTest(unittest.TestCase):
    def test_compare_when_layer_is_isotropic(self):
        wavelength = jnp.asarray([[[0.2]], [[0.3]], [[0.4]]])
        in_plane_wavevector = jnp.asarray([[[0.0, 0.0]], [[0.05, 0.08]]])
        permittivity = jnp.asarray(
            [
                1.0 + jax.random.uniform(jax.random.PRNGKey(0), shape=(64, 60)),
                1.0 + jax.random.uniform(jax.random.PRNGKey(1), shape=(64, 60)),
            ]
        )
        expected = fmm._eigensolve_patterned_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        result = fmm._eigensolve_patterned_general_anisotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivities=(
                permittivity,
                jnp.zeros_like(permittivity),
                jnp.zeros_like(permittivity),
                permittivity,
                permittivity,
            ),
            permeabilities=(
                jnp.ones_like(permittivity),
                jnp.zeros_like(permittivity),
                jnp.zeros_like(permittivity),
                jnp.ones_like(permittivity),
                jnp.ones_like(permittivity),
            ),
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
            vector_field_source=permittivity,
        )
        onp.testing.assert_allclose(result.eigenvalues**2, expected.eigenvalues**2)

    def test_uniform_matches_patterned(self):
        permittivity_xx = jnp.asarray([[2.0]])
        permittivity_xy = jnp.asarray([[0.1]])
        permittivity_yx = jnp.asarray([[0.2]])
        permittivity_yy = jnp.asarray([[2.5]])
        permittivity_zz = jnp.asarray([[3.0]])
        permeability_xx = jnp.asarray([[5.0]])
        permeability_xy = jnp.asarray([[0.2]])
        permeability_yx = jnp.asarray([[0.4]])
        permeability_yy = jnp.asarray([[1.5]])
        permeability_zz = jnp.asarray([[1.3]])
        uniform_result = fmm._eigensolve_uniform_general_anisotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivities=(
                permittivity_xx,
                permittivity_xy,
                permittivity_yx,
                permittivity_yy,
                permittivity_zz,
            ),
            permeabilities=(
                permeability_xx,
                permeability_xy,
                permeability_yx,
                permeability_yy,
                permeability_zz,
            ),
            expansion=EXPANSION,
        )
        uniform_eigenvalues, uniform_eigenvectors = _sort_eigs(
            uniform_result.eigenvalues, uniform_result.eigenvectors
        )
        vector_field_source = (permittivity_xx + permittivity_yy) / 2
        vector_field_source = jnp.broadcast_to(vector_field_source, (64, 64))
        patterned_result = fmm._eigensolve_patterned_general_anisotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivities=(
                jnp.broadcast_to(permittivity_xx, (64, 64)),
                jnp.broadcast_to(permittivity_xy, (64, 64)),
                jnp.broadcast_to(permittivity_yx, (64, 64)),
                jnp.broadcast_to(permittivity_yy, (64, 64)),
                jnp.broadcast_to(permittivity_zz, (64, 64)),
            ),
            permeabilities=(
                jnp.broadcast_to(permeability_xx, (64, 64)),
                jnp.broadcast_to(permeability_xy, (64, 64)),
                jnp.broadcast_to(permeability_yx, (64, 64)),
                jnp.broadcast_to(permeability_yy, (64, 64)),
                jnp.broadcast_to(permeability_zz, (64, 64)),
            ),
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
            vector_field_source=vector_field_source,
        )
        patterned_eigenvalues, patterned_eigenvectors = _sort_eigs(
            patterned_result.eigenvalues, patterned_result.eigenvectors
        )
        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(
                uniform_eigenvalues**2, patterned_eigenvalues**2
            )
        with self.subTest("z_permittivity_matrix"):
            onp.testing.assert_allclose(
                uniform_result.z_permittivity_matrix,
                patterned_result.z_permittivity_matrix,
            )
        with self.subTest("omega_script_k_matrix"):
            onp.testing.assert_allclose(
                uniform_result.omega_script_k_matrix,
                patterned_result.omega_script_k_matrix,
            )


class AnistropicLayerFFTMatrixTest(unittest.TestCase):
    def test_anisotropic_fft_matrices_match_isotropic_for_isotropic_media(self):
        permittivity = 1 + jax.random.uniform(jax.random.PRNGKey(0), (50, 50))
        (
            inverse_z_permittivity_matrix_expected,
            z_permittivity_matrix_expected,
            transverse_permittivity_matrix_expected,
            tangent_vector_field_expected,
        ) = fmm.fourier_matrices_patterned_isotropic_media(
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        (
            inverse_z_permittivity_matrix,
            z_permittivity_matrix,
            transverse_permittivity_matrix,
            _,
            _,
            _,
            tangent_vector_field,
        ) = fmm.fourier_matrices_patterned_anisotropic_media(
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivities=(
                permittivity,
                jnp.zeros_like(permittivity),
                jnp.zeros_like(permittivity),
                permittivity,
                permittivity,
            ),
            permeabilities=(
                jnp.ones_like(permittivity),
                jnp.zeros_like(permittivity),
                jnp.zeros_like(permittivity),
                jnp.ones_like(permittivity),
                jnp.ones_like(permittivity),
            ),
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
            vector_field_source=permittivity,
        )
        onp.testing.assert_array_equal(
            inverse_z_permittivity_matrix, inverse_z_permittivity_matrix_expected
        )
        onp.testing.assert_array_equal(
            z_permittivity_matrix, z_permittivity_matrix_expected
        )
        onp.testing.assert_array_equal(
            transverse_permittivity_matrix, transverse_permittivity_matrix_expected
        )
        onp.testing.assert_array_equal(
            tangent_vector_field, tangent_vector_field_expected
        )


class FourierMatrixBatchMatchesSingleTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.FFT,),
            (fmm.Formulation.JONES_DIRECT,),
            (fmm.Formulation.JONES,),
            (fmm.Formulation.NORMAL,),
            (fmm.Formulation.POL,),
        ]
    )
    def test_single_matches_batch_isotropic(self, formulation):
        x, y = jnp.meshgrid(
            jnp.linspace(-0.5, 0.5),
            jnp.linspace(-0.5, 0.5),
            indexing="ij",
        )
        circle = (jnp.sqrt(x**2 + y**2) <= 0.2).astype(float)
        scale = jnp.arange(1, 5)[:, jnp.newaxis, jnp.newaxis]
        permittivity = 1 + circle * scale

        (
            batch_inverse_z_permittivity_matrix,
            batch_z_permittivity_matrix,
            batch_transverse_permittivity_matrix,
            batch_tangent_vector_field,
        ) = fmm.fourier_matrices_patterned_isotropic_media(
            PRIMITIVE_LATTICE_VECTORS, permittivity, EXPANSION, formulation
        )

        for i, p in enumerate(permittivity):
            (
                inverse_z_permittivity_matrix,
                z_permittivity_matrix,
                transverse_permittivity_matrix,
                tangent_vector_field,
            ) = fmm.fourier_matrices_patterned_isotropic_media(
                PRIMITIVE_LATTICE_VECTORS, p, EXPANSION, formulation
            )
            onp.testing.assert_allclose(
                inverse_z_permittivity_matrix,
                batch_inverse_z_permittivity_matrix[i, ...],
                atol=1e-15,
            )
            onp.testing.assert_allclose(
                z_permittivity_matrix, batch_z_permittivity_matrix[i, ...], atol=1e-15
            )
            onp.testing.assert_allclose(
                transverse_permittivity_matrix,
                batch_transverse_permittivity_matrix[i, ...],
                atol=1e-15,
            )

    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.FFT,),
            (fmm.Formulation.JONES_DIRECT,),
            (fmm.Formulation.JONES,),
            (fmm.Formulation.NORMAL,),
            (fmm.Formulation.POL,),
        ]
    )
    def test_single_matches_batch_anisotropic(self, formulation):
        x, y = jnp.meshgrid(
            jnp.linspace(-0.5, 0.5),
            jnp.linspace(-0.5, 0.5),
            indexing="ij",
        )
        circle = (jnp.sqrt(x**2 + y**2) <= 0.2).astype(float)
        permittivities_scale = jnp.arange(30).reshape(5, 6, 1, 1) * 0.1
        permittivities = 1 + circle * permittivities_scale
        permeabilities_scale = jnp.arange(30).reshape(5, 6, 1, 1) * 0.3
        permeabilities = 1 + circle * permeabilities_scale
        assert permittivities.shape == (5, 6, 50, 50)

        (
            batch_inverse_z_permittivity_matrix,
            batch_z_permittivity_matrix,
            batch_transverse_permittivity_matrix,
            batch_inverse_z_permeability_matrix,
            batch_z_permeability_matrix,
            batch_transverse_permeability_matrix,
            batch_tangent_vector_field,
        ) = fmm.fourier_matrices_patterned_anisotropic_media(
            PRIMITIVE_LATTICE_VECTORS,
            tuple(permittivities),
            tuple(permeabilities),
            EXPANSION,
            formulation,
            permittivities[0, ...],
        )

        for i in range(permittivities.shape[1]):
            (
                inverse_z_permittivity_matrix,
                z_permittivity_matrix,
                transverse_permittivity_matrix,
                inverse_z_permeability_matrix,
                z_permeability_matrix,
                transverse_permeability_matrix,
                tangent_vector_field,
            ) = fmm.fourier_matrices_patterned_anisotropic_media(
                PRIMITIVE_LATTICE_VECTORS,
                tuple(permittivities[:, i, ...]),
                tuple(permeabilities[:, i, ...]),
                EXPANSION,
                formulation,
                permittivities[0, i, ...],
            )
            onp.testing.assert_allclose(
                inverse_z_permittivity_matrix,
                batch_inverse_z_permittivity_matrix[i, ...],
                atol=1e-15,
            )
            onp.testing.assert_allclose(
                z_permittivity_matrix, batch_z_permittivity_matrix[i, ...], atol=1e-15
            )
            onp.testing.assert_allclose(
                transverse_permittivity_matrix,
                batch_transverse_permittivity_matrix[i, ...],
                atol=1e-15,
            )
            onp.testing.assert_allclose(
                inverse_z_permeability_matrix,
                batch_inverse_z_permeability_matrix[i, ...],
                atol=1e-15,
            )
            onp.testing.assert_allclose(
                z_permeability_matrix, batch_z_permeability_matrix[i, ...], atol=1e-15
            )
            onp.testing.assert_allclose(
                transverse_permeability_matrix,
                batch_transverse_permeability_matrix[i, ...],
                atol=1e-15,
            )
            if formulation != fmm.Formulation.FFT:
                onp.testing.assert_allclose(
                    tangent_vector_field[0],
                    batch_tangent_vector_field[0][i, ...],
                    atol=1e-15,
                )
                onp.testing.assert_allclose(
                    tangent_vector_field[1],
                    batch_tangent_vector_field[1][i, ...],
                    atol=1e-15,
                )
            else:
                self.assertEqual(tangent_vector_field, None)


class SignSelectionTest(unittest.TestCase):
    def test_select_sign(self):
        eigenvalues = jnp.array(
            [1 + 1e-7j, 1 - 1e-7j, 1e-3 + 1e-7j, 1e-3 - 1e-7j, 1e-7 - 1e-3j]
        )
        expected = jnp.array(
            [1 + 1e-7j, -1 + 1e-7j, 1e-3 + 1e-7j, -1e-3 + 1e-7j, -1e-7 + 1e-3j]
        )
        result = fmm._select_eigenvalues_sign(eigenvalues)
        onp.testing.assert_array_equal(result, expected)


class LayerSolveResultInputValidationTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("wavelength", jnp.ones((1,))),
            ("in_plane_wavevector", jnp.ones((1,))),
            ("eigenvalues", jnp.ones((1,))),
            ("eigenvectors", jnp.ones((3, 4, 5, 1, 1))),
            ("z_permittivity_matrix", jnp.ones((1,))),
            ("inverse_z_permittivity_matrix", jnp.ones((1,))),
            ("z_permeability_matrix", jnp.ones((1,))),
            ("inverse_z_permeability_matrix", jnp.ones((1,))),
            ("transverse_permeability_matrix", jnp.ones((1,))),
            ("tangent_vector_field", (jnp.ones((1,)), jnp.ones((1,)))),
        ]
    )
    def test_invalid_shape(self, name, invalid_shape):
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            approximate_num_terms=20,
            truncation=basis.Truncation.CIRCULAR,
        )
        num = expansion.num_terms
        kwargs = {
            "wavelength": jnp.ones((3, 1, 1)),
            "in_plane_wavevector": jnp.ones((1, 4, 5, 2)),
            "primitive_lattice_vectors": basis.LatticeVectors(
                u=basis.X * jnp.ones((1, 1, 1, 2)),
                v=basis.Y * jnp.ones((1, 1, 1, 2)),
            ),
            "expansion": expansion,
            "eigenvalues": jnp.ones((3, 4, 5, 2 * num)),
            "eigenvectors": jnp.ones((3, 4, 5, 2 * num, 2 * num)),
            "z_permittivity_matrix": jnp.ones((3, 4, 5, num, num)),
            "inverse_z_permittivity_matrix": jnp.ones((3, 4, 5, num, num)),
            "z_permeability_matrix": jnp.ones((3, 4, 5, num, num)),
            "inverse_z_permeability_matrix": jnp.ones((3, 4, 5, num, num)),
            "transverse_permeability_matrix": jnp.ones((3, 4, 5, 2 * num, 2 * num)),
            "tangent_vector_field": (
                jnp.ones((1, 1, 2, 64, 60)),
                jnp.ones((1, 1, 2, 64, 60)),
            ),
        }
        kwargs[name] = invalid_shape
        with self.assertRaisesRegex(ValueError, f"`{name}` must have "):
            fmm.LayerSolveResult(**kwargs)
