"""Tests for `fmmax.layer`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import grcwa
import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fmm, layer, utils

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


class GrcwaComparisonTest(unittest.TestCase):
    def test_uniform_layer_eigenvalues_and_eigenvectors(self):
        # Compares the eigenvalues and eigenvectors for a uniform layer to those
        # obtained by grcwa.
        permittivity = jnp.asarray([[3.14]])
        solve_result = layer.eigensolve_uniform_isotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
        )

        angular_frequency = utils.angular_frequency_for_wavelength(WAVELENGTH)
        transverse_wavevectors = basis.transverse_wavevectors(
            IN_PLANE_WAVEVECTOR, PRIMITIVE_LATTICE_VECTORS, EXPANSION
        )
        kx = transverse_wavevectors[:, 0]
        ky = transverse_wavevectors[:, 1]
        (
            expected_eigenvalues,
            expected_eigenvectors,
        ) = grcwa.rcwa.SolveLayerEigensystem_uniform(
            omega=angular_frequency,
            kx=kx,
            ky=ky,
            epsilon=onp.squeeze(permittivity),
        )

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(
                solve_result.eigenvalues**2, expected_eigenvalues**2
            )
        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(
                solve_result.eigenvectors, expected_eigenvectors
            )

    def test_patterned_layer_eigenvalues_and_eigenvectors(self):
        # Compares the eigenvalues and eigenvectors for a patterned layer to those
        # obtained by grcwa.
        permittivity = 1.0 + jax.random.uniform(jax.random.PRNGKey(0), shape=(64, 64))
        solve_result = layer.eigensolve_patterned_isotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        eigenvalues, eigenvectors = _sort_eigs(
            solve_result.eigenvalues, solve_result.eigenvectors
        )

        angular_frequency = utils.angular_frequency_for_wavelength(WAVELENGTH)
        transverse_wavevectors = basis.transverse_wavevectors(
            IN_PLANE_WAVEVECTOR, PRIMITIVE_LATTICE_VECTORS, EXPANSION
        )
        kx = transverse_wavevectors[:, 0]
        ky = transverse_wavevectors[:, 1]
        epinv, ep2 = grcwa.fft_funs.Epsilon_fft(
            dN=1 / onp.prod(permittivity.shape),
            eps_grid=permittivity,
            G=EXPANSION.basis_coefficients,
        )
        kp = grcwa.rcwa.MakeKPMatrix(
            omega=angular_frequency,
            layer_type=1,  # `1` is for patterned layers.
            epinv=epinv,
            kx=kx,
            ky=ky,
        )
        expected_eigenvalues, expected_eigenvectors = grcwa.rcwa.SolveLayerEigensystem(
            omega=angular_frequency,
            kx=kx,
            ky=ky,
            kp=kp,
            ep2=ep2,
        )
        expected_eigenvalues, expected_eigenvectors = _sort_eigs(
            expected_eigenvalues, expected_eigenvectors
        )

        with self.subTest("eigenvalues"):
            onp.testing.assert_allclose(eigenvalues**2, expected_eigenvalues**2)
        with self.subTest("eigenvectors"):
            onp.testing.assert_allclose(eigenvectors, expected_eigenvectors)


class LayerTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.FFT,),
            (fmm.Formulation.POL,),
            (fmm.Formulation.NORMAL,),
            (fmm.Formulation.JONES,),
            (fmm.Formulation.JONES_DIRECT,),
        ]
    )
    def test_uniform_matches_patterned(self, formulation):
        permittivity = jnp.asarray([[3.14]])
        uniform_result = layer.eigensolve_uniform_isotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
        )
        uniform_eigenvalues, uniform_eigenvectors = _sort_eigs(
            uniform_result.eigenvalues, uniform_result.eigenvectors
        )
        patterned_result = layer.eigensolve_patterned_isotropic_media(
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
        result = layer.eigensolve_uniform_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
        )
        for i, w in enumerate(wavelength[:, 0, 0]):
            for j, ipwv in enumerate(in_plane_wavevector[:, 0, :]):
                for k, p in enumerate(permittivity[:, :, :]):
                    single_result = layer.eigensolve_uniform_isotropic_media(
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
        [
            (fmm.Formulation.FFT,),
            (fmm.Formulation.POL,),
            (fmm.Formulation.NORMAL,),
            (fmm.Formulation.JONES,),
            (fmm.Formulation.JONES_DIRECT,),
        ]
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
        result = layer.eigensolve_patterned_isotropic_media(
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
                    single_result = layer.eigensolve_patterned_isotropic_media(
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
                    onp.testing.assert_allclose(
                        eigenvalues**2, expected_eigenvalues**2
                    )
                    onp.testing.assert_allclose(
                        result.z_permittivity_matrix[i, j, k, :, :],
                        single_result.z_permittivity_matrix,
                    )
                    onp.testing.assert_allclose(
                        result.omega_script_k_matrix[i, j, k, :, :],
                        single_result.omega_script_k_matrix,
                    )

    def test_shape_insufficient_validation(self):
        min_shape = fmm._min_array_shape_for_expansion(EXPANSION)
        permittivity = jnp.ones(tuple([d - 1 for d in min_shape]))
        with self.assertRaisesRegex(ValueError, "`shape` is insufficient for"):
            layer.eigensolve_patterned_isotropic_media(
                wavelength=WAVELENGTH,
                in_plane_wavevector=IN_PLANE_WAVEVECTOR,
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                permittivity=permittivity,
                expansion=EXPANSION,
                formulation=fmm.Formulation.FFT,
            )


class JitTest(unittest.TestCase):
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
        result = layer.eigensolve_patterned_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=formulation,
        )
        jit_fn = jax.jit(
            functools.partial(
                layer.eigensolve_patterned_isotropic_media,
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
        onp.testing.assert_allclose(
            result.eigenvalues**2, jit_result.eigenvalues**2
        )


class AnistropicLayerTest(unittest.TestCase):
    def test_compare_when_layer_is_isotropic(self):
        wavelength = jnp.asarray([[[0.2]], [[0.3]], [[0.4]]])
        in_plane_wavevector = jnp.asarray([[[0.0, 0.0]], [[0.05, 0.08]]])
        permittivity = jnp.asarray(
            [
                1.0 + jax.random.uniform(jax.random.PRNGKey(0), shape=(64, 60)),
                1.0 + jax.random.uniform(jax.random.PRNGKey(1), shape=(64, 60)),
            ]
        )
        expected = layer.eigensolve_patterned_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        result = layer.eigensolve_patterned_anisotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity_xx=permittivity,
            permittivity_xy=jnp.zeros_like(permittivity),
            permittivity_yx=jnp.zeros_like(permittivity),
            permittivity_yy=permittivity,
            permittivity_zz=permittivity,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        onp.testing.assert_allclose(result.eigenvalues**2, expected.eigenvalues**2)

    def test_uniform_matches_patterned(self):
        permittivity_xx = jnp.asarray([[2.0]])
        permittivity_xy = jnp.asarray([[0.1]])
        permittivity_yx = jnp.asarray([[0.2]])
        permittivity_yy = jnp.asarray([[2.5]])
        permittivity_zz = jnp.asarray([[3.0]])
        uniform_result = layer.eigensolve_uniform_anisotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity_xx=permittivity_xx,
            permittivity_xy=permittivity_xy,
            permittivity_yx=permittivity_yx,
            permittivity_yy=permittivity_yy,
            permittivity_zz=permittivity_zz,
            expansion=EXPANSION,
        )
        uniform_eigenvalues, uniform_eigenvectors = _sort_eigs(
            uniform_result.eigenvalues, uniform_result.eigenvectors
        )
        patterned_result = layer.eigensolve_patterned_anisotropic_media(
            wavelength=WAVELENGTH,
            in_plane_wavevector=IN_PLANE_WAVEVECTOR,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            permittivity_xx=jnp.broadcast_to(permittivity_xx, (64, 64)),
            permittivity_xy=jnp.broadcast_to(permittivity_xy, (64, 64)),
            permittivity_yx=jnp.broadcast_to(permittivity_yx, (64, 64)),
            permittivity_yy=jnp.broadcast_to(permittivity_yy, (64, 64)),
            permittivity_zz=jnp.broadcast_to(permittivity_zz, (64, 64)),
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
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


class UtilityFunctionTests(unittest.TestCase):
    def test_select_sign(self):
        eigenvalues = jnp.array(
            [1 + 1e-7j, 1 - 1e-7j, 1e-3 + 1e-7j, 1e-3 - 1e-7j, 1e-7 - 1e-3j]
        )
        expected = jnp.array(
            [1 + 1e-7j, -1 + 1e-7j, 1e-3 + 1e-7j, -1e-3 + 1e-7j, -1e-7 + 1e-3j]
        )
        result = layer._select_eigenvalues_sign(eigenvalues)
        onp.testing.assert_array_equal(result, expected)


class LayerSolveResultInputValidationTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("wavelength", jnp.ones((1,))),
            ("in_plane_wavevector", jnp.ones((1,))),
            ("eigenvalues", jnp.ones((1,))),
            ("eigenvectors", jnp.ones((3, 4, 5, 1, 1))),
            ("eta_matrix", jnp.ones((1,))),
            ("z_permittivity_matrix", jnp.ones((1,))),
            ("omega_script_k_matrix", jnp.ones((1,))),
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
            "primitive_lattice_vectors": PRIMITIVE_LATTICE_VECTORS,
            "expansion": expansion,
            "eigenvalues": jnp.ones((3, 4, 5, 2 * num)),
            "eigenvectors": jnp.ones((3, 4, 5, 2 * num, 2 * num)),
            "eta_matrix": jnp.ones((3, 4, 5, num, num)),
            "z_permittivity_matrix": jnp.ones((3, 4, 5, num, num)),
            "omega_script_k_matrix": jnp.ones((3, 4, 5, 2 * num, 2 * num)),
        }
        kwargs[name] = invalid_shape
        with self.assertRaisesRegex(ValueError, f"`{name}` must have "):
            layer.LayerSolveResult(**kwargs)
