"""Tests `fmmax.fmm` with comparisons to `grcwa`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import grcwa
import jax
import jax.numpy as jnp
import numpy as onp

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


class GrcwaEigensolveComparisonTest(unittest.TestCase):
    def test_uniform_layer_eigenvalues_and_eigenvectors(self):
        # Compares the eigenvalues and eigenvectors for a uniform layer to those
        # obtained by grcwa.
        permittivity = jnp.asarray([[3.14]])
        solve_result = fmm._eigensolve_uniform_isotropic_media(
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
        solve_result = fmm._eigensolve_patterned_isotropic_media(
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
