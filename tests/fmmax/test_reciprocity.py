"""Tests for testing reciprocity of s matrices.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import basis, fmm, scattering, utils

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
# max absolute tolerance for allclose
MAX_ATOL = 1e-15


class ReciprocityTest(unittest.TestCase):
    def _generate_s_matrix_from_density(
        self, density_array, index_solid, index_void, top_index=0, bottom_index=0
    ):
        permittivities = (
            jnp.full((1, 1), top_index**2),
            utils.interpolate_permittivity(
                permittivity_solid=jnp.asarray(index_solid**2),
                permittivity_void=jnp.asarray(index_void**2),
                density=density_array,
            ),
            jnp.full((1, 1), bottom_index**2),
        )
        layer_solve_results = [
            fmm.eigensolve_isotropic_media(
                wavelength=jnp.asarray(WAVELENGTH),
                in_plane_wavevector=jnp.zeros((2,)),  # normal incidence
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                permittivity=p,
                expansion=EXPANSION,
                formulation=fmm.Formulation.FFT,
            )
            for p in permittivities
        ]

        # Layer thicknesses for the ambient and substrate are set to zero; these do not
        # affect the result of the calculation.
        layer_thicknesses = (jnp.zeros(()), jnp.asarray(0.325), jnp.zeros(()))

        return scattering.stack_s_matrix(layer_solve_results, layer_thicknesses)

    def test_uniform_slightly_lossy(self):
        """Test reciprocity for a stack with uniform lossy permittivity."""
        extinction = 1e-2j
        with self.subTest("vacumm for top and bottom"):
            density_array = jnp.ones((64, 128))
            s_matrix = self._generate_s_matrix_from_density(
                density_array, 1.0 + extinction, 0.0 + extinction
            )
            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)
        with self.subTest("top_index = 1.0, bottom_index = 1.5"):
            density_array = jnp.ones((64, 128))
            s_matrix = self._generate_s_matrix_from_density(
                density_array,
                1.0 + extinction,
                0.0 + extinction,
                top_index=1.0,
                bottom_index=1.5,
            )
            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)

    def test_uniform_lossless(self):
        """Test reciprocity for a stack with uniform lossless permittivity."""
        density_array = jnp.ones((64, 128))
        with self.subTest("epsilon = 1.0"):
            s_matrix = self._generate_s_matrix_from_density(density_array, 1.0, 0.0)
            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)
        with self.subTest("epsilon = 3.0"):
            s_matrix = self._generate_s_matrix_from_density(density_array, 3.0, 0.0)
            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)
        with self.subTest("top_index = 1.0, bottom_index = 1.5"):
            s_matrix = self._generate_s_matrix_from_density(
                density_array, 1.0, 0.0, top_index=1.0, bottom_index=1.5
            )
            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)

    def test_random_lossless(self):
        """Test reciprocity for a stack with random lossless permittivity."""
        key = jax.random.PRNGKey(0)  # Random seed is explicit in JAX
        density_array = jax.random.uniform(key, shape=(64, 128))
        s_matrix = self._generate_s_matrix_from_density(density_array, 1.0, 0.0)

        onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)

    def test_patterned_lossless(self):
        """Test reciprocity for a patterned permittivity."""
        with self.subTest("large structure"):
            # draw a rectangle array
            density_array = jnp.zeros((64, 128))
            density_array.at[20:40, 40:80].set(1)
            s_matrix = self._generate_s_matrix_from_density(density_array, 2.0, 0.0)

            onp.testing.assert_allclose(s_matrix.s12, s_matrix.s21, atol=MAX_ATOL)
