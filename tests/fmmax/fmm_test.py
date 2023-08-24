"""Tests for `fmmax.fmm`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fmm

# Enable 64-bit precision for higher-accuracy.
jax.config.update("jax_enable_x64", True)


PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(
    u=jnp.array([1, 0]), v=jnp.array([0, 1])
)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=30,
    truncation=basis.Truncation.CIRCULAR,
)


class TransversePermittivityTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.JONES_DIRECT,),
            (fmm.Formulation.JONES,),
            (fmm.Formulation.NORMAL,),
            (fmm.Formulation.POL,),
        ]
    )
    def test_single_matches_batch_vector(self, fmm_formulation):
        x, y = jnp.meshgrid(
            jnp.linspace(-0.5, 0.5),
            jnp.linspace(-0.5, 0.5),
            indexing="ij",
        )
        circle = (jnp.sqrt(x**2 + y**2) <= 0.2).astype(float)
        scale = jnp.arange(1, 5)[:, jnp.newaxis, jnp.newaxis]
        permittivity = 1 + circle * scale
        result_batch = fmm._transverse_permittivity_vector(
            PRIMITIVE_LATTICE_VECTORS, permittivity, EXPANSION, fmm_formulation
        )
        result_single = [
            fmm._transverse_permittivity_vector(
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                permittivity=p,
                expansion=EXPANSION,
                formulation=fmm_formulation,
            )
            for p in permittivity
        ]
        onp.testing.assert_allclose(result_batch, result_single, atol=1e-15)


class AnistropicLayerTest(unittest.TestCase):
    def test_compare_when_layer_is_isotropic(self):
        permittivity = 1 + jax.random.uniform(jax.random.PRNGKey(0), (50, 50))
        (
            inverse_z_permittivity_matrix_expected,
            z_permittivity_matrix_expected,
            transverse_permittivity_matrix_expected,
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
