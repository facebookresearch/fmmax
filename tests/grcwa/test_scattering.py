"""Tests for `fmmax.scattering` using `grcwa`..

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import grcwa
import jax
import jax.numpy as jnp
import numpy as onp

from fmmax import basis, fmm, scattering

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
    keys = jax.random.split(key, 8)
    dim = expansion.basis_coefficients.shape[0]
    return fmm.LayerSolveResult(
        wavelength=wavelength,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
        eigenvalues=_random_normal_complex(keys[0], (2 * dim,)),
        eigenvectors=_random_normal_complex(keys[1], (2 * dim, 2 * dim)),
        z_permittivity_matrix=_random_normal_complex(keys[2], (dim, dim)),
        inverse_z_permittivity_matrix=_random_normal_complex(keys[3], (dim, dim)),
        z_permeability_matrix=_random_normal_complex(keys[4], (dim, dim)),
        inverse_z_permeability_matrix=_random_normal_complex(keys[5], (dim, dim)),
        transverse_permeability_matrix=_random_normal_complex(
            keys[6], (2 * dim, 2 * dim)
        ),
        tangent_vector_field=(
            _random_normal_complex(keys[7], (2 * dim, 2 * dim)),
            _random_normal_complex(keys[7], (2 * dim, 2 * dim)),
        ),
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
        fmm.eigensolve_isotropic_media(
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
