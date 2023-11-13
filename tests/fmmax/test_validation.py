"""Validation tests that involve simulations of various structures.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
from parameterized import parameterized

from fmmax import basis, fields, fmm, scattering

# Enable 64-bit precision for higher accuracy.
jax.config.update("jax_enable_x64", True)


def _solve_s_matrix(
    wavelength,
    in_plane_wavevector,
    primitive_lattice_vectors,
    expansion,
    permittivities,
    thicknesses,
    formulation=fmm.Formulation.FFT,
):
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=formulation,
        )
        for p in permittivities
    ]
    s_matrix = scattering.stack_s_matrix(layer_solve_results, thicknesses)
    return s_matrix, layer_solve_results


def _solve_fresnel_reflection(n1, n2, theta_i):
    theta_t = jnp.arcsin(n1 / n2 * jnp.sin(theta_i))
    reflection_s = (
        jnp.abs(
            (n1 * jnp.cos(theta_i) - n2 * jnp.cos(theta_t))
            / (n1 * jnp.cos(theta_i) + n2 * jnp.cos(theta_t))
        )
        ** 2
    )
    reflection_p = (
        jnp.abs(
            (n1 * jnp.cos(theta_t) - n2 * jnp.cos(theta_i))
            / (n1 * jnp.cos(theta_t) + n2 * jnp.cos(theta_i))
        )
        ** 2
    )
    return reflection_s, reflection_p


class ThinFilmComparisonTest(unittest.TestCase):
    @parameterized.expand([((1, 1),), ((64, 64),)])
    def test_normal_incidence_matches_fresnel(self, permittivity_shape):
        n1 = 1.2
        n2 = 3.8
        wavelength = jnp.array(0.238)
        in_plane_wavevector = jnp.zeros((2,))
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.asarray([1.0, 0.0]), v=jnp.asarray([0.0, 0.8])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )
        permittivities = [
            jnp.broadcast_to(jnp.asarray([[n1**2]]), permittivity_shape),
            jnp.broadcast_to(jnp.asarray([[n2**2]]), permittivity_shape),
        ]
        thicknesses = [1.0, 2.0]
        s_matrix, _ = _solve_s_matrix(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            permittivities=permittivities,
            thicknesses=thicknesses,
        )
        # Ensure that the `0`-th basis coefficient is for the zeroth order.
        onp.testing.assert_array_equal(expansion.basis_coefficients[0, :], (0, 0))

        # Extract the zeroth-order reflection.
        reflection = jnp.abs(s_matrix.s12[0, 0]) ** 2
        expected_reflection = jnp.abs((n1 - n2) / (n1 + n2)) ** 2
        onp.testing.assert_allclose(reflection, expected_reflection)
        # Extract the zeroth-order transmission.
        transmission = jnp.abs(s_matrix.s11[0, 0]) ** 2 * n1 / n2
        onp.testing.assert_allclose(transmission, 1 - expected_reflection)

    def test_oblique_incidence_matches_fresenl(self):
        n1 = 1.2
        n2 = 3.8
        wavelength = jnp.array(0.238)
        polar_angle = 0.0
        in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
            wavelength=wavelength,
            polar_angle=polar_angle,
            azimuthal_angle=0.0,
            permittivity=n1**2,
        )
        primitive_lattice_vectors = basis.LatticeVectors(
            u=jnp.asarray([1.0, 0.0]), v=jnp.asarray([0.0, 1.0])
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=5,
            truncation=basis.Truncation.CIRCULAR,
        )
        num_terms = expansion.num_terms
        permittivities = [jnp.asarray([[n1**2]]), jnp.asarray([[n2**2]])]
        thicknesses = [1.0, 1.0]

        s_matrix, layer_solve_results = _solve_s_matrix(
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            permittivities=permittivities,
            thicknesses=thicknesses,
        )

        expected_reflection_s, expected_reflection_p = _solve_fresnel_reflection(
            n1, n2, polar_angle
        )

        # Check that reflection matches expected, using the s-matrix directly.
        with self.subTest("reflection from s-matrix"):
            reflection_s = jnp.abs(s_matrix.s21[0, 0]) ** 2
            reflection_p = jnp.abs(s_matrix.s21[num_terms, num_terms]) ** 2
            onp.testing.assert_allclose(reflection_s, expected_reflection_s)
            onp.testing.assert_allclose(reflection_p, expected_reflection_p)

        # Check that reflection matches expected, using the poynting vector.
        with self.subTest("reflection from poynting flux"):
            forward_amplitude_0_start = jnp.ones((2 * num_terms, 1))
            backward_amplitude_0_end = s_matrix.s21 @ forward_amplitude_0_start
            forward_amplitude_0, backward_amplitude_0 = fields.colocate_amplitudes(
                forward_amplitude_start=forward_amplitude_0_start,
                backward_amplitude_end=backward_amplitude_0_end,
                z_offset=0,
                layer_solve_result=layer_solve_results[0],
                layer_thickness=thicknesses[0],
            )
            s_forward_0, s_backward_0 = fields.amplitude_poynting_flux(
                forward_amplitude_0, backward_amplitude_0, layer_solve_results[0]
            )
            reflection_s = s_backward_0[0, 0] / s_forward_0[0, 0]
            reflection_p = s_backward_0[num_terms, 0] / s_forward_0[num_terms, 0]
            onp.testing.assert_allclose(jnp.abs(reflection_s), expected_reflection_s)
            onp.testing.assert_allclose(jnp.abs(reflection_p), expected_reflection_p)

        # Check that transmission matches expected, using the poynting vector.
        with self.subTest("transmission from poynting flux"):
            forward_amplitude_1_start = s_matrix.s11 @ forward_amplitude_0_start
            backward_amplitude_1_end = jnp.zeros((2 * num_terms, 1))
            forward_amplitude_1, backward_amplitude_1 = fields.colocate_amplitudes(
                forward_amplitude_start=forward_amplitude_1_start,
                backward_amplitude_end=backward_amplitude_1_end,
                z_offset=thicknesses[1],
                layer_solve_result=layer_solve_results[1],
                layer_thickness=thicknesses[1],
            )
            s_forward_1, s_backward_1 = fields.amplitude_poynting_flux(
                forward_amplitude_1, backward_amplitude_1, layer_solve_results[1]
            )
            transmission_s = s_forward_1[0, 0] / s_forward_0[0, 0]
            transmission_p = s_forward_1[num_terms, 0] / s_forward_0[num_terms, 0]
            onp.testing.assert_allclose(
                jnp.abs(transmission_s), 1 - expected_reflection_s
            )
            onp.testing.assert_allclose(
                jnp.abs(transmission_p), 1 - expected_reflection_p
            )


class RotatedGratingTest(unittest.TestCase):
    @parameterized.expand(
        [
            [fmm.Formulation.JONES_DIRECT],
            [fmm.Formulation.JONES_DIRECT_FOURIER],
        ]
    )
    def test_different_unit_cells_give_same_reflection(self, formulation):
        # Tests that different, nominally equivalent unit cell selections
        # actually yield equal reflectivity for a complex, wavy grating
        # structure.
        wavelength = jnp.array(0.4)
        in_plane_wavevector = jnp.array((0, 0))
        approximate_num_terms = 50
        pattern_pitch = 1.0
        pattern_width = 0.3
        pattern_modulation_depth = 0.1
        thicknesses = (0, 1, 0)

        def wavy_pattern(x, y):
            # Returns the density of a "wavy" grating at location `(x, y)`.
            xu = x % pattern_pitch  # x in the unit cell
            yu = y % pattern_pitch  # y in the unit cell
            x_start = (
                pattern_pitch / 2
                - pattern_width / 2
                + pattern_modulation_depth * jnp.cos(yu * 2 * jnp.pi)
            )
            x_end = (
                pattern_pitch / 2
                + pattern_width / 2
                + pattern_modulation_depth * jnp.cos(yu * 2 * jnp.pi)
            )
            return ((xu > x_start) & (xu < x_end)).astype(float)

        def compute_reflection(u, v):
            i, j = jnp.meshgrid(
                jnp.linspace(0, 1, 100),
                jnp.linspace(0, 1, 100),
                indexing="ij",
            )
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            density = wavy_pattern(x, y)
            permittivities = [
                jnp.array([[1]]),
                density * 2.25 + (1 - density),
                jnp.array([[2.25]]),
            ]
            primitive_lattice_vectors = basis.LatticeVectors(
                u=jnp.asarray(u), v=jnp.asarray(v)
            )
            expansion = basis.generate_expansion(
                primitive_lattice_vectors=primitive_lattice_vectors,
                approximate_num_terms=approximate_num_terms,
                truncation=basis.Truncation.CIRCULAR,
            )
            s_matrix, layer_solve_results = _solve_s_matrix(
                wavelength=wavelength,
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=primitive_lattice_vectors,
                expansion=expansion,
                permittivities=permittivities,
                thicknesses=thicknesses,
                formulation=formulation,
            )
            rte = s_matrix.s21[0, 0]
            rtm = s_matrix.s21[expansion.num_terms, expansion.num_terms]
            return rte, rtm

        # Solve where the unit cell is square, with u=x and v=y.
        expected_rte, expected_rtm = compute_reflection(u=(1, 0), v=(0, 1))

        # Solve with a variety of other unit cells, taking care that all selections
        # have the same area. The wavevectors in the expansion are identical, but
        # there can still be slight differences in the reflection since e.g. the
        # vector field can differ.

        rte, rtm = compute_reflection(u=(0, 1), v=(1, 0))
        with self.subTest("rotated square TE"):
            onp.testing.assert_allclose(rte, expected_rte)
        with self.subTest("rotated square TM"):
            onp.testing.assert_allclose(rtm, expected_rtm)

        rte, rtm = compute_reflection(u=(1, 0), v=(1, 1))
        with self.subTest("parallelogram TE"):
            onp.testing.assert_allclose(rte, expected_rte, atol=3e-3)
        with self.subTest("parallelogram TM"):
            onp.testing.assert_allclose(rtm, expected_rtm, atol=3e-3)

        rte, rtm = compute_reflection(u=(1, 1), v=(2, 1))
        with self.subTest("skewed parallelogram TE"):
            onp.testing.assert_allclose(rte, expected_rte, atol=2e-3)
        with self.subTest("skewed parallelogram TM"):
            onp.testing.assert_allclose(rtm, expected_rtm, atol=4e-3)
