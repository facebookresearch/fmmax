"""Tests related to anisotropic and magnetic materials.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fmm, scattering, utils

# Enable 64-bit precision for higher-accuracy.
jax.config.update("jax_enable_x64", True)


class AnisotropicMatchesIsotropicGratingTest(unittest.TestCase):
    def compute_grating_reflection(self, formulation, grating_angle):
        # Computes the TE- and TM-reflection from a metallic grating.
        permittivity_ambient = jnp.asarray([[1.0 + 0.0j]])
        permittivity_passivation = jnp.asarray([[2.25 + 0.0j]])
        permittivity_metal = jnp.asarray([[-7.632 + 0.731j]])

        # Permittivity of the grating layer.
        x, y = jnp.meshgrid(
            jnp.linspace(-0.5, 0.5), jnp.linspace(-0.5, 0.5), indexing="ij"
        )
        position = x * jnp.cos(grating_angle) + y * jnp.sin(grating_angle)
        density = (-0.2 < position) & (position < 0.2)
        permittivity_grating = utils.interpolate_permittivity(
            permittivity_solid=permittivity_metal,
            permittivity_void=permittivity_passivation,
            density=density,
        )

        # Thickensses of ambient, passivation, grating, and metal substrate.
        thicknesses = [
            jnp.asarray(0.0),
            jnp.asarray(0.02),
            jnp.asarray(0.06),
            jnp.asarray(0.0),
        ]

        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=220,
            truncation=basis.Truncation.CIRCULAR,
        )
        eigensolve_kwargs = {
            "wavelength": jnp.asarray(0.63),
            "in_plane_wavevector": jnp.zeros((2,)),
            "primitive_lattice_vectors": primitive_lattice_vectors,
            "expansion": expansion,
            "formulation": formulation,
        }
        solve_result_ambient = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_ambient, **eigensolve_kwargs
        )
        solve_result_passivation = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_passivation, **eigensolve_kwargs
        )
        solve_result_metal = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_metal, **eigensolve_kwargs
        )

        # Perform the isotropic grating eigensolve and compute the zeroth-order reflectivity.
        solve_result_grating_isotropic = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_grating, **eigensolve_kwargs
        )
        s_matrix_isotropic = scattering.stack_s_matrix(
            layer_solve_results=[
                solve_result_ambient,
                solve_result_passivation,
                solve_result_grating_isotropic,
                solve_result_metal,
            ],
            layer_thicknesses=thicknesses,
        )
        n = expansion.num_terms
        r_te_isotropic = s_matrix_isotropic.s21[0, 0]
        r_tm_isotropic = s_matrix_isotropic.s21[n, n]

        # Perform the anisotropic grating eigensolve and compute the zeroth-order reflectivity.
        solve_result_grating_anisotropic = fmm.eigensolve_general_anisotropic_media(
            permittivity_xx=permittivity_grating,
            permittivity_xy=jnp.zeros_like(permittivity_grating),
            permittivity_yx=jnp.zeros_like(permittivity_grating),
            permittivity_yy=permittivity_grating,
            permittivity_zz=permittivity_grating,
            permeability_xx=jnp.ones_like(permittivity_grating),
            permeability_xy=jnp.zeros_like(permittivity_grating),
            permeability_yx=jnp.zeros_like(permittivity_grating),
            permeability_yy=jnp.ones_like(permittivity_grating),
            permeability_zz=jnp.ones_like(permittivity_grating),
            **eigensolve_kwargs,
        )
        s_matrix_anisotropic = scattering.stack_s_matrix(
            layer_solve_results=[
                solve_result_ambient,
                solve_result_passivation,
                solve_result_grating_anisotropic,  # Use results of anisotropic eigensolve.
                solve_result_metal,
            ],
            layer_thicknesses=thicknesses,
        )
        r_te_anisotropic = s_matrix_anisotropic.s21[0, 0]
        r_tm_anisotropic = s_matrix_anisotropic.s21[n, n]

        return (r_te_anisotropic, r_tm_anisotropic), (r_te_isotropic, r_tm_isotropic)

    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.FFT, 0.0),
            (fmm.Formulation.FFT, jnp.pi / 4),
            (fmm.Formulation.FFT, jnp.pi / 2),
            (fmm.Formulation.JONES_DIRECT, 0.0),
            (fmm.Formulation.JONES_DIRECT, jnp.pi / 2),
            (fmm.Formulation.JONES_DIRECT_FOURIER, 0.0),
            (fmm.Formulation.JONES_DIRECT_FOURIER, jnp.pi / 2),
        ]
    )
    def test_reflection_with_anisotropic_eignensolve_matches_isotropic_tight_tolerance(
        self, formulation, grating_angle
    ):
        # Checks that the zeroth order reflection of a grating computed using the anisotropic
        # codepath matches that using the isotropic material codepath.
        (
            (r_te_anisotropic, r_tm_anisotropic),
            (r_te_isotropic, r_tm_isotropic),
        ) = self.compute_grating_reflection(formulation, grating_angle)
        onp.testing.assert_allclose(r_te_anisotropic, r_te_isotropic, rtol=1e-4)
        onp.testing.assert_allclose(r_tm_anisotropic, r_tm_isotropic, rtol=1e-4)

    @parameterized.parameterized.expand(
        [
            (fmm.Formulation.JONES_DIRECT, jnp.pi / 3),
            (fmm.Formulation.JONES_DIRECT, 2 * jnp.pi / 3),
            (fmm.Formulation.JONES_DIRECT_FOURIER, jnp.pi / 3),
            (fmm.Formulation.JONES_DIRECT_FOURIER, 2 * jnp.pi / 3),
        ]
    )
    def test_reflection_with_anisotropic_eignensolve_matches_isotropic_loose_tolerance(
        self, formulation, grating_angle
    ):
        # Checks that the zeroth order reflection of a grating computed using the anisotropic
        # codepath matches that using the isotropic material codepath. Gratings that are rotated
        # result in slightly larger differences between the anisotropic and isotropic codepaths.
        (
            (r_te_anisotropic, r_tm_anisotropic),
            (r_te_isotropic, r_tm_isotropic),
        ) = self.compute_grating_reflection(formulation, grating_angle)
        onp.testing.assert_allclose(r_te_anisotropic, r_te_isotropic, rtol=3e-2)
        onp.testing.assert_allclose(r_tm_anisotropic, r_tm_isotropic, rtol=3e-2)


class AnisotropicMagneticFresnelReflectionTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # Cases with permittivity and permeability arrays with shape (1, 1)
            # exercise the uniform media eigensolve.
            [1.0, 1.0, 0.0, (1, 1), fmm.Formulation.FFT],
            [10.0, 1.0, 0.0, (1, 1), fmm.Formulation.FFT],
            [1.0, 10.0, 0.0, (1, 1), fmm.Formulation.FFT],
            [1.0, 1.0, jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [10.0, 1.0, jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [1.0, 10.0, jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [10.0, 10.0, jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [1.0, 1.0, jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            [10.0, 1.0, jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            [1.0, 10.0, jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            [10.0, 10.0, jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            # Cases with permittivity and permeability arrays with shape larger
            # than (1, 1) exercise the patterned media eigensolve.
            [1.0, 1.0, 0.0, (10, 10), fmm.Formulation.FFT],
            [10.0, 1.0, 0.0, (10, 10), fmm.Formulation.FFT],
            [1.0, 10.0, 0.0, (10, 10), fmm.Formulation.FFT],
            [1.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [10.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [1.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [10.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [1.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            [10.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            [1.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            [10.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            # Patterned media, vector formulation.
            [1.0, 1.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT],
            [10.0, 1.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT],
            [1.0, 10.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT],
            [1.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [10.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [1.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [10.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [1.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            [10.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            [1.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            [10.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            # Patterned media, vector formulation.
            [1.0, 1.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [10.0, 1.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [1.0, 10.0, 0.0, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [1.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [10.0, 1.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [1.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [10.0, 10.0, jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [1.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [10.0, 1.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [1.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [10.0, 10.0, jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
        ]
    )
    def test_reflection_matches_expected(
        self, permittivity, permeability, incident_angle, shape, formulation
    ):
        # Checks that the FMM-computed Fresnel reflection matches the analytical
        # result, including the case of magnetic materials.
        z1 = 1.0 + 0.0j
        n1 = 1.0 + 0.0j
        n2 = jnp.sqrt(jnp.asarray(permittivity * permeability, dtype=complex))
        z2 = jnp.sqrt(jnp.asarray(permeability / permittivity, dtype=complex))
        transmitted_angle = jnp.arcsin(n1 * jnp.sin(incident_angle) / n2)
        reflection_s = (
            z2 * jnp.cos(incident_angle) - z1 * jnp.cos(transmitted_angle)
        ) / (z2 * jnp.cos(incident_angle) + z1 * jnp.cos(transmitted_angle))
        reflection_p = (
            z2 * jnp.cos(transmitted_angle) - z1 * jnp.cos(incident_angle)
        ) / (z2 * jnp.cos(transmitted_angle) + z1 * jnp.cos(incident_angle))

        # Compute reflection using the FMM scheme.
        wavelength = jnp.asarray(0.63)
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=5,
            truncation=basis.Truncation.CIRCULAR,
        )
        eigensolve_kwargs = {
            "wavelength": wavelength,
            "in_plane_wavevector": basis.plane_wave_in_plane_wavevector(
                wavelength=wavelength,
                polar_angle=incident_angle,
                azimuthal_angle=jnp.asarray(0.0),
                permittivity=jnp.asarray(1.0 + 0.0j),
            ),
            "primitive_lattice_vectors": primitive_lattice_vectors,
            "expansion": expansion,
            "formulation": formulation,
        }

        permittivity_ambient = jnp.asarray([[z1]])
        solve_result_ambient = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_ambient, **eigensolve_kwargs
        )

        permittivity = jnp.full(shape, permittivity)
        permeability = jnp.full(shape, permeability)
        solve_result_substrate = fmm.eigensolve_general_anisotropic_media(
            permittivity_xx=permittivity,
            permittivity_xy=jnp.zeros_like(permittivity),
            permittivity_yx=jnp.zeros_like(permittivity),
            permittivity_yy=permittivity,
            permittivity_zz=permittivity,
            permeability_xx=permeability,
            permeability_xy=jnp.zeros_like(permeability),
            permeability_yx=jnp.zeros_like(permeability),
            permeability_yy=permeability,
            permeability_zz=permeability,
            **eigensolve_kwargs,
        )
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[
                solve_result_ambient,
                solve_result_substrate,
            ],
            layer_thicknesses=[jnp.zeros(()), jnp.zeros(())],
        )
        n = expansion.num_terms
        r_te = s_matrix.s21[0, 0]
        r_tm = s_matrix.s21[n, n]

        onp.testing.assert_allclose(
            jnp.abs(r_tm) ** 2, jnp.abs(reflection_p) ** 2, rtol=1e-6, atol=1e-6
        )
        onp.testing.assert_allclose(
            jnp.abs(r_te) ** 2, jnp.abs(reflection_s) ** 2, rtol=1e-6, atol=1e-6
        )

    @parameterized.parameterized.expand(
        [
            # Cases with permittivity and permeability arrays with shape (1, 1)
            # exercise the uniform media eigensolve.
            [0.0, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 5, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 2, (1, 1), fmm.Formulation.FFT],
            # Cases with permittivity and permeability arrays with shape larger
            # than (1, 1) exercise the patterned media eigensolve.
            [0.0, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 5, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 2, (10, 10), fmm.Formulation.FFT],
            [0.0, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 5, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 2, (10, 10), fmm.Formulation.JONES_DIRECT],
            [0.0, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 5, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 2, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
        ]
    )
    def test_reflection_anisotropic_permittivity_matches_expected(
        self, rotation_angle, shape, formulation
    ):
        n1 = 1.0 + 0.0j
        n2e = 2.2 + 0.0001j
        n2o = 1.8 + 0.0001j
        reflection_e = (n1 - n2e) / (n1 + n2e)
        reflection_o = (n1 - n2o) / (n1 + n2o)

        # Compute reflection using the FMM scheme.
        wavelength = jnp.asarray(0.63)
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=5,
            truncation=basis.Truncation.CIRCULAR,
        )
        eigensolve_kwargs = {
            "wavelength": wavelength,
            "in_plane_wavevector": jnp.zeros((2,)),
            "primitive_lattice_vectors": primitive_lattice_vectors,
            "expansion": expansion,
            "formulation": formulation,
        }

        permittivity_ambient = jnp.asarray([[n1**2]])
        solve_result_ambient = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_ambient, **eigensolve_kwargs
        )

        permittivity_tensor_eo = jnp.block([[n2e**2, 0.0], [0.0, n2o**2]])
        # The optical axis is rotated from the x-axis by the `rotation_angle`.
        # The rotation matrix is defined so that [Dx, Dy]^T = R [De, Do]^T.
        rotation_matrix = jnp.block(
            [
                [jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
                [jnp.sin(rotation_angle), jnp.cos(rotation_angle)],
            ]
        )
        # Compute the rotated permittivity tensor. This is found by,
        #       [De, Do]^T = eps_eo [Ee, Eo]^T
        #       [Dx, Dy]^T = R eps_eo R^-1 [Ex, Ey]^T
        permittivity_tensor = (
            rotation_matrix @ permittivity_tensor_eo @ jnp.linalg.inv(rotation_matrix)
        )
        solve_result_substrate = fmm.eigensolve_general_anisotropic_media(
            permittivity_xx=jnp.full(shape, permittivity_tensor[0, 0]),
            permittivity_xy=jnp.full(shape, permittivity_tensor[0, 1]),
            permittivity_yx=jnp.full(shape, permittivity_tensor[1, 0]),
            permittivity_yy=jnp.full(shape, permittivity_tensor[1, 1]),
            permittivity_zz=jnp.full(shape, n2o**2),
            permeability_xx=jnp.ones(shape),
            permeability_xy=jnp.zeros(shape),
            permeability_yx=jnp.zeros(shape),
            permeability_yy=jnp.ones(shape),
            permeability_zz=jnp.ones(shape),
            **eigensolve_kwargs,
        )
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result_ambient, solve_result_substrate],
            layer_thicknesses=[jnp.zeros(()), jnp.zeros(())],
        )
        # Excite the structure with normal incidence waves that are polarized parallel
        # or perpendicular to the optical axes.
        n = expansion.num_terms
        amplitudes = jnp.zeros((2 * n, 2))
        # Magnetic field parallel to the extraordinary axis, i.e. electric field
        # perpendicular to the extrordinary axis. Should experience "ordinary" reflection.
        amplitudes = amplitudes.at[0, 0].set(jnp.cos(rotation_angle))
        amplitudes = amplitudes.at[n, 0].set(jnp.sin(rotation_angle))
        # Electric field parallel to the extraordinary axis.
        amplitudes = amplitudes.at[0, 1].set(-jnp.sin(rotation_angle))
        amplitudes = amplitudes.at[n, 1].set(jnp.cos(rotation_angle))

        reflected_amplitudes = s_matrix.s21 @ amplitudes

        onp.testing.assert_allclose(
            jnp.sum(jnp.abs(reflected_amplitudes[:, 0]) ** 2),
            jnp.abs(reflection_o) ** 2,
            rtol=1e-6,
            atol=1e-6,
        )
        onp.testing.assert_allclose(
            jnp.sum(jnp.abs(reflected_amplitudes[:, 1]) ** 2),
            jnp.abs(reflection_e) ** 2,
            rtol=1e-6,
            atol=1e-6,
        )

    @parameterized.parameterized.expand(
        [
            # Cases with permittivity and permeability arrays with shape (1, 1)
            # exercise the uniform media eigensolve.
            [0.0, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 5, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 4, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 3, (1, 1), fmm.Formulation.FFT],
            [jnp.pi / 2, (1, 1), fmm.Formulation.FFT],
            # Cases with permittivity and permeability arrays with shape larger
            # than (1, 1) exercise the patterned media eigensolve.
            [0.0, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 5, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 4, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 3, (10, 10), fmm.Formulation.FFT],
            [jnp.pi / 2, (10, 10), fmm.Formulation.FFT],
            [0.0, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 5, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT],
            [jnp.pi / 2, (10, 10), fmm.Formulation.JONES_DIRECT],
            [0.0, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 5, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 4, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 3, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
            [jnp.pi / 2, (10, 10), fmm.Formulation.JONES_DIRECT_FOURIER],
        ]
    )
    def test_reflection_anisotropic_permeability_matches_expected(
        self, rotation_angle, shape, formulation
    ):
        n1 = 1.0 + 0.0j
        n2e = 2.2 + 0.0001j
        n2o = 1.8 + 0.0001j
        reflection_e = (n1 - n2e) / (n1 + n2e)
        reflection_o = (n1 - n2o) / (n1 + n2o)

        # Compute reflection using the FMM scheme.
        wavelength = jnp.asarray(0.63)
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )
        eigensolve_kwargs = {
            "wavelength": wavelength,
            "in_plane_wavevector": jnp.zeros((2,)),
            "primitive_lattice_vectors": primitive_lattice_vectors,
            "expansion": expansion,
            "formulation": formulation,
        }

        permittivity_ambient = jnp.asarray([[n1**2]])
        solve_result_ambient = fmm.eigensolve_isotropic_media(
            permittivity=permittivity_ambient, **eigensolve_kwargs
        )

        permeability_tensor_eo = jnp.block([[n2e**2, 0.0], [0.0, n2o**2]])
        # The optical axis is rotated from the x-axis by the `rotation_angle`.
        # The rotation matrix is defined so that [Bx, By]^T = R [Be, Bo]^T.
        rotation_matrix = jnp.block(
            [
                [jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
                [jnp.sin(rotation_angle), jnp.cos(rotation_angle)],
            ]
        )
        # Compute the rotated permittivity tensor. This is found by,
        #       [Be, Bo]^T = mu_eo [He, Ho]^T
        #       [Bx, By]^T = R mu_eo R^-1 [Hx, Hy]^T
        permeability_tensor = (
            rotation_matrix @ permeability_tensor_eo @ jnp.linalg.inv(rotation_matrix)
        )
        solve_result_substrate = fmm.eigensolve_general_anisotropic_media(
            permittivity_xx=jnp.ones(shape),
            permittivity_xy=jnp.zeros(shape),
            permittivity_yx=jnp.zeros(shape),
            permittivity_yy=jnp.ones(shape),
            permittivity_zz=jnp.ones(shape),
            permeability_xx=jnp.full(shape, permeability_tensor[0, 0]),
            permeability_xy=jnp.full(shape, permeability_tensor[0, 1]),
            permeability_yx=jnp.full(shape, permeability_tensor[1, 0]),
            permeability_yy=jnp.full(shape, permeability_tensor[1, 1]),
            permeability_zz=jnp.full(shape, n2o**2),
            **eigensolve_kwargs,
        )
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=[solve_result_ambient, solve_result_substrate],
            layer_thicknesses=[jnp.zeros(()), jnp.zeros(())],
        )
        # Excite the structure with normal incidence waves that are polarized parallel
        # or perpendicular to the optical axes.
        n = expansion.num_terms
        amplitudes = jnp.zeros((2 * n, 2))
        # Magnetic field parallel to the extrordinary axis. Should experience
        # extrordinary reflection.
        amplitudes = amplitudes.at[0, 0].set(jnp.cos(rotation_angle))
        amplitudes = amplitudes.at[n, 0].set(jnp.sin(rotation_angle))
        # Magnetic field perpendicular to the extrordinary axis.
        amplitudes = amplitudes.at[0, 1].set(-jnp.sin(rotation_angle))
        amplitudes = amplitudes.at[n, 1].set(jnp.cos(rotation_angle))

        reflected_amplitudes = s_matrix.s21 @ amplitudes

        onp.testing.assert_allclose(
            jnp.sum(jnp.abs(reflected_amplitudes[:, 0]) ** 2),
            jnp.abs(reflection_e) ** 2,
            rtol=1e-6,
            atol=1e-6,
        )
        onp.testing.assert_allclose(
            jnp.sum(jnp.abs(reflected_amplitudes[:, 1]) ** 2),
            jnp.abs(reflection_o) ** 2,
            rtol=1e-6,
            atol=1e-6,
        )
