"""Tests for `fmmax.sources`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fft, fields, fmm, scattering, sources

WAVELENGTH = jnp.array(0.314)
PRIMITIVE_LATTICE_VECTORS = basis.LatticeVectors(u=basis.X, v=basis.Y)
EXPANSION = basis.generate_expansion(
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    approximate_num_terms=20,
    truncation=basis.Truncation.CIRCULAR,
)

LAYER_SOLVE_RESULT = fmm.eigensolve_isotropic_media(
    wavelength=WAVELENGTH,
    in_plane_wavevector=jnp.zeros((2,)),
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    permittivity=jnp.ones((50, 50)) * 2,
    expansion=EXPANSION,
    formulation=fmm.Formulation.FFT,
)

BATCH_LAYER_SOLVE_RESULT = fmm.eigensolve_isotropic_media(
    wavelength=WAVELENGTH,
    in_plane_wavevector=jnp.zeros((2,)),
    primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
    permittivity=jnp.ones((1, 1, 2, 50, 50)) * 2,
    expansion=EXPANSION,
    formulation=fmm.Formulation.FFT,
)


class FieldSourcesTest(unittest.TestCase):
    def test_amplitudes_match_expected(self):
        # Generate random amplitudes, compute the resulting fields, and extract
        # the amplitudes resulting from those fields. Compare the extracted
        # amplitudes to the original amplitudes.
        brillouin_grid_shape = (5, 5)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=WAVELENGTH,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        fwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(0),
            brillouin_grid_shape + (2 * EXPANSION.num_terms, 3),
            dtype=complex,
        )
        bwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(1),
            brillouin_grid_shape + (2 * EXPANSION.num_terms, 3),
            dtype=complex,
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            fwd_amplitude, bwd_amplitude, layer_solve_result
        )
        (ex, ey, _), (hx, hy, _), _ = fields.fields_on_grid(
            efield,
            hfield,
            layer_solve_result,
            shape=(20, 20),
            num_unit_cells=brillouin_grid_shape,
        )
        ex = jnp.mean(ex, axis=(0, 1))
        ey = jnp.mean(ey, axis=(0, 1))
        hx = jnp.mean(hx, axis=(0, 1))
        hy = jnp.mean(hy, axis=(0, 1))
        (
            fwd_amplitude_extracted,
            bwd_amplitude_extracted,
        ) = sources.amplitudes_for_fields(
            ex,
            ey,
            hx,
            hy,
            layer_solve_result,
            brillouin_grid_axes=(0, 1),
        )
        onp.testing.assert_allclose(fwd_amplitude_extracted, fwd_amplitude, rtol=5e-3)
        onp.testing.assert_allclose(bwd_amplitude_extracted, bwd_amplitude, rtol=5e-3)

    def test_amplitudes_match_expected_no_bz_grid(self):
        # Generate random amplitudes, compute the resulting fields, and extract
        # the amplitudes resulting from those fields. Compare the extracted
        # amplitudes to the original amplitudes.
        in_plane_wavevector = jnp.zeros((2,))
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=WAVELENGTH,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        fwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(0), (2 * EXPANSION.num_terms, 3), dtype=complex
        )
        bwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(1), (2 * EXPANSION.num_terms, 3), dtype=complex
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            fwd_amplitude, bwd_amplitude, layer_solve_result
        )
        (ex, ey, _), (hx, hy, _), _ = fields.fields_on_grid(
            efield,
            hfield,
            layer_solve_result,
            shape=(20, 20),
            num_unit_cells=(1, 1),
        )
        (
            fwd_amplitude_extracted,
            bwd_amplitude_extracted,
        ) = sources.amplitudes_for_fields(
            ex,
            ey,
            hx,
            hy,
            layer_solve_result,
            brillouin_grid_axes=None,
        )
        onp.testing.assert_allclose(fwd_amplitude_extracted, fwd_amplitude, rtol=5e-3)
        onp.testing.assert_allclose(bwd_amplitude_extracted, bwd_amplitude, rtol=5e-3)

    def test_amplitudes_match_expected_wavelength_batch(self):
        # With a size-2 wavelength batch, generate random amplitudes, compute the
        # resulting fields, and extract the amplitudes resulting from those fields.
        # Compare the extracted amplitudes to the original amplitudes.
        brillouin_grid_shape = (5, 5)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=jnp.asarray([0.277, 0.314])[:, jnp.newaxis, jnp.newaxis],
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        fwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(0),
            (2,) + brillouin_grid_shape + (2 * EXPANSION.num_terms, 3),
            dtype=complex,
        )
        bwd_amplitude = jax.random.normal(
            jax.random.PRNGKey(1),
            (2,) + brillouin_grid_shape + (2 * EXPANSION.num_terms, 3),
            dtype=complex,
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            fwd_amplitude, bwd_amplitude, layer_solve_result
        )
        (ex, ey, _), (hx, hy, _), _ = fields.fields_on_grid(
            efield,
            hfield,
            layer_solve_result,
            shape=(20, 20),
            num_unit_cells=brillouin_grid_shape,
        )
        ex = jnp.mean(ex, axis=(1, 2), keepdims=True)
        ey = jnp.mean(ey, axis=(1, 2), keepdims=True)
        hx = jnp.mean(hx, axis=(1, 2), keepdims=True)
        hy = jnp.mean(hy, axis=(1, 2), keepdims=True)
        (
            fwd_amplitude_extracted,
            bwd_amplitude_extracted,
        ) = sources.amplitudes_for_fields(
            ex,
            ey,
            hx,
            hy,
            layer_solve_result,
            brillouin_grid_axes=(1, 2),
        )
        onp.testing.assert_allclose(fwd_amplitude_extracted, fwd_amplitude, rtol=5e-3)
        onp.testing.assert_allclose(bwd_amplitude_extracted, bwd_amplitude, rtol=5e-3)

    def test_field_shape_validation(self):
        brillouin_grid_shape = (5, 5)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=WAVELENGTH,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        with self.assertRaisesRegex(
            ValueError, "All fields must have rank of at least 3"
        ):
            sources.amplitudes_for_fields(
                ex=jnp.ones((20, 20, 1)),
                ey=jnp.ones((20, 20, 1)),
                hx=jnp.ones((20, 20, 1)),
                hy=jnp.ones((20, 20)),
                layer_solve_result=layer_solve_result,
                brillouin_grid_axes=(0, 1),
            )

    def test_field_shape_validation_wavelength_batch(self):
        brillouin_grid_shape = (5, 5)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=jnp.ones((2, 1, 1)),  # Batch of wavelengths
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        with self.assertRaisesRegex(
            ValueError, "Fields must be batch-compatible with `layer_solve_result`"
        ):
            sources.amplitudes_for_fields(
                ex=jnp.ones((2, 2, 1, 20, 20, 1)),
                ey=jnp.ones((2, 2, 1, 20, 20, 1)),
                hx=jnp.ones((2, 2, 1, 20, 20, 1)),
                hy=jnp.ones((2, 2, 1, 20, 20, 1)),
                layer_solve_result=layer_solve_result,
                brillouin_grid_axes=(0, 1),
            )

    def test_field_shape_brillouin_grid_compatible_validation(self):
        brillouin_grid_shape = (3, 3)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=WAVELENGTH,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        with self.assertRaisesRegex(
            ValueError, "Field shapes must be evenly divisible by the Brillouin"
        ):
            sources.amplitudes_for_fields(
                ex=jnp.ones((20, 20, 1)),
                ey=jnp.ones((20, 20, 1)),
                hx=jnp.ones((20, 20, 1)),
                hy=jnp.ones((20, 20, 1)),
                layer_solve_result=layer_solve_result,
                brillouin_grid_axes=(0, 1),
            )


class InternalSourcesTest(unittest.TestCase):
    @parameterized.parameterized.expand([[(2,)], [(1, 3)], [(1, 3, 2)]])
    def test_gaussian_location_shape_validation(self, invalid_shape):
        with self.assertRaisesRegex(
            ValueError, "`location` must be rank-2 with a trailing axis size of 2"
        ):
            sources.gaussian_source(
                fwhm=0.0,
                location=jnp.ones(invalid_shape),
                in_plane_wavevector=jnp.zeros((1, 2)),
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                expansion=EXPANSION,
            )

    @parameterized.parameterized.expand([[(2,)], [(1, 3)], [(1, 3, 2)]])
    def test_dirac_delta_location_shape_validation(self, invalid_shape):
        with self.assertRaisesRegex(
            ValueError, "`location` must be rank-2 with a trailing axis size of 2"
        ):
            sources.dirac_delta_source(
                location=jnp.ones(invalid_shape),
                in_plane_wavevector=jnp.zeros((1, 2)),
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                expansion=EXPANSION,
            )

    def test_gaussian_wavevector_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`in_plane_wavevector` must have a trailing axis size of 2"
        ):
            sources.gaussian_source(
                fwhm=0.0,
                location=jnp.ones((1, 2)),
                in_plane_wavevector=jnp.zeros((3,)),
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                expansion=EXPANSION,
            )

    def test_dirac_delta_wavevector_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`in_plane_wavevector` must have a trailing axis size of 2"
        ):
            sources.dirac_delta_source(
                location=jnp.ones((1, 2)),
                in_plane_wavevector=jnp.zeros((3,)),
                primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
                expansion=EXPANSION,
            )

    @parameterized.parameterized.expand(
        [
            [(1, 1), 0.4, (0.5, 0.5)],
            [(2, 2), 0.4, (0.5, 0.5)],
            [(3, 4), 0.4, (1.5, 1.5)],
            [(3, 4), 0.5, (1.5, 1.5)],
            [(3, 4), 1.2, (1.5, 1.5)],
        ]
    )
    def test_gaussian_source_matches_expected(
        self, brillouin_grid_shape, fwhm, dipole_location
    ):
        # Performs a simulation of x- and y-oriented Gaussian dipoles in a vacuum,
        # and computes the magnetic field magnitude in the plane of the dipole. We
        # validate that the resulting magnetic field magnitude is actually Gaussian
        # with the specified full width.
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=brillouin_grid_shape,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        dipole = sources.gaussian_source(
            fwhm=fwhm,
            location=jnp.asarray([dipole_location]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )

        zeros = jnp.zeros_like(dipole)
        jx = jnp.concatenate([dipole, zeros], axis=-1)
        jy = jnp.concatenate([zeros, dipole], axis=-1)
        jz = jnp.concatenate([zeros, zeros], axis=-1)

        # Perform the eigendecomposition of a vacuum fmm.
        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=WAVELENGTH,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
            formulation=fmm.Formulation.FFT,
        )
        s_matrices_before_source = s_matrices_after_source = (
            scattering.stack_s_matrices_interior([layer_solve_result], [1.0])
        )
        s_matrix_before_source = s_matrices_before_source[-1][0]
        s_matrix_after_source = s_matrices_after_source[-1][0]
        bwd_amplitude_ambient_end, *_ = sources.amplitudes_for_source(
            jx=jx,
            jy=jy,
            jz=jz,
            s_matrix_before_source=s_matrix_before_source,
            s_matrix_after_source=s_matrix_after_source,
        )

        efield, hfield = fields.fields_from_wave_amplitudes(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=layer_solve_result,
        )
        efield, hfield, (x, y) = fields.fields_on_grid(
            electric_field=efield,
            magnetic_field=hfield,
            layer_solve_result=layer_solve_result,
            shape=(21, 21),
            num_unit_cells=brillouin_grid_shape,
        )
        x = jnp.squeeze(x)  # Remove batch dimensions from BZ grid.
        y = jnp.squeeze(y)

        # Perform Brillouin zone integration and compute magnetic field magnitude.
        hfield = jnp.mean(jnp.asarray(hfield), axis=(1, 2))
        hmag = jnp.sqrt(jnp.sum(jnp.abs(hfield) ** 2, axis=0))

        # Add the field for x- and y-oriented dipoles.
        hmag = jnp.sum(hmag, axis=-1)
        hmag /= jnp.amax(hmag)

        self.assertSequenceEqual(hmag.shape, [21 * d for d in brillouin_grid_shape])

        # Compute the expected magnetic field magnitude, which is just a Gaussian.
        x0, y0 = dipole_location
        distance = jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        expected = jnp.exp(-(distance**2) / (2 * sigma**2))

        onp.testing.assert_allclose(hmag, expected, atol=0.05)

    @parameterized.parameterized.expand([[(0, 0)], [(0.5, 0.5)]])
    def test_dirac_delta_matches_gaussian_with_zero_fwhm(self, dipole_location):
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(3, 4),
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
        )
        gaussian_dipole = sources.gaussian_source(
            fwhm=0.0,
            location=jnp.asarray([dipole_location]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        dirac_delta_dipole = sources.dirac_delta_source(
            location=jnp.asarray([dipole_location]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=PRIMITIVE_LATTICE_VECTORS,
            expansion=EXPANSION,
        )
        onp.testing.assert_allclose(gaussian_dipole, dirac_delta_dipole)


class AmplitudesFromInternalSourcesTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (1, 0, True, False),
            (0, 1, False, True),
            (1, 1, False, False),
        ]
    )
    def test_emission_planar_source_in_uniform_media(
        self, jx, jy, te_expected_zero, tm_expected_zero
    ):
        # Use the same scattering matrix before and after the source.
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=(LAYER_SOLVE_RESULT,) * 2,
            layer_thicknesses=(0.2, 0.3),
        )
        (
            bwd_amplitude_0_end,
            fwd_amplitude_before_start,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            bwd_amplitude_after_end,
            fwd_amplitude_N_start,
        ) = sources.amplitudes_for_source(
            jx=fft.fft(jnp.ones((20, 20, 1)), expansion=EXPANSION, axes=(-3, -2)) * jx,
            jy=fft.fft(jnp.ones((20, 20, 1)), expansion=EXPANSION, axes=(-3, -2)) * jy,
            jz=fft.fft(jnp.zeros((20, 20, 1)), expansion=EXPANSION, axes=(-3, -2)),
            s_matrix_before_source=s_matrix,
            s_matrix_after_source=s_matrix,
        )
        with self.subTest("exiting power is nonzero"):
            self.assertFalse(onp.allclose(bwd_amplitude_before_end, 0.0))
            self.assertFalse(onp.allclose(fwd_amplitude_after_start, 0.0))
        with self.subTest("fwd and bwd power at source are equal"):
            onp.testing.assert_allclose(
                jnp.abs(bwd_amplitude_before_end),
                jnp.abs(fwd_amplitude_after_start),
                rtol=1e-05,
                atol=1e-08,
            )
        with self.subTest("bwd power in first layer and at source match"):
            onp.testing.assert_allclose(
                jnp.abs(bwd_amplitude_before_end),
                jnp.abs(bwd_amplitude_0_end),
                rtol=1e-05,
                atol=1e-08,
            )
        with self.subTest("fwd power in last layer and source match"):
            onp.testing.assert_allclose(
                jnp.abs(fwd_amplitude_after_start),
                jnp.abs(fwd_amplitude_N_start),
                rtol=1e-05,
                atol=1e-08,
            )
        with self.subTest("no reflection in the stack"):
            onp.testing.assert_allclose(fwd_amplitude_before_start, 0.0, atol=1e-12)
            onp.testing.assert_allclose(bwd_amplitude_after_end, 0.0, atol=1e-12)
        with self.subTest("check polarization"):
            te, tm = jnp.split(bwd_amplitude_before_end, 2, axis=-2)
            if te_expected_zero:
                onp.testing.assert_allclose(te, 0.0, rtol=1e-05, atol=1e-06)
            else:
                self.assertFalse(onp.allclose(te, 0.0))
            if tm_expected_zero:
                onp.testing.assert_allclose(tm, 0.0, rtol=1e-05, atol=1e-06)
            else:
                self.assertFalse(onp.allclose(tm, 0.0))

    def test_power_xyz_dipole_orientations_match(self):
        def _compute_power(pitch, permittivity, wavelength):
            primitive_lattice_vectors = basis.LatticeVectors(
                u=basis.X * pitch, v=basis.Y * pitch
            )
            layer_solve_result = fmm.eigensolve_isotropic_media(
                wavelength=wavelength,
                in_plane_wavevector=jnp.zeros((2,)),
                primitive_lattice_vectors=primitive_lattice_vectors,
                permittivity=jnp.asarray([[permittivity]]),
                expansion=basis.generate_expansion(
                    primitive_lattice_vectors=primitive_lattice_vectors,
                    approximate_num_terms=500,
                    truncation=basis.Truncation.CIRCULAR,
                ),
                formulation=fmm.Formulation.FFT,
            )

            s_matrix = scattering.stack_s_matrix(
                layer_solve_results=(layer_solve_result,),
                layer_thicknesses=(jnp.asarray(pitch / 2),),
            )

            mask = jnp.zeros((100, 100)).at[49:51, 49:51].set(1)

            mask = fft.fft(mask, expansion=layer_solve_result.expansion)
            zeros = jnp.zeros_like(mask)
            jx = jnp.stack([mask, zeros, zeros], axis=-1)
            jy = jnp.stack([zeros, mask, zeros], axis=-1)
            jz = jnp.stack([zeros, zeros, mask], axis=-1)

            (
                bwd_amplitude_0_end,
                fwd_amplitude_before_start,
                bwd_amplitude_before_end,
                fwd_amplitude_after_start,
                bwd_amplitude_after_end,
                fwd_amplitude_N_start,
            ) = sources.amplitudes_for_source(
                jx=jx,
                jy=jy,
                jz=jz,
                s_matrix_before_source=s_matrix,
                s_matrix_after_source=s_matrix,
            )

            forward_power, _ = fields.amplitude_poynting_flux(
                forward_amplitude=fwd_amplitude_after_start,
                backward_amplitude=jnp.zeros_like(fwd_amplitude_after_start),
                layer_solve_result=layer_solve_result,
            )
            _, backward_power = fields.amplitude_poynting_flux(
                forward_amplitude=jnp.zeros_like(bwd_amplitude_before_end),
                backward_amplitude=bwd_amplitude_before_end,
                layer_solve_result=layer_solve_result,
            )
            forward_power = jnp.sum(forward_power, axis=-2)
            backward_power = jnp.sum(backward_power, axis=-2)
            return forward_power, backward_power

        forward_power, backward_power = _compute_power(
            pitch=2.4, permittivity=1, wavelength=jnp.asarray([0.295, 0.59])
        )

        with self.subTest("Compare x- and y-oriented dipoles"):
            # x- and y-oriented dipoles should match almost exactly.
            onp.testing.assert_allclose(
                forward_power[:, 1], forward_power[:, 0], rtol=1e-6
            )
            onp.testing.assert_allclose(
                backward_power[:, 1], backward_power[:, 0], rtol=1e-6
            )

        with self.subTest("Compare x- and z-oriented dipoles"):
            # x- and z-oriented dipoles may differ, since the structure is periodic in x/y.
            onp.testing.assert_allclose(
                forward_power[:, 2], forward_power[:, 0], rtol=0.2
            )

        # Check that the calculation is scale-invariant as expected. Increase the refractive
        # index by 2, and reduce the pitch by 2. The power should be lower by a factor of 2.
        forward_power_scaled, backward_power_scaled = _compute_power(
            pitch=1.2, permittivity=4, wavelength=jnp.asarray([0.295, 0.59])
        )

        with self.subTest("Check scale invariant"):
            onp.testing.assert_allclose(
                forward_power_scaled, forward_power / 2, rtol=1e-05, atol=1e-08
            )
            onp.testing.assert_allclose(
                backward_power_scaled, backward_power / 2, rtol=1e-05, atol=1e-08
            )

    def test_polarization_terms_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`jx`, `jy`, and `jz` must have matching shape"
        ):
            jx = jnp.ones((20, 20, 1))
            jy = jnp.ones((20, 20, 1))
            jz = jnp.ones((20, 20))
            sources.polarization_terms(jx, jy, jz, LAYER_SOLVE_RESULT)

    @parameterized.parameterized.expand(
        [
            [(20, 30, 1), (-1, 1)],
            [(30, 20, 2), (-1, 2)],
            [(10, 1, 30, 20, 2), (10, 1, -1, 2)],
        ]
    )
    def test_polarization_terms_with_dipole_batch(self, shape, expected_shape):
        pol = sources.polarization_terms(
            jx=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            jy=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            jz=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            layer_solve_result=LAYER_SOLVE_RESULT,
        )
        expected_shape = list(expected_shape)
        expected_shape[-2] = 4 * LAYER_SOLVE_RESULT.expansion.num_terms
        self.assertSequenceEqual(pol.shape, expected_shape)

    @parameterized.parameterized.expand(
        [
            [(20, 30, 1), (1, 1, 2, -1, 1)],
            [(30, 20, 2), (1, 1, 2, -1, 2)],
            [(10, 1, 30, 20, 2), (1, 10, 2, -1, 2)],
        ]
    )
    def test_polarization_terms_with_layer_batch(self, shape, expected_shape):
        pol = sources.polarization_terms(
            jx=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            jy=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            jz=fft.fft(jnp.ones(shape), expansion=EXPANSION, axes=(-3, -2)),
            layer_solve_result=BATCH_LAYER_SOLVE_RESULT,
        )
        expected_shape = list(expected_shape)
        expected_shape[-2] = 4 * BATCH_LAYER_SOLVE_RESULT.expansion.num_terms
        self.assertSequenceEqual(pol.shape, expected_shape)

    def test_emission_matrix_output_shape(self):
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=(LAYER_SOLVE_RESULT,) * 2,
            layer_thicknesses=(0.2, 0.3),
        )
        mat = sources.emission_matrix(s_matrix, s_matrix)
        n = LAYER_SOLVE_RESULT.expansion.num_terms
        self.assertSequenceEqual(mat.shape, (4 * n, 4 * n))

    def test_emission_matrix_batch_output_shape(self):
        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=(LAYER_SOLVE_RESULT,) * 2,
            layer_thicknesses=(0.2, 0.3),
        )
        s_matrix_batch = scattering.stack_s_matrix(
            layer_solve_results=(BATCH_LAYER_SOLVE_RESULT,) * 2,
            layer_thicknesses=(0.2, 0.3),
        )
        mat = sources.emission_matrix(s_matrix, s_matrix_batch)
        n = LAYER_SOLVE_RESULT.expansion.num_terms
        self.assertSequenceEqual(mat.shape, (1, 1, 2, 4 * n, 4 * n))
