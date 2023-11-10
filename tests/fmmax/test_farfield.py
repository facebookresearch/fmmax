"""Tests for `fmmax.farfield`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import functools
import unittest

import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, farfield, fields, fmm, scattering, sources, utils


class FarfieldProfileTest(unittest.TestCase):
    def test_dipole_farfield_matches_analytical_calculation(self):
        # Calculate the farfield for a dipole in vacuum, and compare to an analytical result.
        wavelength = jnp.asarray(0.63)
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=50,
            truncation=basis.Truncation.CIRCULAR,
        )
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            (7, 7), primitive_lattice_vectors
        )

        dipole = sources.dirac_delta_source(
            location=jnp.asarray([[0.5, 0.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )
        zeros = jnp.zeros_like(dipole)
        jx = jnp.concatenate([dipole, zeros, zeros], axis=-1)
        jy = jnp.concatenate([zeros, dipole, zeros], axis=-1)
        jz = jnp.concatenate([zeros, zeros, dipole], axis=-1)

        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        s_matrix_before_source = scattering.stack_s_matrix([layer_solve_result], [1.0])
        s_matrix_after_source = s_matrix_before_source
        bwd_amplitude_ambient_end, *_ = sources.amplitudes_for_source(
            jx=jx,
            jy=jy,
            jz=jz,
            s_matrix_before_source=s_matrix_before_source,
            s_matrix_after_source=s_matrix_after_source,
        )
        forward_flux, backward_flux = fields.directional_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=layer_solve_result,
        )
        (
            polar_angle,
            azimuthal_angle,
            solid_angle,
            farfield_flux,
        ) = farfield.farfield_profile(
            flux=-backward_flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=(0, 1),
        )
        # Sum the farfield contributions from the two polarizations, and normalize.
        farfield_flux = jnp.sum(farfield_flux, axis=-2)
        mask = ~jnp.isnan(farfield_flux)
        farfield_flux /= jnp.amax(farfield_flux[mask])

        # Compute an analytical farfield, which has amplitude `sin(theta)`, when
        # theta is the angle with respect to the polarization axis.
        x = jnp.sin(polar_angle) * jnp.cos(azimuthal_angle)
        y = jnp.sin(polar_angle) * jnp.sin(azimuthal_angle)
        z_flux = jnp.sin(polar_angle) ** 2
        x_flux = jnp.sin(jnp.arccos(x)) ** 2
        y_flux = jnp.sin(jnp.arccos(y)) ** 2
        analytical_farfield_flux = jnp.stack([x_flux, y_flux, z_flux], axis=-1)

        onp.testing.assert_allclose(
            farfield_flux[mask],
            analytical_farfield_flux[mask],
            rtol=1e-5,
        )

    @parameterized.parameterized.expand(
        (
            [(1, 1), (0, 1), jnp.pi / 2],
            [(4, 5), (0, 1), jnp.pi / 2],
            [(4, 5), (-4, -3), jnp.pi / 2],
            [(2, 4, 5), (-4, -3), jnp.pi / 2],
            [(4, 5, 2), (-5, -4), jnp.pi / 2],
            [(1, 4, 5, 2, 1), (-6, -5), jnp.pi / 2],
            [(4, 5), (0, 1), 0.8 * jnp.pi / 2],  # Integrals over fraction of farfield.
            [(4, 5), (0, 1), 0.5 * jnp.pi / 2],
            [(4, 5), (0, 1), 0.2 * jnp.pi / 2],
        )
    )
    def test_farfield_integral_in_different_domains(
        self, batch_shape, brillouin_grid_axes, polar_angle_cutoff
    ):
        # Checks that the farfield can be computed for a variety of batch
        # shapes and brillouin zone axes. Also checks that the integral of
        # flux in k space and in the angular domain agree.
        absolute_axes = utils.absolute_axes(brillouin_grid_axes, len(batch_shape) + 2)

        # Compute shapes of dummy variables.
        wavelength_shape = tuple(
            [1 if i in absolute_axes else d for i, d in enumerate(batch_shape)]
        )
        bz_grid_shape = tuple(
            [d for i, d in enumerate(batch_shape) if i in absolute_axes]
        )

        # Generate dummy variable shapes.
        wavelength = 1 + jnp.arange(jnp.prod(jnp.asarray(wavelength_shape)))
        wavelength = wavelength.reshape(wavelength_shape)
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_grid_shape,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        in_plane_wavevector = jnp.expand_dims(
            in_plane_wavevector,
            [i for i in range(len(batch_shape)) if i not in absolute_axes],
        )

        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )
        transverse_wavevectors = basis.transverse_wavevectors(
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        kx = transverse_wavevectors[..., 0]
        ky = transverse_wavevectors[..., 1]
        kt = jnp.sqrt(kx**2 + ky**2)
        kz = jnp.sqrt((2 * jnp.pi / wavelength[..., jnp.newaxis]) ** 2 - kt**2)
        kt_cutoff = jnp.sin(polar_angle_cutoff) * 2 * jnp.pi / wavelength

        # Construct a dummy flux which is 1 for all orders that have nonzero kz,
        # i.e. those which actually carry power into the farfield.
        flux = jnp.concatenate([jnp.where(jnp.isnan(kz), 0, 1)] * 2, axis=-1)
        flux = flux[..., jnp.newaxis]

        # Compute the flux within the specified angular cone.
        k_space_mask = kt < kt_cutoff[..., jnp.newaxis]
        k_space_mask = jnp.concatenate([k_space_mask, k_space_mask], axis=-1)
        k_space_mask = k_space_mask[..., jnp.newaxis]

        # Integrate the flux over k-space. The differential area element in
        # k-space is independent of k.
        integrated_flux_k_space = jnp.sum(
            jnp.where(k_space_mask, flux, 0), axis=brillouin_grid_axes + (-2,)
        )

        # Compute the farfield angles, solid angle, and associated flux.
        (
            polar_angle,
            azimuthal_angle,
            solid_angle,
            farfield_flux,
        ) = farfield.farfield_profile(
            flux=flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=brillouin_grid_axes,
        )
        farfield_flux = jnp.where(jnp.isnan(farfield_flux), 0, farfield_flux)
        solid_angle = solid_angle[..., jnp.newaxis, jnp.newaxis]

        angle_mask = polar_angle < polar_angle_cutoff
        angle_mask = jnp.where(jnp.isnan(angle_mask), False, angle_mask)
        angle_mask = angle_mask[..., jnp.newaxis, jnp.newaxis]

        # Integrate the flux in the angular domain. The differential element
        # is `solid_angle`, i.e. the area of the cell associated with each
        # element in `farfield_flux`.
        integrated_flux_spherical_coords = jnp.sum(
            jnp.where(angle_mask, farfield_flux * solid_angle, 0), axis=(-4, -3, -2)
        )
        onp.testing.assert_allclose(
            integrated_flux_spherical_coords, integrated_flux_k_space
        )


class IntegratedFluxTest(unittest.TestCase):
    def test_resize(self):
        # Directly tests resizing, as this can fail if some GPU libraries are missing.
        arr = jnp.ones((10, 10))
        upsampled = jax.image.resize(arr, shape=(30, 30), method="linear")
        self.assertSequenceEqual(upsampled.shape, (30, 30))

    @parameterized.parameterized.expand(
        (
            [(4, 4), (0, 1), 1, 1],
            [(2, 4, 4), (1, 2), 1, 1],
            [(4, 4, 2), (0, 1), 1, 1],
            [(4, 4, 2), (0, 1), 5, 1],
            [(4, 2, 4), (0, 2), 5, 1],
            [(4, 4, 2), (0, 1), 1, 2],
        )
    )
    def test_integrated_flux_matches_direct_calculation(
        self, batch_shape, brillouin_grid_axes, num_sources, upsample_factor
    ):
        # Checks that the calculation of integrated flux "directly" matches that
        # when calcuated by first computing weights, and then taking the inner
        # product.

        # Dummy values to test the farfield calculation.
        primitive_lattice_vectors = basis.LatticeVectors(basis.X * 3, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )
        flux = jax.random.uniform(
            jax.random.PRNGKey(0),
            batch_shape + (2 * expansion.num_terms, num_sources),
        )
        wavelength = jax.random.uniform(
            jax.random.PRNGKey(1),
            [(1 if i in brillouin_grid_axes else d) for i, d in enumerate(batch_shape)],
        )
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=tuple([batch_shape[i] for i in brillouin_grid_axes]),
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        in_plane_wavevector = jnp.expand_dims(
            in_plane_wavevector,
            axis=[i for i in range(len(batch_shape)) if i not in brillouin_grid_axes],
        )

        def angle_bounds_fn(polar_angle, azimuthal_angle):
            del azimuthal_angle
            return jnp.full(polar_angle.shape, True)

        # Calculate the integrated flux by the default method, which first
        # computes weights and then does the integration by taking the weighted
        # sum of flux.
        integrated_flux = farfield.integrated_flux(
            flux=flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=brillouin_grid_axes,
            angle_bounds_fn=angle_bounds_fn,
            upsample_factor=upsample_factor,
        )

        # Compute the integrated flux directly.
        integrated_flux_direct = farfield._integrated_flux_upsampled(
            flux=flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=brillouin_grid_axes,
            angle_bounds_fn=angle_bounds_fn,
            upsample_factor=upsample_factor,
        )
        onp.testing.assert_allclose(integrated_flux, integrated_flux_direct, rtol=1e-6)

    def test_upsample_scale(self):
        # Check that scaling of the integrated power when using upsampling is correct.
        primitive_lattice_vectors = basis.LatticeVectors(basis.X * 3, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.CIRCULAR,
        )
        flux = jax.random.uniform(
            jax.random.PRNGKey(0),
            (4, 4, 2 * expansion.num_terms, 1),
        )
        wavelength = jnp.asarray(0.63)
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=(4, 4),
            primitive_lattice_vectors=primitive_lattice_vectors,
        )

        def angle_bounds_fn(polar_angle, azimuthal_angle):
            del azimuthal_angle
            return jnp.full(polar_angle.shape, True)

        integrated_flux_fn = functools.partial(
            farfield.integrated_flux,
            flux=flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=(0, 1),
            angle_bounds_fn=angle_bounds_fn,
        )
        integrated_flux_no_upsample = integrated_flux_fn(upsample_factor=1)
        integrated_flux_upsample = integrated_flux_fn(upsample_factor=10)
        onp.testing.assert_allclose(
            integrated_flux_no_upsample, integrated_flux_upsample
        )

    def test_all_angles_matches_total_flux(self):
        wavelength = jnp.asarray(0.63)
        primitive_lattice_vectors = basis.LatticeVectors(basis.X, basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=50,
            truncation=basis.Truncation.CIRCULAR,
        )
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            (7, 7), primitive_lattice_vectors
        )

        dipole = sources.dirac_delta_source(
            location=jnp.asarray([[0.5, 0.5]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )
        zeros = jnp.zeros_like(dipole)
        jx = jnp.concatenate([dipole, zeros, zeros], axis=-1)
        jy = jnp.concatenate([zeros, dipole, zeros], axis=-1)
        jz = jnp.concatenate([zeros, zeros, dipole], axis=-1)

        layer_solve_result = fmm.eigensolve_isotropic_media(
            permittivity=jnp.asarray([[1.0]]),
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        s_matrix_before_source = scattering.stack_s_matrix([layer_solve_result], [1.0])
        s_matrix_after_source = s_matrix_before_source
        (
            bwd_amplitude_ambient_end,
            fwd_amplitude_before_start,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            bwd_amplitude_after_end,
            fwd_amplitude_substrate_start,
        ) = sources.amplitudes_for_source(
            jx=jx,
            jy=jy,
            jz=jz,
            s_matrix_before_source=s_matrix_before_source,
            s_matrix_after_source=s_matrix_after_source,
        )
        forward_flux, backward_flux = fields.directional_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=layer_solve_result,
        )
        (
            polar_angle,
            azimuthal_angle,
            solid_angle,
            farfield_flux,
        ) = farfield.farfield_profile(
            flux=-backward_flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=(0, 1),
        )

        integrated_flux = farfield.integrated_flux(
            flux=-backward_flux,
            wavelength=wavelength,
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
            brillouin_grid_axes=(0, 1),
            angle_bounds_fn=lambda polar_angle, _: jnp.full(polar_angle.shape, True),
            upsample_factor=10,
        )
        onp.testing.assert_allclose(
            jnp.sum(-backward_flux),
            jnp.sum(integrated_flux),
            rtol=1e-6,
        )


class UnflattenTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        (
            ([[1, 2, 3], [4, 5, 6]],),
            ([[[1, 2, 3], [4, 5, 6]]],),
            ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],),
            ([[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]],),
        )
    )
    def test_unflatten_matches_expected(self, bz_data):
        bz_data = jnp.asarray(bz_data)

        expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(basis.X * 3, basis.Y),
            approximate_num_terms=100,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )

        # Create a dummy array to be unstacked, where all the Fourier orders for a
        # given Brillouin zone point just take the value in `bz_data`.
        data = jnp.broadcast_to(
            bz_data[..., jnp.newaxis], bz_data.shape + (expansion.num_terms,)
        )
        unstacked = farfield.unflatten(data, expansion)

        i_min = onp.amin(expansion.basis_coefficients[:, 0])
        i_max = onp.amax(expansion.basis_coefficients[:, 0])
        j_min = onp.amin(expansion.basis_coefficients[:, 1])
        j_max = onp.amax(expansion.basis_coefficients[:, 1])

        batch_shape = bz_data.shape[:-2]
        expected_shape = batch_shape + (
            bz_data.shape[-2] * (i_max - i_min + 1),
            bz_data.shape[-1] * (j_max - j_min + 1),
        )
        self.assertSequenceEqual(unstacked.shape, expected_shape)

        # The unflattening of dummy data is just the tiling of the original `bz_data`.
        expected = onp.tile(
            bz_data,
            (1,) * (bz_data.ndim - 2) + (i_max - i_min + 1, j_max - j_min + 1),
        )
        onp.testing.assert_array_equal(expected, unstacked)

    @parameterized.parameterized.expand(
        (
            [(4, 5), (0, 1)],
            [(2, 4, 5), (1, 2)],
            [(1, 3, 2, 4), (1, 2)],
            [(1, 3, 2, 4), (-5, -4)],
        )
    )
    def test_unflatten_flux_matches_expected(self, batch_shape, bz_axes):
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y * 3),
            approximate_num_terms=100,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )

        i_min = onp.amin(expansion.basis_coefficients[:, 0])
        i_max = onp.amax(expansion.basis_coefficients[:, 0])
        j_min = onp.amin(expansion.basis_coefficients[:, 1])
        j_max = onp.amax(expansion.basis_coefficients[:, 1])
        nkx = i_max - i_min + 1
        nky = j_max - j_min + 1

        # Create a dummy flux array with batch dimensions and four sources.
        # This array is missing the Fourier order axis; we will broadcast and
        # reshape so that all Fourier orders for a given batch element and
        # polarization are equal.
        num_sources = 4
        batch_flux = onp.arange(onp.prod(batch_shape) * 2 * num_sources)
        batch_flux = batch_flux.reshape(batch_shape + (2, num_sources))

        # Broadcast and reshape to have the proper shape of a flux array.
        flux = onp.broadcast_to(
            batch_flux[..., jnp.newaxis, :],
            batch_shape + (2, expansion.num_terms, num_sources),
        )
        flux = flux.reshape(batch_shape + (2 * expansion.num_terms, num_sources))

        unstacked = farfield.unflatten_flux(flux, expansion, bz_axes)

        # Compute expected shape of the unflattend flux array.
        bz_axes = tuple([b % flux.ndim for b in bz_axes])
        unstacked_batch_shape = tuple(
            [d for i, d in enumerate(batch_shape) if i not in bz_axes]
        )
        expected_shape = unstacked_batch_shape + (
            nkx * batch_shape[bz_axes[0]],
            nky * batch_shape[bz_axes[1]],
            2,
            num_sources,
        )
        self.assertSequenceEqual(unstacked.shape, expected_shape)

        # Compute the expected unflattened flux array.
        transposed_axes = (
            tuple([i for i in range(batch_flux.ndim - 2) if i not in bz_axes])
            + bz_axes
            + (flux.ndim - 2, flux.ndim - 1)
        )
        transposed_batch_flux = onp.transpose(batch_flux, axes=transposed_axes)
        expected = onp.tile(
            transposed_batch_flux, (1,) * len(unstacked_batch_shape) + (nkx, nky, 1, 1)
        )
        onp.testing.assert_array_equal(expected, unstacked)

    @parameterized.parameterized.expand(
        (
            [(4, 5), (0, 1)],
            [(2, 4, 5), (1, 2)],
            [(1, 3, 2, 4), (1, 2)],
            [(1, 3, 2, 4), (-5, -4)],
        )
    )
    def test_unflatten_wavevector_matches_expected(self, batch_shape, bz_axes):
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(basis.X, basis.Y * 3),
            approximate_num_terms=100,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )

        i_min = onp.amin(expansion.basis_coefficients[:, 0])
        i_max = onp.amax(expansion.basis_coefficients[:, 0])
        j_min = onp.amin(expansion.basis_coefficients[:, 1])
        j_max = onp.amax(expansion.basis_coefficients[:, 1])
        nkx = i_max - i_min + 1
        nky = j_max - j_min + 1

        # Create a dummy transverse wavevector array with batch dimensions.
        # This array is missing the Fourier order axis; we will broadcast and
        # reshape so that all Fourier orders for a given batch element are equal.
        batch_wavevectors = onp.arange(onp.prod(batch_shape) * 2)
        batch_wavevectors = batch_wavevectors.reshape(batch_shape + (2,))

        # Broadcast and reshape to have the proper shape of a wavevector array.
        wavevectors = onp.broadcast_to(
            batch_wavevectors[..., jnp.newaxis, :],
            batch_shape + (expansion.num_terms, 2),
        )

        unstacked = farfield.unflatten_transverse_wavevectors(
            wavevectors, expansion, bz_axes
        )

        # Compute expected shape of the unflattend flux array.
        bz_axes = tuple([b % wavevectors.ndim for b in bz_axes])
        unstacked_batch_shape = tuple(
            [d for i, d in enumerate(batch_shape) if i not in bz_axes]
        )
        expected_shape = unstacked_batch_shape + (
            nkx * batch_shape[bz_axes[0]],
            nky * batch_shape[bz_axes[1]],
            2,
        )
        self.assertSequenceEqual(unstacked.shape, expected_shape)

        # Compute the expected unflattened transverse wavevector array.
        transposed_axes = (
            tuple([i for i in range(batch_wavevectors.ndim - 1) if i not in bz_axes])
            + bz_axes
            + (batch_wavevectors.ndim - 1,)
        )
        transposed_flux = onp.transpose(batch_wavevectors, axes=transposed_axes)
        expected = onp.tile(
            transposed_flux, (1,) * len(unstacked_batch_shape) + (nkx, nky, 1)
        )
        onp.testing.assert_array_equal(expected, unstacked)

    @parameterized.parameterized.expand(
        (
            [(1, 1), basis.LatticeVectors(basis.X, basis.Y)],
            [(3, 4), basis.LatticeVectors(basis.X, basis.Y)],
            [(1, 1), basis.LatticeVectors(basis.X, -basis.Y)],
            [(3, 4), basis.LatticeVectors(basis.X, -basis.Y)],
            [(1, 1), basis.LatticeVectors(basis.X, basis.Y * 3)],
            [(3, 4), basis.LatticeVectors(basis.X, basis.Y * 3)],
            [(1, 1), basis.LatticeVectors(basis.X + basis.Y, basis.X - basis.Y)],
            [(3, 4), basis.LatticeVectors(basis.X + basis.Y, basis.X - basis.Y)],
        )
    )
    def test_unstacked_wavevectors_are_sorted(
        self, bz_shape, primitive_lattice_vectors
    ):
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=100,
            truncation=basis.Truncation.PARALLELOGRAMIC,
        )
        in_plane_wavevector = basis.brillouin_zone_in_plane_wavevector(
            brillouin_grid_shape=bz_shape,
            primitive_lattice_vectors=primitive_lattice_vectors,
        )
        transverse_wavevectors = basis.transverse_wavevectors(
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )
        unflattened_transverse_wavevectors = farfield.unflatten_transverse_wavevectors(
            transverse_wavevectors=transverse_wavevectors,
            expansion=expansion,
            brillouin_grid_axes=(0, 1),
        )
        kx = unflattened_transverse_wavevectors[..., 0]
        ky = unflattened_transverse_wavevectors[..., 1]
        self.assertEqual(kx.ndim, 2)
        dkx = kx[1:, :] - kx[:-1, :]
        dky = ky[:, 1:] - ky[:, :-1]
        # The wavevectors should be monotonically increasing or decreasing,
        # depending upon the choice of basis vectors.
        self.assertTrue(onp.all(dkx > 0) or onp.all(dkx < 0))
        self.assertTrue(onp.all(dky > 0) or onp.all(dky < 0))
