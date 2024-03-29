"""Tests for `fmmax.pml`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from fmmax import basis, beams, fields, fmm, pml, scattering, sources


class GaussianBeamInPMLDecayTest(unittest.TestCase):
    def simulate_beam_1d(
        self, use_pml, polar_angle=jnp.pi / 3, approximate_num_terms=400
    ):
        width = 20.0  # Simulation unit cell width
        width_pml = 2.0  # Width of perfectly matched layers on edges of unit cell.
        grid_spacing = 0.02

        beam_waist = 1.0
        permittivity_ambient = 1.0 + 0.0j
        thickness_ambient = 20.0

        wavelength = jnp.asarray(0.45)
        in_plane_wavevector = jnp.zeros((2,))
        primitive_lattice_vectors = basis.LatticeVectors(u=width * basis.X, v=basis.Y)
        dim = int(width / grid_spacing)
        grid_shape = (dim, 1)
        formulation = fmm.Formulation.FFT

        # Manually generate the expansion for a one-dimensional simulation.
        nmax = approximate_num_terms // 2
        ix = onp.zeros((2 * nmax + 1,), dtype=int)
        ix[1::2] = -jnp.arange(1, nmax + 1, dtype=int)
        ix[2::2] = jnp.arange(1, nmax + 1, dtype=int)
        assert tuple(ix[:5].tolist()) == (0, -1, 1, -2, 2)
        expansion = basis.Expansion(
            basis_coefficients=onp.stack([ix, onp.zeros_like(ix)], axis=-1)
        )

        def eigensolve_fn(permittivity: jnp.ndarray) -> fmm.LayerSolveResult:
            if use_pml:
                permittivities_pml, permeabilities_pml = pml.apply_uniaxial_pml(
                    permittivity=permittivity,
                    pml_params=pml.PMLParams(
                        num_x=int(width_pml / grid_spacing), num_y=0
                    ),
                )
                return fmm.eigensolve_general_anisotropic_media(
                    wavelength,
                    in_plane_wavevector,
                    primitive_lattice_vectors,
                    *permittivities_pml,
                    *permeabilities_pml,
                    expansion=expansion,
                    formulation=formulation,
                    vector_field_source=jnp.mean(
                        jnp.asarray(permittivities_pml), axis=0
                    ),
                )
            return fmm.eigensolve_isotropic_media(
                wavelength=wavelength,
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=primitive_lattice_vectors,
                permittivity=permittivity,
                expansion=expansion,
                formulation=formulation,
            )

        layer_solve_result = eigensolve_fn(
            permittivity=jnp.full(grid_shape, permittivity_ambient)
        )

        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=[layer_solve_result],
            layer_thicknesses=[jnp.asarray(thickness_ambient)],
        )

        def _paraxial_gaussian_field_fn(x, y, z):
            # Returns the fields of a z-propagating, x-polarized Gaussian beam.
            wavelengths_padded = wavelength[..., jnp.newaxis, jnp.newaxis]
            k = 2 * jnp.pi / wavelengths_padded
            z_r = (
                jnp.pi
                * beam_waist**2
                * jnp.sqrt(permittivity_ambient)
                / wavelengths_padded
            )
            w_z = beam_waist * jnp.sqrt(1 + (z / z_r) ** 2)
            r = jnp.sqrt(x**2 + y**2)
            ex = (
                beam_waist
                / w_z
                * jnp.exp(-(r**2) / w_z**2)
                * jnp.exp(
                    1j
                    * (
                        (k * z)  # Phase
                        + (k * r**2 / 2) * z / (z**2 + z_r**2)  # Wavefront curvature
                        - jnp.arctan(z / z_r)  # Gouy phase
                    )
                )
            )
            ey = jnp.zeros_like(ex)
            ez = jnp.zeros_like(ex)
            hx = jnp.zeros_like(ex)
            hy = ex / jnp.sqrt(permittivity_ambient)
            hz = jnp.zeros_like(ex)
            return (ex, ey, ez), (hx, hy, hz)

        # Solve for the fields of the beam with the desired rotation and shift.
        x, y = basis.unit_cell_coordinates(
            primitive_lattice_vectors=primitive_lattice_vectors,
            shape=grid_shape,  # type: ignore[arg-type]
            num_unit_cells=(1, 1),
        )
        (beam_ex, beam_ey, _), (beam_hx, beam_hy, _) = beams.shifted_rotated_fields(
            field_fn=_paraxial_gaussian_field_fn,
            x=x,
            y=y,
            z=jnp.full(x.shape, thickness_ambient),
            beam_origin_x=jnp.amax(x) / 4,
            beam_origin_y=jnp.amax(y) / 2,
            beam_origin_z=thickness_ambient - 0,
            polar_angle=jnp.asarray(jnp.pi - polar_angle),
            azimuthal_angle=jnp.asarray(0.0),
            polarization_angle=jnp.asarray(0.0),
        )

        _, bwd_amplitude_ambient_end = sources.amplitudes_for_fields(
            ex=beam_ex[..., jnp.newaxis],
            ey=beam_ey[..., jnp.newaxis],
            hx=beam_hx[..., jnp.newaxis],
            hy=beam_hy[..., jnp.newaxis],
            layer_solve_result=layer_solve_result,
            brillouin_grid_axes=None,
        )

        fwd_flux_end, bwd_flux_end = fields.amplitude_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=bwd_amplitude_ambient_end,
            layer_solve_result=layer_solve_result,
        )
        fwd_flux_start, bwd_flux_start = fields.amplitude_poynting_flux(
            forward_amplitude=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude=fields.propagate_amplitude(
                amplitude=bwd_amplitude_ambient_end,
                distance=thickness_ambient,
                layer_solve_result=layer_solve_result,
            ),
            layer_solve_result=layer_solve_result,
        )
        power_at_end = -jnp.sum(bwd_flux_end)
        power_at_start = -jnp.sum(bwd_flux_start)

        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=jnp.zeros_like(bwd_amplitude_ambient_end),
            backward_amplitude_N_end=bwd_amplitude_ambient_end,
        )
        layer_znum = int(jnp.round(thickness_ambient / grid_spacing) + 1)
        (ex, ey, ez), (hx, hy, hz), _ = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[layer_solve_result],
            layer_thicknesses=[jnp.asarray(thickness_ambient)],
            layer_znum=[layer_znum],
            grid_shape=grid_shape,
            num_unit_cells=(1, 1),
        )
        return power_at_end, power_at_start, (ex, ey, ez), (hx, hy, hz)

    def test_pml_absorbs(self):
        # Simulate a beam that is propagating at an angle. It strikes the PML and
        # must be absorbed before reaching the bottom of the simulation domain. The
        # beam is incident from the "end" of the simulation domain, and propagates
        # toward the "start" of the simulation domain.
        power_at_end, power_at_start, _, _ = self.simulate_beam_1d(use_pml=True)
        power_at_end_no_pml, power_at_start_no_pml, _, _ = self.simulate_beam_1d(
            use_pml=False
        )
        # With or without PML, the incident power is identical.
        onp.testing.assert_allclose(power_at_end, power_at_end_no_pml, rtol=1e-2)
        # With no PML, the power at the end is equal to the power at the start.
        onp.testing.assert_allclose(power_at_end_no_pml, power_at_start_no_pml)
        # With PML, the power at the start is decayed by at least 1e4.
        self.assertGreater(onp.abs(power_at_end), 1e4 * onp.abs(power_at_start))


class DipoleFieldsInPMLDecayTest(unittest.TestCase):
    def simulate_dipole_in_vacuum_with_pml(self, pml_params):
        pitch = 2.0
        primitive_lattice_vectors = basis.LatticeVectors(
            u=pitch * basis.X, v=pitch * basis.Y
        )
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=500,
            truncation=basis.Truncation.CIRCULAR,
        )
        in_plane_wavevector = jnp.zeros((2,))

        grid_shape = (100, 100)
        permittivity = 1.0 + 0.00001j
        permittivities_pml, permeabilities_pml = pml.apply_uniaxial_pml(
            permittivity=jnp.full(grid_shape, permittivity), pml_params=pml_params
        )
        solve_result_ambient = fmm.eigensolve_general_anisotropic_media(
            wavelength=jnp.asarray(0.63),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity_xx=permittivities_pml[0],
            permittivity_xy=permittivities_pml[1],
            permittivity_yx=permittivities_pml[2],
            permittivity_yy=permittivities_pml[3],
            permittivity_zz=permittivities_pml[4],
            permeability_xx=permeabilities_pml[0],
            permeability_xy=permeabilities_pml[1],
            permeability_yx=permeabilities_pml[2],
            permeability_yy=permeabilities_pml[3],
            permeability_zz=permeabilities_pml[4],
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
            vector_field_source=None,  # Automatically choose the vector field source.
        )

        # Compute interior scattering matrices to enable field calculations.
        s_matrices_interior_before_source = scattering.stack_s_matrices_interior(
            layer_solve_results=[solve_result_ambient],
            layer_thicknesses=[jnp.asarray(1.0)],
        )
        s_matrices_interior_after_source = s_matrices_interior_before_source

        # Extract the scattering matrices relating fields at the two ends of each substack.
        s_matrix_before_source = s_matrices_interior_before_source[-1][0]
        s_matrix_after_source = s_matrices_interior_after_source[-1][0]

        # Generate the Fourier representation of a point dipole.
        dipole = sources.dirac_delta_source(
            location=jnp.asarray([[pitch / 2, pitch / 2]]),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            expansion=expansion,
        )

        # Compute backward eigenmode amplitudes at the end of the layer before the
        # source, and the forward amplitudes the start of the layer after the source.
        (
            _,
            _,
            bwd_amplitude_before_end,
            fwd_amplitude_after_start,
            _,
            _,
        ) = sources.amplitudes_for_source(
            jx=jnp.zeros_like(dipole),
            jy=dipole,
            jz=jnp.zeros_like(dipole),
            s_matrix_before_source=s_matrix_before_source,
            s_matrix_after_source=s_matrix_after_source,
        )

        # Solve for the eigenmode amplitudes in every layer of the stack.
        amplitudes_interior = fields.stack_amplitudes_interior_with_source(
            s_matrices_interior_before_source=s_matrices_interior_before_source,
            s_matrices_interior_after_source=s_matrices_interior_after_source,
            backward_amplitude_before_end=bwd_amplitude_before_end,
            forward_amplitude_after_start=fwd_amplitude_after_start,
        )
        # Coordinates where fields are to be evaluated.
        (ex, ey, ez), (hx, hy, hz), (x, y, z) = fields.stack_fields_3d(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=[solve_result_ambient, solve_result_ambient],
            grid_shape=grid_shape,
            layer_thicknesses=[jnp.asarray(1.0), jnp.asarray(1.0)],
            layer_znum=[20, 20],
            num_unit_cells=(1, 1),
        )
        return (ex, ey, ez), (hx, hy, hz), (x, y, z)

    def test_pml_fields_decay(self):
        (ex_pml, ey_pml, ez_pml), _, _ = self.simulate_dipole_in_vacuum_with_pml(
            pml.PMLParams(num_x=30, num_y=30)
        )
        e_pml = jnp.sqrt(
            jnp.abs(ex_pml) ** 2 + jnp.abs(ey_pml) ** 2 + jnp.abs(ez_pml) ** 2
        )
        e_pml_borders = jnp.concatenate(
            [
                e_pml[0, :, :, 0].flatten(),
                e_pml[-1, :, :, 0].flatten(),
                e_pml[:, 0, :, 0].flatten(),
                e_pml[:, -1, :, 0].flatten(),
            ]
        )

        (ex, ey, ez), _, _ = self.simulate_dipole_in_vacuum_with_pml(
            pml.PMLParams(num_x=0, num_y=0)
        )
        e = jnp.sqrt(jnp.abs(ex) ** 2 + jnp.abs(ey) ** 2 + jnp.abs(ez) ** 2)
        e_borders = jnp.concatenate(
            [
                e[0, :, :, 0].flatten(),
                e[-1, :, :, 0].flatten(),
                e[:, 0, :, 0].flatten(),
                e[:, -1, :, 0].flatten(),
            ]
        )

        self.assertSequenceEqual(e_pml.shape, (100, 100, 40, 1))
        self.assertSequenceEqual(e.shape, (100, 100, 40, 1))

        # Check that the simulation with PMLs has decayed fields at the borders.
        self.assertGreater(jnp.mean(e_borders), 10 * jnp.mean(e_pml_borders))


class PMLParamsTest(unittest.TestCase):
    def test_can_flatten_unflatten(self):
        params = pml.PMLParams(num_x=10, num_y=10)
        leaves, treedef = tree_util.tree_flatten(params)
        restored_params = tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(params, restored_params)


class CropAndPadTest(unittest.TestCase):
    def test_crop_and_pad_matches_expected(self):
        arr = jnp.asarray(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 16],
                [17, 18, 19, 20, 21, 22],
                [23, 24, 25, 26, 27, 28],
                [29, 30, 31, 32, 33, 34],
            ]
        )
        expected = jnp.asarray(
            [
                [8, 8, 8, 9, 9, 9],
                [8, 8, 8, 9, 9, 9],
                [14, 14, 14, 15, 15, 15],
                [19, 19, 19, 20, 20, 20],
                [25, 25, 25, 26, 26, 26],
                [25, 25, 25, 26, 26, 26],
            ]
        )
        onp.testing.assert_array_equal(
            pml._crop_and_edge_pad_pml_region(
                permittivity=arr,
                widths=(1, 2),
            ),
            expected,
        )


class DistanceTest(unittest.TestCase):
    shape = (7, 8)
    widths = (3, 2)
    expected_dx = (
        jnp.asarray(
            [
                [3, 3, 3, 3, 3, 3, 3, 3],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3],
            ]
        )
        / 3
    )
    expected_dy = (
        jnp.asarray(
            [
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
                [2, 1, 0, 0, 0, 0, 1, 2],
            ]
        )
        / 2
    )
    dx, dy = pml._normalized_distance_into_pml(shape, widths)
    onp.testing.assert_array_equal(dx, expected_dx)
    onp.testing.assert_array_equal(dy, expected_dy)
