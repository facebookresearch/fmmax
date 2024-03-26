"""Tests for `fmmax.fields`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import dataclasses
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp
import parameterized

from fmmax import basis, fields, fmm, scattering


def example_solve(permittivity_batch_shape, wavelength_batch_shape):
    # Solves for the interior scattering matrices of a simple structure.
    random_density = jax.random.uniform(
        jax.random.PRNGKey(0), permittivity_batch_shape + (20, 20)
    )
    permittivities = [
        jnp.ones(permittivity_batch_shape + (1, 1), dtype=complex),
        jnp.ones(permittivity_batch_shape + (1, 1), dtype=complex) * 2.25,
        random_density + 2,
        jnp.ones(permittivity_batch_shape + (1, 1), dtype=complex),
    ]
    thicknesses = [10, 10, 10, 10]
    in_plane_wavevector = jnp.zeros((2,))
    primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=10,
        truncation=basis.Truncation.CIRCULAR,
    )
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.full(wavelength_batch_shape, 100.0),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=p,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )
        for p in permittivities
    ]

    s_matrices_interior = scattering.stack_s_matrices_interior(
        layer_solve_results, thicknesses
    )
    return layer_solve_results, thicknesses, s_matrices_interior


def time_average_z_poynting_flux(electric_fields, magnetic_fields):
    ex, ey, _ = electric_fields
    hx, hy, _ = magnetic_fields
    smz = ex * jnp.conj(hy) - ey * jnp.conj(hx)
    return jnp.real(smz)


class ShapesTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # permittivity, wavelength, and excitation batch shape.
            [(), (), (1,)],
            [(3,), (), (1,)],
            [(), (), (2,)],
            [(3,), (), (2,)],
            [(2, 1, 3), (), (2,)],
            [(), (3,), (1,)],
            [(3,), (2, 1), (1,)],
            [(), (3,), (2,)],
            [(3,), (2, 1), (2,)],
            [(2, 1, 3), (2, 1, 1), (2,)],
        ]
    )
    def test_shapes_match_expected(
        self, permittivity_batch_shape, wavelength_batch_shape, excitation_batch_shape
    ):
        # Tests various functions in the `fields` module, validating compatibility
        # with various batch sizes for the excitation and simulation.
        layer_solve_results, thicknesses, s_matrices_interior = example_solve(
            permittivity_batch_shape, wavelength_batch_shape
        )

        num_terms = layer_solve_results[0].expansion.num_terms
        forward_amplitude_0_start = jnp.zeros(
            (2 * num_terms,) + excitation_batch_shape, dtype=complex
        )
        forward_amplitude_0_start = forward_amplitude_0_start.at[0, :].set(1)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)

        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )

        for (
            (forward_amplitude_start, backward_amplitude_end),
            layer_solve_result,
            thickness,
        ) in zip(ab, layer_solve_results, thicknesses):
            a, b = fields.colocate_amplitudes(
                forward_amplitude_start,
                backward_amplitude_end,
                z_offset=thickness / 2,
                layer_solve_result=layer_solve_result,
                layer_thickness=thickness,
            )
            batch_shape = jnp.broadcast_shapes(
                permittivity_batch_shape, wavelength_batch_shape
            )
            expected_shape = batch_shape + (num_terms * 2,) + excitation_batch_shape
            self.assertSequenceEqual(a.shape, expected_shape)
            self.assertSequenceEqual(b.shape, expected_shape)

            # Compute the electric and magnetic field Fourier coefficeints.
            ef, hf = fields.fields_from_wave_amplitudes(
                a, b, layer_solve_result=layer_solve_result
            )
            ex, ey, ez = ef
            hx, hy, hz = hf
            expected_shape = batch_shape + (num_terms,) + excitation_batch_shape
            self.assertSequenceEqual(ex.shape, expected_shape)
            self.assertSequenceEqual(ey.shape, expected_shape)
            self.assertSequenceEqual(ez.shape, expected_shape)
            self.assertSequenceEqual(hx.shape, expected_shape)
            self.assertSequenceEqual(hy.shape, expected_shape)
            self.assertSequenceEqual(hz.shape, expected_shape)

            # Compute the electric and magnetic fields on the grid.
            grid_shape = (10, 11)
            eg, hg, (x, y) = fields.fields_on_grid(
                ef,
                hf,
                layer_solve_result=layer_solve_result,
                shape=grid_shape,
                num_unit_cells=(1, 1),
            )
            egx, egy, egz = eg
            hgx, hgy, hgz = hg
            expected_shape = batch_shape + grid_shape + excitation_batch_shape
            self.assertSequenceEqual(egx.shape, expected_shape)
            self.assertSequenceEqual(egy.shape, expected_shape)
            self.assertSequenceEqual(egz.shape, expected_shape)
            self.assertSequenceEqual(hgx.shape, expected_shape)
            self.assertSequenceEqual(hgy.shape, expected_shape)
            self.assertSequenceEqual(hgz.shape, expected_shape)
            self.assertSequenceEqual(x.shape[-2:], grid_shape)
            self.assertSequenceEqual(y.shape[-2:], grid_shape)


class FieldsOnCoordinatesTest(unittest.TestCase):
    @parameterized.parameterized.expand([[()], [(3,)]])
    def test_fields_on_coordinates_match_fields_on_grid(self, batch_shape):
        in_plane_wavevector = jnp.zeros((2,))
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )

        permittivity = 1 + jax.random.uniform(
            jax.random.PRNGKey(0), batch_shape + (20, 20)
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(()),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        ex, ey, ez, hx, hy, hz = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=(6,) + batch_shape + (layer_solve_result.expansion.num_terms, 5),
            dtype=complex,
        )
        expected_efield, expected_hfield, (x, y) = fields.fields_on_grid(
            electric_field=(ex, ey, ez),
            magnetic_field=(hx, hy, hz),
            layer_solve_result=layer_solve_result,
            shape=(20, 20),
            num_unit_cells=(1, 1),
        )
        with self.subTest("xy on a square grid"):
            efield, hfield, (x, y) = fields.fields_on_coordinates(
                electric_field=(ex, ey, ez),
                magnetic_field=(hx, hy, hz),
                layer_solve_result=layer_solve_result,
                x=jnp.squeeze(x),
                y=jnp.squeeze(y),
            )
            onp.testing.assert_allclose(efield, expected_efield, rtol=1e-4)
            onp.testing.assert_allclose(hfield, expected_hfield, rtol=1e-4)

        with self.subTest("xy as flat arrays"):
            efield, hfield, _ = fields.fields_on_coordinates(
                electric_field=(ex, ey, ez),
                magnetic_field=(hx, hy, hz),
                layer_solve_result=layer_solve_result,
                x=x.flatten(),
                y=y.flatten(),
            )
            efield = onp.reshape(efield, onp.shape(expected_efield))
            hfield = onp.reshape(hfield, onp.shape(expected_efield))
            onp.testing.assert_allclose(efield, expected_efield, rtol=1e-4)
            onp.testing.assert_allclose(hfield, expected_hfield, rtol=1e-4)

    @parameterized.parameterized.expand([[()], [(3,)]])
    def test_layer_fields_on_coordinates_match_fields_on_grid(self, batch_shape):
        in_plane_wavevector = jnp.zeros((2,))
        primitive_lattice_vectors = basis.LatticeVectors(u=basis.X, v=basis.Y)
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=primitive_lattice_vectors,
            approximate_num_terms=10,
            truncation=basis.Truncation.CIRCULAR,
        )

        permittivity = 1 + jax.random.uniform(
            jax.random.PRNGKey(0), batch_shape + (20, 20)
        )
        layer_solve_result = fmm.eigensolve_isotropic_media(
            wavelength=jnp.ones(()),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=fmm.Formulation.FFT,
        )

        fwd_amplitude_start, bwd_amplitude_end = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=(2,) + batch_shape + (layer_solve_result.expansion.num_terms * 2, 5),
            dtype=complex,
        )
        expected_efield, expected_hfield, (x, y, _) = fields.layer_fields_3d(
            forward_amplitude_start=fwd_amplitude_start,
            backward_amplitude_end=bwd_amplitude_end,
            layer_solve_result=layer_solve_result,
            layer_thickness=jnp.ones(()),
            layer_znum=10,
            grid_shape=(20, 20),
            num_unit_cells=(1, 1),
        )

        with self.subTest("xy on a square grid"):
            efield, hfield, _ = fields.layer_fields_3d_on_coordinates(
                forward_amplitude_start=fwd_amplitude_start,
                backward_amplitude_end=bwd_amplitude_end,
                layer_solve_result=layer_solve_result,
                layer_thickness=jnp.ones(()),
                layer_znum=10,
                x=jnp.squeeze(x),
                y=jnp.squeeze(y),
            )
            onp.testing.assert_allclose(efield, expected_efield, rtol=2e-4)
            onp.testing.assert_allclose(hfield, expected_hfield, rtol=2e-4)

        with self.subTest("xy as flat arrays"):
            efield, hfield, _ = fields.layer_fields_3d_on_coordinates(
                forward_amplitude_start=fwd_amplitude_start,
                backward_amplitude_end=bwd_amplitude_end,
                layer_solve_result=layer_solve_result,
                layer_thickness=jnp.ones(()),
                layer_znum=10,
                x=x.flatten(),
                y=y.flatten(),
            )
            efield = onp.reshape(efield, onp.shape(expected_efield))
            hfield = onp.reshape(hfield, onp.shape(expected_efield))
            onp.testing.assert_allclose(efield, expected_efield, rtol=2e-4)
            onp.testing.assert_allclose(hfield, expected_hfield, rtol=2e-4)

    @parameterized.parameterized.expand([[()], [(3,)]])
    def test_stack_fields_on_coordinates_match_fields_on_grid(self, batch_shape):
        layer_solve_results, thicknesses, s_matrices_interior = example_solve(
            batch_shape, ()
        )
        num_terms = layer_solve_results[0].expansion.num_terms
        forward_amplitude_0_start = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=batch_shape + (2 * num_terms, 1),
            dtype=complex,
        )
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )
        expected_efield, expected_hfield, (x, y, z) = fields.stack_fields_3d(
            amplitudes_interior=ab,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            layer_znum=[10] * len(thicknesses),
            grid_shape=(10, 10),
            num_unit_cells=(1, 1),
        )
        efield, hfield, (x, y, z) = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=ab,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            layer_znum=[10] * len(thicknesses),
            x=jnp.squeeze(x),
            y=jnp.squeeze(y),
        )
        onp.testing.assert_allclose(efield, expected_efield, rtol=1e-4)
        onp.testing.assert_allclose(hfield, expected_hfield, rtol=1e-4)


class PoyntingFluxTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            [(), ()],
            [(3,), ()],
            [(), ()],
            [(3,), ()],
            [(2, 1, 3), ()],
            [(), (3,)],
            [(3,), (2, 1)],
            [(2, 1, 3), (2, 1, 1)],
        ]
    )
    def test_eigenmode_poynting_flux(
        self, permittivity_batch_shape, wavelength_batch_shape
    ):
        layer_solve_results, _, _ = example_solve(
            permittivity_batch_shape, wavelength_batch_shape
        )
        layer_solve_result = layer_solve_results[0]
        eigenmode_flux = fields.eigenmode_poynting_flux(layer_solve_result)
        num_eigenmodes = layer_solve_result.eigenvalues.shape[-1]
        one_hot_amplitude = jnp.eye(num_eigenmodes)
        expected_eigenmode_flux, _ = fields.amplitude_poynting_flux(
            forward_amplitude=one_hot_amplitude,
            backward_amplitude=jnp.zeros_like(one_hot_amplitude),
            layer_solve_result=layer_solve_result,
        )
        expected_eigenmode_flux = jnp.sum(expected_eigenmode_flux, axis=-2)
        onp.testing.assert_allclose(eigenmode_flux, expected_eigenmode_flux)

    @parameterized.parameterized.expand(
        [
            [(), (), (1,)],
            [(3,), (), (1,)],
            [(), (), (2,)],
            [(3,), (), (2,)],
            [(2, 1, 3), (), (2,)],
            [(), (3,), (1,)],
            [(3,), (2, 1), (1,)],
            [(), (3,), (2,)],
            [(3,), (2, 1), (2,)],
            [(2, 1, 3), (2, 1, 1), (2,)],
        ]
    )
    def test_amplitude_poynting_flux(
        self, permittivity_batch_shape, wavelength_batch_shape, excitation_batch_shape
    ):
        layer_solve_results, thicknesses, s_matrices_interior = example_solve(
            permittivity_batch_shape, wavelength_batch_shape
        )

        num_eigenmodes = layer_solve_results[0].eigenvalues.shape[-1]
        amplitude_shape = (num_eigenmodes,) + excitation_batch_shape
        forward_amplitude_0_start = jax.random.uniform(
            jax.random.PRNGKey(0), amplitude_shape
        ).astype(complex)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )

        for (
            (forward_amplitude_start, backward_amplitude_end),
            layer_solve_result,
            thickness,
        ) in zip(ab, layer_solve_results, thicknesses):
            a, b = fields.colocate_amplitudes(
                forward_amplitude_start,
                backward_amplitude_end,
                z_offset=thickness / 2,
                layer_solve_result=layer_solve_result,
                layer_thickness=thickness,
            )
            ef, hf = fields.fields_from_wave_amplitudes(
                a, b, layer_solve_result=layer_solve_result
            )
            grid_shape = (10, 11)
            eg, hg, _ = fields.fields_on_grid(
                ef,
                hf,
                layer_solve_result=layer_solve_result,
                shape=grid_shape,
                num_unit_cells=(1, 1),
            )
            sgz = time_average_z_poynting_flux(eg, hg)
            s_forward, s_backward = fields.amplitude_poynting_flux(
                a, b, layer_solve_result
            )
            onp.testing.assert_allclose(
                jnp.sum(s_forward + s_backward, axis=-2),
                jnp.mean(sgz, axis=(-3, -2)),
                rtol=1e-5,
            )

    @parameterized.parameterized.expand(
        [
            [(), (), (1,)],
            [(3,), (), (1,)],
            [(), (), (2,)],
            [(3,), (), (2,)],
            [(2, 1, 3), (), (2,)],
            [(), (3,), (1,)],
            [(3,), (2, 1), (1,)],
            [(), (3,), (2,)],
            [(3,), (2, 1), (2,)],
            [(2, 1, 3), (2, 1, 1), (2,)],
        ]
    )
    def test_directional_poynting_flux(
        self, permittivity_batch_shape, wavelength_batch_shape, excitation_batch_shape
    ):
        layer_solve_results, thicknesses, s_matrices_interior = example_solve(
            permittivity_batch_shape, wavelength_batch_shape
        )

        num_eigenmodes = layer_solve_results[0].eigenvalues.shape[-1]
        amplitude_shape = (num_eigenmodes,) + excitation_batch_shape
        forward_amplitude_0_start = jax.random.uniform(
            jax.random.PRNGKey(0), amplitude_shape
        ).astype(complex)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )

        for (
            (forward_amplitude_start, backward_amplitude_end),
            layer_solve_result,
            thickness,
        ) in zip(ab, layer_solve_results, thicknesses):
            a, b = fields.colocate_amplitudes(
                forward_amplitude_start,
                backward_amplitude_end,
                z_offset=thickness / 2,
                layer_solve_result=layer_solve_result,
                layer_thickness=thickness,
            )
            ef, hf = fields.fields_from_wave_amplitudes(
                a, b, layer_solve_result=layer_solve_result
            )
            grid_shape = (10, 11)
            eg, hg, _ = fields.fields_on_grid(
                ef,
                hf,
                layer_solve_result=layer_solve_result,
                shape=grid_shape,
                num_unit_cells=(1, 1),
            )
            sgz = time_average_z_poynting_flux(eg, hg)
            s_forward, s_backward = fields.directional_poynting_flux(
                a, b, layer_solve_result
            )
            onp.testing.assert_allclose(
                jnp.sum(s_forward + s_backward, axis=-2),
                jnp.mean(sgz, axis=(-3, -2)),
                rtol=1e-5,
            )

            # Randomly flip the sign of some eigenvalues, and check that the
            # directional Poynting flux is unchanged.
            q = layer_solve_result.eigenvalues
            mask = jax.random.uniform(jax.random.PRNGKey(0), (q.shape[-1],))
            mask = jnp.ones_like(mask)
            a_swapped = jnp.where(mask[:, jnp.newaxis], a, b)
            b_swapped = jnp.where(mask[:, jnp.newaxis], b, a)
            layer_solve_result_swapped = dataclasses.replace(
                layer_solve_result, eigenvalues=jnp.where(mask, q, -q)
            )
            s_forward_swapped, s_backward_swapped = fields.directional_poynting_flux(
                a_swapped, b_swapped, layer_solve_result_swapped
            )

            onp.testing.assert_allclose(s_forward, s_forward_swapped)
            onp.testing.assert_allclose(s_backward, s_backward_swapped)
            onp.testing.assert_allclose(
                jnp.sum(s_forward_swapped + s_backward_swapped, axis=-2),
                jnp.mean(sgz, axis=(-3, -2)),
                rtol=1e-5,
            )


class Fields3DTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # permittivity, wavelength, and excitation batch shape.
            [(), (), (1,), (1, 1)],
            [(3,), (), (1,), (1, 1)],
            [(), (), (2,), (1, 1)],
            [(3,), (), (2,), (1, 1)],
            [(2, 1, 3), (), (2,), (1, 1)],
            [(), (3,), (1,), (1, 1)],
            [(3,), (2, 1), (1,), (1, 1)],
            [(), (3,), (2,), (1, 1)],
            [(3,), (2, 1), (2,), (1, 1)],
            [(2, 1, 3), (2, 1, 1), (2,), (1, 1)],
            [(2, 1, 3), (2, 1, 1), (2,), (3, 2)],
        ]
    )
    def test_shapes_match_expected(
        self,
        permittivity_batch_shape,
        wavelength_batch_shape,
        excitation_batch_shape,
        num_unit_cells,
    ):
        layer_solve_results, thicknesses, s_matrices_interior = example_solve(
            permittivity_batch_shape, wavelength_batch_shape
        )
        num_terms = layer_solve_results[0].expansion.num_terms
        forward_amplitude_0_start = jnp.zeros(
            (2 * num_terms,) + excitation_batch_shape, dtype=complex
        )
        forward_amplitude_0_start = forward_amplitude_0_start.at[0, :].set(1)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )
        ef, hf, (x, y, z) = fields.stack_fields_3d_auto_grid(
            amplitudes_interior=ab,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            grid_spacing=0.2,
            num_unit_cells=num_unit_cells,
        )
        batch_shape = jnp.broadcast_shapes(
            permittivity_batch_shape, wavelength_batch_shape
        )
        expected_grid_shape = (5 * num_unit_cells[0], 5 * num_unit_cells[1])
        self.assertSequenceEqual(
            ef.shape,
            (3,) + batch_shape + expected_grid_shape + (200,) + excitation_batch_shape,
        )

        # Check that `z` is monotonic.
        self.assertTrue(not onp.any(onp.diff(z) < 0))

    def test_batch_excitation_matches_single(self):
        layer_solve_results, thicknesses, s_matrices_interior = example_solve((), ())
        num_terms = layer_solve_results[0].expansion.num_terms

        # Single amplitude solve.
        forward_amplitude_0_start = jnp.zeros((2 * num_terms, 1), dtype=complex)
        forward_amplitude_0_start = forward_amplitude_0_start.at[0, :].set(1)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )
        ef_single, hf_single, (x, y, z) = fields.stack_fields_3d_auto_grid(
            amplitudes_interior=ab,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            grid_spacing=0.2,
            num_unit_cells=(1, 1),
        )

        # Batch amplitude solve.
        forward_amplitude_0_start = jnp.zeros((2 * num_terms, 1), dtype=complex)
        forward_amplitude_0_start = forward_amplitude_0_start.at[0, :].set(1)
        backward_amplitude_N_end = jnp.zeros_like(forward_amplitude_0_start)
        ab = fields.stack_amplitudes_interior(
            s_matrices_interior, forward_amplitude_0_start, backward_amplitude_N_end
        )
        ef_batch, hf_batch, (x, y, z) = fields.stack_fields_3d_auto_grid(
            amplitudes_interior=ab,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            grid_spacing=0.2,
            num_unit_cells=(1, 1),
        )

        onp.testing.assert_array_equal(ef_batch[..., 0], ef_single[..., 0])
        onp.testing.assert_array_equal(hf_batch[..., 0], hf_single[..., 0])
        onp.testing.assert_array_equal(ef_batch[..., 1], ef_single[..., 0])
        onp.testing.assert_array_equal(hf_batch[..., 1], hf_single[..., 0])

    def test_sequence_lengths_match_validation(self):
        with self.assertRaisesRegex(ValueError, "Encountered incompatible lengths"):
            fields.stack_fields_3d(
                amplitudes_interior=range(5),
                layer_solve_results=range(5),
                layer_thicknesses=range(5),
                layer_znum=range(4),
                grid_shape=(10, 10),
                num_unit_cells=(1, 1),
            )
