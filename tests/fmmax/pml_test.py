"""Tests for `fmmax.pml`.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import unittest

import jax.numpy as jnp
import numpy as onp
from jax import tree_util

from fmmax import basis, fields, fmm, layer, pml, scattering, sources


class FieldsInPMLDecayTest(unittest.TestCase):
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
        solve_result_ambient = layer.eigensolve_general_anisotropic_media(
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
